"""
Tradier data provider for options chain data and brokerage operations.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional

import aiohttp
import structlog

from trading_agent.core.config import settings
from trading_agent.core.models import OptionContract

logger = structlog.get_logger()


class TradierDataProvider:
    """
    Tradier API client for options data and account management.

    Supports:
    - Options chain data with Greeks
    - Account positions and balances
    - Market quotes
    """

    def __init__(
        self,
        account_id: Optional[str] = None,
        access_token: Optional[str] = None,
        sandbox: Optional[bool] = None,
    ):
        self.account_id = account_id or settings.tradier_account_id
        self.access_token = access_token or settings.tradier_access_token
        self.sandbox = sandbox if sandbox is not None else settings.tradier_sandbox

        self.base_url = (
            "https://sandbox.tradier.com/v1"
            if self.sandbox
            else "https://api.tradier.com/v1"
        )

        self.headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Accept": "application/json",
        }

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[dict] = None,
        data: Optional[dict] = None,
    ) -> Optional[dict]:
        """Make authenticated request to Tradier API."""
        url = f"{self.base_url}{endpoint}"

        async with aiohttp.ClientSession() as session:
            async with session.request(
                method,
                url,
                headers=self.headers,
                params=params,
                data=data,
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(
                        "Tradier API error",
                        status=resp.status,
                        endpoint=endpoint,
                        error=error_text,
                    )
                    return None

                return await resp.json()

    # =========================================================================
    # Options Data
    # =========================================================================

    async def get_options_expirations(self, symbol: str) -> list[str]:
        """
        Get available options expiration dates for a symbol.

        Returns list of dates in YYYY-MM-DD format.
        """
        data = await self._request(
            "GET",
            "/markets/options/expirations",
            params={"symbol": symbol.upper()},
        )

        if not data:
            return []

        expirations = data.get("expirations", {}).get("date", [])
        if isinstance(expirations, str):
            expirations = [expirations]

        return expirations

    async def get_options_chain(
        self,
        symbol: str,
        expiration: Optional[str] = None,
        min_dte: int = 3,
        max_dte: int = 30,
    ) -> list[OptionContract]:
        """
        Get options chain with Greeks for a symbol.

        Args:
            symbol: Underlying symbol
            expiration: Specific expiration date (YYYY-MM-DD)
            min_dte: Minimum days to expiry (default: 3)
            max_dte: Maximum days to expiry (default: 30)

        Returns:
            List of OptionContract objects
        """
        today = datetime.now().date()

        # Get expirations if not specified
        if expiration:
            expirations = [expiration]
        else:
            all_expirations = await self.get_options_expirations(symbol)
            expirations = []
            for exp in all_expirations:
                exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
                dte = (exp_date - today).days
                if min_dte <= dte <= max_dte:
                    expirations.append(exp)

        if not expirations:
            logger.warning(
                "No valid expirations found",
                symbol=symbol,
                min_dte=min_dte,
                max_dte=max_dte,
            )
            return []

        all_options = []

        for exp in expirations:
            data = await self._request(
                "GET",
                "/markets/options/chains",
                params={
                    "symbol": symbol.upper(),
                    "expiration": exp,
                    "greeks": "true",
                },
            )

            if not data:
                continue

            options = data.get("options", {}).get("option", [])
            if isinstance(options, dict):
                options = [options]

            exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
            dte = (exp_date - today).days

            for opt in options:
                greeks = opt.get("greeks", {}) or {}

                contract = OptionContract(
                    symbol=opt["symbol"],
                    underlying=symbol.upper(),
                    option_type=opt["option_type"],
                    strike=opt["strike"],
                    expiration=exp,
                    days_to_expiry=dte,
                    bid=opt.get("bid", 0) or 0,
                    ask=opt.get("ask", 0) or 0,
                    last=opt.get("last", 0) or 0,
                    volume=opt.get("volume", 0) or 0,
                    open_interest=opt.get("open_interest", 0) or 0,
                    delta=greeks.get("delta", 0) or 0,
                    gamma=greeks.get("gamma", 0) or 0,
                    theta=greeks.get("theta", 0) or 0,
                    vega=greeks.get("vega", 0) or 0,
                    iv=greeks.get("mid_iv", 0) or 0,
                )
                all_options.append(contract)

        logger.info(
            "Fetched options chain",
            symbol=symbol,
            count=len(all_options),
            expirations=expirations,
        )

        return all_options

    async def select_optimal_contract(
        self,
        symbol: str,
        direction: str,  # 'long' or 'short'
        target_delta: float = 0.55,
        min_delta: float = 0.45,
        max_delta: float = 0.70,
        min_dte: int = 7,
        max_dte: int = 14,
        min_oi: int = 500,
        max_spread_pct: float = 0.10,
    ) -> Optional[OptionContract]:
        """
        Select optimal options contract based on Bill Fanter's criteria.

        Selection Rules:
        1. EXPIRATION: 7-14 DTE for short-term trades
        2. STRIKE: ATM or slightly ITM (delta 0.50-0.70)
        3. LIQUIDITY: Bid-ask spread <10%, OI >500

        Args:
            symbol: Underlying symbol
            direction: 'long' (buy calls) or 'short' (buy puts)
            target_delta: Ideal delta (default: 0.55)
            min_delta: Minimum acceptable delta
            max_delta: Maximum acceptable delta
            min_dte: Minimum days to expiry
            max_dte: Maximum days to expiry
            min_oi: Minimum open interest
            max_spread_pct: Maximum bid-ask spread as percentage

        Returns:
            Best matching OptionContract or None
        """
        chain = await self.get_options_chain(
            symbol, min_dte=min_dte, max_dte=max_dte
        )

        if not chain:
            return None

        # Filter by option type (calls for long, puts for short)
        option_type = "call" if direction == "long" else "put"
        candidates = [c for c in chain if c.option_type == option_type]

        # Filter by delta
        candidates = [
            c for c in candidates
            if min_delta <= abs(c.delta) <= max_delta
        ]

        # Filter by liquidity
        candidates = [
            c for c in candidates
            if c.open_interest >= min_oi and c.spread_percent < max_spread_pct
        ]

        if not candidates:
            logger.warning(
                "No suitable contracts found",
                symbol=symbol,
                direction=direction,
            )
            return None

        # Score remaining candidates
        scored = []
        for contract in candidates:
            score = 0.0

            # Delta score: prefer delta around target
            delta_diff = abs(abs(contract.delta) - target_delta)
            delta_score = 100 - (delta_diff * 200)
            score += delta_score

            # Spread score: prefer tighter spreads
            spread_score = (max_spread_pct - contract.spread_percent) * 500
            score += spread_score

            # OI score: prefer higher open interest
            oi_score = min(50, contract.open_interest / 100)
            score += oi_score

            # DTE score: prefer middle of range
            mid_dte = (min_dte + max_dte) / 2
            dte_diff = abs(contract.days_to_expiry - mid_dte)
            dte_score = 20 - dte_diff
            score += dte_score

            scored.append((score, contract))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        best = scored[0][1]
        logger.info(
            "Selected optimal contract",
            symbol=symbol,
            contract=best.symbol,
            strike=best.strike,
            expiration=best.expiration,
            delta=best.delta,
            spread_pct=f"{best.spread_percent:.2%}",
        )

        return best

    # =========================================================================
    # Market Data
    # =========================================================================

    async def get_quote(self, symbol: str) -> Optional[dict]:
        """Get current quote for a symbol."""
        data = await self._request(
            "GET",
            "/markets/quotes",
            params={"symbols": symbol.upper()},
        )

        if not data:
            return None

        quote = data.get("quotes", {}).get("quote")
        if not quote:
            return None

        return {
            "symbol": quote["symbol"],
            "last": quote.get("last"),
            "bid": quote.get("bid"),
            "ask": quote.get("ask"),
            "open": quote.get("open"),
            "high": quote.get("high"),
            "low": quote.get("low"),
            "close": quote.get("close"),
            "volume": quote.get("volume"),
            "change": quote.get("change"),
            "change_percent": quote.get("change_percentage"),
        }

    async def get_quotes(self, symbols: list[str]) -> dict[str, dict]:
        """Get quotes for multiple symbols."""
        data = await self._request(
            "GET",
            "/markets/quotes",
            params={"symbols": ",".join(s.upper() for s in symbols)},
        )

        if not data:
            return {}

        quotes_data = data.get("quotes", {}).get("quote", [])
        if isinstance(quotes_data, dict):
            quotes_data = [quotes_data]

        quotes = {}
        for q in quotes_data:
            quotes[q["symbol"]] = {
                "symbol": q["symbol"],
                "last": q.get("last"),
                "bid": q.get("bid"),
                "ask": q.get("ask"),
                "volume": q.get("volume"),
                "change_percent": q.get("change_percentage"),
            }

        return quotes

    # =========================================================================
    # Account Data
    # =========================================================================

    async def get_account_balance(self) -> Optional[dict]:
        """Get account balance information."""
        data = await self._request(
            "GET",
            f"/accounts/{self.account_id}/balances",
        )

        if not data:
            return None

        balances = data.get("balances", {})

        return {
            "total_equity": balances.get("total_equity"),
            "total_cash": balances.get("total_cash"),
            "option_buying_power": balances.get("option_buying_power"),
            "stock_buying_power": balances.get("stock_buying_power"),
            "pending_orders_count": balances.get("pending_orders_count"),
        }

    async def get_positions(self) -> list[dict]:
        """Get all open positions."""
        data = await self._request(
            "GET",
            f"/accounts/{self.account_id}/positions",
        )

        if not data:
            return []

        positions = data.get("positions", {}).get("position", [])
        if isinstance(positions, dict):
            positions = [positions]
        if positions is None:
            positions = []

        return [
            {
                "symbol": p["symbol"],
                "quantity": p["quantity"],
                "cost_basis": p["cost_basis"],
                "date_acquired": p.get("date_acquired"),
            }
            for p in positions
        ]

    async def get_orders(self, status: str = "pending") -> list[dict]:
        """
        Get orders by status.

        Status options: pending, open, filled, cancelled, expired, rejected, all
        """
        data = await self._request(
            "GET",
            f"/accounts/{self.account_id}/orders",
        )

        if not data:
            return []

        orders = data.get("orders", {}).get("order", [])
        if isinstance(orders, dict):
            orders = [orders]
        if orders is None:
            orders = []

        if status != "all":
            orders = [o for o in orders if o.get("status") == status]

        return orders
