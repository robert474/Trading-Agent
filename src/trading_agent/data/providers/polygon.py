"""
Polygon.io data provider for real-time market data.
Handles WebSocket connections and REST API calls.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import AsyncIterator, Callable, Optional

import aiohttp
import structlog
import websockets
from websockets.exceptions import ConnectionClosed

from trading_agent.core.config import settings
from trading_agent.core.models import Candle

logger = structlog.get_logger()


class PolygonDataProvider:
    """
    Real-time market data provider using Polygon.io.

    Supports:
    - WebSocket streaming for real-time quotes and trades
    - REST API for historical data and snapshots
    - Automatic reconnection on disconnect
    """

    WEBSOCKET_URL = "wss://socket.polygon.io/stocks"
    REST_URL = "https://api.polygon.io"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.polygon_api_key
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._callbacks: list[Callable] = []
        self._subscribed_symbols: set[str] = set()
        self._running = False
        self._reconnect_delay = 1.0
        self._max_reconnect_delay = 60.0

    async def connect(self) -> None:
        """Establish WebSocket connection to Polygon."""
        logger.info("Connecting to Polygon WebSocket...")

        try:
            self._ws = await websockets.connect(
                self.WEBSOCKET_URL,
                ping_interval=30,
                ping_timeout=10,
            )

            # Authenticate
            auth_msg = json.dumps({"action": "auth", "params": self.api_key})
            await self._ws.send(auth_msg)

            # Wait for auth response
            response = await self._ws.recv()
            data = json.loads(response)

            if data[0].get("status") == "auth_success":
                logger.info("Polygon authentication successful")
                self._reconnect_delay = 1.0  # Reset delay on successful connect
            else:
                logger.error("Polygon authentication failed", response=data)
                raise ConnectionError(f"Auth failed: {data}")

        except Exception as e:
            logger.error("Failed to connect to Polygon", error=str(e))
            raise

    async def disconnect(self) -> None:
        """Close WebSocket connection."""
        self._running = False
        if self._ws:
            await self._ws.close()
            self._ws = None
        logger.info("Disconnected from Polygon")

    async def subscribe(self, symbols: list[str]) -> None:
        """
        Subscribe to real-time data for symbols.

        Subscribes to:
        - Q.{symbol} - Quotes (bid/ask)
        - T.{symbol} - Trades
        - AM.{symbol} - Minute aggregates
        """
        if not self._ws:
            await self.connect()

        # Build subscription channels
        channels = []
        for symbol in symbols:
            symbol = symbol.upper()
            channels.extend([f"Q.{symbol}", f"T.{symbol}", f"AM.{symbol}"])
            self._subscribed_symbols.add(symbol)

        sub_msg = json.dumps({"action": "subscribe", "params": ",".join(channels)})
        await self._ws.send(sub_msg)

        logger.info("Subscribed to symbols", symbols=symbols)

    async def unsubscribe(self, symbols: list[str]) -> None:
        """Unsubscribe from symbols."""
        if not self._ws:
            return

        channels = []
        for symbol in symbols:
            symbol = symbol.upper()
            channels.extend([f"Q.{symbol}", f"T.{symbol}", f"AM.{symbol}"])
            self._subscribed_symbols.discard(symbol)

        unsub_msg = json.dumps({"action": "unsubscribe", "params": ",".join(channels)})
        await self._ws.send(unsub_msg)

        logger.info("Unsubscribed from symbols", symbols=symbols)

    def add_callback(self, callback: Callable) -> None:
        """Add a callback function for incoming data."""
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable) -> None:
        """Remove a callback function."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    async def start_streaming(self) -> None:
        """Start processing incoming WebSocket messages."""
        self._running = True

        while self._running:
            try:
                if not self._ws:
                    await self.connect()
                    # Resubscribe to previously subscribed symbols
                    if self._subscribed_symbols:
                        await self.subscribe(list(self._subscribed_symbols))

                async for message in self._ws:
                    if not self._running:
                        break

                    data = json.loads(message)
                    await self._process_message(data)

            except ConnectionClosed:
                logger.warning(
                    "WebSocket connection closed, reconnecting...",
                    delay=self._reconnect_delay,
                )
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(
                    self._reconnect_delay * 2, self._max_reconnect_delay
                )
                self._ws = None

            except Exception as e:
                logger.error("Error in WebSocket stream", error=str(e))
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(
                    self._reconnect_delay * 2, self._max_reconnect_delay
                )
                self._ws = None

    async def _process_message(self, data: list) -> None:
        """Process incoming WebSocket messages."""
        for event in data:
            event_type = event.get("ev")

            if event_type == "Q":  # Quote
                quote = {
                    "type": "quote",
                    "symbol": event["sym"],
                    "bid": event.get("bp", 0),
                    "ask": event.get("ap", 0),
                    "bid_size": event.get("bs", 0),
                    "ask_size": event.get("as", 0),
                    "timestamp": datetime.fromtimestamp(event["t"] / 1000),
                }
                await self._emit("quote", quote)

            elif event_type == "T":  # Trade
                trade = {
                    "type": "trade",
                    "symbol": event["sym"],
                    "price": event["p"],
                    "size": event["s"],
                    "timestamp": datetime.fromtimestamp(event["t"] / 1000),
                }
                await self._emit("trade", trade)

            elif event_type == "AM":  # Minute aggregate
                candle = Candle(
                    symbol=event["sym"],
                    timeframe="1m",
                    timestamp=datetime.fromtimestamp(event["s"] / 1000),
                    open=event["o"],
                    high=event["h"],
                    low=event["l"],
                    close=event["c"],
                    volume=event["v"],
                )
                await self._emit("candle", candle)

            elif event_type == "status":
                logger.debug("Polygon status", message=event.get("message"))

    async def _emit(self, event_type: str, data: dict | Candle) -> None:
        """Emit event to all registered callbacks."""
        for callback in self._callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event_type, data)
                else:
                    callback(event_type, data)
            except Exception as e:
                logger.error(
                    "Error in callback", callback=callback.__name__, error=str(e)
                )

    # =========================================================================
    # REST API Methods
    # =========================================================================

    async def get_historical_candles(
        self,
        symbol: str,
        timeframe: str = "1",  # minutes
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        limit: int = 5000,
    ) -> list[Candle]:
        """
        Fetch historical candle data from Polygon REST API.

        Args:
            symbol: Stock symbol
            timeframe: Candle timeframe in minutes (1, 5, 15, 60, etc.)
            from_date: Start date (default: 30 days ago)
            to_date: End date (default: now)
            limit: Max candles to return

        Returns:
            List of Candle objects
        """
        if from_date is None:
            from_date = datetime.now() - timedelta(days=30)
        if to_date is None:
            to_date = datetime.now()

        # Convert timeframe to Polygon format
        multiplier = int(timeframe) if timeframe.isdigit() else 1
        timespan = "minute"

        if timeframe in ["60", "1h"]:
            multiplier = 1
            timespan = "hour"
        elif timeframe in ["D", "1D"]:
            multiplier = 1
            timespan = "day"

        url = (
            f"{self.REST_URL}/v2/aggs/ticker/{symbol.upper()}/range/"
            f"{multiplier}/{timespan}/"
            f"{from_date.strftime('%Y-%m-%d')}/{to_date.strftime('%Y-%m-%d')}"
        )

        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": limit,
            "apiKey": self.api_key,
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(
                        "Polygon API error",
                        status=resp.status,
                        error=error_text,
                    )
                    return []

                data = await resp.json()

        candles = []
        for bar in data.get("results", []):
            candles.append(
                Candle(
                    symbol=symbol.upper(),
                    timeframe=f"{multiplier}m" if timespan == "minute" else timeframe,
                    timestamp=datetime.fromtimestamp(bar["t"] / 1000),
                    open=bar["o"],
                    high=bar["h"],
                    low=bar["l"],
                    close=bar["c"],
                    volume=bar["v"],
                )
            )

        logger.info(
            "Fetched historical candles",
            symbol=symbol,
            count=len(candles),
            timeframe=timeframe,
        )

        return candles

    async def get_snapshot(self, symbol: str) -> Optional[dict]:
        """
        Get current market snapshot for a symbol.

        Returns quote, last trade, and day stats.
        """
        url = f"{self.REST_URL}/v2/snapshot/locale/us/markets/stocks/tickers/{symbol.upper()}"
        params = {"apiKey": self.api_key}

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    return None

                data = await resp.json()

        ticker = data.get("ticker", {})

        return {
            "symbol": symbol.upper(),
            "last_price": ticker.get("lastTrade", {}).get("p"),
            "bid": ticker.get("lastQuote", {}).get("p"),
            "ask": ticker.get("lastQuote", {}).get("P"),
            "day_open": ticker.get("day", {}).get("o"),
            "day_high": ticker.get("day", {}).get("h"),
            "day_low": ticker.get("day", {}).get("l"),
            "day_close": ticker.get("day", {}).get("c"),
            "day_volume": ticker.get("day", {}).get("v"),
            "prev_close": ticker.get("prevDay", {}).get("c"),
            "timestamp": datetime.now(),
        }

    async def get_multiple_snapshots(self, symbols: list[str]) -> dict[str, dict]:
        """Get snapshots for multiple symbols."""
        url = f"{self.REST_URL}/v2/snapshot/locale/us/markets/stocks/tickers"
        params = {
            "tickers": ",".join(s.upper() for s in symbols),
            "apiKey": self.api_key,
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    return {}

                data = await resp.json()

        snapshots = {}
        for ticker in data.get("tickers", []):
            symbol = ticker.get("ticker")
            snapshots[symbol] = {
                "symbol": symbol,
                "last_price": ticker.get("lastTrade", {}).get("p"),
                "bid": ticker.get("lastQuote", {}).get("p"),
                "ask": ticker.get("lastQuote", {}).get("P"),
                "day_volume": ticker.get("day", {}).get("v"),
                "prev_close": ticker.get("prevDay", {}).get("c"),
            }

        return snapshots

    async def get_options_chain(
        self,
        symbol: str,
        option_type: Optional[str] = None,  # 'call', 'put', or None for both
        expiration_gte: Optional[str] = None,  # YYYY-MM-DD
        expiration_lte: Optional[str] = None,
        strike_gte: Optional[float] = None,
        strike_lte: Optional[float] = None,
        limit: int = 100,
    ) -> list[dict]:
        """
        Fetch options chain from Polygon REST API.

        Args:
            symbol: Underlying stock symbol
            option_type: 'call', 'put', or None for both
            expiration_gte: Earliest expiration date
            expiration_lte: Latest expiration date
            strike_gte: Minimum strike price
            strike_lte: Maximum strike price
            limit: Max contracts to return

        Returns:
            List of option contract dicts
        """
        from datetime import timedelta

        url = f"{self.REST_URL}/v3/reference/options/contracts"

        # Default to options expiring in next 30 days if not specified
        if expiration_gte is None:
            expiration_gte = datetime.now().strftime("%Y-%m-%d")
        if expiration_lte is None:
            expiration_lte = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")

        params = {
            "underlying_ticker": symbol.upper(),
            "expiration_date.gte": expiration_gte,
            "expiration_date.lte": expiration_lte,
            "limit": limit,
            "apiKey": self.api_key,
        }

        if option_type:
            params["contract_type"] = option_type

        if strike_gte:
            params["strike_price.gte"] = strike_gte
        if strike_lte:
            params["strike_price.lte"] = strike_lte

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(
                        "Options chain API error",
                        status=resp.status,
                        error=error_text,
                    )
                    return []

                data = await resp.json()

        contracts = []
        for contract in data.get("results", []):
            contracts.append(
                {
                    "ticker": contract.get("ticker"),
                    "underlying": contract.get("underlying_ticker"),
                    "contract_type": contract.get("contract_type"),
                    "strike": contract.get("strike_price"),
                    "expiration": contract.get("expiration_date"),
                    "shares_per_contract": contract.get("shares_per_contract", 100),
                }
            )

        logger.info(
            "Fetched options chain",
            symbol=symbol,
            count=len(contracts),
            type=option_type,
        )

        return contracts

    async def get_option_quote(self, option_ticker: str) -> Optional[dict]:
        """
        Get current quote for an options contract.

        Args:
            option_ticker: OCC-style option symbol (e.g., O:SPY251219C00600000)

        Returns:
            Dict with bid, ask, last, greeks, etc.
        """
        url = f"{self.REST_URL}/v3/snapshot/options/{option_ticker}"
        params = {"apiKey": self.api_key}

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    return None

                data = await resp.json()

        result = data.get("results", {})
        if not result:
            return None

        return {
            "ticker": option_ticker,
            "underlying_price": result.get("underlying_asset", {}).get("price"),
            "bid": result.get("day", {}).get("bid"),
            "ask": result.get("day", {}).get("ask"),
            "last": result.get("day", {}).get("last"),
            "volume": result.get("day", {}).get("volume", 0),
            "open_interest": result.get("open_interest", 0),
            "implied_volatility": result.get("implied_volatility"),
            "delta": result.get("greeks", {}).get("delta"),
            "gamma": result.get("greeks", {}).get("gamma"),
            "theta": result.get("greeks", {}).get("theta"),
            "vega": result.get("greeks", {}).get("vega"),
        }

    async def get_options_near_price(
        self,
        symbol: str,
        current_price: float,
        option_type: str,  # 'call' or 'put'
        days_out: int = 7,
        num_strikes: int = 5,
    ) -> list[dict]:
        """
        Get options near the current price for strike selection.

        Fetches contracts around current price with expirations
        in the next `days_out` days.
        """
        from datetime import timedelta

        # Get nearest expiration (aim for 5-14 days out per Bill's methodology)
        min_days = max(5, days_out - 2)
        max_days = days_out + 7

        expiration_gte = (datetime.now() + timedelta(days=min_days)).strftime("%Y-%m-%d")
        expiration_lte = (datetime.now() + timedelta(days=max_days)).strftime("%Y-%m-%d")

        # Get strikes within 3% of current price
        strike_range = current_price * 0.03
        strike_gte = current_price - strike_range
        strike_lte = current_price + strike_range

        contracts = await self.get_options_chain(
            symbol=symbol,
            option_type=option_type,
            expiration_gte=expiration_gte,
            expiration_lte=expiration_lte,
            strike_gte=strike_gte,
            strike_lte=strike_lte,
            limit=50,
        )

        # Sort by distance to current price
        for c in contracts:
            c["distance_to_price"] = abs(c["strike"] - current_price)

        contracts.sort(key=lambda x: x["distance_to_price"])

        # Get quotes for top contracts
        results = []
        for contract in contracts[:num_strikes]:
            quote = await self.get_option_quote(contract["ticker"])
            if quote:
                contract.update(quote)
                results.append(contract)
            await asyncio.sleep(0.2)  # Rate limiting

        return results
