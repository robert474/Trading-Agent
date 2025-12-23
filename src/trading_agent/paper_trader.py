"""
Paper Trading Engine - Simulates live trading with Bill Fanter's methodology.

This module combines:
1. Real-time market data from Polygon (websocket or polling)
2. Zone detection with confirmation filters
3. Options chain integration for contract selection
4. Position tracking and P&L simulation

Usage:
    # WebSocket mode (real-time)
    python -m trading_agent.paper_trader --watchlist SPY,QQQ,AAPL,NVDA

    # Polling mode (simpler, good for testing)
    python -m trading_agent.paper_trader --watchlist SPY,QQQ --polling --interval 30
"""

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from uuid import uuid4

import structlog

from trading_agent.analysis.zone_detector import ZoneDetector
from trading_agent.core.config import settings
from trading_agent.core.models import (
    Candle,
    OptionContract,
    TradeDirection,
    TradeSignal,
    Zone,
    ZoneType,
)
from trading_agent.data.providers.polygon import PolygonDataProvider

logger = structlog.get_logger()


class PaperPosition:
    """Represents a simulated position."""

    def __init__(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        quantity: int,
        stop_loss: float,
        target_price: float,
        confirmation_score: int,
        confirmations: list[str],
        option_contract: Optional[dict] = None,
    ):
        self.id = str(uuid4())[:8]
        self.symbol = symbol
        self.direction = direction  # "long" or "short"
        self.entry_price = entry_price
        self.quantity = quantity
        self.stop_loss = stop_loss
        self.target_price = target_price
        self.confirmation_score = confirmation_score
        self.confirmations = confirmations
        self.entry_time = datetime.now()
        self.current_price = entry_price
        self.unrealized_pnl = 0.0
        self.unrealized_pnl_pct = 0.0
        self.highest_price = entry_price  # For trailing stop
        self.lowest_price = entry_price
        self.trailing_stop = None

        # Options-specific fields
        self.option_contract = option_contract
        self.option_ticker = option_contract.get("ticker") if option_contract else None
        self.option_strike = option_contract.get("strike") if option_contract else None
        self.option_expiry = option_contract.get("expiration") if option_contract else None
        self.option_entry_premium = option_contract.get("ask") if option_contract else None
        self.option_current_premium = self.option_entry_premium

    def update_price(self, price: float) -> dict:
        """Update position with new price and check exits."""
        self.current_price = price

        # Track high/low for trailing stop
        if price > self.highest_price:
            self.highest_price = price
        if price < self.lowest_price:
            self.lowest_price = price

        # Calculate P&L
        if self.direction == "long":
            self.unrealized_pnl = (price - self.entry_price) * self.quantity
            self.unrealized_pnl_pct = (price - self.entry_price) / self.entry_price * 100
        else:
            self.unrealized_pnl = (self.entry_price - price) * self.quantity
            self.unrealized_pnl_pct = (self.entry_price - price) / self.entry_price * 100

        # Update trailing stop after 50% of target reached
        target_move = abs(self.target_price - self.entry_price)
        current_move = abs(price - self.entry_price)

        if current_move >= target_move * 0.5:
            # Activate trailing stop at 50% of profit
            if self.direction == "long":
                new_trail = price - (target_move * 0.25)  # Trail 25% of target move
                if self.trailing_stop is None or new_trail > self.trailing_stop:
                    self.trailing_stop = new_trail
            else:
                new_trail = price + (target_move * 0.25)
                if self.trailing_stop is None or new_trail < self.trailing_stop:
                    self.trailing_stop = new_trail

        # Check exit conditions
        return self._check_exit(price)

    def _check_exit(self, price: float) -> dict:
        """Check if position should be exited."""
        result = {"should_exit": False, "reason": None, "exit_price": price}

        is_long = self.direction == "long"

        # 1. STOP LOSS (highest priority)
        if is_long and price <= self.stop_loss:
            result = {"should_exit": True, "reason": "STOP LOSS", "exit_price": price}
        elif not is_long and price >= self.stop_loss:
            result = {"should_exit": True, "reason": "STOP LOSS", "exit_price": price}

        # 2. TARGET
        elif is_long and price >= self.target_price:
            result = {"should_exit": True, "reason": "TARGET HIT", "exit_price": price}
        elif not is_long and price <= self.target_price:
            result = {"should_exit": True, "reason": "TARGET HIT", "exit_price": price}

        # 3. TRAILING STOP
        elif self.trailing_stop:
            if is_long and price <= self.trailing_stop:
                result = {"should_exit": True, "reason": "TRAILING STOP", "exit_price": price}
            elif not is_long and price >= self.trailing_stop:
                result = {"should_exit": True, "reason": "TRAILING STOP", "exit_price": price}

        return result

    def to_dict(self) -> dict:
        result = {
            "id": self.id,
            "symbol": self.symbol,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "quantity": self.quantity,
            "stop_loss": self.stop_loss,
            "target_price": self.target_price,
            "trailing_stop": self.trailing_stop,
            "unrealized_pnl": self.unrealized_pnl,
            "unrealized_pnl_pct": self.unrealized_pnl_pct,
            "confirmation_score": self.confirmation_score,
            "confirmations": self.confirmations,
            "entry_time": self.entry_time.isoformat(),
        }

        # Add options data if present
        if self.option_ticker:
            result.update(
                {
                    "option_ticker": self.option_ticker,
                    "option_strike": self.option_strike,
                    "option_expiry": self.option_expiry,
                    "option_entry_premium": self.option_entry_premium,
                    "option_current_premium": self.option_current_premium,
                }
            )

        return result


class PaperTrader:
    """
    Paper trading engine using Bill Fanter's methodology.

    Features:
    - Real-time price monitoring via Polygon (websocket or polling)
    - Automatic zone detection and confirmation checking
    - Options chain integration for contract selection
    - Position management with stops and targets
    - P&L tracking and trade journaling
    """

    def __init__(
        self,
        watchlist: list[str],
        starting_capital: float = 10000.0,
        position_size_pct: float = 5.0,  # 5% per trade
        min_confirmation_score: int = 50,  # Require 50+ for trades
        data_dir: str = "data/paper_trading",
        use_polling: bool = False,  # Use polling instead of websocket
        poll_interval: int = 30,  # Seconds between polls
        trade_options: bool = True,  # Trade options vs underlying
    ):
        self.watchlist = [s.upper() for s in watchlist]
        self.starting_capital = starting_capital
        self.capital = starting_capital
        self.position_size_pct = position_size_pct
        self.min_confirmation_score = min_confirmation_score
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Mode settings
        self.use_polling = use_polling
        self.poll_interval = poll_interval
        self.trade_options = trade_options

        # Components
        self.polygon = PolygonDataProvider()
        self.zone_detector = ZoneDetector()

        # State
        self.positions: dict[str, PaperPosition] = {}
        self.closed_trades: list[dict] = []
        self.candle_buffers: dict[str, list[Candle]] = {s: [] for s in self.watchlist}
        self.current_prices: dict[str, float] = {}
        self.detected_zones: dict[str, list[Zone]] = {s: [] for s in self.watchlist}

        # Stats
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0

        self._running = False

    async def start(self) -> None:
        """Start the paper trading engine."""
        logger.info(
            "Starting Paper Trader",
            watchlist=self.watchlist,
            capital=self.capital,
            min_confirmation=self.min_confirmation_score,
            mode="polling" if self.use_polling else "websocket",
            trade_options=self.trade_options,
        )

        self._running = True

        # Load historical candles for zone detection
        await self._load_historical_data()

        # Detect initial zones
        for symbol in self.watchlist:
            self._detect_zones(symbol)

        # Start monitoring based on mode
        try:
            if self.use_polling:
                # Polling mode - simpler, good for testing
                await self._polling_loop()
            else:
                # WebSocket mode - real-time streaming
                await self.polygon.connect()
                await self.polygon.subscribe(self.watchlist)
                self.polygon.add_callback(self._on_market_data)

                await asyncio.gather(
                    self.polygon.start_streaming(),
                    self._monitoring_loop(),
                )
        except KeyboardInterrupt:
            logger.info("Shutting down paper trader...")
        finally:
            await self.stop()

    async def _polling_loop(self) -> None:
        """Poll for price updates instead of streaming."""
        logger.info(f"Starting polling loop (interval: {self.poll_interval}s)")

        while self._running:
            try:
                # Get snapshots for all symbols
                snapshots = await self.polygon.get_multiple_snapshots(self.watchlist)

                for symbol, snapshot in snapshots.items():
                    price = snapshot.get("last_price")
                    if price:
                        self.current_prices[symbol] = price

                        # Check positions
                        await self._check_positions(symbol, price)

                # Check for new setups
                for symbol in self.watchlist:
                    await self._check_setups(symbol)

                # Print status
                self._print_positions_summary()

            except Exception as e:
                logger.error(f"Polling error: {e}")

            await asyncio.sleep(self.poll_interval)

    async def stop(self) -> None:
        """Stop the paper trading engine."""
        self._running = False
        await self.polygon.disconnect()
        self._save_state()
        logger.info("Paper trader stopped")

    async def _load_historical_data(self) -> None:
        """Load historical candles for zone detection."""
        logger.info("Loading historical data...")

        for symbol in self.watchlist:
            try:
                # Get last 30 days of 15-minute candles
                candles = await self.polygon.get_historical_candles(
                    symbol=symbol,
                    timeframe="15",
                    from_date=datetime.now() - timedelta(days=30),
                    to_date=datetime.now(),
                )
                self.candle_buffers[symbol] = candles
                logger.info(f"Loaded {len(candles)} candles for {symbol}")

            except Exception as e:
                logger.error(f"Failed to load history for {symbol}: {e}")

            await asyncio.sleep(0.5)  # Rate limiting

    def _detect_zones(self, symbol: str) -> None:
        """Detect supply/demand zones for a symbol."""
        candles = self.candle_buffers.get(symbol, [])
        if len(candles) < 20:
            return

        zones = self.zone_detector.detect_zones(candles, "15m")
        self.detected_zones[symbol] = zones

        logger.info(
            f"Detected {len(zones)} zones for {symbol}",
            demand=[z for z in zones if z.zone_type == ZoneType.DEMAND],
            supply=[z for z in zones if z.zone_type == ZoneType.SUPPLY],
        )

    async def _on_market_data(self, event_type: str, data) -> None:
        """Handle incoming market data."""
        if event_type == "trade":
            symbol = data["symbol"]
            price = data["price"]
            self.current_prices[symbol] = price

            # Check positions
            await self._check_positions(symbol, price)

        elif event_type == "candle":
            candle = data
            symbol = candle.symbol

            # Update candle buffer
            if symbol in self.candle_buffers:
                self.candle_buffers[symbol].append(candle)
                # Keep last 500 candles
                if len(self.candle_buffers[symbol]) > 500:
                    self.candle_buffers[symbol] = self.candle_buffers[symbol][-500:]

            # Re-detect zones every 10 candles
            if len(self.candle_buffers[symbol]) % 10 == 0:
                self._detect_zones(symbol)

            # Check for new trade setups
            await self._check_setups(symbol)

    async def _check_positions(self, symbol: str, price: float) -> None:
        """Check and update positions for a symbol."""
        positions_to_close = []

        for pos_id, position in self.positions.items():
            if position.symbol != symbol:
                continue

            exit_signal = position.update_price(price)

            if exit_signal["should_exit"]:
                positions_to_close.append((pos_id, exit_signal))

        # Close positions that hit exit conditions
        for pos_id, exit_signal in positions_to_close:
            await self._close_position(pos_id, exit_signal["exit_price"], exit_signal["reason"])

    async def _check_setups(self, symbol: str) -> None:
        """Check for new trade setups on a symbol."""
        # Skip if we already have a position in this symbol
        if any(p.symbol == symbol for p in self.positions.values()):
            return

        # Skip if we don't have enough capital
        position_value = self.capital * (self.position_size_pct / 100)
        if position_value < 100:
            return

        current_price = self.current_prices.get(symbol)
        if not current_price:
            return

        zones = self.detected_zones.get(symbol, [])
        candles = self.candle_buffers.get(symbol, [])

        if len(candles) < 10:
            return

        # Find nearest zones
        nearest = self.zone_detector.find_nearest_zones(current_price, zones)

        # Check for setups at demand zones (long)
        demand_zone = nearest.get("nearest_demand")
        if demand_zone:
            setup = self.zone_detector.check_entry_conditions(
                current_price=current_price,
                zone=demand_zone,
                htf_trend="neutral",  # Could integrate HTF analysis
                recent_candles=candles[-10:],
            )

            if setup["has_setup"] and setup["confirmation_score"] >= self.min_confirmation_score:
                await self._enter_trade(
                    symbol=symbol,
                    direction="long",
                    entry_price=current_price,
                    stop_loss=setup["stop_loss"],
                    zone=demand_zone,
                    confirmation_score=setup["confirmation_score"],
                    confirmations=setup["confirmations"],
                )

        # Check for setups at supply zones (short)
        supply_zone = nearest.get("nearest_supply")
        if supply_zone:
            setup = self.zone_detector.check_entry_conditions(
                current_price=current_price,
                zone=supply_zone,
                htf_trend="neutral",
                recent_candles=candles[-10:],
            )

            if setup["has_setup"] and setup["confirmation_score"] >= self.min_confirmation_score:
                await self._enter_trade(
                    symbol=symbol,
                    direction="short",
                    entry_price=current_price,
                    stop_loss=setup["stop_loss"],
                    zone=supply_zone,
                    confirmation_score=setup["confirmation_score"],
                    confirmations=setup["confirmations"],
                )

    async def _enter_trade(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        zone: Zone,
        confirmation_score: int,
        confirmations: list[str],
    ) -> None:
        """Enter a new paper trade."""
        # Calculate position size
        position_value = self.capital * (self.position_size_pct / 100)

        # Calculate target (2:1 R:R minimum)
        risk = abs(entry_price - stop_loss)
        if direction == "long":
            target_price = entry_price + (risk * 2)
        else:
            target_price = entry_price - (risk * 2)

        # Fetch options contract if trading options
        option_contract = None
        if self.trade_options:
            option_type = "call" if direction == "long" else "put"
            try:
                options = await self.polygon.get_options_near_price(
                    symbol=symbol,
                    current_price=entry_price,
                    option_type=option_type,
                    days_out=7,  # Bill prefers 5-14 DTE
                    num_strikes=3,
                )

                if options:
                    # Select the most liquid ATM option
                    for opt in options:
                        if opt.get("bid") and opt.get("ask"):
                            spread_pct = (opt["ask"] - opt["bid"]) / opt["ask"]
                            if spread_pct < 0.10:  # Require tight spread
                                option_contract = opt
                                break

                    if option_contract:
                        # For options, quantity = contracts, cost = premium * 100
                        premium = option_contract.get("ask", 0)
                        quantity = max(1, int(position_value / (premium * 100)))
                        logger.info(
                            f"Selected option: {option_contract.get('ticker')}",
                            strike=option_contract.get("strike"),
                            expiry=option_contract.get("expiration"),
                            premium=premium,
                            delta=option_contract.get("delta"),
                        )
                    else:
                        logger.warning(f"No liquid options found for {symbol}, using shares")

            except Exception as e:
                logger.warning(f"Failed to fetch options for {symbol}: {e}")

        # Fallback to shares if no option contract
        if option_contract is None:
            quantity = int(position_value / entry_price)

        if quantity < 1:
            return

        position = PaperPosition(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            quantity=quantity,
            stop_loss=stop_loss,
            target_price=target_price,
            confirmation_score=confirmation_score,
            confirmations=confirmations,
            option_contract=option_contract,
        )

        self.positions[position.id] = position
        self.total_trades += 1

        logger.info(
            "PAPER TRADE ENTERED",
            id=position.id,
            symbol=symbol,
            direction=direction,
            entry=entry_price,
            stop=stop_loss,
            target=target_price,
            quantity=quantity,
            confirmation=confirmation_score,
            option=option_contract.get("ticker") if option_contract else None,
        )

        self._print_position(position)

    async def _close_position(self, position_id: str, exit_price: float, reason: str) -> None:
        """Close a paper position."""
        position = self.positions.pop(position_id, None)
        if not position:
            return

        # Calculate final P&L
        if position.direction == "long":
            pnl = (exit_price - position.entry_price) * position.quantity
            pnl_pct = (exit_price - position.entry_price) / position.entry_price * 100
        else:
            pnl = (position.entry_price - exit_price) * position.quantity
            pnl_pct = (position.entry_price - exit_price) / position.entry_price * 100

        # Update stats
        self.total_pnl += pnl
        self.capital += pnl
        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

        # Record trade
        trade_record = {
            "id": position.id,
            "symbol": position.symbol,
            "direction": position.direction,
            "entry_price": position.entry_price,
            "exit_price": exit_price,
            "quantity": position.quantity,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "confirmation_score": position.confirmation_score,
            "confirmations": position.confirmations,
            "exit_reason": reason,
            "entry_time": position.entry_time.isoformat(),
            "exit_time": datetime.now().isoformat(),
            "duration_minutes": (datetime.now() - position.entry_time).total_seconds() / 60,
        }
        self.closed_trades.append(trade_record)

        win_rate = self.winning_trades / max(1, self.total_trades) * 100

        logger.info(
            f"PAPER TRADE CLOSED - {reason}",
            id=position.id,
            symbol=position.symbol,
            pnl=f"${pnl:+.2f}",
            pnl_pct=f"{pnl_pct:+.2f}%",
            total_pnl=f"${self.total_pnl:+.2f}",
            win_rate=f"{win_rate:.1f}%",
        )

        self._save_state()

    async def _monitoring_loop(self) -> None:
        """Periodic monitoring and status updates."""
        while self._running:
            await asyncio.sleep(60)  # Every minute

            if self.positions:
                self._print_positions_summary()

    def _print_position(self, position: PaperPosition) -> None:
        """Print position details to console."""
        print("\n" + "=" * 60)
        print(f"NEW PAPER TRADE: {position.symbol} {position.direction.upper()}")
        print("=" * 60)
        print(f"  Entry:  ${position.entry_price:.2f}")
        print(f"  Stop:   ${position.stop_loss:.2f}")
        print(f"  Target: ${position.target_price:.2f}")
        print(f"  Qty:    {position.quantity}")
        print(f"  Conf:   {position.confirmation_score}/100")
        print(f"  Signals: {', '.join(position.confirmations)}")

        if position.option_ticker:
            print(f"  Option: {position.option_ticker}")
            print(f"  Strike: ${position.option_strike:.2f}")
            print(f"  Expiry: {position.option_expiry}")
            print(f"  Premium: ${position.option_entry_premium:.2f}")

        print("=" * 60 + "\n")

    def _print_positions_summary(self) -> None:
        """Print summary of all positions."""
        if not self.positions:
            return

        print("\n" + "-" * 60)
        print(f"OPEN POSITIONS ({len(self.positions)})")
        print("-" * 60)

        for pos in self.positions.values():
            arrow = "" if pos.unrealized_pnl >= 0 else ""
            print(
                f"  {pos.symbol:6} {pos.direction:5} | "
                f"Entry: ${pos.entry_price:.2f} | "
                f"Now: ${pos.current_price:.2f} | "
                f"P&L: {arrow} ${pos.unrealized_pnl:+.2f} ({pos.unrealized_pnl_pct:+.1f}%)"
            )

        print("-" * 60)
        print(f"Capital: ${self.capital:.2f} | Total P&L: ${self.total_pnl:+.2f}")
        print("-" * 60 + "\n")

    def _save_state(self) -> None:
        """Save trading state to disk."""
        state = {
            "timestamp": datetime.now().isoformat(),
            "capital": self.capital,
            "starting_capital": self.starting_capital,
            "total_pnl": self.total_pnl,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.winning_trades / max(1, self.total_trades) * 100,
            "open_positions": [p.to_dict() for p in self.positions.values()],
            "closed_trades": self.closed_trades,
        }

        state_file = self.data_dir / "paper_trading_state.json"
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)

    def get_stats(self) -> dict:
        """Get trading statistics."""
        win_rate = self.winning_trades / max(1, self.total_trades) * 100
        avg_win = 0
        avg_loss = 0

        wins = [t for t in self.closed_trades if t["pnl"] > 0]
        losses = [t for t in self.closed_trades if t["pnl"] <= 0]

        if wins:
            avg_win = sum(t["pnl"] for t in wins) / len(wins)
        if losses:
            avg_loss = sum(t["pnl"] for t in losses) / len(losses)

        return {
            "capital": self.capital,
            "total_pnl": self.total_pnl,
            "total_pnl_pct": (self.capital - self.starting_capital) / self.starting_capital * 100,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "open_positions": len(self.positions),
        }


async def main():
    """Run paper trader from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Bill Fanter Paper Trader")
    parser.add_argument(
        "--watchlist",
        type=str,
        default="SPY,QQQ,AAPL,NVDA,TSLA,META,GOOGL,AMZN,MSFT,AMD",
        help="Comma-separated list of symbols to trade",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=10000.0,
        help="Starting paper trading capital",
    )
    parser.add_argument(
        "--position-size",
        type=float,
        default=5.0,
        help="Position size as percentage of capital",
    )
    parser.add_argument(
        "--min-confirmation",
        type=int,
        default=50,
        help="Minimum confirmation score to take trades (0-100)",
    )
    parser.add_argument(
        "--polling",
        action="store_true",
        help="Use polling instead of websocket (simpler, good for testing)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Polling interval in seconds (only used with --polling)",
    )
    parser.add_argument(
        "--no-options",
        action="store_true",
        help="Trade underlying shares instead of options",
    )

    args = parser.parse_args()

    watchlist = [s.strip() for s in args.watchlist.split(",")]

    trader = PaperTrader(
        watchlist=watchlist,
        starting_capital=args.capital,
        position_size_pct=args.position_size,
        min_confirmation_score=args.min_confirmation,
        use_polling=args.polling,
        poll_interval=args.interval,
        trade_options=not args.no_options,
    )

    mode = "POLLING" if args.polling else "WEBSOCKET"
    instrument = "SHARES" if args.no_options else "OPTIONS"

    print("\n" + "=" * 60)
    print("BILL FANTER PAPER TRADER")
    print("=" * 60)
    print(f"Mode: {mode}")
    print(f"Instrument: {instrument}")
    print(f"Watchlist: {', '.join(watchlist)}")
    print(f"Capital: ${args.capital:,.2f}")
    print(f"Position Size: {args.position_size}%")
    print(f"Min Confirmation: {args.min_confirmation}/100")
    if args.polling:
        print(f"Poll Interval: {args.interval}s")
    print("=" * 60)
    print("Starting... Press Ctrl+C to stop\n")

    await trader.start()


if __name__ == "__main__":
    asyncio.run(main())
