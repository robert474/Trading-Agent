#!/usr/bin/env python3
"""
Backtest the trading strategy on historical data.

Usage:
    python scripts/backtest.py --symbol AAPL --start 2024-01-01 --end 2024-12-01
"""

import argparse
import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

import structlog
from trading_agent.analysis.zone_detector import ZoneDetector
from trading_agent.core.models import TradeDirection, ZoneType
from trading_agent.data.providers.polygon import PolygonDataProvider
from trading_agent.monitoring.logger import setup_logging

logger = structlog.get_logger()


class BacktestEngine:
    """Simple backtesting engine for zone-based strategy."""

    def __init__(self):
        self.polygon = PolygonDataProvider()
        self.zone_detector = ZoneDetector()

        # Results tracking
        self.trades = []
        self.zones_detected = []

    async def run(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        initial_balance: float = 10000.0,
    ) -> dict:
        """
        Run backtest on historical data.

        Args:
            symbol: Stock symbol to backtest
            start_date: Backtest start date
            end_date: Backtest end date
            initial_balance: Starting balance

        Returns:
            Backtest results dict
        """
        logger.info(
            "Starting backtest",
            symbol=symbol,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
        )

        # Fetch historical data
        candles_15m = await self.polygon.get_historical_candles(
            symbol=symbol,
            timeframe="15",
            from_date=start_date,
            to_date=end_date,
            limit=50000,
        )

        if not candles_15m:
            logger.error("No historical data found")
            return {"error": "No data"}

        logger.info(f"Fetched {len(candles_15m)} candles")

        # Track balance and positions
        balance = initial_balance
        position = None
        wins = 0
        losses = 0

        # Walk through data
        for i in range(50, len(candles_15m)):
            # Get lookback window
            lookback = candles_15m[i-50:i]
            current_candle = candles_15m[i]
            current_price = current_candle.close

            # Detect zones on lookback
            zones = self.zone_detector.detect_zones(lookback, "15m")

            # Store new zones
            for zone in zones:
                if zone not in self.zones_detected:
                    self.zones_detected.append(zone)

            # Check for exit if we have a position
            if position:
                exit_signal = self._check_exit(position, current_price)
                if exit_signal:
                    # Calculate P&L
                    if position["direction"] == "long":
                        pnl = (current_price - position["entry"]) * position["quantity"]
                    else:
                        pnl = (position["entry"] - current_price) * position["quantity"]

                    balance += pnl

                    if pnl > 0:
                        wins += 1
                    else:
                        losses += 1

                    self.trades.append({
                        "symbol": symbol,
                        "direction": position["direction"],
                        "entry": position["entry"],
                        "exit": current_price,
                        "quantity": position["quantity"],
                        "pnl": pnl,
                        "exit_reason": exit_signal,
                        "entry_time": position["entry_time"],
                        "exit_time": current_candle.timestamp,
                    })

                    logger.info(
                        f"Trade closed: {position['direction']} @ {position['entry']:.2f} -> {current_price:.2f}, P&L: ${pnl:.2f}"
                    )

                    position = None
                    continue

            # Check for entry if no position
            if not position:
                entry_signal = self._check_entry(current_price, self.zones_detected)
                if entry_signal:
                    # Calculate position size (1% risk)
                    risk_per_share = abs(entry_signal["entry"] - entry_signal["stop"])
                    max_risk = balance * 0.01
                    quantity = int(max_risk / risk_per_share)

                    if quantity > 0:
                        position = {
                            "direction": entry_signal["direction"],
                            "entry": entry_signal["entry"],
                            "stop": entry_signal["stop"],
                            "target": entry_signal["target"],
                            "quantity": quantity,
                            "entry_time": current_candle.timestamp,
                        }

                        logger.info(
                            f"Entry signal: {entry_signal['direction']} @ {entry_signal['entry']:.2f}, "
                            f"stop: {entry_signal['stop']:.2f}, target: {entry_signal['target']:.2f}"
                        )

        # Calculate results
        total_trades = len(self.trades)
        win_rate = wins / total_trades * 100 if total_trades > 0 else 0
        total_pnl = balance - initial_balance
        return_pct = total_pnl / initial_balance * 100

        results = {
            "symbol": symbol,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "initial_balance": initial_balance,
            "final_balance": balance,
            "total_pnl": total_pnl,
            "return_pct": return_pct,
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "zones_detected": len(self.zones_detected),
            "trades": self.trades,
        }

        return results

    def _check_entry(self, current_price: float, zones: list) -> dict | None:
        """Check for entry at zones."""
        for zone in zones:
            if not zone.is_valid:
                continue

            distance_pct = abs(zone.distance_to_price(current_price)) / current_price * 100

            # Entry when price is within 0.3% of zone
            if distance_pct > 0.3:
                continue

            if zone.quality_score < 50:
                continue

            if zone.zone_type == ZoneType.DEMAND:
                # Long entry
                risk = current_price - zone.zone_low
                target = current_price + (risk * 2.5)  # 2.5:1 R:R

                return {
                    "direction": "long",
                    "entry": current_price,
                    "stop": zone.zone_low * 0.995,
                    "target": target,
                }

            elif zone.zone_type == ZoneType.SUPPLY:
                # Short entry
                risk = zone.zone_high - current_price
                target = current_price - (risk * 2.5)

                return {
                    "direction": "short",
                    "entry": current_price,
                    "stop": zone.zone_high * 1.005,
                    "target": target,
                }

        return None

    def _check_exit(self, position: dict, current_price: float) -> str | None:
        """Check exit conditions."""
        if position["direction"] == "long":
            if current_price <= position["stop"]:
                return "stop_loss"
            if current_price >= position["target"]:
                return "target"
        else:
            if current_price >= position["stop"]:
                return "stop_loss"
            if current_price <= position["target"]:
                return "target"

        return None


async def main():
    parser = argparse.ArgumentParser(description="Backtest trading strategy")
    parser.add_argument("--symbol", default="AAPL", help="Stock symbol")
    parser.add_argument("--start", default="2024-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2024-12-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--balance", type=float, default=10000, help="Initial balance")

    args = parser.parse_args()

    setup_logging(level="INFO")

    engine = BacktestEngine()
    results = await engine.run(
        symbol=args.symbol,
        start_date=datetime.strptime(args.start, "%Y-%m-%d"),
        end_date=datetime.strptime(args.end, "%Y-%m-%d"),
        initial_balance=args.balance,
    )

    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"Symbol: {results['symbol']}")
    print(f"Period: {results['start_date']} to {results['end_date']}")
    print(f"Initial Balance: ${results['initial_balance']:,.2f}")
    print(f"Final Balance: ${results['final_balance']:,.2f}")
    print(f"Total P&L: ${results['total_pnl']:+,.2f} ({results['return_pct']:+.2f}%)")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Win Rate: {results['win_rate']:.1f}%")
    print(f"Wins/Losses: {results['wins']}/{results['losses']}")
    print(f"Zones Detected: {results['zones_detected']}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
