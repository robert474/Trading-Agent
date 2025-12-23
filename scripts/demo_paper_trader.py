#!/usr/bin/env python3
"""
Demo Paper Trader - Shows what the paper trader does using historical data.

This script simulates paper trading by replaying recent price action,
so you can see how the system detects zones, checks confirmations,
and would enter/exit trades.

Usage:
    python scripts/demo_paper_trader.py
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trading_agent.analysis.zone_detector import ZoneDetector
from trading_agent.data.providers.polygon import PolygonDataProvider
from trading_agent.core.models import ZoneType


async def demo():
    """Run a demo of the paper trading system."""
    print("\n" + "=" * 70)
    print("              BILL FANTER PAPER TRADER - DEMO MODE")
    print("=" * 70)
    print("This demo shows how the paper trader analyzes markets and finds trades.")
    print("=" * 70 + "\n")

    polygon = PolygonDataProvider()
    zone_detector = ZoneDetector()

    symbols = ["SPY", "QQQ", "AAPL", "NVDA"]

    for symbol in symbols:
        print(f"\n{'='*70}")
        print(f"  ANALYZING: {symbol}")
        print(f"{'='*70}")

        # Fetch historical candles
        print(f"\n  [1] Fetching 15-minute candles for last 30 days...")
        candles = await polygon.get_historical_candles(
            symbol=symbol,
            timeframe="15",
            from_date=datetime.now() - timedelta(days=30),
            to_date=datetime.now(),
        )

        if not candles:
            print(f"      No data available for {symbol}")
            continue

        print(f"      Loaded {len(candles)} candles")

        # Current price (last candle close)
        current_price = candles[-1].close
        print(f"      Last price: ${current_price:.2f}")

        # Detect zones
        print(f"\n  [2] Detecting supply/demand zones...")
        zones = zone_detector.detect_zones(candles, "15m")

        supply_zones = [z for z in zones if z.zone_type == ZoneType.SUPPLY]
        demand_zones = [z for z in zones if z.zone_type == ZoneType.DEMAND]

        print(f"      Found {len(supply_zones)} supply zones (resistance)")
        print(f"      Found {len(demand_zones)} demand zones (support)")

        # Show nearest zones
        nearest = zone_detector.find_nearest_zones(current_price, zones)

        print(f"\n  [3] Key zones near current price (${current_price:.2f}):")

        if nearest.get("nearest_demand"):
            z = nearest["nearest_demand"]
            dist_pct = abs(current_price - z.zone_high) / current_price * 100
            print(f"\n      NEAREST DEMAND (support):")
            print(f"        Zone: ${z.zone_low:.2f} - ${z.zone_high:.2f}")
            print(f"        Distance: {dist_pct:.1f}% below current price")
            print(f"        Quality: {z.quality_score:.0f}/100")
            print(f"        Status: {z.freshness.value.upper()}")

        if nearest.get("nearest_supply"):
            z = nearest["nearest_supply"]
            dist_pct = abs(current_price - z.zone_low) / current_price * 100
            print(f"\n      NEAREST SUPPLY (resistance):")
            print(f"        Zone: ${z.zone_low:.2f} - ${z.zone_high:.2f}")
            print(f"        Distance: {dist_pct:.1f}% above current price")
            print(f"        Quality: {z.quality_score:.0f}/100")
            print(f"        Status: {z.freshness.value.upper()}")

        # Check for current setups
        print(f"\n  [4] Checking for trade setups...")

        setup_found = False

        if nearest.get("nearest_demand"):
            z = nearest["nearest_demand"]
            setup = zone_detector.check_entry_conditions(
                current_price=current_price,
                zone=z,
                htf_trend="neutral",
                recent_candles=candles[-10:],
            )

            if setup["has_setup"]:
                setup_found = True
                print(f"\n      {'*'*50}")
                print(f"      LONG SETUP AT DEMAND ZONE!")
                print(f"      {'*'*50}")
                print(f"        Entry Zone: ${z.zone_low:.2f} - ${z.zone_high:.2f}")
                print(f"        Stop Loss: ${setup['stop_loss']:.2f}")
                print(f"        Confirmation Score: {setup['confirmation_score']}/100")
                print(f"        Signals: {', '.join(setup['confirmations']) if setup['confirmations'] else 'None'}")

                if setup['confirmation_score'] >= 50:
                    print(f"\n        >> WOULD ENTER TRADE (score >= 50)")
                else:
                    print(f"\n        >> Would skip (score < 50)")

        if nearest.get("nearest_supply"):
            z = nearest["nearest_supply"]
            setup = zone_detector.check_entry_conditions(
                current_price=current_price,
                zone=z,
                htf_trend="neutral",
                recent_candles=candles[-10:],
            )

            if setup["has_setup"]:
                setup_found = True
                print(f"\n      {'*'*50}")
                print(f"      SHORT SETUP AT SUPPLY ZONE!")
                print(f"      {'*'*50}")
                print(f"        Entry Zone: ${z.zone_low:.2f} - ${z.zone_high:.2f}")
                print(f"        Stop Loss: ${setup['stop_loss']:.2f}")
                print(f"        Confirmation Score: {setup['confirmation_score']}/100")
                print(f"        Signals: {', '.join(setup['confirmations']) if setup['confirmations'] else 'None'}")

                if setup['confirmation_score'] >= 50:
                    print(f"\n        >> WOULD ENTER TRADE (score >= 50)")
                else:
                    print(f"\n        >> Would skip (score < 50)")

        if not setup_found:
            print(f"\n      No active setups - price not at a zone")
            print(f"      Waiting for price to reach a supply or demand zone...")

        await asyncio.sleep(0.5)  # Rate limiting

    print("\n" + "=" * 70)
    print("                          DEMO COMPLETE")
    print("=" * 70)
    print("""
HOW IT WORKS:

1. ZONE DETECTION
   - Scans price history for swing highs/lows with strong departures
   - These become supply (resistance) and demand (support) zones

2. CONFIRMATION FILTERS (each worth 25 points)
   - Wick Rejection: Price wicks into zone then closes outside
   - Volume Spike: 1.5x+ average volume when hitting zone
   - Reversal Pattern: Engulfing, Hammer, or Shooting Star
   - Trend Confirmation: Higher lows (long) or lower highs (short)

3. TRADE ENTRY
   - Only enters when confirmation score >= 50 (at least 2 signals)
   - Sets stop loss below/above zone
   - Sets target for 2:1 risk/reward minimum

4. POSITION MANAGEMENT
   - Tracks P&L in real-time
   - Activates trailing stop at 50% of target
   - Exits at target, stop loss, or trailing stop

TO RUN LIVE PAPER TRADING:
   python -m trading_agent.paper_trader --polling --watchlist SPY,QQQ,AAPL

TO MONITOR POSITIONS:
   python scripts/paper_dashboard.py --watch
""")


if __name__ == "__main__":
    asyncio.run(demo())
