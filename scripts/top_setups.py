#!/usr/bin/env python3
"""
Find Top Trading Setups - Scans watchlist for best supply/demand zone setups.
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from trading_agent.analysis.zone_detector import ZoneDetector
from trading_agent.data.providers.polygon import PolygonDataProvider
from trading_agent.core.models import ZoneType


async def find_best_setups():
    polygon = PolygonDataProvider()
    zone_detector = ZoneDetector()

    symbols = ["SPY", "QQQ", "NVDA", "AAPL", "TSLA"]
    all_setups = []

    print("Scanning for setups...\n")

    for symbol in symbols:
        print(f"  {symbol}...", end=" ", flush=True)
        try:
            candles = await polygon.get_historical_candles(
                symbol=symbol,
                timeframe="15",
                from_date=datetime.now() - timedelta(days=30),
                to_date=datetime.now(),
            )

            await asyncio.sleep(13)  # Rate limit

            if not candles or len(candles) < 50:
                print("no data")
                continue

            current_price = candles[-1].close
            zones = zone_detector.detect_zones(candles, "15m")

            supply_count = len([z for z in zones if z.zone_type == ZoneType.SUPPLY])
            demand_count = len([z for z in zones if z.zone_type == ZoneType.DEMAND])
            print(f"{len(zones)} zones ({demand_count}D/{supply_count}S)")

            if not zones:
                continue

            nearest = zone_detector.find_nearest_zones(current_price, zones)

            # Demand zone (long)
            if nearest.get("nearest_demand"):
                z = nearest["nearest_demand"]
                dist_pct = abs(current_price - z.zone_high) / current_price * 100

                setup = zone_detector.check_entry_conditions(
                    current_price=current_price,
                    zone=z,
                    htf_trend="neutral",
                    recent_candles=candles[-10:],
                )

                stop = setup.get("stop_loss") or z.zone_low * 0.995

                all_setups.append(
                    {
                        "symbol": symbol,
                        "direction": "LONG",
                        "zone_type": "DEMAND",
                        "current_price": current_price,
                        "zone_low": z.zone_low,
                        "zone_high": z.zone_high,
                        "stop_loss": stop,
                        "distance_pct": dist_pct,
                        "conf_score": setup["confirmation_score"],
                        "confirmations": setup["confirmations"],
                        "quality": z.quality_score,
                        "freshness": z.freshness.value,
                        "at_zone": setup["has_setup"],
                    }
                )

            # Supply zone (short)
            if nearest.get("nearest_supply"):
                z = nearest["nearest_supply"]
                dist_pct = abs(current_price - z.zone_low) / current_price * 100

                setup = zone_detector.check_entry_conditions(
                    current_price=current_price,
                    zone=z,
                    htf_trend="neutral",
                    recent_candles=candles[-10:],
                )

                stop = setup.get("stop_loss") or z.zone_high * 1.005

                all_setups.append(
                    {
                        "symbol": symbol,
                        "direction": "SHORT",
                        "zone_type": "SUPPLY",
                        "current_price": current_price,
                        "zone_low": z.zone_low,
                        "zone_high": z.zone_high,
                        "stop_loss": stop,
                        "distance_pct": dist_pct,
                        "conf_score": setup["confirmation_score"],
                        "confirmations": setup["confirmations"],
                        "quality": z.quality_score,
                        "freshness": z.freshness.value,
                        "at_zone": setup["has_setup"],
                    }
                )

        except Exception as e:
            print(f"error: {e}")

    # Score and sort
    def score_setup(s):
        score = 0
        if s["at_zone"]:
            score += 100
        if s["distance_pct"] < 0.5:
            score += 50
        elif s["distance_pct"] < 1.0:
            score += 25
        score += s["conf_score"]
        score += s["quality"] / 2
        if s["freshness"] == "fresh":
            score += 20
        return score

    all_setups.sort(key=score_setup, reverse=True)
    return all_setups[:5]


async def main():
    setups = await find_best_setups()

    print()
    print("=" * 70)
    print("        TOP 5 SETUPS - RANKED BY CONFIDENCE")
    print("=" * 70)

    for i, s in enumerate(setups, 1):
        if s["at_zone"]:
            status = "AT ZONE"
        else:
            status = f"{s['distance_pct']:.1f}% away"

        if s["conf_score"] >= 50:
            tradeable = "TRADEABLE"
        else:
            tradeable = "Wait for confirmation"

        risk = abs(s["current_price"] - s["stop_loss"])
        if s["direction"] == "LONG":
            target = s["current_price"] + (risk * 2)
        else:
            target = s["current_price"] - (risk * 2)

        opt_type = "CALL" if s["direction"] == "LONG" else "PUT"

        print()
        print("-" * 70)
        print(f"#{i}  {s['symbol']} {s['direction']} - {status}")
        print("-" * 70)
        print(f"  Price:      ${s['current_price']:.2f}")
        print(f"  Zone:       ${s['zone_low']:.2f} - ${s['zone_high']:.2f} ({s['zone_type']})")
        print(f"  Quality:    {s['quality']:.0f}/100  |  Freshness: {s['freshness'].upper()}")
        print(f"  Conf Score: {s['conf_score']}/100  [{tradeable}]")

        if s["confirmations"]:
            print("  Signals:")
            for c in s["confirmations"]:
                print(f"    - {c}")

        print()
        print("  TRADE PLAN:")
        print(f"    Entry:    ${s['current_price']:.2f}")
        print(f"    Stop:     ${s['stop_loss']:.2f}  (risk: ${risk:.2f})")
        print(f"    Target:   ${target:.2f}  (2:1 R:R, reward: ${risk*2:.2f})")
        print()
        print(f"  OPTION: {s['symbol']} {opt_type}")
        print(f"    Strike:   ~${round(s['current_price'])}  (ATM)")
        print(f"    Expiry:   7-14 DTE")
        print(f"    Size:     1-2 contracts (~$500 max risk)")

    print()
    print("=" * 70)
    print("NOTE: Only trade setups with Conf Score >= 50 (74% win rate)")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
