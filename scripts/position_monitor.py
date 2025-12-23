#!/usr/bin/env python3
"""
Position Monitor - Uses vision to monitor active trades.

Workflow:
1. Every 2 minutes, check active positions
2. For each position, fetch 5-min chart
3. Analyze with Claude Vision for exit signals
4. Send alerts/recommendations via the dashboard
"""

import asyncio
import aiohttp
import json
import os
from datetime import datetime, time
from pathlib import Path
from typing import Optional

from vision_chart_analyzer import (
    fetch_finviz_chart,
    analyze_5min_chart_for_exit,
    get_current_price
)

DATA_DIR = Path(__file__).parent.parent / "data"
PAPER_TRADING_STATE = DATA_DIR / "paper_trading" / "paper_trading_state.json"


def load_positions() -> list:
    """Load current open positions."""
    if PAPER_TRADING_STATE.exists():
        with open(PAPER_TRADING_STATE) as f:
            state = json.load(f)
            return state.get("open_positions", [])
    return []


def is_market_hours() -> bool:
    """Check if we're within US market hours (9:30 AM - 4:00 PM ET)."""
    now = datetime.now()
    # Simple check - would need proper timezone handling for production
    market_open = time(9, 30)
    market_close = time(16, 0)
    return market_open <= now.time() <= market_close


async def monitor_position(position: dict) -> dict:
    """
    Monitor a single position using 5-minute chart analysis.
    """
    symbol = position.get("symbol", "")
    position_id = position.get("id", "")

    print(f"\nMonitoring {symbol} ({position_id})...")

    # Get current price
    current_price = await get_current_price(symbol)
    if not current_price:
        return {"symbol": symbol, "error": "Could not fetch price"}

    # Update position with current price
    entry_price = position.get("entry_price", 0)
    direction = position.get("direction", "long")

    if direction == "long":
        pnl_pct = ((current_price - entry_price) / entry_price) * 100
    else:
        pnl_pct = ((entry_price - current_price) / entry_price) * 100

    position["current_price"] = current_price
    position["unrealized_pnl_pct"] = pnl_pct

    # Fetch 5-minute chart
    chart_bytes = await fetch_finviz_chart(symbol, "i5")

    if not chart_bytes:
        print(f"  Could not fetch 5-min chart for {symbol}")
        return {
            "symbol": symbol,
            "current_price": current_price,
            "pnl_pct": pnl_pct,
            "recommendation": "HOLD",
            "reason": "No chart available for analysis"
        }

    # Analyze with vision
    analysis = await analyze_5min_chart_for_exit(chart_bytes, symbol, position)

    # Add position context
    analysis["position_id"] = position_id
    analysis["current_price"] = current_price
    analysis["pnl_pct"] = pnl_pct
    analysis["entry_price"] = entry_price
    analysis["direction"] = direction

    # Print recommendation
    rec = analysis.get("recommendation", "HOLD")
    reason = analysis.get("reason", "")

    if rec == "HOLD":
        print(f"  ✓ HOLD - {reason}")
    elif rec == "TAKE_PROFIT":
        print(f"  💰 TAKE PROFIT - {reason}")
    elif rec == "TIGHTEN_STOP":
        new_stop = analysis.get("new_stop", "N/A")
        print(f"  ⚠️  TIGHTEN STOP to ${new_stop} - {reason}")
    elif rec == "EXIT_NOW":
        print(f"  🚨 EXIT NOW - {reason}")

    return analysis


async def monitor_all_positions() -> list:
    """
    Monitor all open positions.
    """
    positions = load_positions()

    if not positions:
        print("No open positions to monitor.")
        return []

    print(f"\n{'='*60}")
    print(f"POSITION MONITOR - {datetime.now().strftime('%H:%M:%S')}")
    print(f"Monitoring {len(positions)} position(s)")
    print(f"{'='*60}")

    results = []
    for position in positions:
        try:
            result = await monitor_position(position)
            results.append(result)
            # Small delay between positions
            await asyncio.sleep(0.5)
        except Exception as e:
            print(f"Error monitoring {position.get('symbol')}: {e}")
            results.append({
                "symbol": position.get("symbol"),
                "error": str(e),
                "recommendation": "HOLD"
            })

    # Save monitoring results
    output = {
        "timestamp": datetime.now().isoformat(),
        "positions_monitored": len(positions),
        "results": results
    }

    output_path = DATA_DIR / "position_monitor_log.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY:")
    for r in results:
        symbol = r.get("symbol", "?")
        rec = r.get("recommendation", "?")
        pnl = r.get("pnl_pct", 0)
        print(f"  {symbol}: {rec} (P&L: {pnl:+.1f}%)")
    print(f"{'='*60}\n")

    return results


async def run_continuous_monitor(interval_seconds: int = 120):
    """
    Run continuous position monitoring during market hours.
    Default: check every 2 minutes.
    """
    print(f"\n{'='*60}")
    print("STARTING CONTINUOUS POSITION MONITOR")
    print(f"Interval: {interval_seconds} seconds")
    print(f"{'='*60}\n")

    while True:
        if not is_market_hours():
            print(f"[{datetime.now().strftime('%H:%M')}] Market closed. Sleeping...")
            await asyncio.sleep(300)  # Check every 5 minutes when closed
            continue

        positions = load_positions()
        if positions:
            await monitor_all_positions()
        else:
            print(f"[{datetime.now().strftime('%H:%M')}] No open positions.")

        # Wait for next check
        await asyncio.sleep(interval_seconds)


async def check_for_entries(levels: list) -> list:
    """
    Check if price is approaching any detected levels.
    Alert when within 0.5% of entry zone.
    """
    alerts = []

    for level in levels:
        symbol = level.get("symbol", "")
        entry = level.get("entry", 0)
        direction = level.get("direction", "LONG")

        current_price = await get_current_price(symbol)
        if not current_price:
            continue

        # Calculate distance to entry
        distance_pct = abs(current_price - entry) / entry * 100

        if distance_pct <= 0.5:
            # Near entry zone - alert!
            alerts.append({
                "symbol": symbol,
                "direction": direction,
                "entry": entry,
                "current_price": current_price,
                "distance_pct": distance_pct,
                "target": level.get("target"),
                "stop": level.get("stop"),
                "alert_type": "ENTRY_APPROACHING",
                "timestamp": datetime.now().isoformat()
            })

            print(f"\n🚨 ALERT: {symbol} {direction} approaching entry!")
            print(f"   Entry: ${entry:.2f} | Current: ${current_price:.2f} | Distance: {distance_pct:.2f}%")

    return alerts


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--continuous":
        # Run continuous monitoring
        asyncio.run(run_continuous_monitor())
    else:
        # One-time check
        asyncio.run(monitor_all_positions())
