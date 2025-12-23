#!/usr/bin/env python3
"""
Paper Trading Dashboard - Monitor paper trading performance.

Usage:
    python scripts/paper_dashboard.py
    python scripts/paper_dashboard.py --watch  # Auto-refresh every 5s
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_state(data_dir: str = "data/paper_trading") -> dict:
    """Load the paper trading state from disk."""
    state_file = Path(data_dir) / "paper_trading_state.json"

    if not state_file.exists():
        return None

    with open(state_file) as f:
        return json.load(f)


def clear_screen():
    """Clear terminal screen."""
    os.system("cls" if os.name == "nt" else "clear")


def format_pnl(pnl: float) -> str:
    """Format P&L with color codes."""
    if pnl >= 0:
        return f"\033[92m${pnl:+,.2f}\033[0m"  # Green
    else:
        return f"\033[91m${pnl:+,.2f}\033[0m"  # Red


def format_pct(pct: float) -> str:
    """Format percentage with color codes."""
    if pct >= 0:
        return f"\033[92m{pct:+.2f}%\033[0m"  # Green
    else:
        return f"\033[91m{pct:+.2f}%\033[0m"  # Red


def display_dashboard(state: dict):
    """Display the paper trading dashboard."""
    print("\n" + "=" * 70)
    print("                    BILL FANTER PAPER TRADER DASHBOARD")
    print("=" * 70)

    # Last update time
    last_update = state.get("timestamp", "N/A")
    if last_update != "N/A":
        dt = datetime.fromisoformat(last_update)
        last_update = dt.strftime("%Y-%m-%d %H:%M:%S")
    print(f"Last Update: {last_update}")
    print()

    # Account Summary
    print("-" * 70)
    print("ACCOUNT SUMMARY")
    print("-" * 70)

    starting = state.get("starting_capital", 10000)
    current = state.get("capital", starting)
    total_pnl = state.get("total_pnl", 0)
    pnl_pct = (current - starting) / starting * 100

    print(f"  Starting Capital:  ${starting:,.2f}")
    print(f"  Current Capital:   ${current:,.2f}")
    print(f"  Total P&L:         {format_pnl(total_pnl)} ({format_pct(pnl_pct)})")
    print()

    # Trade Stats
    print("-" * 70)
    print("TRADE STATISTICS")
    print("-" * 70)

    total_trades = state.get("total_trades", 0)
    winning = state.get("winning_trades", 0)
    losing = state.get("losing_trades", 0)
    win_rate = state.get("win_rate", 0)

    print(f"  Total Trades:  {total_trades}")
    print(f"  Winning:       {winning}")
    print(f"  Losing:        {losing}")
    print(f"  Win Rate:      {win_rate:.1f}%")
    print()

    # Open Positions
    positions = state.get("open_positions", [])
    print("-" * 70)
    print(f"OPEN POSITIONS ({len(positions)})")
    print("-" * 70)

    if positions:
        print(
            f"  {'Symbol':<8} {'Dir':<6} {'Entry':>10} {'Current':>10} "
            f"{'P&L':>12} {'Conf':>6}"
        )
        print("  " + "-" * 60)

        for pos in positions:
            symbol = pos.get("symbol", "???")
            direction = pos.get("direction", "???").upper()[:5]
            entry = pos.get("entry_price", 0)
            current = pos.get("current_price", entry)
            pnl = pos.get("unrealized_pnl", 0)
            pnl_pct = pos.get("unrealized_pnl_pct", 0)
            conf = pos.get("confirmation_score", 0)

            pnl_str = f"${pnl:+.2f} ({pnl_pct:+.1f}%)"
            if pnl >= 0:
                pnl_display = f"\033[92m{pnl_str:>12}\033[0m"
            else:
                pnl_display = f"\033[91m{pnl_str:>12}\033[0m"

            print(
                f"  {symbol:<8} {direction:<6} ${entry:>9.2f} ${current:>9.2f} "
                f"{pnl_display} {conf:>5}/100"
            )

            # Show option info if present
            if pos.get("option_ticker"):
                print(
                    f"           └─ Option: {pos['option_ticker']} "
                    f"Strike: ${pos.get('option_strike', 0):.2f} "
                    f"Exp: {pos.get('option_expiry', 'N/A')}"
                )
    else:
        print("  No open positions")
    print()

    # Recent Closed Trades
    closed = state.get("closed_trades", [])
    print("-" * 70)
    print(f"RECENT CLOSED TRADES (Last 5 of {len(closed)})")
    print("-" * 70)

    if closed:
        recent = closed[-5:][::-1]  # Last 5, newest first
        print(
            f"  {'Symbol':<8} {'Dir':<6} {'Entry':>9} {'Exit':>9} "
            f"{'P&L':>10} {'Reason':<12}"
        )
        print("  " + "-" * 60)

        for trade in recent:
            symbol = trade.get("symbol", "???")
            direction = trade.get("direction", "???").upper()[:5]
            entry = trade.get("entry_price", 0)
            exit_p = trade.get("exit_price", 0)
            pnl = trade.get("pnl", 0)
            reason = trade.get("exit_reason", "???")[:12]

            pnl_str = f"${pnl:+.2f}"
            if pnl >= 0:
                pnl_display = f"\033[92m{pnl_str:>10}\033[0m"
            else:
                pnl_display = f"\033[91m{pnl_str:>10}\033[0m"

            print(
                f"  {symbol:<8} {direction:<6} ${entry:>8.2f} ${exit_p:>8.2f} "
                f"{pnl_display} {reason:<12}"
            )
    else:
        print("  No closed trades yet")

    print()
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Paper Trading Dashboard")
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Auto-refresh dashboard every 5 seconds",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/paper_trading",
        help="Path to paper trading data directory",
    )

    args = parser.parse_args()

    if args.watch:
        print("Watching paper trading state (Ctrl+C to stop)...")
        try:
            while True:
                clear_screen()
                state = load_state(args.data_dir)
                if state:
                    display_dashboard(state)
                else:
                    print("\nNo paper trading state found.")
                    print(f"Looking in: {args.data_dir}/paper_trading_state.json")
                    print("\nStart the paper trader first:")
                    print("  python -m trading_agent.paper_trader --polling")

                print(f"\n[Auto-refresh in 5s... Press Ctrl+C to exit]")
                time.sleep(5)
        except KeyboardInterrupt:
            print("\nExiting dashboard...")
    else:
        state = load_state(args.data_dir)
        if state:
            display_dashboard(state)
        else:
            print("\nNo paper trading state found.")
            print(f"Looking in: {args.data_dir}/paper_trading_state.json")
            print("\nStart the paper trader first:")
            print("  python -m trading_agent.paper_trader --polling")


if __name__ == "__main__":
    main()
