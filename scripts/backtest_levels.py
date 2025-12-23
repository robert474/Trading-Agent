#!/usr/bin/env python3
"""
Backtest Bill Fanter's price levels against actual market data.

This script:
1. Loads extracted signals from Bill's videos
2. Pulls historical candle data from Polygon.io
3. Checks if price levels were hit and how they performed
4. Calculates win rates and statistics
"""

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
import requests

# Configuration
API_KEY = os.getenv("POLYGON_API_KEY", "oZrkzpyhfzeU2vtkuxdpoLSXpMe7sRqu")
BASE_URL = "https://api.polygon.io"
SIGNALS_DIR = Path("data/transcriber/signals")
RESULTS_DIR = Path("data/backtest")

# Rate limiting for free tier (5 calls/min)
RATE_LIMIT_DELAY = 12.5  # seconds between calls


def get_candles(ticker: str, start_date: str, end_date: str, timeframe: str = "1", timespan: str = "day") -> list:
    """Fetch candle data from Polygon.io"""
    url = f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/{timeframe}/{timespan}/{start_date}/{end_date}"
    params = {"apiKey": API_KEY, "adjusted": "true", "sort": "asc", "limit": 5000}

    try:
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("results", [])
        else:
            print(f"  Error fetching {ticker}: {resp.status_code}")
            return []
    except Exception as e:
        print(f"  Exception fetching {ticker}: {e}")
        return []


def check_entry_confirmation(candles: list, hit_idx: int, zone_type: str, price: float) -> dict:
    """
    Check for entry confirmation patterns at zone hit.

    Returns confirmation score (0-100) and list of confirmations found.
    """
    result = {
        "score": 0,
        "confirmations": [],
        "has_wick_rejection": False,
        "has_volume_spike": False,
        "has_pattern": False,
        "has_trend_confirm": False,
    }

    if hit_idx < 2 or hit_idx >= len(candles):
        return result

    candle = candles[hit_idx]
    prev_candle = candles[hit_idx - 1]

    low = candle.get("l", 0)
    high = candle.get("h", 0)
    close = candle.get("c", 0)
    open_price = candle.get("o", 0)
    volume = candle.get("v", 0)

    candle_range = high - low
    body = abs(close - open_price)
    is_bullish = close > open_price

    is_support = zone_type in ["support", "demand", "gap_bottom", "fib_50", "fib_618", "fib_golden"]

    # 1. Wick Rejection Check (+25 points)
    if candle_range > 0:
        if is_support:
            lower_wick = min(open_price, close) - low
            if lower_wick / candle_range >= 0.4 and close > price:
                result["score"] += 25
                result["confirmations"].append("Wick rejection")
                result["has_wick_rejection"] = True
        else:
            upper_wick = high - max(open_price, close)
            if upper_wick / candle_range >= 0.4 and close < price:
                result["score"] += 25
                result["confirmations"].append("Wick rejection")
                result["has_wick_rejection"] = True

    # 2. Volume Confirmation (+25 points)
    if hit_idx >= 5:
        prev_volumes = [c.get("v", 0) for c in candles[hit_idx-5:hit_idx]]
        avg_volume = sum(prev_volumes) / len(prev_volumes) if prev_volumes else 0
        if avg_volume > 0 and volume >= avg_volume * 1.5:
            result["score"] += 25
            result["confirmations"].append(f"Volume spike {volume/avg_volume:.1f}x")
            result["has_volume_spike"] = True

    # 3. Reversal Pattern Check (+25 points)
    prev_body = abs(prev_candle.get("c", 0) - prev_candle.get("o", 0))
    prev_bullish = prev_candle.get("c", 0) > prev_candle.get("o", 0)

    if is_support:
        # Bullish engulfing
        if not prev_bullish and is_bullish and body > prev_body:
            if close > prev_candle.get("o", 0) and open_price < prev_candle.get("c", 0):
                result["score"] += 25
                result["confirmations"].append("Bullish Engulfing")
                result["has_pattern"] = True
        # Hammer
        elif is_bullish and body > 0:
            lower_wick = min(open_price, close) - low
            upper_wick = high - max(open_price, close)
            if lower_wick >= body * 2 and upper_wick < body * 0.5:
                result["score"] += 25
                result["confirmations"].append("Hammer")
                result["has_pattern"] = True
    else:
        # Bearish engulfing
        if prev_bullish and not is_bullish and body > prev_body:
            if close < prev_candle.get("o", 0) and open_price > prev_candle.get("c", 0):
                result["score"] += 25
                result["confirmations"].append("Bearish Engulfing")
                result["has_pattern"] = True
        # Shooting star
        elif not is_bullish and body > 0:
            upper_wick = high - max(open_price, close)
            lower_wick = min(open_price, close) - low
            if upper_wick >= body * 2 and lower_wick < body * 0.5:
                result["score"] += 25
                result["confirmations"].append("Shooting Star")
                result["has_pattern"] = True

    # 4. Trend Confirmation (+25 points)
    if hit_idx >= 3:
        recent_lows = [c.get("l", 0) for c in candles[hit_idx-2:hit_idx+1]]
        recent_highs = [c.get("h", 0) for c in candles[hit_idx-2:hit_idx+1]]

        if is_support:
            # Higher lows forming
            if recent_lows[-1] >= recent_lows[-2] or recent_lows[-1] >= min(recent_lows[:-1]):
                result["score"] += 25
                result["confirmations"].append("Higher lows")
                result["has_trend_confirm"] = True
        else:
            # Lower highs forming
            if recent_highs[-1] <= recent_highs[-2] or recent_highs[-1] <= max(recent_highs[:-1]):
                result["score"] += 25
                result["confirmations"].append("Lower highs")
                result["has_trend_confirm"] = True

    return result


def check_level_hit(candles: list, price: float, zone_type: str, tolerance: float = 0.002) -> dict:
    """
    Check if a price level was hit and what happened after.

    Returns:
        dict with hit status, direction, result, and confirmation score
    """
    result = {
        "hit": False,
        "hit_date": None,
        "hit_price": None,
        "direction": None,
        "bounce": False,
        "break": False,
        "max_move_after": 0,
        "close_after": None,
        "confirmation_score": 0,
        "confirmations": [],
        "confirmed_bounce": False,  # Bounce with confirmation >= 50
    }

    for i, candle in enumerate(candles):
        low = candle.get("l", 0)
        high = candle.get("h", 0)
        close = candle.get("c", 0)
        open_price = candle.get("o", 0)
        timestamp = candle.get("t", 0)

        # Check if price was within tolerance of our level
        price_low = price * (1 - tolerance)
        price_high = price * (1 + tolerance)

        if low <= price_high and high >= price_low:
            result["hit"] = True
            result["hit_date"] = datetime.fromtimestamp(timestamp / 1000).strftime("%Y-%m-%d")
            result["hit_price"] = price

            # Check for entry confirmations
            confirmation = check_entry_confirmation(candles, i, zone_type, price)
            result["confirmation_score"] = confirmation["score"]
            result["confirmations"] = confirmation["confirmations"]

            # Determine if support or resistance based on zone_type
            is_support = zone_type in ["support", "demand", "gap_bottom", "fib_50", "fib_618", "fib_golden"]

            if is_support:
                # For support: did it bounce (close above level)?
                if close > price:
                    result["bounce"] = True
                    result["direction"] = "long"
                    # Mark as confirmed if score >= 50
                    if confirmation["score"] >= 50:
                        result["confirmed_bounce"] = True
                else:
                    result["break"] = True
                    result["direction"] = "short"
            else:
                # For resistance: did it reject (close below level)?
                if close < price:
                    result["bounce"] = True
                    result["direction"] = "short"
                    if confirmation["score"] >= 50:
                        result["confirmed_bounce"] = True
                else:
                    result["break"] = True
                    result["direction"] = "long"

            # Check next few candles for max move
            if i + 1 < len(candles):
                next_candle = candles[i + 1]
                result["close_after"] = next_candle.get("c", 0)

                # Calculate max move in expected direction
                if result["direction"] == "long":
                    result["max_move_after"] = (next_candle.get("h", 0) - price) / price * 100
                else:
                    result["max_move_after"] = (price - next_candle.get("l", 0)) / price * 100

            break

    return result


def backtest_video(signal_file: Path) -> dict:
    """Backtest all levels from a single video's signals"""

    with open(signal_file) as f:
        signals = json.load(f)

    # Skip duplicates
    if signals.get("is_duplicate_of"):
        return {"skipped": True, "reason": "duplicate"}

    video_id = signals.get("video_id", "unknown")
    video_date = signals.get("date", "2025-01-01")
    title = signals.get("title", "Unknown")

    print(f"\n{'='*60}")
    print(f"Backtesting: {title}")
    print(f"Video Date: {video_date}")
    print(f"{'='*60}")

    # Calculate the week after the video for testing
    start_date = datetime.strptime(video_date, "%Y-%m-%d")
    end_date = start_date + timedelta(days=7)

    results = {
        "video_id": video_id,
        "title": title,
        "video_date": video_date,
        "test_period": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
        "levels_tested": 0,
        "levels_hit": 0,
        "bounces": 0,
        "breaks": 0,
        "win_rate": 0,
        # NEW: Confirmation-filtered stats
        "confirmed_bounces": 0,
        "confirmed_breaks": 0,
        "confirmed_win_rate": 0,
        "avg_confirmation_score": 0,
        "details": []
    }

    zones = signals.get("zone_levels", [])

    # Group zones by ticker
    tickers = {}
    for zone in zones:
        ticker = zone.get("ticker", "SPY")
        if ticker not in tickers:
            tickers[ticker] = []
        tickers[ticker].append(zone)

    # Test each ticker
    for ticker, ticker_zones in tickers.items():
        # Skip futures and crypto for now
        if ticker in ["ES", "NQ", "BTC", "BTCUSD"]:
            continue

        print(f"\n  {ticker}: {len(ticker_zones)} levels")

        # Fetch daily candles for the week after video
        candles = get_candles(
            ticker,
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
            "1", "day"
        )

        if not candles:
            print(f"    No data available")
            time.sleep(RATE_LIMIT_DELAY)
            continue

        # Also get 15-minute candles for more precision
        candles_15m = get_candles(
            ticker,
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
            "15", "minute"
        )
        time.sleep(RATE_LIMIT_DELAY)  # Rate limit

        # Test each zone
        for zone in ticker_zones:
            price = zone.get("price", 0)
            zone_type = zone.get("zone_type", "support")
            notes = zone.get("notes", "")

            # Use 15m candles if available, else daily
            test_candles = candles_15m if candles_15m else candles

            result = check_level_hit(test_candles, price, zone_type)

            results["levels_tested"] += 1

            if result["hit"]:
                results["levels_hit"] += 1
                conf_score = result.get("confirmation_score", 0)

                if result["bounce"]:
                    results["bounces"] += 1
                    status = "BOUNCE"
                    # Track confirmed bounces (score >= 50)
                    if result.get("confirmed_bounce"):
                        results["confirmed_bounces"] += 1
                else:
                    results["breaks"] += 1
                    status = "BREAK"
                    # A break with high confirmation means we avoided a bad trade
                    if conf_score >= 50:
                        results["confirmed_breaks"] += 1

                conf_str = f" [Conf: {conf_score}]" if conf_score > 0 else ""
                confirms = ", ".join(result.get("confirmations", []))
                print(f"    ${price:.2f} ({zone_type}): {status} on {result['hit_date']} | Move: {result['max_move_after']:.2f}%{conf_str}")
                if confirms:
                    print(f"      Confirmations: {confirms}")

                results["details"].append({
                    "ticker": ticker,
                    "price": price,
                    "zone_type": zone_type,
                    "notes": notes,
                    "hit": True,
                    "status": status,
                    "hit_date": result["hit_date"],
                    "max_move": result["max_move_after"],
                    "confirmation_score": conf_score,
                    "confirmations": result.get("confirmations", []),
                    "confirmed_bounce": result.get("confirmed_bounce", False)
                })
            else:
                print(f"    ${price:.2f} ({zone_type}): NOT HIT")
                results["details"].append({
                    "ticker": ticker,
                    "price": price,
                    "zone_type": zone_type,
                    "notes": notes,
                    "hit": False
                })

    # Calculate win rate (bounces are wins for Bill's methodology)
    if results["levels_hit"] > 0:
        results["win_rate"] = results["bounces"] / results["levels_hit"] * 100

    # Calculate confirmed win rate (only trades with confirmation >= 50)
    confirmed_total = results["confirmed_bounces"] + results["confirmed_breaks"]
    if confirmed_total > 0:
        results["confirmed_win_rate"] = results["confirmed_bounces"] / confirmed_total * 100

    # Calculate average confirmation score
    conf_scores = [d.get("confirmation_score", 0) for d in results["details"] if d.get("hit")]
    if conf_scores:
        results["avg_confirmation_score"] = sum(conf_scores) / len(conf_scores)

    print(f"\n  Summary: {results['levels_hit']}/{results['levels_tested']} levels hit")
    print(f"  Bounces: {results['bounces']} | Breaks: {results['breaks']}")
    print(f"  Win Rate: {results['win_rate']:.1f}%")
    print(f"  --- WITH CONFIRMATION FILTERS ---")
    print(f"  Confirmed Bounces: {results['confirmed_bounces']} | Confirmed Breaks: {results['confirmed_breaks']}")
    if confirmed_total > 0:
        print(f"  Confirmed Win Rate: {results['confirmed_win_rate']:.1f}%")
    print(f"  Avg Confirmation Score: {results['avg_confirmation_score']:.1f}")

    return results


def main():
    """Run backtest on all signal files"""

    # Create results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Get all signal files
    signal_files = sorted(SIGNALS_DIR.glob("*_signals.json"))

    print(f"Found {len(signal_files)} signal files")
    print(f"API Key: {API_KEY[:10]}...")

    all_results = []

    for signal_file in signal_files:
        result = backtest_video(signal_file)
        if not result.get("skipped"):
            all_results.append(result)

    # Aggregate statistics
    total_tested = sum(r["levels_tested"] for r in all_results)
    total_hit = sum(r["levels_hit"] for r in all_results)
    total_bounces = sum(r["bounces"] for r in all_results)
    total_breaks = sum(r["breaks"] for r in all_results)
    total_confirmed_bounces = sum(r["confirmed_bounces"] for r in all_results)
    total_confirmed_breaks = sum(r["confirmed_breaks"] for r in all_results)
    total_confirmed = total_confirmed_bounces + total_confirmed_breaks

    print(f"\n{'='*60}")
    print("OVERALL RESULTS")
    print(f"{'='*60}")
    print(f"Videos analyzed: {len(all_results)}")
    print(f"Total levels tested: {total_tested}")
    print(f"Levels hit: {total_hit} ({total_hit/total_tested*100:.1f}%)")
    print(f"Bounces (wins): {total_bounces}")
    print(f"Breaks (losses): {total_breaks}")
    if total_hit > 0:
        print(f"Overall Win Rate: {total_bounces/total_hit*100:.1f}%")

    print(f"\n{'='*60}")
    print("WITH CONFIRMATION FILTERS (Score >= 50)")
    print(f"{'='*60}")
    print(f"Confirmed Bounces: {total_confirmed_bounces}")
    print(f"Confirmed Breaks: {total_confirmed_breaks}")
    if total_confirmed > 0:
        confirmed_win_rate = total_confirmed_bounces / total_confirmed * 100
        print(f"Confirmed Win Rate: {confirmed_win_rate:.1f}%")
        print(f"\n  >> Improvement: {confirmed_win_rate - (total_bounces/total_hit*100):.1f}% higher than raw")

    # Save results
    output_file = RESULTS_DIR / f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    raw_win_rate = total_bounces/total_hit*100 if total_hit > 0 else 0
    conf_win_rate = total_confirmed_bounces / total_confirmed * 100 if total_confirmed > 0 else 0

    with open(output_file, "w") as f:
        json.dump({
            "run_date": datetime.now().isoformat(),
            "summary": {
                "videos_analyzed": len(all_results),
                "total_levels_tested": total_tested,
                "levels_hit": total_hit,
                "bounces": total_bounces,
                "breaks": total_breaks,
                "win_rate": raw_win_rate,
                # Confirmation-filtered stats
                "confirmed_bounces": total_confirmed_bounces,
                "confirmed_breaks": total_confirmed_breaks,
                "confirmed_total": total_confirmed,
                "confirmed_win_rate": conf_win_rate,
                "improvement": conf_win_rate - raw_win_rate if total_confirmed > 0 else 0,
            },
            "video_results": all_results
        }, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
