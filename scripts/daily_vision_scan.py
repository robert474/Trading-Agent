#!/usr/bin/env python3
"""
Daily Vision Scanner - Runs at 7am to analyze charts for breakout setups.

1. Downloads Finviz daily charts for top 35 stocks
2. Uses Claude Vision to identify supply/demand zones
3. Determines breakout setups (LONG above resistance, SHORT below support)
4. Saves setups to vision_levels.json for dashboard
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
import aiohttp
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).parent.parent / ".env")

import anthropic

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trading_agent.data.providers.polygon import PolygonDataProvider

# Bill Fanter's key stocks from his videos
BILL_FANTER_WATCHLIST = [
    # From his weekly watchlist
    "SPY", "QQQ", "AAPL", "AMZN", "GOOGL", "TSLA", "META", "NVDA", "AMD",
    "AVGO", "ORCL", "PLTR", "NKE", "SBUX", "HD",
    # Other tech he mentions
    "MSFT", "CRM", "INTC", "MU", "ADBE",
    # Financials
    "JPM", "GS", "BAC",
    # Consumer
    "COST", "WMT", "MCD", "LULU",
    # Risk-off indicators
    "MSTR", "HOOD", "COIN",
    # ETFs
    "IWM", "XLF", "TQQQ",
    # Other movers he watches
    "DIS", "FDX", "UPS",
]

# Limit to 35 for daily scan
DAILY_WATCHLIST = BILL_FANTER_WATCHLIST[:35]

CHARTS_DIR = Path(__file__).parent.parent / "data" / "charts"
DATA_DIR = Path(__file__).parent.parent / "data"


async def download_finviz_chart(symbol: str, session: aiohttp.ClientSession, timeframe: str = "d") -> str | None:
    """
    Download chart from Finviz.

    timeframe: 'd' = daily, 'i5' = 5-minute intraday
    """
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    if timeframe == "d":
        url = f"https://finviz.com/chart.ashx?t={symbol}&ty=c&ta=1&p=d&s=l"
        filename = f"{symbol}_daily.png"
    else:
        url = f"https://finviz.com/chart.ashx?t={symbol}&ty=c&ta=1&p=i5&s=l"
        filename = f"{symbol}_5min.png"

    filepath = CHARTS_DIR / filename

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }
        async with session.get(url, headers=headers) as resp:
            if resp.status == 200:
                content = await resp.read()
                with open(filepath, "wb") as f:
                    f.write(content)
                return str(filepath)
    except Exception as e:
        print(f"  Error downloading {symbol} chart: {e}")

    return None


async def get_current_price(symbol: str, polygon: PolygonDataProvider, session: aiohttp.ClientSession) -> float | None:
    """Get current price from Polygon or Yahoo as fallback."""
    try:
        price = await polygon.get_current_price(symbol)
        if price:
            return price
    except:
        pass

    # Fallback to Yahoo Finance
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=1d"
        headers = {"User-Agent": "Mozilla/5.0"}
        async with session.get(url, headers=headers) as resp:
            if resp.status == 200:
                data = await resp.json()
                price = data["chart"]["result"][0]["meta"].get("regularMarketPrice")
                if price:
                    return float(price)
    except Exception as e:
        pass

    return None


def analyze_chart_with_vision(image_path: str, symbol: str, current_price: float) -> dict | None:
    """
    Use Claude Vision to analyze chart and identify supply/demand zones.
    Returns breakout setup recommendations.
    """
    client = anthropic.Anthropic()

    # Read image
    with open(image_path, "rb") as f:
        image_data = f.read()

    import base64
    image_base64 = base64.b64encode(image_data).decode("utf-8")

    prompt = f"""Analyze this daily chart for {symbol} (current price: ${current_price:.2f}).

You are looking for BREAKOUT setups like Bill Fanter trades:
- LONG BREAKOUT: Price needs to break ABOVE a resistance level
- SHORT BREAKDOWN: Price needs to break BELOW a support level

Identify:
1. Key resistance levels above current price (for LONG breakouts)
2. Key support levels below current price (for SHORT breakdowns)
3. The quality of each zone (fresh, tested multiple times, etc.)

For each setup, provide:
- direction: "LONG" or "SHORT"
- entry: The breakout/breakdown price level
- stop: Where to place stop loss
- target: Price target (aim for 3:1 R:R minimum)
- quality: 1-100 score for zone strength
- notes: Brief explanation

Current price: ${current_price:.2f}

Only include setups where entry is within 5% of current price.

Respond in JSON format:
{{
    "symbol": "{symbol}",
    "current_price": {current_price},
    "analysis": "Brief market structure analysis",
    "setups": [
        {{
            "direction": "LONG",
            "entry": 123.45,
            "stop": 120.00,
            "target": 133.00,
            "quality": 75,
            "notes": "Breaking above resistance with volume"
        }}
    ]
}}

If no good setups exist within 5%, return empty setups array."""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_base64,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
        )

        # Parse response
        text = response.content[0].text

        # Extract JSON from response
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        return json.loads(text.strip())

    except Exception as e:
        print(f"  Vision analysis error for {symbol}: {e}")
        return None


def convert_to_levels(analysis: dict) -> list:
    """Convert vision analysis to standard levels format."""
    levels = []
    symbol = analysis.get("symbol", "")
    current_price = analysis.get("current_price", 0)

    for setup in analysis.get("setups", []):
        direction = setup.get("direction", "LONG")
        entry = setup.get("entry", 0)
        stop = setup.get("stop", 0)
        target = setup.get("target", 0)

        # Calculate distance percentage
        if direction == "LONG":
            # LONG: entry above current price
            distance_pct = (entry - current_price) / current_price * 100
            if current_price >= entry:
                status = "MISSED"
            else:
                status = "PENDING"
        else:
            # SHORT: entry below current price
            distance_pct = (current_price - entry) / current_price * 100
            if current_price <= entry:
                status = "MISSED"
            else:
                status = "PENDING"

        # Calculate R:R
        risk = abs(entry - stop)
        reward = abs(target - entry)
        rr_ratio = reward / risk if risk > 0 else 0

        levels.append({
            "symbol": symbol,
            "direction": direction,
            "entry": round(entry, 2),
            "stop": round(stop, 2),
            "target": round(target, 2),
            "current_price": round(current_price, 2),
            "distance_pct": round(abs(distance_pct), 2),
            "status": status,
            "rr_ratio": round(rr_ratio, 1),
            "quality": setup.get("quality", 50),
            "notes": setup.get("notes", ""),
            "source": "Vision AI",
            "chart_image": f"{symbol}_daily.png",
            "detected_at": datetime.now().isoformat(),
        })

    return levels


async def run_daily_scan(symbols: list = None):
    """Run the daily vision scan."""
    if symbols is None:
        symbols = DAILY_WATCHLIST

    print("=" * 60)
    print(f"DAILY VISION SCAN - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Scanning {len(symbols)} symbols...")
    print("=" * 60)

    polygon = PolygonDataProvider()
    all_levels = []

    async with aiohttp.ClientSession() as session:
        for symbol in symbols:
            print(f"\nScanning {symbol}...")

            # 1. Get current price
            current_price = await get_current_price(symbol, polygon, session)
            if not current_price:
                print(f"  ⚠ Could not get price for {symbol}")
                continue

            print(f"  Price: ${current_price:.2f}")

            # 2. Download Finviz chart
            chart_path = await download_finviz_chart(symbol, session, "d")
            if not chart_path:
                print(f"  ⚠ Could not download chart for {symbol}")
                continue

            print(f"  Chart: {chart_path}")

            # 3. Analyze with Vision AI
            analysis = analyze_chart_with_vision(chart_path, symbol, current_price)
            if not analysis:
                print(f"  ⚠ Vision analysis failed for {symbol}")
                continue

            # 4. Convert to levels
            levels = convert_to_levels(analysis)
            if levels:
                all_levels.extend(levels)
                for level in levels:
                    print(f"  ✓ {level['direction']} @ ${level['entry']} ({level['status']}, {level['distance_pct']}% away)")
            else:
                print(f"  - No setups within range")

            # Rate limit
            await asyncio.sleep(1)

    # Save results
    output = {
        "scan_time": datetime.now().isoformat(),
        "symbols_scanned": len(symbols),
        "levels_found": len(all_levels),
        "levels": all_levels,
    }

    output_path = DATA_DIR / "vision_levels.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 60)
    print(f"SCAN COMPLETE")
    print(f"Setups found: {len(all_levels)}")
    print(f"Saved to: {output_path}")
    print("=" * 60)

    # Print summary of actionable setups
    pending = [l for l in all_levels if l["status"] == "PENDING" and l["distance_pct"] <= 3.0]
    if pending:
        print("\n📊 ACTIONABLE SETUPS (within 3%):")
        pending.sort(key=lambda x: x["distance_pct"])
        for l in pending[:10]:
            print(f"  {l['symbol']:6} {l['direction']:5} @ ${l['entry']:.2f} ({l['distance_pct']:.1f}% away) R:R {l['rr_ratio']:.1f}")

    return all_levels


async def scan_single_stock(symbol: str, timeframe: str = "d"):
    """Scan a single stock (useful for testing)."""
    polygon = PolygonDataProvider()

    async with aiohttp.ClientSession() as session:
        current_price = await get_current_price(symbol, polygon, session)
        if not current_price:
            print(f"Could not get price for {symbol}")
            return None

        chart_path = await download_finviz_chart(symbol, session, timeframe)
        if not chart_path:
            print(f"Could not download chart for {symbol}")
            return None

        analysis = analyze_chart_with_vision(chart_path, symbol, current_price)
        return analysis


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Daily Vision Scanner")
    parser.add_argument("--symbol", "-s", help="Scan single symbol")
    parser.add_argument("--full", action="store_true", help="Run full daily scan")
    parser.add_argument("--test", action="store_true", help="Test with 5 symbols")

    args = parser.parse_args()

    if args.symbol:
        result = asyncio.run(scan_single_stock(args.symbol))
        if result:
            print(json.dumps(result, indent=2))
    elif args.full:
        asyncio.run(run_daily_scan())
    elif args.test:
        asyncio.run(run_daily_scan(["TSLA", "AAPL", "NVDA", "AMD", "AMZN"]))
    else:
        print("Usage:")
        print("  python daily_vision_scan.py --full     # Full 35-stock scan")
        print("  python daily_vision_scan.py --test     # Test with 5 stocks")
        print("  python daily_vision_scan.py -s TSLA    # Single stock")
