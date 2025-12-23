#!/usr/bin/env python3
"""
Vision-based Chart Analyzer using Claude Vision API.

This module:
1. Fetches chart images from free sources (Finviz, Yahoo)
2. Analyzes them with Claude Vision to detect supply/demand zones
3. Saves detected zones in the same format as Bill Fanter levels
4. Monitors active positions with 5-minute charts
"""

import asyncio
import aiohttp
import anthropic
import base64
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
import io

# Load environment from project root
import os
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key] = value

DATA_DIR = Path(__file__).parent.parent / "data"
CHARTS_DIR = DATA_DIR / "charts"
CHARTS_DIR.mkdir(exist_ok=True)

# Anthropic client
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


async def fetch_finviz_chart(symbol: str, timeframe: str = "d") -> Optional[bytes]:
    """
    Fetch chart image from Finviz.

    Timeframes:
    - 'd' = daily
    - 'w' = weekly
    - 'm' = monthly
    - 'i1' = 1 minute intraday
    - 'i5' = 5 minute intraday
    """
    # Finviz chart URL pattern
    if timeframe.startswith("i"):
        # Intraday charts
        url = f"https://finviz.com/chart.ashx?t={symbol}&ty=c&ta=0&p={timeframe}&s=l"
    else:
        # Daily/weekly/monthly
        url = f"https://finviz.com/chart.ashx?t={symbol}&ty=c&ta=0&p={timeframe}&s=l"

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as resp:
                if resp.status == 200:
                    return await resp.read()
                else:
                    print(f"Finviz error for {symbol}: {resp.status}")
                    return None
    except Exception as e:
        print(f"Error fetching Finviz chart for {symbol}: {e}")
        return None


async def fetch_bigcharts_image(symbol: str, timeframe: str = "1y") -> Optional[bytes]:
    """
    Fetch chart from BigCharts (MarketWatch).
    More reliable than Finviz for some cases.

    Timeframes: 1d, 5d, 1m, 3m, 6m, 1y, 2y, 5y
    """
    url = f"https://api.wsj.net/api/kaavio/charts/big.chart?nosettings=1&symb={symbol}&uf=0&type=4&style=320&size=3&time={timeframe}&freq=1&comp=&compidx=&ma=0&maession=hours&lf=1&lf2=0&lf3=0&height=335&width=579&mocession=hours"

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as resp:
                if resp.status == 200:
                    return await resp.read()
                else:
                    print(f"BigCharts error for {symbol}: {resp.status}")
                    return None
    except Exception as e:
        print(f"Error fetching BigCharts for {symbol}: {e}")
        return None


async def fetch_stockcharts_image(symbol: str) -> Optional[bytes]:
    """
    Fetch free chart from StockCharts.
    Shows candlestick with volume - good for S/D analysis.
    """
    # StockCharts SharpChart URL
    url = f"https://stockcharts.com/c-sc/sc?s={symbol}&p=D&yr=0&mn=6&dy=0&i=t4779906435c&r=1703"

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Referer": "https://stockcharts.com/"
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as resp:
                if resp.status == 200:
                    return await resp.read()
                else:
                    print(f"StockCharts error for {symbol}: {resp.status}")
                    return None
    except Exception as e:
        print(f"Error fetching StockCharts for {symbol}: {e}")
        return None


def analyze_chart_with_vision(image_bytes: bytes, symbol: str, current_price: float, timeframe: str = "daily") -> dict:
    """
    Use Claude Vision to analyze a chart image and detect supply/demand zones.

    Returns structured zone data compatible with Bill Fanter levels format.
    """
    # Encode image to base64
    image_base64 = base64.standard_b64encode(image_bytes).decode("utf-8")

    # Determine media type (most chart sources return PNG)
    media_type = "image/png"

    prompt = f"""Analyze this {timeframe} chart for {symbol} (current price: ${current_price:.2f}).

You are a professional supply/demand zone trader following Bill Fanter's methodology. Identify:

1. **SUPPLY ZONES** (resistance/selling areas):
   - Areas where price consolidated then dropped sharply
   - Look for bases/ranges before big down moves
   - The "launch pad" areas before sell-offs

2. **DEMAND ZONES** (support/buying areas):
   - Areas where price consolidated then rallied sharply
   - Look for bases/ranges before big up moves
   - The "launch pad" areas before rallies

3. **ZONE QUALITY FACTORS**:
   - Fresh zones (never retested) are strongest
   - Volume confirmation at the zone
   - Speed of departure from zone
   - Multiple touches weakens a zone

For each zone found, provide:
- Price range (zone_low to zone_high)
- Direction: "LONG" for demand zone, "SHORT" for supply zone
- Quality score (1-100)
- Notes on why this zone is significant

Respond in this exact JSON format:
{{
  "symbol": "{symbol}",
  "current_price": {current_price},
  "analysis_time": "{datetime.now().isoformat()}",
  "overall_bias": "BULLISH" or "BEARISH" or "NEUTRAL",
  "zones": [
    {{
      "type": "demand" or "supply",
      "direction": "LONG" or "SHORT",
      "zone_low": <price>,
      "zone_high": <price>,
      "quality": <1-100>,
      "fresh": true/false,
      "notes": "<why this zone matters>"
    }}
  ],
  "key_levels": {{
    "resistance": [<price levels>],
    "support": [<price levels>]
  }},
  "trade_idea": {{
    "bias": "LONG" or "SHORT" or "WAIT",
    "entry_zone": <price or null>,
    "target": <price or null>,
    "stop": <price or null>,
    "rationale": "<brief explanation>"
  }}
}}

Only include zones that are clearly visible and significant. Quality over quantity.
If no clear zones are visible, return empty zones array with "WAIT" trade idea."""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",  # Using Sonnet for cost efficiency
            max_tokens=2000,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_base64,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ],
                }
            ],
        )

        # Parse the response
        response_text = response.content[0].text

        # Extract JSON from response (handle markdown code blocks)
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_str = response_text.split("```")[1].split("```")[0].strip()
        else:
            json_str = response_text.strip()

        return json.loads(json_str)

    except Exception as e:
        print(f"Vision analysis error for {symbol}: {e}")
        return {
            "symbol": symbol,
            "current_price": current_price,
            "error": str(e),
            "zones": [],
            "trade_idea": {"bias": "WAIT"}
        }


async def analyze_5min_chart_for_exit(image_bytes: bytes, symbol: str, position: dict) -> dict:
    """
    Analyze 5-minute chart for position management.
    Look for exit signals: volume fade, reversal patterns, target approach.
    """
    image_base64 = base64.standard_b64encode(image_bytes).decode("utf-8")

    direction = position.get("direction", "long")
    entry_price = position.get("entry_price", 0)
    target = position.get("target_price", 0)
    stop = position.get("stop_loss", 0)
    current_pnl_pct = position.get("unrealized_pnl_pct", 0)

    prompt = f"""Analyze this 5-minute intraday chart for {symbol}.

CURRENT POSITION:
- Direction: {direction.upper()}
- Entry: ${entry_price:.2f}
- Target: ${target:.2f}
- Stop: ${stop:.2f}
- Current P&L: {current_pnl_pct:.1f}%

Evaluate the position and recommend action:

1. **HOLD** - Price action healthy, momentum intact
2. **TAKE PROFIT** - Near target or showing reversal signs
3. **TIGHTEN STOP** - Move stop to breakeven or trail
4. **EXIT NOW** - Clear reversal, volume fade, or breakdown

Look for:
- Volume patterns (fading = exit signal)
- Candlestick reversal patterns
- Support/resistance reactions
- Momentum divergence

Respond in JSON:
{{
  "symbol": "{symbol}",
  "recommendation": "HOLD" or "TAKE_PROFIT" or "TIGHTEN_STOP" or "EXIT_NOW",
  "confidence": <1-100>,
  "new_stop": <price or null>,
  "reason": "<brief explanation>",
  "volume_assessment": "STRONG" or "FADING" or "NEUTRAL",
  "momentum": "WITH_TRADE" or "AGAINST_TRADE" or "NEUTRAL"
}}"""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
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
                        {"type": "text", "text": prompt}
                    ],
                }
            ],
        )

        response_text = response.content[0].text
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_str = response_text.split("```")[1].split("```")[0].strip()
        else:
            json_str = response_text.strip()

        return json.loads(json_str)

    except Exception as e:
        print(f"5min analysis error for {symbol}: {e}")
        return {"recommendation": "HOLD", "reason": f"Analysis error: {e}"}


async def scan_stock_for_zones(symbol: str, current_price: float) -> dict:
    """
    Complete workflow: fetch chart and analyze for S/D zones.
    """
    print(f"Scanning {symbol} @ ${current_price:.2f}...")

    # Try multiple chart sources
    image_bytes = await fetch_finviz_chart(symbol, "d")

    if not image_bytes:
        image_bytes = await fetch_bigcharts_image(symbol, "6m")

    if not image_bytes:
        print(f"  Could not fetch chart for {symbol}")
        return {"symbol": symbol, "error": "No chart available", "zones": []}

    # Save chart for reference
    chart_path = CHARTS_DIR / f"{symbol}_daily.png"
    with open(chart_path, "wb") as f:
        f.write(image_bytes)

    # Analyze with Vision
    analysis = analyze_chart_with_vision(image_bytes, symbol, current_price)
    analysis["chart_path"] = str(chart_path)

    return analysis


def convert_to_bill_fanter_format(vision_analysis: dict) -> list:
    """
    Convert vision analysis results to Bill Fanter levels format.

    BREAKOUT LOGIC:
    - LONG: Entry is ABOVE current price (breakout above resistance)
            Price must break UP through entry to trigger
    - SHORT: Entry is BELOW current price (breakdown below support)
            Price must break DOWN through entry to trigger

    Status:
    - PENDING: Price approaching entry level
    - READY: Price near entry + confidence >= 75%
    - TRIGGERED: Price hit entry, order placed
    - MISSED: Price blew past entry (LONG: current > entry, SHORT: current < entry)

    Vision detects:
    - Supply zones (resistance) = potential LONG breakout (break above)
    - Demand zones (support) = potential SHORT breakdown (break below)
    """
    levels = []
    symbol = vision_analysis.get("symbol", "")
    current_price = vision_analysis.get("current_price", 0)

    for zone in vision_analysis.get("zones", []):
        zone_low = zone.get("zone_low", 0)
        zone_high = zone.get("zone_high", 0)
        zone_type = zone.get("type", "demand")  # "demand" or "supply"

        if zone_type == "supply":
            # Supply zone (resistance) = LONG BREAKOUT setup
            # Entry is at TOP of supply zone (breakout above resistance)
            # We wait for price to break ABOVE this level
            entry = zone_high
            stop = zone_low * 0.995  # Just below zone
            risk = entry - stop
            target = entry + (risk * 3)  # 3:1 RR
            direction = "LONG"

        elif zone_type == "demand":
            # Demand zone (support) = SHORT BREAKDOWN setup
            # Entry is at BOTTOM of demand zone (breakdown below support)
            # We wait for price to break BELOW this level
            entry = zone_low
            stop = zone_high * 1.005  # Just above zone
            risk = stop - entry
            target = entry - (risk * 3)  # 3:1 RR
            direction = "SHORT"
        else:
            continue

        # Calculate distance and status for BREAKOUT style
        if direction == "LONG":
            # LONG breakout: entry should be ABOVE current price (waiting for breakout)
            # distance_pct > 0 means entry is above current (pending)
            # distance_pct < 0 means entry is below current (missed)
            distance_pct = (entry - current_price) / current_price * 100
            if current_price >= entry:
                status = "MISSED"  # Price already broke out, we missed it
            elif distance_pct <= 1.0:
                status = "PENDING"  # Close to breakout level
            else:
                status = "PENDING"
        else:
            # SHORT breakdown: entry should be BELOW current price (waiting for breakdown)
            # distance_pct > 0 means entry is below current (pending)
            # distance_pct < 0 means entry is above current (missed)
            distance_pct = (current_price - entry) / current_price * 100
            if current_price <= entry:
                status = "MISSED"  # Price already broke down, we missed it
            elif distance_pct <= 1.0:
                status = "PENDING"  # Close to breakdown level
            else:
                status = "PENDING"

        # Only include setups within 5% of entry (or missed by up to 3%)
        if status == "MISSED" and abs(distance_pct) > 3:
            continue  # Too far past, not worth tracking
        if status != "MISSED" and distance_pct > 5:
            continue  # Too far away to be actionable

        levels.append({
            "symbol": symbol,
            "direction": direction,
            "entry": round(entry, 2),
            "stop": round(stop, 2),
            "target": round(target, 2),
            "current_price": round(current_price, 2),
            "zone_low": round(zone_low, 2),
            "zone_high": round(zone_high, 2),
            "distance_pct": round(abs(distance_pct), 2),
            "status": status,
            "rr_ratio": round(abs(target - entry) / abs(entry - stop), 1) if abs(entry - stop) > 0 else 0,
            "quality": zone.get("quality", 50),
            "fresh": zone.get("fresh", False),
            "notes": f"Vision: {zone.get('notes', '')}",
            "source": "Vision AI",
            "detected_at": datetime.now().isoformat(),
            "chart_image": f"{symbol}_daily.png"
        })

    return levels


async def run_daily_scan(symbols: list[str]) -> dict:
    """
    Run the daily scan for all symbols.
    Fetches prices, charts, and analyzes each stock.
    """
    print(f"\n{'='*60}")
    print(f"DAILY VISION SCAN - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Scanning {len(symbols)} symbols...")
    print(f"{'='*60}\n")

    all_levels = []
    scan_results = []

    async with aiohttp.ClientSession() as session:
        for symbol in symbols:
            try:
                # Get current price from Yahoo
                url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1m&range=1d"
                headers = {"User-Agent": "Mozilla/5.0"}

                async with session.get(url, headers=headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        result = data.get("chart", {}).get("result", [])
                        if result:
                            current_price = result[0].get("meta", {}).get("regularMarketPrice", 0)
                        else:
                            print(f"  No price data for {symbol}")
                            continue
                    else:
                        print(f"  Price fetch failed for {symbol}")
                        continue

                # Analyze with vision
                analysis = await scan_stock_for_zones(symbol, current_price)
                scan_results.append(analysis)

                # Convert to levels format
                levels = convert_to_bill_fanter_format(analysis)
                all_levels.extend(levels)

                # Print summary
                trade_idea = analysis.get("trade_idea", {})
                if trade_idea.get("bias") != "WAIT":
                    print(f"  ✓ {symbol}: {trade_idea.get('bias')} @ ${trade_idea.get('entry_zone', 'N/A')}")
                    print(f"    Zones found: {len(analysis.get('zones', []))}")

                # Rate limit - be gentle with chart sources
                await asyncio.sleep(1.0)

            except Exception as e:
                print(f"  Error scanning {symbol}: {e}")
                continue

    # Save results
    output = {
        "scan_time": datetime.now().isoformat(),
        "symbols_scanned": len(symbols),
        "levels_found": len(all_levels),
        "levels": all_levels,
        "detailed_results": scan_results
    }

    # Save to vision_levels.json
    output_path = DATA_DIR / "vision_levels.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"SCAN COMPLETE")
    print(f"Levels found: {len(all_levels)}")
    print(f"Saved to: {output_path}")
    print(f"{'='*60}\n")

    return output


async def get_current_price(symbol: str) -> Optional[float]:
    """Get current price from Yahoo Finance."""
    async with aiohttp.ClientSession() as session:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1m&range=1d"
        headers = {"User-Agent": "Mozilla/5.0"}
        async with session.get(url, headers=headers) as resp:
            if resp.status == 200:
                data = await resp.json()
                result = data.get("chart", {}).get("result", [])
                if result:
                    return result[0].get("meta", {}).get("regularMarketPrice")
    return None


# Test symbols - start small
TEST_SYMBOLS = ["NVDA", "TSLA", "AAPL", "META", "GOOGL"]

# Full watchlist
FULL_WATCHLIST = [
    # MAG7
    "AAPL", "MSFT", "NVDA", "AMD", "TSLA", "META", "GOOGL", "AMZN", "AVGO",
    # Major tech
    "ORCL", "CRM", "ADBE", "INTC", "QCOM", "MU", "PLTR",
    # Financials
    "JPM", "BAC", "GS", "V", "MA",
    # Consumer
    "NKE", "SBUX", "MCD", "HD", "COST",
    # Healthcare
    "UNH", "LLY", "MRNA",
    # Energy
    "XOM", "CVX",
    # ETFs
    "SPY", "QQQ", "IWM"
]


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        symbols = FULL_WATCHLIST
    else:
        symbols = TEST_SYMBOLS
        print("Running test scan (5 symbols). Use --full for complete scan.")

    asyncio.run(run_daily_scan(symbols))
