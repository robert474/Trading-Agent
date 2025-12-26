#!/usr/bin/env python3
"""
Model Comparison Test: Compare Haiku, Sonnet, and Opus on chart analysis.

Tests all three Claude models on the same chart image to compare:
- Pattern detection accuracy
- Zone identification
- Trade recommendations
- Response quality
"""

import asyncio
import aiohttp
import anthropic
import base64
import json
import os
import time
from datetime import datetime
from pathlib import Path

# Load environment from project root
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key] = value

# Anthropic client
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Model definitions with pricing (per 1M tokens)
MODELS = {
    "haiku": {
        "id": "claude-3-5-haiku-20241022",
        "name": "Claude 3.5 Haiku",
        "input_cost": 0.25,
        "output_cost": 1.25,
    },
    "sonnet": {
        "id": "claude-sonnet-4-20250514",
        "name": "Claude Sonnet 4",
        "input_cost": 3.00,
        "output_cost": 15.00,
    },
    "opus": {
        "id": "claude-opus-4-20250514",
        "name": "Claude Opus 4",
        "input_cost": 15.00,
        "output_cost": 75.00,
    }
}

DATA_DIR = Path(__file__).parent.parent / "data"
CHARTS_DIR = DATA_DIR / "charts"
CHARTS_DIR.mkdir(exist_ok=True)


async def fetch_finviz_chart(symbol: str, timeframe: str = "d") -> bytes | None:
    """Fetch chart image from Finviz."""
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
        print(f"Error fetching chart for {symbol}: {e}")
        return None


async def get_current_price(symbol: str) -> float | None:
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


def analyze_with_model(model_key: str, image_bytes: bytes, symbol: str, current_price: float) -> dict:
    """
    Analyze chart with a specific model.
    Returns analysis results plus timing and token usage.
    """
    model_info = MODELS[model_key]
    image_base64 = base64.standard_b64encode(image_bytes).decode("utf-8")

    prompt = f"""Analyze this daily chart for {symbol} (current price: ${current_price:.2f}).

As a professional supply/demand zone trader, identify:

1. **CHART PATTERNS** - What patterns do you see?
   - Double top/bottom
   - Higher highs/higher lows (uptrend) or lower highs/lower lows (downtrend)
   - Breakout/breakdown from consolidation
   - Head and shoulders
   - Bull/bear flags
   - Wedges or triangles

2. **SUPPLY ZONES** (resistance areas where selling occurred)
3. **DEMAND ZONES** (support areas where buying occurred)

4. **OVERALL TREND** - Bullish, bearish, or ranging?

5. **TRADE SETUP** - If you were trading this:
   - Direction (LONG/SHORT/WAIT)
   - Entry level
   - Stop loss
   - Target
   - Confidence (1-100)

Respond in JSON format:
{{
  "symbol": "{symbol}",
  "current_price": {current_price},
  "patterns_detected": ["list of patterns you see"],
  "trend": "BULLISH" or "BEARISH" or "RANGING",
  "trend_strength": "STRONG" or "MODERATE" or "WEAK",
  "supply_zones": [
    {{"high": <price>, "low": <price>, "quality": 1-100, "notes": "why significant"}}
  ],
  "demand_zones": [
    {{"high": <price>, "low": <price>, "quality": 1-100, "notes": "why significant"}}
  ],
  "key_levels": {{
    "resistance": [<prices>],
    "support": [<prices>]
  }},
  "trade_idea": {{
    "direction": "LONG" or "SHORT" or "WAIT",
    "entry": <price or null>,
    "stop": <price or null>,
    "target": <price or null>,
    "confidence": 1-100,
    "rationale": "brief explanation"
  }},
  "additional_observations": "any other notable features of the chart"
}}"""

    start_time = time.time()

    try:
        response = client.messages.create(
            model=model_info["id"],
            max_tokens=2000,
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

        elapsed = time.time() - start_time
        response_text = response.content[0].text

        # Parse JSON from response
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_str = response_text.split("```")[1].split("```")[0].strip()
        else:
            json_str = response_text.strip()

        analysis = json.loads(json_str)

        # Calculate cost
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cost = (input_tokens * model_info["input_cost"] / 1_000_000) + \
               (output_tokens * model_info["output_cost"] / 1_000_000)

        return {
            "model": model_info["name"],
            "model_id": model_info["id"],
            "success": True,
            "analysis": analysis,
            "elapsed_seconds": round(elapsed, 2),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cost_usd": round(cost, 4),
        }

    except anthropic.APIError as e:
        elapsed = time.time() - start_time
        return {
            "model": model_info["name"],
            "model_id": model_info["id"],
            "success": False,
            "error": str(e),
            "elapsed_seconds": round(elapsed, 2),
        }
    except json.JSONDecodeError as e:
        elapsed = time.time() - start_time
        return {
            "model": model_info["name"],
            "model_id": model_info["id"],
            "success": False,
            "error": f"JSON parse error: {e}",
            "raw_response": response_text[:500] if 'response_text' in locals() else None,
            "elapsed_seconds": round(elapsed, 2),
        }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "model": model_info["name"],
            "model_id": model_info["id"],
            "success": False,
            "error": str(e),
            "elapsed_seconds": round(elapsed, 2),
        }


def print_comparison(results: list[dict], symbol: str):
    """Print a formatted comparison of all model results."""
    print("\n" + "=" * 80)
    print(f"MODEL COMPARISON TEST: {symbol}")
    print(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    for result in results:
        print(f"\n{'─' * 80}")
        print(f"📊 {result['model']} ({result['model_id']})")
        print(f"{'─' * 80}")

        if not result["success"]:
            print(f"❌ ERROR: {result['error']}")
            continue

        analysis = result["analysis"]

        # Patterns
        patterns = analysis.get("patterns_detected", [])
        print(f"\n🔍 PATTERNS DETECTED: {len(patterns)}")
        for p in patterns:
            print(f"   • {p}")

        # Trend
        trend = analysis.get("trend", "N/A")
        strength = analysis.get("trend_strength", "N/A")
        print(f"\n📈 TREND: {trend} ({strength})")

        # Zones
        supply = analysis.get("supply_zones", [])
        demand = analysis.get("demand_zones", [])
        print(f"\n📍 ZONES FOUND:")
        print(f"   Supply Zones: {len(supply)}")
        for z in supply[:3]:  # Show top 3
            print(f"      ${z.get('low', 0):.2f} - ${z.get('high', 0):.2f} (Q:{z.get('quality', 0)})")
        print(f"   Demand Zones: {len(demand)}")
        for z in demand[:3]:  # Show top 3
            print(f"      ${z.get('low', 0):.2f} - ${z.get('high', 0):.2f} (Q:{z.get('quality', 0)})")

        # Key Levels
        levels = analysis.get("key_levels", {})
        resistance = levels.get("resistance", [])
        support = levels.get("support", [])
        print(f"\n📏 KEY LEVELS:")
        print(f"   Resistance: {', '.join([f'${x:.2f}' for x in resistance[:5]])}")
        print(f"   Support: {', '.join([f'${x:.2f}' for x in support[:5]])}")

        # Trade Idea
        trade = analysis.get("trade_idea", {})
        print(f"\n💡 TRADE IDEA:")
        print(f"   Direction: {trade.get('direction', 'N/A')}")
        if trade.get("entry"):
            print(f"   Entry: ${trade.get('entry', 0):.2f}")
            print(f"   Stop: ${trade.get('stop', 0):.2f}")
            print(f"   Target: ${trade.get('target', 0):.2f}")
        print(f"   Confidence: {trade.get('confidence', 'N/A')}%")
        print(f"   Rationale: {trade.get('rationale', 'N/A')}")

        # Additional observations
        obs = analysis.get("additional_observations", "")
        if obs:
            print(f"\n📝 ADDITIONAL: {obs[:200]}...")

        # Performance metrics
        print(f"\n⏱️  PERFORMANCE:")
        print(f"   Time: {result['elapsed_seconds']}s")
        print(f"   Tokens: {result.get('total_tokens', 'N/A')} (in:{result.get('input_tokens', 0)}, out:{result.get('output_tokens', 0)})")
        print(f"   Cost: ${result.get('cost_usd', 0):.4f}")

    # Summary comparison
    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON")
    print("=" * 80)

    print(f"\n{'Model':<25} {'Patterns':<10} {'Zones':<10} {'Conf':<8} {'Time':<8} {'Cost':<10}")
    print("-" * 80)

    for r in results:
        if r["success"]:
            a = r["analysis"]
            patterns = len(a.get("patterns_detected", []))
            zones = len(a.get("supply_zones", [])) + len(a.get("demand_zones", []))
            conf = a.get("trade_idea", {}).get("confidence", "N/A")
            print(f"{r['model']:<25} {patterns:<10} {zones:<10} {conf}%{'':<5} {r['elapsed_seconds']:<8}s ${r['cost_usd']:<9.4f}")
        else:
            print(f"{r['model']:<25} {'ERROR':<10} {'-':<10} {'-':<8} {r['elapsed_seconds']:<8}s $0.0000")

    total_cost = sum(r.get("cost_usd", 0) for r in results)
    print("-" * 80)
    print(f"{'TOTAL':<25} {'':<10} {'':<10} {'':<8} {'':<8} ${total_cost:.4f}")


async def run_comparison(symbol: str = "AMZN"):
    """Run the full comparison test for a symbol."""
    print(f"\n🚀 Starting model comparison test for {symbol}...")

    # Get current price
    print(f"   Fetching current price...")
    current_price = await get_current_price(symbol)
    if not current_price:
        print(f"❌ Could not get price for {symbol}")
        return

    print(f"   Current price: ${current_price:.2f}")

    # Fetch chart
    print(f"   Fetching chart image...")
    image_bytes = await fetch_finviz_chart(symbol)
    if not image_bytes:
        print(f"❌ Could not fetch chart for {symbol}")
        return

    # Save chart for reference
    chart_path = CHARTS_DIR / f"{symbol}_comparison_test.png"
    with open(chart_path, "wb") as f:
        f.write(image_bytes)
    print(f"   Chart saved: {chart_path}")

    # Test each model
    results = []
    for model_key in ["haiku", "sonnet", "opus"]:
        print(f"\n   Testing {MODELS[model_key]['name']}...")
        result = analyze_with_model(model_key, image_bytes, symbol, current_price)
        results.append(result)

        if result["success"]:
            print(f"   ✓ Completed in {result['elapsed_seconds']}s (${result['cost_usd']:.4f})")
        else:
            print(f"   ✗ Failed: {result['error']}")

    # Print comparison
    print_comparison(results, symbol)

    # Save results
    output_path = DATA_DIR / f"{symbol}_model_comparison.json"
    with open(output_path, "w") as f:
        json.dump({
            "symbol": symbol,
            "current_price": current_price,
            "test_time": datetime.now().isoformat(),
            "results": results
        }, f, indent=2)
    print(f"\n📄 Full results saved to: {output_path}")

    return results


if __name__ == "__main__":
    import sys

    symbol = sys.argv[1] if len(sys.argv) > 1 else "AMZN"
    asyncio.run(run_comparison(symbol))
