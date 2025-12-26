#!/usr/bin/env python3
"""
Blind Evaluation Framework: Compare multiple AI models on Bill Fanter's watchlist.

This script runs a head-to-head competition between different model configurations
to see which best matches Bill Fanter's actual trade recommendations.

Model Configurations Tested:
1. Gemini 2.5 Flash alone
2. Gemini 2.5 Flash -> Claude Sonnet (tiered)
3. Gemini 1.5 Flash alone
4. Gemini 1.5 Flash -> Claude Sonnet (tiered)
5. Claude Haiku alone
6. Claude Haiku -> Claude Sonnet (tiered)
7. Claude Sonnet alone
8. Claude Opus alone

Usage:
    # Run evaluation on specific symbols
    python scripts/blind_evaluation.py NVDA TSLA AAPL META GOOGL

    # Run with Bill's actual recommendations for comparison
    python scripts/blind_evaluation.py --bill-file data/bill_watchlist.json
"""

import asyncio
import aiohttp
import json
import os
import time
import base64
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

# Load environment from project root
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
EVAL_DIR = DATA_DIR / "evaluations"
EVAL_DIR.mkdir(exist_ok=True)


# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

CLAUDE_MODELS = {
    "haiku": {
        "id": "claude-3-5-haiku-20241022",
        "name": "Claude 3.5 Haiku",
        "provider": "anthropic",
        "input_cost": 0.25,
        "output_cost": 1.25,
    },
    "sonnet": {
        "id": "claude-sonnet-4-20250514",
        "name": "Claude Sonnet 4",
        "provider": "anthropic",
        "input_cost": 3.00,
        "output_cost": 15.00,
    },
    "opus": {
        "id": "claude-opus-4-20250514",
        "name": "Claude Opus 4",
        "provider": "anthropic",
        "input_cost": 15.00,
        "output_cost": 75.00,
    },
}

GEMINI_MODELS = {
    "gemini-2.5-flash": {
        "id": "gemini-2.5-flash-preview-05-20",
        "name": "Gemini 2.5 Flash",
        "provider": "google",
        "input_cost": 0.15,
        "output_cost": 0.60,
    },
    "gemini-1.5-flash": {
        "id": "gemini-1.5-flash",
        "name": "Gemini 1.5 Flash",
        "provider": "google",
        "input_cost": 0.075,
        "output_cost": 0.30,
    },
}

# Model configurations to test
MODEL_CONFIGS = {
    "gemini-2.5-flash-alone": {
        "name": "Gemini 2.5 Flash (alone)",
        "screening": "gemini-2.5-flash",
        "detail": None,
    },
    "gemini-2.5-flash-sonnet": {
        "name": "Gemini 2.5 Flash -> Sonnet",
        "screening": "gemini-2.5-flash",
        "detail": "sonnet",
    },
    "gemini-1.5-flash-alone": {
        "name": "Gemini 1.5 Flash (alone)",
        "screening": "gemini-1.5-flash",
        "detail": None,
    },
    "gemini-1.5-flash-sonnet": {
        "name": "Gemini 1.5 Flash -> Sonnet",
        "screening": "gemini-1.5-flash",
        "detail": "sonnet",
    },
    "haiku-alone": {
        "name": "Claude Haiku (alone)",
        "screening": "haiku",
        "detail": None,
    },
    "haiku-sonnet": {
        "name": "Haiku -> Sonnet",
        "screening": "haiku",
        "detail": "sonnet",
    },
    "sonnet-alone": {
        "name": "Claude Sonnet (alone)",
        "screening": "sonnet",
        "detail": None,
    },
    "opus-alone": {
        "name": "Claude Opus (alone)",
        "screening": "opus",
        "detail": None,
    },
}


@dataclass
class AnalysisResult:
    """Result from a single model analysis."""
    symbol: str
    model_config: str
    model_name: str

    # Core outputs
    direction: str  # LONG, SHORT, WAIT
    confidence: int  # 1-100
    patterns_detected: list
    trend: str  # BULLISH, BEARISH, RANGING

    # Trade setup (if direction != WAIT)
    entry: Optional[float]
    stop: Optional[float]
    target: Optional[float]

    # Zones found
    supply_zones: list
    demand_zones: list

    # Performance metrics
    elapsed_seconds: float
    total_cost: float

    # Raw rationale
    rationale: str


# ============================================================================
# API CLIENTS
# ============================================================================

def get_anthropic_client():
    """Get Anthropic client."""
    import anthropic
    return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def get_gemini_client():
    """Get Google Gemini client."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        return genai
    except ImportError:
        raise ImportError("google-generativeai not installed. Run: pip install google-generativeai")


# ============================================================================
# CHART FETCHING
# ============================================================================

async def fetch_finviz_chart(symbol: str) -> bytes | None:
    """Fetch daily chart from Finviz."""
    url = f"https://finviz.com/chart.ashx?t={symbol}&ty=c&ta=0&p=d&s=l"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as resp:
                if resp.status == 200:
                    return await resp.read()
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


# ============================================================================
# ANALYSIS PROMPT
# ============================================================================

def get_analysis_prompt(symbol: str, current_price: float) -> str:
    """Get the standard analysis prompt for all models."""
    return f"""Analyze this daily chart for {symbol} (current price: ${current_price:.2f}).

You are a professional supply/demand zone trader. Provide your analysis:

1. **PATTERNS** - What chart patterns do you see?
   - Double top/bottom, higher highs/lows, breakouts, flags, wedges, etc.

2. **TREND** - Overall trend direction and strength

3. **SUPPLY ZONES** - Key resistance/selling areas with price ranges

4. **DEMAND ZONES** - Key support/buying areas with price ranges

5. **TRADE RECOMMENDATION** - Your trading bias:
   - LONG: Buy setup with entry/stop/target
   - SHORT: Sell setup with entry/stop/target
   - WAIT: No clear setup, stay on sidelines

Respond in this exact JSON format:
{{
  "symbol": "{symbol}",
  "current_price": {current_price},
  "patterns_detected": ["list of patterns"],
  "trend": "BULLISH" or "BEARISH" or "RANGING",
  "trend_strength": "STRONG" or "MODERATE" or "WEAK",
  "supply_zones": [
    {{"high": <price>, "low": <price>, "quality": 1-100}}
  ],
  "demand_zones": [
    {{"high": <price>, "low": <price>, "quality": 1-100}}
  ],
  "direction": "LONG" or "SHORT" or "WAIT",
  "confidence": 1-100,
  "entry": <price or null>,
  "stop": <price or null>,
  "target": <price or null>,
  "rationale": "brief explanation of your recommendation"
}}

Be decisive. If you see a setup, take it. Only say WAIT if there's genuinely no clear opportunity."""


# ============================================================================
# MODEL ANALYZERS
# ============================================================================

def analyze_with_claude(model_key: str, image_bytes: bytes, symbol: str, current_price: float) -> dict:
    """Analyze chart with Claude model."""
    model_info = CLAUDE_MODELS[model_key]
    client = get_anthropic_client()

    image_base64 = base64.standard_b64encode(image_bytes).decode("utf-8")
    prompt = get_analysis_prompt(symbol, current_price)

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

        # Parse JSON
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
            "success": True,
            "analysis": analysis,
            "elapsed": elapsed,
            "cost": cost,
            "tokens": {"input": input_tokens, "output": output_tokens},
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "elapsed": time.time() - start_time,
            "cost": 0,
        }


def analyze_with_gemini(model_key: str, image_bytes: bytes, symbol: str, current_price: float) -> dict:
    """Analyze chart with Gemini model."""
    model_info = GEMINI_MODELS[model_key]
    genai = get_gemini_client()

    prompt = get_analysis_prompt(symbol, current_price)

    start_time = time.time()

    try:
        # Create model
        model = genai.GenerativeModel(model_info["id"])

        # Create image part
        import PIL.Image
        import io
        image = PIL.Image.open(io.BytesIO(image_bytes))

        # Generate response
        response = model.generate_content([prompt, image])

        elapsed = time.time() - start_time
        response_text = response.text

        # Parse JSON
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_str = response_text.split("```")[1].split("```")[0].strip()
        else:
            json_str = response_text.strip()

        analysis = json.loads(json_str)

        # Estimate cost (Gemini doesn't return exact token counts easily)
        # Rough estimate: ~1000 input tokens for image + prompt, ~500 output
        estimated_input = 1500
        estimated_output = 500
        cost = (estimated_input * model_info["input_cost"] / 1_000_000) + \
               (estimated_output * model_info["output_cost"] / 1_000_000)

        return {
            "success": True,
            "analysis": analysis,
            "elapsed": elapsed,
            "cost": cost,
            "tokens": {"input": estimated_input, "output": estimated_output},
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "elapsed": time.time() - start_time,
            "cost": 0,
        }


def analyze_with_model(model_key: str, image_bytes: bytes, symbol: str, current_price: float) -> dict:
    """Route to appropriate analyzer based on model."""
    if model_key in CLAUDE_MODELS:
        return analyze_with_claude(model_key, image_bytes, symbol, current_price)
    elif model_key in GEMINI_MODELS:
        return analyze_with_gemini(model_key, image_bytes, symbol, current_price)
    else:
        return {"success": False, "error": f"Unknown model: {model_key}"}


# ============================================================================
# TIERED ANALYSIS
# ============================================================================

def run_tiered_analysis(
    config: dict,
    image_bytes: bytes,
    symbol: str,
    current_price: float
) -> AnalysisResult:
    """
    Run analysis with a model configuration.

    For tiered configs, first runs screening model, then detail model if promising.
    """
    config_name = config["name"]
    screening_model = config["screening"]
    detail_model = config.get("detail")

    total_cost = 0.0
    total_time = 0.0

    # Step 1: Screening analysis
    print(f"      Screening with {screening_model}...")
    screening_result = analyze_with_model(screening_model, image_bytes, symbol, current_price)

    if not screening_result["success"]:
        return AnalysisResult(
            symbol=symbol,
            model_config=config_name,
            model_name=screening_model,
            direction="ERROR",
            confidence=0,
            patterns_detected=[],
            trend="UNKNOWN",
            entry=None,
            stop=None,
            target=None,
            supply_zones=[],
            demand_zones=[],
            elapsed_seconds=screening_result["elapsed"],
            total_cost=0,
            rationale=f"Error: {screening_result.get('error', 'Unknown')}",
        )

    total_cost += screening_result["cost"]
    total_time += screening_result["elapsed"]
    analysis = screening_result["analysis"]

    # Step 2: If tiered and screening shows promise, run detail model
    if detail_model and analysis.get("direction") != "WAIT" and analysis.get("confidence", 0) >= 50:
        print(f"      Detail analysis with {detail_model}...")
        detail_result = analyze_with_model(detail_model, image_bytes, symbol, current_price)

        if detail_result["success"]:
            total_cost += detail_result["cost"]
            total_time += detail_result["elapsed"]
            analysis = detail_result["analysis"]  # Use detail model's analysis

    # Build result
    return AnalysisResult(
        symbol=symbol,
        model_config=config_name,
        model_name=detail_model or screening_model,
        direction=analysis.get("direction", "WAIT"),
        confidence=analysis.get("confidence", 0),
        patterns_detected=analysis.get("patterns_detected", []),
        trend=analysis.get("trend", "UNKNOWN"),
        entry=analysis.get("entry"),
        stop=analysis.get("stop"),
        target=analysis.get("target"),
        supply_zones=analysis.get("supply_zones", []),
        demand_zones=analysis.get("demand_zones", []),
        elapsed_seconds=round(total_time, 2),
        total_cost=round(total_cost, 4),
        rationale=analysis.get("rationale", ""),
    )


# ============================================================================
# EVALUATION RUNNER
# ============================================================================

async def run_blind_evaluation(symbols: list[str], configs_to_run: list[str] = None) -> dict:
    """
    Run blind evaluation on a list of symbols with all model configurations.

    Returns comprehensive results for comparison.
    """
    if configs_to_run is None:
        configs_to_run = list(MODEL_CONFIGS.keys())

    print(f"\n{'='*70}")
    print(f"BLIND EVALUATION - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Configurations: {len(configs_to_run)}")
    print(f"{'='*70}\n")

    results = {
        "evaluation_time": datetime.now().isoformat(),
        "symbols": symbols,
        "configs_tested": configs_to_run,
        "by_symbol": {},
        "by_config": {cfg: [] for cfg in configs_to_run},
        "summary": {},
    }

    for symbol in symbols:
        print(f"\n[{symbol}] Fetching data...")

        # Get price and chart
        current_price = await get_current_price(symbol)
        if not current_price:
            print(f"   Could not get price for {symbol}, skipping")
            continue

        image_bytes = await fetch_finviz_chart(symbol)
        if not image_bytes:
            print(f"   Could not fetch chart for {symbol}, skipping")
            continue

        # Save chart
        chart_path = CHARTS_DIR / f"{symbol}_eval.png"
        with open(chart_path, "wb") as f:
            f.write(image_bytes)

        print(f"   Price: ${current_price:.2f}")

        results["by_symbol"][symbol] = {
            "current_price": current_price,
            "analyses": {},
        }

        # Run each configuration
        for config_key in configs_to_run:
            config = MODEL_CONFIGS[config_key]
            print(f"   [{config['name']}]")

            try:
                result = run_tiered_analysis(config, image_bytes, symbol, current_price)
                results["by_symbol"][symbol]["analyses"][config_key] = asdict(result)
                results["by_config"][config_key].append(asdict(result))

                # Print summary
                print(f"      -> {result.direction} ({result.confidence}%) | {result.elapsed_seconds}s | ${result.total_cost:.4f}")

            except Exception as e:
                print(f"      -> ERROR: {e}")
                error_result = {
                    "symbol": symbol,
                    "model_config": config["name"],
                    "direction": "ERROR",
                    "error": str(e),
                }
                results["by_symbol"][symbol]["analyses"][config_key] = error_result

        # Small delay between symbols
        await asyncio.sleep(0.5)

    # Calculate summary statistics
    results["summary"] = calculate_summary(results)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = EVAL_DIR / f"blind_eval_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"EVALUATION COMPLETE")
    print(f"Results saved to: {output_path}")
    print(f"{'='*70}")

    return results


def calculate_summary(results: dict) -> dict:
    """Calculate summary statistics for the evaluation."""
    summary = {}

    for config_key, analyses in results["by_config"].items():
        if not analyses:
            continue

        valid = [a for a in analyses if a.get("direction") != "ERROR"]

        summary[config_key] = {
            "total_analyses": len(analyses),
            "successful": len(valid),
            "long_calls": len([a for a in valid if a.get("direction") == "LONG"]),
            "short_calls": len([a for a in valid if a.get("direction") == "SHORT"]),
            "wait_calls": len([a for a in valid if a.get("direction") == "WAIT"]),
            "avg_confidence": sum(a.get("confidence", 0) for a in valid) / len(valid) if valid else 0,
            "total_cost": sum(a.get("total_cost", 0) for a in valid),
            "avg_time": sum(a.get("elapsed_seconds", 0) for a in valid) / len(valid) if valid else 0,
        }

    return summary


def print_comparison_table(results: dict):
    """Print a formatted comparison table."""
    print(f"\n{'='*90}")
    print("COMPARISON TABLE")
    print(f"{'='*90}")

    summary = results.get("summary", {})

    print(f"\n{'Config':<30} {'LONG':<6} {'SHORT':<6} {'WAIT':<6} {'Avg Conf':<10} {'Cost':<10} {'Time':<8}")
    print("-" * 90)

    for config_key, stats in summary.items():
        config_name = MODEL_CONFIGS.get(config_key, {}).get("name", config_key)[:28]
        print(f"{config_name:<30} {stats['long_calls']:<6} {stats['short_calls']:<6} {stats['wait_calls']:<6} "
              f"{stats['avg_confidence']:<10.1f} ${stats['total_cost']:<9.4f} {stats['avg_time']:<8.2f}s")

    print("-" * 90)


def compare_to_bill(results: dict, bill_recommendations: dict) -> dict:
    """
    Compare model results to Bill Fanter's actual recommendations.

    Args:
        results: Evaluation results from run_blind_evaluation
        bill_recommendations: Dict of {symbol: {direction: "LONG"/"SHORT"/"WAIT", ...}}

    Returns:
        Accuracy metrics for each model configuration.
    """
    accuracy = {}

    for config_key in results["by_config"]:
        matches = 0
        total = 0

        for analysis in results["by_config"][config_key]:
            symbol = analysis.get("symbol")
            if symbol not in bill_recommendations:
                continue

            bill_dir = bill_recommendations[symbol].get("direction", "").upper()
            model_dir = analysis.get("direction", "").upper()

            if bill_dir and model_dir:
                total += 1
                if bill_dir == model_dir:
                    matches += 1

        accuracy[config_key] = {
            "matches": matches,
            "total": total,
            "accuracy_pct": (matches / total * 100) if total > 0 else 0,
        }

    return accuracy


# ============================================================================
# MAIN
# ============================================================================

async def main():
    """Run the blind evaluation."""
    import sys

    # Default test symbols
    symbols = ["NVDA", "TSLA", "AAPL", "META", "GOOGL"]

    # Parse command line args
    if len(sys.argv) > 1:
        if sys.argv[1] == "--bill-file":
            # Load symbols from Bill's watchlist file
            bill_file = Path(sys.argv[2]) if len(sys.argv) > 2 else DATA_DIR / "bill_watchlist.json"
            if bill_file.exists():
                with open(bill_file) as f:
                    bill_data = json.load(f)
                symbols = list(bill_data.get("symbols", {}).keys())
                print(f"Loaded {len(symbols)} symbols from Bill's watchlist")
        else:
            # Use provided symbols
            symbols = [s.upper() for s in sys.argv[1:]]

    # Run evaluation
    results = await run_blind_evaluation(symbols)

    # Print comparison table
    print_comparison_table(results)

    # If Bill's recommendations are available, compare
    bill_file = DATA_DIR / "bill_watchlist.json"
    if bill_file.exists():
        with open(bill_file) as f:
            bill_data = json.load(f)

        if "symbols" in bill_data:
            accuracy = compare_to_bill(results, bill_data["symbols"])

            print(f"\n{'='*70}")
            print("ACCURACY vs BILL FANTER")
            print(f"{'='*70}")
            print(f"\n{'Config':<30} {'Matches':<10} {'Total':<10} {'Accuracy':<10}")
            print("-" * 70)

            for config_key, stats in sorted(accuracy.items(), key=lambda x: -x[1]["accuracy_pct"]):
                config_name = MODEL_CONFIGS.get(config_key, {}).get("name", config_key)[:28]
                print(f"{config_name:<30} {stats['matches']:<10} {stats['total']:<10} {stats['accuracy_pct']:.1f}%")


if __name__ == "__main__":
    asyncio.run(main())
