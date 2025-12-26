#!/usr/bin/env python3
"""
Quick Evaluation: Test a subset of models on a single symbol.

Useful for quick testing before running the full blind evaluation.

Usage:
    python scripts/quick_eval.py NVDA
    python scripts/quick_eval.py NVDA TSLA
"""

import asyncio
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from blind_evaluation import (
    run_blind_evaluation,
    print_comparison_table,
    MODEL_CONFIGS,
)


async def main():
    # Get symbols from args
    symbols = sys.argv[1:] if len(sys.argv) > 1 else ["NVDA"]
    symbols = [s.upper() for s in symbols]

    # Run subset of configs for quick test
    quick_configs = [
        "gemini-2.5-flash-alone",
        "haiku-alone",
        "sonnet-alone",
    ]

    print(f"\nQuick evaluation on: {', '.join(symbols)}")
    print(f"Configs: {', '.join(quick_configs)}")

    results = await run_blind_evaluation(symbols, configs_to_run=quick_configs)
    print_comparison_table(results)


if __name__ == "__main__":
    asyncio.run(main())
