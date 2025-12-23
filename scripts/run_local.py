#!/usr/bin/env python3
"""
Run the trading agent locally (without Docker).

Useful for development and testing.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from trading_agent.main import main

if __name__ == "__main__":
    print("Starting Trading Agent locally...")
    print("Press Ctrl+C to stop")
    print()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown requested...")
