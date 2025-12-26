# Bill Fanter Trading Bot

An AI-powered options trading system that combines:
- **Vision AI** - Claude analyzes chart images to detect supply/demand zones and patterns
- **RAG Pipeline** - Learns from Bill Fanter's YouTube trading methodology
- **Putt Indicator** - Validates setups against historical patterns and Bill's teachings
- **Paper Trading** - Simulates trades with real-time market data

---

## Trading Methodology

### Supply/Demand Zone Trading (Bill Fanter Style)

This system follows Bill Fanter's supply/demand methodology:

1. **Supply Zones** (Resistance)
   - Areas where price consolidated then dropped sharply
   - "Launch pads" before major sell-offs
   - Entry: SHORT when price returns to zone from below

2. **Demand Zones** (Support)
   - Areas where price consolidated then rallied sharply
   - "Launch pads" before major rallies
   - Entry: LONG when price returns to zone from above

3. **Zone Quality Factors**
   - **Fresh zones** (never retested) are strongest
   - Volume confirmation at the zone
   - Speed of departure from zone
   - Multiple touches weaken a zone

4. **Entry Confirmation**
   - Wait for price to enter zone
   - Look for reversal candle patterns
   - Volume confirmation
   - Don't chase - let price come to you

### Pattern Recognition

The Vision AI detects these patterns:
- Double top/bottom
- Higher highs/higher lows (uptrend)
- Lower highs/lower lows (downtrend)
- Breakout/breakdown from consolidation
- Head and shoulders
- Bull/bear flags
- Wedges and triangles

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DAILY WORKFLOW                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   1. VISION SCAN                                                │
│   ┌─────────────┐    ┌──────────────┐    ┌─────────────────┐   │
│   │ Fetch Chart │───▶│ Claude Vision │───▶│ Detect Patterns │   │
│   │   (Finviz)  │    │   Analysis    │    │    & Zones      │   │
│   └─────────────┘    └──────────────┘    └─────────────────┘   │
│                                                  │               │
│   2. RAG VALIDATION                              ▼               │
│   ┌─────────────┐    ┌──────────────┐    ┌─────────────────┐   │
│   │  Pattern    │◀───│ Putt         │◀───│  Setup Found    │   │
│   │  Lessons    │    │ Indicator    │    │  (LONG/SHORT)   │   │
│   └─────────────┘    └──────────────┘    └─────────────────┘   │
│         │                   │                                   │
│         ▼                   ▼                                   │
│   ┌─────────────┐    ┌──────────────┐                          │
│   │ Bill's      │    │ Confidence   │                          │
│   │ Methodology │───▶│ Adjustment   │                          │
│   └─────────────┘    └──────────────┘                          │
│                             │                                   │
│   3. TRADE EXECUTION        ▼                                   │
│   ┌─────────────┐    ┌──────────────┐    ┌─────────────────┐   │
│   │   Paper     │◀───│  Validated   │    │  Discord Alert  │   │
│   │   Trader    │    │   Setup      │───▶│  (if enabled)   │   │
│   └─────────────┘    └──────────────┘    └─────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Vision Chart Analyzer (`scripts/vision_chart_analyzer.py`)

Uses Claude Vision API to analyze chart images and detect:
- Supply/demand zones with quality scores
- Chart patterns (double top, higher lows, etc.)
- Key support/resistance levels
- Trade recommendations (LONG/SHORT/WAIT)

```python
# Analyze a single chart
analysis = await scan_stock_for_zones("NVDA", current_price=140.50)

# Run daily scan on watchlist
results = await run_daily_scan(["NVDA", "TSLA", "AAPL", "META"])
```

**Cost per analysis:**
| Model | Cost/Chart | Speed | Quality |
|-------|------------|-------|---------|
| Haiku | $0.0008 | 7s | Basic screening |
| Sonnet | $0.011 | 12s | Good detail |
| Opus | $0.056 | 16s | Maximum detail |

### 2. RAG Database (`src/rag_pipeline/rag_database.py`)

ChromaDB-powered vector database storing Bill Fanter's methodology:
- **Transcript segments** - Searchable video transcripts
- **Pattern lessons** - How to trade specific patterns
- **Risk rules** - Position sizing, stop placement
- **Entry confirmations** - What to look for before entry
- **Sector correlations** - How related stocks move together

```python
from src.rag_pipeline.rag_database import BillFanterRAG

rag = BillFanterRAG()

# Get pattern-specific lessons
context = rag.get_pattern_context("double_top", trend="downtrend")

# Get sector correlations for a symbol
correlations = rag.get_sector_correlation_context("NVDA")
```

**Sector Groups:**
- `mega_tech`: AAPL, MSFT, GOOGL, AMZN, META, NVDA
- `semiconductors`: NVDA, AMD, AVGO, INTC, MU, QCOM, TSM
- `crypto_adjacent`: COIN, HOOD, MSTR, MARA, RIOT
- `meme_retail`: TSLA, PLTR, GME, AMC, HOOD
- `banks`: JPM, BAC, WFC, GS, MS, C
- And more...

### 3. Putt Indicator (`src/trading_agent/putt_indicator.py`)

The "brain" that validates setups against historical data:

```python
from src.trading_agent.putt_indicator import PuttIndicator

putt = PuttIndicator()

# Analyze a setup
context = putt.analyze_setup(
    symbol="NVDA",
    direction="long",
    zone_type="demand",
    zone_level=135.0,
    base_confidence=65.0,
    pattern_type="higher_low",  # From vision analysis
    trend="uptrend"
)

print(f"Final Confidence: {context.final_confidence}")
print(f"Win Rate: {context.win_rate}%")
print(f"Summary: {context.summary}")
```

**Confidence Adjustments:**
| Factor | Adjustment |
|--------|------------|
| Pattern lesson found | +10 |
| High win rate (>70%) | +10 |
| Pattern recognized | +5 |
| Similar trades found | +1 to +5 |
| Bill mentioned symbol | +3 |
| Low win rate (<40%) | -10 |

### 4. Paper Trader (`src/trading_agent/paper_trader.py`)

Simulates live trading with real market data:
- Real-time price updates via Polygon
- Zone detection with confirmation filters
- Options chain integration
- Position tracking and P&L

```bash
# Start paper trading
python -m trading_agent.paper_trader --watchlist SPY,QQQ,AAPL,NVDA

# Polling mode (simpler)
python -m trading_agent.paper_trader --watchlist SPY,QQQ --polling --interval 30
```

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- Anthropic API key (for Claude Vision)
- Polygon.io API key (for market data)
- Optional: Tradier account (for live trading)

### Installation

```bash
# Clone repository
git clone <repo-url>
cd trading-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -e .

# Or install specific packages
pip install anthropic chromadb sentence-transformers aiohttp structlog
```

### Configuration

Copy `.env.example` to `.env` and fill in your API keys:

```bash
# Required
ANTHROPIC_API_KEY=sk-ant-api...
POLYGON_API_KEY=your_polygon_key

# Optional
TRADIER_ACCESS_TOKEN=your_tradier_token
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
```

### Initialize RAG Database

```bash
# Index Bill Fanter's methodology from transcripts
python -c "
from src.rag_pipeline.rag_database import BillFanterRAG
rag = BillFanterRAG()
rag.index_all_data()
"
```

---

## Daily Usage

### 1. Run Vision Scan

```bash
# Test scan (5 symbols)
python scripts/vision_chart_analyzer.py

# Full scan (all watchlist)
python scripts/vision_chart_analyzer.py --full
```

### 2. Model Comparison Test

Compare Haiku, Sonnet, and Opus on any chart:

```bash
python scripts/model_comparison_test.py AMZN
python scripts/model_comparison_test.py NVDA
```

### 3. View Top Setups

```bash
python scripts/top_setups.py
```

### 4. Start Paper Trading

```bash
python scripts/demo_paper_trader.py
```

---

## Cost Optimization

### Recommended: Tiered Approach

1. **Screen with Haiku** (~$0.04 for 53 symbols)
   - Quick pattern/trend identification
   - Filter to promising setups

2. **Analyze with Sonnet** (~$0.11 per setup)
   - Detailed zone analysis
   - Specific entry/stop/target levels

3. **Skip Opus** unless you need maximum detail
   - 5x cost of Sonnet
   - Only 2% better on vision benchmarks

### Monthly Cost Estimates (20 trading days)

| Strategy | Daily Cost | Monthly Cost |
|----------|-----------|--------------|
| Haiku only | $0.04 | $0.80 |
| Tiered (Haiku + Sonnet for 5 setups) | $0.60 | $12 |
| Sonnet all | $0.60 | $12 |
| Opus all | $3.00 | $60 |

---

## Project Structure

```
trading-agent/
├── src/
│   ├── rag_pipeline/           # RAG database and indexing
│   │   ├── rag_database.py     # ChromaDB vector store
│   │   ├── pattern_extractor.py # Extract lessons from transcripts
│   │   ├── transcribe.py       # YouTube transcription
│   │   └── align_data.py       # Align transcripts with charts
│   │
│   └── trading_agent/
│       ├── putt_indicator.py   # Setup validation with RAG
│       ├── paper_trader.py     # Paper trading engine
│       ├── analysis/
│       │   ├── zone_detector.py    # S/D zone detection
│       │   └── market_analyzer.py  # Market regime analysis
│       ├── data/
│       │   └── providers/
│       │       ├── polygon.py      # Market data
│       │       └── tradier.py      # Brokerage
│       ├── execution/
│       │   ├── order_manager.py    # Order execution
│       │   └── position_manager.py # Position tracking
│       └── monitoring/
│           └── discord_alerts.py   # Trade notifications
│
├── scripts/
│   ├── vision_chart_analyzer.py    # Main vision scan script
│   ├── model_comparison_test.py    # Compare Haiku/Sonnet/Opus
│   ├── daily_vision_scan.py        # Daily scan automation
│   ├── top_setups.py               # View best setups
│   └── demo_paper_trader.py        # Paper trading demo
│
├── data/
│   ├── charts/                 # Saved chart images
│   ├── vision_levels.json      # Detected zones
│   ├── transcriber/
│   │   ├── transcripts/        # Video transcripts
│   │   ├── signals/            # Extracted trade signals
│   │   └── patterns/           # Pattern lessons
│   └── rag_pipeline/
│       └── chroma_db/          # Vector database
│
├── config/                     # Configuration files
├── dashboard/                  # Web dashboard (optional)
├── tests/                      # Unit tests
├── .env.example               # Environment template
├── pyproject.toml             # Package config
└── docker-compose.yml         # Docker setup
```

---

## API Reference

### Vision Chart Analyzer

```python
# Fetch and analyze chart
analysis = await scan_stock_for_zones(symbol: str, current_price: float) -> dict

# Returns:
{
    "symbol": "NVDA",
    "current_price": 140.50,
    "overall_bias": "BULLISH",
    "zones": [
        {
            "type": "demand",
            "direction": "LONG",
            "zone_low": 135.0,
            "zone_high": 138.0,
            "quality": 85,
            "fresh": True,
            "notes": "Strong bounce with volume"
        }
    ],
    "key_levels": {
        "resistance": [145.0, 150.0],
        "support": [135.0, 130.0]
    },
    "trade_idea": {
        "bias": "LONG",
        "entry_zone": 136.0,
        "target": 145.0,
        "stop": 133.0,
        "rationale": "..."
    }
}
```

### Putt Indicator

```python
context = putt.analyze_setup(...) -> PuttContext

# PuttContext fields:
- symbol: str
- direction: str ("long" or "short")
- zone_type: str ("demand" or "supply")
- base_confidence: float (0-100)
- putt_adjustment: float (-20 to +25)
- final_confidence: float (0-100)
- similar_trades: list
- win_rate: Optional[float]
- avg_rr: Optional[float]
- key_insights: list
- zone_history: list
- pattern_lessons: list  # Bill's lessons for this pattern
- sector_context: dict   # Related stocks
- bill_mentioned: bool
- pattern_recognized: bool
- summary: str
```

### RAG Database

```python
# Query for similar content
results = rag.query(
    query_text: str,
    n_results: int = 10,
    doc_type: Optional[str] = None  # "pattern_lesson", "trade_signal", etc.
) -> list[dict]

# Get pattern-specific context
context = rag.get_pattern_context(
    pattern_type: str,  # "double_top", "higher_low", etc.
    trend: Optional[str] = None,
    market_regime: Optional[str] = None
) -> dict

# Get sector correlations
context = rag.get_sector_correlation_context(symbol: str) -> dict
```

---

## Troubleshooting

### API Credit Issues

```
Error: Your credit balance is too low to access the Anthropic API
```

Add credits at https://console.anthropic.com/settings/billing (separate from Claude.ai subscription).

### ChromaDB Import Error

```bash
pip install chromadb sentence-transformers
```

### Missing Environment Variables

Ensure `.env` file exists with required keys:
```bash
cp .env.example .env
# Edit .env with your API keys
```

---

## License

MIT License - See LICENSE file for details.

---

## Acknowledgments

- Bill Fanter for the supply/demand methodology
- Anthropic for Claude Vision API
- Polygon.io for market data
