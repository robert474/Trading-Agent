"""
Trading Dashboard API - FastAPI backend for the trading dashboard.

Provides:
- Real-time position data
- Pending setups near trigger zones
- Historical trade performance
- WebSocket for live updates

Run with:
    uvicorn dashboard.api:app --reload --port 8000
"""

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trading_agent.analysis.zone_detector import ZoneDetector
from trading_agent.data.providers.polygon import PolygonDataProvider
from trading_agent.core.models import ZoneType
from trading_agent.putt_indicator import PuttIndicator

app = FastAPI(title="Bill Fanter Trading Dashboard")

# CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data directories
DATA_DIR = Path(__file__).parent.parent / "data" / "paper_trading"
CHARTS_DIR = Path(__file__).parent.parent / "data" / "charts"
CHARTS_DIR.mkdir(exist_ok=True)

# Mount static files for chart images
app.mount("/charts", StaticFiles(directory=str(CHARTS_DIR)), name="charts")

# Cache for setups (to avoid rate limiting)
_setups_cache = {"data": [], "timestamp": None}
CACHE_TTL_SECONDS = 60  # Refresh every 1 minute (lightweight - just price updates)

# In-memory price overrides for extended hours
_price_overrides: dict[str, float] = {}

# Initialize Putt Indicator (RAG-based validation)
_putt_indicator = None

def get_putt_indicator() -> PuttIndicator:
    """Lazy load Putt Indicator to avoid slow startup."""
    global _putt_indicator
    if _putt_indicator is None:
        _putt_indicator = PuttIndicator()
    return _putt_indicator


async def get_yahoo_price(symbol: str, session) -> float | None:
    """Fetch real-time price from Yahoo Finance (free, no API key needed)."""
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1m&range=1d"
        headers = {"User-Agent": "Mozilla/5.0"}
        async with session.get(url, headers=headers) as resp:
            if resp.status == 200:
                data = await resp.json()
                result = data.get("chart", {}).get("result", [])
                if result:
                    meta = result[0].get("meta", {})
                    # regularMarketPrice is the live price
                    return meta.get("regularMarketPrice")
    except Exception as e:
        print(f"Yahoo price error for {symbol}: {e}")
    return None


class ConnectionManager:
    """Manage WebSocket connections for real-time updates."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass


manager = ConnectionManager()


def load_state() -> dict:
    """Load paper trading state from disk."""
    state_file = DATA_DIR / "paper_trading_state.json"
    if not state_file.exists():
        return {
            "capital": 10000,
            "starting_capital": 10000,
            "total_pnl": 0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0,
            "open_positions": [],
            "closed_trades": [],
            "timestamp": datetime.now().isoformat(),
        }

    with open(state_file) as f:
        return json.load(f)


@app.get("/api/status")
async def get_status():
    """Get current trading status and account summary."""
    state = load_state()

    return {
        "status": "running" if state.get("open_positions") else "idle",
        "capital": state.get("capital", 10000),
        "starting_capital": state.get("starting_capital", 10000),
        "total_pnl": state.get("total_pnl", 0),
        "total_pnl_pct": (state.get("capital", 10000) - state.get("starting_capital", 10000)) / state.get("starting_capital", 10000) * 100,
        "total_trades": state.get("total_trades", 0),
        "winning_trades": state.get("winning_trades", 0),
        "losing_trades": state.get("losing_trades", 0),
        "win_rate": state.get("win_rate", 0),
        "open_positions_count": len(state.get("open_positions", [])),
        "last_update": state.get("timestamp"),
    }


@app.get("/api/positions")
async def get_positions():
    """Get all open positions with current P&L from live prices."""
    import aiohttp

    state = load_state()
    positions = state.get("open_positions", [])
    polygon = PolygonDataProvider()

    enriched = []
    for pos in positions:
        symbol = pos.get("symbol", "")
        entry_price = pos.get("entry_price", 0)
        direction = pos.get("direction", "long")

        # Check for manual price override first
        if symbol in _price_overrides:
            current_price = _price_overrides[symbol]
        else:
            # Try Yahoo Finance first (real-time), fall back to Polygon
            current_price = pos.get("current_price", entry_price)
            try:
                async with aiohttp.ClientSession() as session:
                    # Try Yahoo Finance for real-time prices
                    yahoo_price = await get_yahoo_price(symbol, session)
                    if yahoo_price:
                        current_price = yahoo_price
                    else:
                        # Fall back to Polygon snapshot
                        url = f"https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}?apiKey={polygon.api_key}"
                        async with session.get(url) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                ticker_data = data.get("ticker", {})
                                if ticker_data.get("day", {}).get("c"):
                                    current_price = ticker_data["day"]["c"]
                                elif ticker_data.get("min", {}).get("c"):
                                    current_price = ticker_data["min"]["c"]
                                elif ticker_data.get("prevDay", {}).get("c"):
                                    current_price = ticker_data["prevDay"]["c"]
            except Exception as e:
                print(f"Error fetching price for {symbol}: {e}")

        # Calculate P&L based on direction
        if direction == "long":
            pnl = (current_price - entry_price) * pos.get("quantity", 100)
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
        else:  # short
            pnl = (entry_price - current_price) * pos.get("quantity", 100)
            pnl_pct = ((entry_price - current_price) / entry_price) * 100

        # Calculate option P&L estimate (simplified - delta ~0.5 for ATM)
        option_entry = pos.get("option_entry_premium", 0)
        contracts = pos.get("option_contracts", 1)
        if option_entry > 0:
            # Rough estimate: option moves ~50% of stock move for ATM
            stock_move_pct = pnl_pct
            option_pnl_pct = stock_move_pct * 2.5  # Options leverage ~2.5x for ATM weekly
            option_current = option_entry * (1 + option_pnl_pct / 100)
            option_pnl = (option_current - option_entry) * contracts * 100
        else:
            option_pnl = 0
            option_pnl_pct = 0
            option_current = 0

        enriched.append({
            **pos,
            "current_price": round(current_price, 2),
            "unrealized_pnl": round(pnl, 2),
            "unrealized_pnl_pct": round(pnl_pct, 2),
            "option_current_premium": round(option_current, 2) if option_entry > 0 else None,
            "option_pnl": round(option_pnl, 2) if option_entry > 0 else None,
            "option_pnl_pct": round(option_pnl_pct, 2) if option_entry > 0 else None,
            "pnl_color": "green" if pnl >= 0 else "red",
            "time_in_trade": _time_since(pos.get("entry_time")),
            "chart_5min": f"{symbol}_5min.png",  # 5-min chart for position monitoring
        })

        await asyncio.sleep(0.15)  # Rate limiting

    return {"positions": enriched}


@app.get("/api/trades")
async def get_trades(limit: int = 50):
    """Get historical closed trades."""
    state = load_state()
    trades = state.get("closed_trades", [])

    # Sort by exit time, newest first
    trades = sorted(trades, key=lambda x: x.get("exit_time", ""), reverse=True)

    return {"trades": trades[:limit]}


@app.get("/api/performance")
async def get_performance():
    """Get performance metrics and equity curve data."""
    state = load_state()
    trades = state.get("closed_trades", [])

    # Build equity curve
    equity_curve = []
    running_equity = state.get("starting_capital", 10000)

    for trade in sorted(trades, key=lambda x: x.get("exit_time", "")):
        # Support both 'pnl' and 'realized_pnl' field names
        trade_pnl = trade.get("realized_pnl", trade.get("pnl", 0))
        running_equity += trade_pnl
        equity_curve.append({
            "time": trade.get("exit_time"),
            "equity": running_equity,
            "trade_pnl": trade_pnl,
            "symbol": trade.get("symbol", ""),
            "result": trade.get("result", "WIN" if trade_pnl > 0 else "LOSS"),
        })

    # Calculate stats - support both field names
    def get_pnl(t):
        return t.get("realized_pnl", t.get("pnl", 0))

    wins = [t for t in trades if get_pnl(t) > 0]
    losses = [t for t in trades if get_pnl(t) < 0]

    avg_win = sum(get_pnl(t) for t in wins) / len(wins) if wins else 0
    avg_loss = sum(get_pnl(t) for t in losses) / len(losses) if losses else 0

    return {
        "equity_curve": equity_curve,
        "stats": {
            "total_trades": len(trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": len(wins) / len(trades) * 100 if trades else 0,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": abs(avg_win / avg_loss) if avg_loss != 0 else 0,
            "largest_win": max((get_pnl(t) for t in trades), default=0),
            "largest_loss": min((get_pnl(t) for t in trades), default=0),
        }
    }


def load_bill_fanter_levels() -> dict:
    """Load Bill Fanter's levels from the latest watchlist."""
    levels_file = Path(__file__).parent.parent / "data" / "bill_fanter_levels.json"
    if levels_file.exists():
        with open(levels_file) as f:
            return json.load(f)
    return {}


def calculate_confidence_score(
    wick_rejection: bool = False,
    volume_spike: bool = False,
    candle_pattern: str | None = None,
    trend_confirmed: bool = False,
) -> tuple[int, list[str]]:
    """
    Calculate CONFIDENCE score based on the 4 entry confirmation indicators.

    This answers: "Should I enter NOW?"

    The 4 indicators (each worth 25 pts, max 100):
    1. Wick Rejection - Price tests zone, shows rejection wick (+25 pts)
    2. Volume Spike - Higher volume = institutional activity (+25 pts)
    3. Candle Pattern - Engulfing, pin bars, hammer, etc. (+25 pts)
    4. Trend Aligned - Higher timeframe trend confirmation (+25 pts)

    Returns (confidence_score 0-100, list of active confirmations)
    """
    score = 0
    confirmations = []

    # 1. Wick Rejection (+25 points)
    if wick_rejection:
        score += 25
        confirmations.append("wick_rejection")

    # 2. Volume Spike (+25 points)
    if volume_spike:
        score += 25
        confirmations.append("volume_spike")

    # 3. Candle Pattern (+25 points)
    if candle_pattern:
        score += 25
        confirmations.append(f"pattern_{candle_pattern.lower().replace(' ', '_')}")

    # 4. Trend Aligned (+25 points)
    if trend_confirmed:
        score += 25
        confirmations.append("trend_aligned")

    return score, confirmations


async def check_entry_confirmations(
    symbol: str,
    entry_price: float,
    direction: str,
    polygon: PolygonDataProvider,
) -> tuple[int, list[str]]:
    """
    Check the 4 entry confirmation indicators for a setup approaching entry.

    Returns (confidence_score 0-100, list of confirmations)
    """
    try:
        # Get recent 5-minute candles for analysis
        candles = await polygon.get_historical_candles(
            symbol=symbol,
            timeframe="5",
            from_date=datetime.now() - timedelta(hours=4),
            to_date=datetime.now(),
        )

        if not candles or len(candles) < 10:
            return 0, []

        # Get the most recent candles
        recent = candles[-10:]  # Last 10 candles (50 minutes)
        latest = candles[-1]
        prev = candles[-2] if len(candles) >= 2 else None

        wick_rejection = False
        volume_spike = False
        candle_pattern = None
        trend_confirmed = False

        # 1. Check Wick Rejection
        # For LONG: Price tested near entry and bounced up (lower wick > body)
        # For SHORT: Price tested near entry and rejected down (upper wick > body)
        if direction == "LONG":
            body_size = abs(latest.close - latest.open)
            lower_wick = min(latest.open, latest.close) - latest.low
            # Wick rejection if lower wick is > 2x body and price near entry
            if lower_wick > body_size * 2 and latest.low <= entry_price * 1.01:
                wick_rejection = True
        else:  # SHORT
            body_size = abs(latest.close - latest.open)
            upper_wick = latest.high - max(latest.open, latest.close)
            if upper_wick > body_size * 2 and latest.high >= entry_price * 0.99:
                wick_rejection = True

        # 2. Check Volume Spike
        # Volume should be significantly higher than average
        avg_volume = sum(c.volume for c in recent[:-1]) / len(recent[:-1])
        if latest.volume > avg_volume * 1.5:
            volume_spike = True

        # 3. Check Candle Patterns
        if prev:
            # Bullish patterns for LONG
            if direction == "LONG":
                # Bullish engulfing
                if (prev.close < prev.open and  # prev was red
                    latest.close > latest.open and  # current is green
                    latest.close > prev.open and  # current close > prev open
                    latest.open < prev.close):  # current open < prev close
                    candle_pattern = "bullish_engulfing"
                # Hammer (small body, long lower wick)
                elif (body_size > 0 and
                      lower_wick > body_size * 2 and
                      (latest.high - max(latest.open, latest.close)) < body_size * 0.5):
                    candle_pattern = "hammer"
            else:  # SHORT
                # Bearish engulfing
                if (prev.close > prev.open and  # prev was green
                    latest.close < latest.open and  # current is red
                    latest.close < prev.open and  # current close < prev open
                    latest.open > prev.close):  # current open > prev close
                    candle_pattern = "bearish_engulfing"
                # Shooting star (small body, long upper wick)
                elif (body_size > 0 and
                      upper_wick > body_size * 2 and
                      (min(latest.open, latest.close) - latest.low) < body_size * 0.5):
                    candle_pattern = "shooting_star"

        # 4. Check Trend Alignment (using 20-period SMA direction)
        if len(candles) >= 20:
            sma_20 = sum(c.close for c in candles[-20:]) / 20
            sma_10 = sum(c.close for c in candles[-10:]) / 10
            if direction == "LONG" and sma_10 > sma_20:
                trend_confirmed = True
            elif direction == "SHORT" and sma_10 < sma_20:
                trend_confirmed = True

        return calculate_confidence_score(
            wick_rejection=wick_rejection,
            volume_spike=volume_spike,
            candle_pattern=candle_pattern,
            trend_confirmed=trend_confirmed,
        )

    except Exception as e:
        print(f"Error checking confirmations for {symbol}: {e}")
        return 0, []


def calculate_trade_plan(level: dict, current_price: float) -> dict:
    """Calculate entry, target, stop based on Bill Fanter's level."""
    symbol = level.get("symbol", "")
    key_levels = level.get("key_levels", {})
    bias = level.get("bias", "neutral")

    # Determine direction and entry based on Bill's calls
    entry = None
    target = None
    stop = None
    direction = None

    # Bullish setups - breakout entries
    if key_levels.get("calls_above"):
        entry = key_levels["calls_above"]
        direction = "LONG"
        # Target is next level up or +5%
        target = key_levels.get("target", key_levels.get("target1", entry * 1.05))
        # Stop is entry - 1.5% or below support
        stop = key_levels.get("support", entry * 0.985)

    elif key_levels.get("breakout"):
        entry = key_levels["breakout"]
        direction = "LONG"
        target = key_levels.get("target", key_levels.get("target1", entry * 1.05))
        stop = entry * 0.985

    elif key_levels.get("hold_for_gap"):
        entry = key_levels["hold_for_gap"]
        direction = "LONG"
        target = entry * 1.05  # Gap fill target
        stop = entry * 0.985

    # Bearish setups
    elif key_levels.get("short_at"):
        entry = key_levels["short_at"]
        direction = "SHORT"
        target = key_levels.get("target1", key_levels.get("target", entry * 0.95))
        stop = entry * 1.02

    elif key_levels.get("puts_below"):
        entry = key_levels["puts_below"]
        direction = "SHORT"
        target = key_levels.get("target", entry * 0.95)
        stop = entry * 1.015

    elif key_levels.get("puts_at"):
        entry = key_levels["puts_at"]
        direction = "SHORT"
        target = entry * 0.95
        stop = entry * 1.02

    # Range plays
    elif key_levels.get("calls_at") and key_levels.get("puts_at"):
        # Range - determine based on current price
        if current_price <= key_levels["calls_at"] * 1.01:
            entry = key_levels["calls_at"]
            direction = "LONG"
            target = key_levels["puts_at"]
            stop = entry * 0.985
        elif current_price >= key_levels["puts_at"] * 0.99:
            entry = key_levels["puts_at"]
            direction = "SHORT"
            target = key_levels["calls_at"]
            stop = entry * 1.015

    # Calculate distance and status for BREAKOUT style
    # LONG: entry should be ABOVE current price (waiting for breakout)
    # SHORT: entry should be BELOW current price (waiting for breakdown)
    status = "PENDING"
    if entry:
        if direction == "LONG":
            # LONG breakout: entry above current = pending, entry below current = missed
            distance_pct = (entry - current_price) / current_price * 100
            if current_price >= entry:
                status = "MISSED"  # Price already broke out
            elif distance_pct <= 1.0:
                status = "PENDING"  # Close to breakout
            else:
                status = "PENDING"
        else:
            # SHORT breakdown: entry below current = pending, entry above current = missed
            distance_pct = (current_price - entry) / current_price * 100
            if current_price <= entry:
                status = "MISSED"  # Price already broke down
            elif distance_pct <= 1.0:
                status = "PENDING"  # Close to breakdown
            else:
                status = "PENDING"
    else:
        distance_pct = None

    # Risk/Reward calculation
    if entry and target and stop:
        risk = abs(entry - stop)
        reward = abs(target - entry)
        rr_ratio = reward / risk if risk > 0 else 0
    else:
        rr_ratio = 0

    return {
        "entry": entry,
        "target": target,
        "stop": stop,
        "direction": direction,
        "distance_pct": distance_pct,
        "status": status,
        "rr_ratio": rr_ratio,
    }


def is_bill_levels_stale(bill_levels: dict, days_valid: int = 5) -> bool:
    """Check if Bill Fanter's levels are stale (older than days_valid)."""
    date_str = bill_levels.get("date", "")
    if not date_str:
        return True
    try:
        level_date = datetime.strptime(date_str, "%Y-%m-%d")
        return (datetime.now() - level_date).days > days_valid
    except:
        return True


async def get_auto_detected_setup(
    symbol: str,
    current_price: float,
    polygon: PolygonDataProvider,
    zone_detector: ZoneDetector,
) -> list[dict]:
    """Auto-detect supply/demand zones for a symbol."""
    setups = []

    try:
        # Get candles for zone detection
        candles = await polygon.get_historical_candles(
            symbol=symbol,
            timeframe="15",
            from_date=datetime.now() - timedelta(days=30),
            to_date=datetime.now(),
        )

        if not candles or len(candles) < 50:
            return []

        # Detect zones
        zones = zone_detector.detect_zones(candles, "15m")
        nearest = zone_detector.find_nearest_zones(current_price, zones)

        # Check supply zone (LONG BREAKOUT) - break above resistance
        if nearest.get("nearest_supply"):
            z = nearest["nearest_supply"]
            # LONG breakout: Entry at TOP of supply zone (break above resistance)
            entry = z.zone_high
            # LONG: entry should be ABOVE current price (waiting for breakout)
            dist_pct = (entry - current_price) / current_price * 100

            # Determine status
            if current_price >= entry:
                status = "MISSED"  # Already broke out
            else:
                status = f"PENDING ({abs(dist_pct):.1f}% away)"

            # Only show if within 3% of entry or recently missed
            if abs(dist_pct) < 3.0:
                # Target: +3% from entry
                target = entry * 1.03
                # Stop: below zone
                stop = z.zone_low * 0.995

                risk = abs(entry - stop)
                reward = abs(target - entry)
                rr_ratio = reward / risk if risk > 0 else 0

                # Check confidence indicators when close to entry (within 2%)
                confidence = 0
                confirmations = []
                if status != "MISSED" and abs(dist_pct) <= 2.0:
                    confidence, confirmations = await check_entry_confirmations(
                        symbol=symbol,
                        entry_price=entry,
                        direction="LONG",
                        polygon=polygon,
                    )
                    if confidence >= 75:
                        status = f"READY ({confidence}%)"

                setups.append({
                    "symbol": symbol,
                    "direction": "LONG",
                    "bias": "auto-supply",
                    "current_price": round(current_price, 2),
                    "entry": round(entry, 2),
                    "target": round(target, 2),
                    "stop": round(stop, 2),
                    "zone_low": round(z.zone_low, 2),
                    "zone_high": round(z.zone_high, 2),
                    "distance_pct": round(abs(dist_pct), 2),
                    "rr_ratio": round(rr_ratio, 2),
                    "notes": f"Auto-detected supply zone (breakout), quality: {z.quality_score:.0f}",
                    "status": status,
                    "source": "Auto-Detected Zone",
                    "quality": z.quality_score,
                    "freshness": z.freshness.value,
                    "confidence_score": confidence,
                    "confirmations": confirmations,
                })

        # Check demand zone (SHORT BREAKDOWN) - break below support
        if nearest.get("nearest_demand"):
            z = nearest["nearest_demand"]
            # SHORT breakdown: Entry at BOTTOM of demand zone (break below support)
            entry = z.zone_low
            # SHORT: entry should be BELOW current price (waiting for breakdown)
            dist_pct = (current_price - entry) / current_price * 100

            # Determine status
            if current_price <= entry:
                status = "MISSED"  # Already broke down
            else:
                status = f"PENDING ({abs(dist_pct):.1f}% away)"

            if abs(dist_pct) < 3.0:
                target = entry * 0.97
                stop = z.zone_high * 1.005

                risk = abs(entry - stop)
                reward = abs(target - entry)
                rr_ratio = reward / risk if risk > 0 else 0

                # Check confidence indicators when close to entry (within 2%)
                confidence = 0
                confirmations = []
                if status != "MISSED" and abs(dist_pct) <= 2.0:
                    confidence, confirmations = await check_entry_confirmations(
                        symbol=symbol,
                        entry_price=entry,
                        direction="SHORT",
                        polygon=polygon,
                    )
                    if confidence >= 75:
                        status = f"READY ({confidence}%)"

                setups.append({
                    "symbol": symbol,
                    "direction": "SHORT",
                    "bias": "auto-demand",
                    "current_price": round(current_price, 2),
                    "entry": round(entry, 2),
                    "target": round(target, 2),
                    "stop": round(stop, 2),
                    "zone_low": round(z.zone_low, 2),
                    "zone_high": round(z.zone_high, 2),
                    "distance_pct": round(abs(dist_pct), 2),
                    "rr_ratio": round(rr_ratio, 2),
                    "notes": f"Auto-detected demand zone (breakdown), quality: {z.quality_score:.0f}",
                    "status": status,
                    "source": "Auto-Detected Zone",
                    "quality": z.quality_score,
                    "freshness": z.freshness.value,
                    "confidence_score": confidence,
                    "confirmations": confirmations,
                })

    except Exception as e:
        print(f"Error auto-detecting zones for {symbol}: {e}")

    return setups


@app.get("/api/setups")
async def get_pending_setups():
    """Get pending setups combining Bill Fanter's levels AND auto-detected zones."""
    global _setups_cache

    # Check cache first
    if (_setups_cache["timestamp"] and
        (datetime.now() - _setups_cache["timestamp"]).total_seconds() < CACHE_TTL_SECONDS):
        return {"setups": _setups_cache["data"], "cached": True}

    try:
        polygon = PolygonDataProvider()
        zone_detector = ZoneDetector()
        bill_levels = load_bill_fanter_levels()

        # Track which symbols have Bill's levels
        bill_symbols = set()
        bill_stale = is_bill_levels_stale(bill_levels)

        setups = []
        import aiohttp

        # 1. First, add Bill Fanter's explicit levels (highest priority)
        if bill_levels.get("levels") and not bill_stale:
            for level in bill_levels.get("levels", []):
                symbol = level.get("symbol", "")
                bill_symbols.add(symbol)

                try:
                    async with aiohttp.ClientSession() as session:
                        # Use Yahoo Finance for real-time prices
                        current_price = await get_yahoo_price(symbol, session)
                        if not current_price:
                            # Fall back to Polygon
                            url = f"https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}?apiKey={polygon.api_key}"
                            async with session.get(url) as resp:
                                if resp.status == 200:
                                    data = await resp.json()
                                    ticker_data = data.get("ticker", {})
                                    if ticker_data.get("day", {}).get("c"):
                                        current_price = ticker_data["day"]["c"]
                                    elif ticker_data.get("min", {}).get("c"):
                                        current_price = ticker_data["min"]["c"]
                                    elif ticker_data.get("prevDay", {}).get("c"):
                                        current_price = ticker_data["prevDay"]["c"]
                                    else:
                                        continue
                                else:
                                    continue

                    await asyncio.sleep(0.1)  # Faster with Yahoo

                    trade_plan = calculate_trade_plan(level, current_price)

                    if not trade_plan.get("entry"):
                        continue

                    # Use status from trade_plan (PENDING, READY, MISSED)
                    status = trade_plan["status"]
                    dist = abs(trade_plan["distance_pct"]) if trade_plan["distance_pct"] else 0

                    # Check confidence indicators when close to entry (within 2%)
                    confidence = 0
                    confirmations = []
                    if status == "PENDING" and dist <= 2.0:
                        # Check the 4 entry confirmation indicators
                        confidence, confirmations = await check_entry_confirmations(
                            symbol=symbol,
                            entry_price=trade_plan["entry"],
                            direction=trade_plan["direction"],
                            polygon=polygon,
                        )
                        # If confidence >= 75%, mark as READY
                        if confidence >= 75:
                            status = f"READY ({confidence}%)"
                        else:
                            status = f"PENDING ({dist:.1f}% away)"
                    elif status == "PENDING":
                        status = f"PENDING ({dist:.1f}% away)"

                    # === PUTT INDICATOR VALIDATION ===
                    # Query RAG for historical context and confidence adjustment
                    try:
                        putt = get_putt_indicator()
                        putt_context = putt.analyze_setup(
                            symbol=symbol,
                            direction=trade_plan["direction"].lower(),
                            zone_type="demand" if trade_plan["direction"] == "LONG" else "supply",
                            zone_level=trade_plan["entry"],
                            base_confidence=float(confidence),
                        )
                        putt_data = {
                            "putt_adjustment": putt_context.putt_adjustment,
                            "putt_confidence": putt_context.final_confidence,
                            "putt_win_rate": putt_context.win_rate,
                            "putt_similar_trades": len(putt_context.similar_trades),
                            "putt_bill_mentioned": putt_context.bill_mentioned,
                            "putt_summary": putt_context.summary,
                            "putt_insights": putt_context.key_insights[:2] if putt_context.key_insights else [],
                        }
                    except Exception as e:
                        print(f"Putt Indicator error for {symbol}: {e}")
                        putt_data = {}

                    setups.append({
                        "symbol": symbol,
                        "direction": trade_plan["direction"],
                        "bias": level.get("bias", "neutral"),
                        "current_price": round(current_price, 2),
                        "entry": round(trade_plan["entry"], 2),
                        "target": round(trade_plan["target"], 2) if trade_plan["target"] else None,
                        "stop": round(trade_plan["stop"], 2) if trade_plan["stop"] else None,
                        "distance_pct": round(dist, 2) if trade_plan["distance_pct"] else None,
                        "rr_ratio": round(trade_plan["rr_ratio"], 2),
                        "notes": level.get("notes", ""),
                        "status": status,
                        "source": "Bill Fanter",
                        "date": bill_levels.get("date", ""),
                        "confidence_score": confidence,
                        "confirmations": confirmations,
                        **putt_data,  # Add Putt Indicator data
                    })

                except Exception as e:
                    print(f"Error processing Bill's {symbol}: {e}")
                    continue

        # 2. Add Vision AI detected levels (from daily scan)
        # This is LIGHTWEIGHT - just load pre-scanned levels and update prices
        vision_file = Path(__file__).parent.parent / "data" / "vision_levels.json"
        if vision_file.exists():
            try:
                with open(vision_file) as f:
                    vision_data = json.load(f)

                # Check if vision scan is recent (within 24 hours)
                scan_time = vision_data.get("scan_time", "")
                if scan_time:
                    from datetime import datetime as dt
                    try:
                        scan_dt = dt.fromisoformat(scan_time)
                        hours_old = (datetime.now() - scan_dt).total_seconds() / 3600
                        if hours_old < 24:
                            # Get fresh prices for vision levels
                            async with aiohttp.ClientSession() as session:
                                for level in vision_data.get("levels", []):
                                    symbol = level.get("symbol", "")
                                    # Skip if Bill already has this symbol
                                    if symbol in bill_symbols:
                                        continue

                                    # Get current price
                                    current_price = await get_yahoo_price(symbol, session)
                                    if not current_price:
                                        current_price = level.get("current_price", 0)

                                    entry = level.get("entry", 0)
                                    direction = level.get("direction", "LONG")

                                    # Recalculate distance and status with fresh price
                                    if direction == "LONG":
                                        distance_pct = (entry - current_price) / current_price * 100
                                        if current_price >= entry:
                                            status = "MISSED"
                                        else:
                                            status = f"PENDING ({abs(distance_pct):.1f}% away)"
                                    else:
                                        distance_pct = (current_price - entry) / current_price * 100
                                        if current_price <= entry:
                                            status = "MISSED"
                                        else:
                                            status = f"PENDING ({abs(distance_pct):.1f}% away)"

                                    # === PUTT INDICATOR FOR VISION LEVELS ===
                                    try:
                                        putt = get_putt_indicator()
                                        putt_context = putt.analyze_setup(
                                            symbol=symbol,
                                            direction=direction.lower(),
                                            zone_type="demand" if direction == "LONG" else "supply",
                                            zone_level=entry,
                                            base_confidence=50.0,  # Vision starts at 50%
                                        )
                                        vision_putt_data = {
                                            "putt_adjustment": putt_context.putt_adjustment,
                                            "putt_confidence": putt_context.final_confidence,
                                            "putt_win_rate": putt_context.win_rate,
                                            "putt_similar_trades": len(putt_context.similar_trades),
                                            "putt_bill_mentioned": putt_context.bill_mentioned,
                                            "putt_summary": putt_context.summary,
                                            "putt_insights": putt_context.key_insights[:2] if putt_context.key_insights else [],
                                        }
                                    except Exception as e:
                                        print(f"Putt error for Vision {symbol}: {e}")
                                        vision_putt_data = {}

                                    # Add vision level as a setup
                                    setups.append({
                                        "symbol": symbol,
                                        "direction": direction,
                                        "bias": "vision",
                                        "current_price": round(current_price, 2),
                                        "entry": entry,
                                        "target": level.get("target"),
                                        "stop": level.get("stop"),
                                        "zone_low": level.get("zone_low"),
                                        "zone_high": level.get("zone_high"),
                                        "distance_pct": round(abs(distance_pct), 2),
                                        "rr_ratio": level.get("rr_ratio", 3.0),
                                        "notes": level.get("notes", ""),
                                        "status": status,
                                        "source": "Vision AI",
                                        "quality": level.get("quality", 50),
                                        "chart_image": level.get("chart_image"),
                                        "confidence_score": 0,
                                        "confirmations": [],
                                        **vision_putt_data,  # Add Putt Indicator data
                                    })

                                    await asyncio.sleep(0.05)  # Small delay between price fetches
                    except Exception as e:
                        print(f"Error parsing vision scan time: {e}")
            except Exception as e:
                print(f"Error loading vision levels: {e}")

        # Sort: Bill's levels first, then Vision, then by status (PENDING > MISSED), then by distance
        def sort_key(x):
            source_priority = {"Bill Fanter": 0, "Vision AI": 1}.get(x["source"], 2)
            # PENDING setups first, then MISSED
            status = x.get("status", "")
            status_priority = 0 if "PENDING" in status else 1
            distance = x.get("distance_pct") or 999
            return (source_priority, status_priority, distance)

        setups.sort(key=sort_key)

        # Update cache
        _setups_cache["data"] = setups
        _setups_cache["timestamp"] = datetime.now()

        return {
            "setups": setups,
            "bill_fanter_source": bill_levels.get("title", ""),
            "bill_fanter_date": bill_levels.get("date", ""),
            "bill_fanter_stale": bill_stale,
            "market_context": bill_levels.get("market_context", {}),
            "risk_off_signals": bill_levels.get("risk_off_signals", {}),
        }

    except Exception as e:
        return {"setups": [], "error": str(e)}


class PriceOverride(BaseModel):
    """Manual price override for extended hours trading."""
    symbol: str
    price: float


class ClosePositionRequest(BaseModel):
    """Request to close a position."""
    position_id: str
    exit_price: float
    reason: str = "manual_close"


def save_state(state: dict):
    """Save paper trading state to disk."""
    state_file = DATA_DIR / "paper_trading_state.json"
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)


@app.post("/api/price/override")
async def set_price_override(override: PriceOverride):
    """Set a manual price override for extended hours trading."""
    _price_overrides[override.symbol] = override.price
    return {"status": "ok", "symbol": override.symbol, "price": override.price}


@app.get("/api/price/overrides")
async def get_price_overrides():
    """Get all current price overrides."""
    return {"overrides": _price_overrides}


@app.delete("/api/price/override/{symbol}")
async def clear_price_override(symbol: str):
    """Clear a price override."""
    if symbol in _price_overrides:
        del _price_overrides[symbol]
    return {"status": "ok", "symbol": symbol}


@app.post("/api/positions/close")
async def close_position(request: ClosePositionRequest):
    """Close a position and record the trade."""
    state = load_state()
    positions = state.get("open_positions", [])

    # Find the position
    pos_to_close = None
    pos_index = None
    for i, pos in enumerate(positions):
        if pos.get("id") == request.position_id:
            pos_to_close = pos
            pos_index = i
            break

    if not pos_to_close:
        return {"error": f"Position {request.position_id} not found"}

    # Calculate P&L
    entry_price = pos_to_close.get("entry_price", 0)
    direction = pos_to_close.get("direction", "long")
    quantity = pos_to_close.get("quantity", 100)

    if direction == "long":
        pnl = (request.exit_price - entry_price) * quantity
        pnl_pct = ((request.exit_price - entry_price) / entry_price) * 100
    else:
        pnl = (entry_price - request.exit_price) * quantity
        pnl_pct = ((entry_price - request.exit_price) / entry_price) * 100

    # Calculate option P&L estimate
    option_entry = pos_to_close.get("option_entry_premium", 0)
    option_contracts = pos_to_close.get("option_contracts", 1)
    if option_entry > 0:
        option_pnl_pct = pnl_pct * 2.5  # Simplified leverage estimate
        option_exit = option_entry * (1 + option_pnl_pct / 100)
        option_pnl = (option_exit - option_entry) * option_contracts * 100
    else:
        option_exit = 0
        option_pnl = 0
        option_pnl_pct = 0

    # Determine result
    result = "WIN" if pnl > 0 else "LOSS" if pnl < 0 else "BREAKEVEN"

    # Create closed trade record
    closed_trade = {
        **pos_to_close,
        "exit_price": request.exit_price,
        "exit_time": datetime.now().isoformat(),
        "realized_pnl": round(pnl, 2),
        "realized_pnl_pct": round(pnl_pct, 2),
        "option_exit_premium": round(option_exit, 2) if option_entry > 0 else None,
        "option_pnl": round(option_pnl, 2) if option_entry > 0 else None,
        "option_pnl_pct": round(option_pnl_pct, 2) if option_entry > 0 else None,
        "exit_reason": request.reason,
        "result": result,
    }

    # Update state
    positions.pop(pos_index)
    state["open_positions"] = positions
    state["closed_trades"].append(closed_trade)
    state["total_trades"] = state.get("total_trades", 0) + 1
    state["total_pnl"] = state.get("total_pnl", 0) + pnl
    state["capital"] = state.get("capital", 10000) + pnl

    if pnl > 0:
        state["winning_trades"] = state.get("winning_trades", 0) + 1
    elif pnl < 0:
        state["losing_trades"] = state.get("losing_trades", 0) + 1

    total_trades = state.get("total_trades", 0)
    if total_trades > 0:
        state["win_rate"] = (state.get("winning_trades", 0) / total_trades) * 100

    state["timestamp"] = datetime.now().isoformat()
    save_state(state)

    return {
        "status": "closed",
        "trade": closed_trade,
        "account": {
            "capital": state["capital"],
            "total_pnl": state["total_pnl"],
            "win_rate": state["win_rate"],
            "total_trades": state["total_trades"],
        }
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates."""
    await manager.connect(websocket)

    try:
        # Send initial state
        state = load_state()
        await websocket.send_json({
            "type": "initial",
            "data": state,
        })

        # Keep connection alive and send updates
        last_state = json.dumps(state)
        while True:
            await asyncio.sleep(2)  # Check for updates every 2 seconds

            current_state = load_state()
            current_json = json.dumps(current_state)

            if current_json != last_state:
                await websocket.send_json({
                    "type": "update",
                    "data": current_state,
                })
                last_state = current_json

    except WebSocketDisconnect:
        manager.disconnect(websocket)


def _time_since(iso_time: str) -> str:
    """Calculate human-readable time since a timestamp."""
    if not iso_time:
        return "N/A"

    try:
        dt = datetime.fromisoformat(iso_time)
        diff = datetime.now() - dt

        if diff.days > 0:
            return f"{diff.days}d ago"
        elif diff.seconds > 3600:
            return f"{diff.seconds // 3600}h ago"
        elif diff.seconds > 60:
            return f"{diff.seconds // 60}m ago"
        else:
            return "Just now"
    except:
        return "N/A"


# ============================================================================
# VISION-BASED CHART ANALYSIS ENDPOINTS
# ============================================================================

# Add scripts directory to path for vision imports
SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

# Lazy import vision module (may not be installed in all environments)
_vision_module = None

def get_vision_module():
    global _vision_module
    if _vision_module is None:
        try:
            import vision_chart_analyzer as vcm
            _vision_module = vcm
        except ImportError as e:
            print(f"Vision module not available: {e}")
            return None
    return _vision_module


class VisionScanRequest(BaseModel):
    """Request to run vision scan on symbols."""
    symbols: list[str] = []  # Empty = use default watchlist
    full_scan: bool = False  # True = scan all ~35 stocks


@app.post("/api/vision/scan")
async def run_vision_scan(request: VisionScanRequest):
    """
    Run vision-based chart analysis.
    Fetches daily charts and uses Claude Vision to detect S/D zones.
    """
    vision = get_vision_module()
    if not vision:
        return {"error": "Vision module not available"}

    symbols = request.symbols
    if not symbols:
        symbols = vision.FULL_WATCHLIST if request.full_scan else vision.TEST_SYMBOLS

    try:
        results = await vision.run_daily_scan(symbols)
        return {
            "status": "success",
            "symbols_scanned": len(symbols),
            "levels_found": len(results.get("levels", [])),
            "results": results
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/vision/levels")
async def get_vision_levels():
    """Get the most recent vision-detected levels."""
    vision_file = Path(__file__).parent.parent / "data" / "vision_levels.json"

    if not vision_file.exists():
        return {"levels": [], "message": "No vision scan results yet. Run /api/vision/scan first."}

    with open(vision_file) as f:
        data = json.load(f)

    return data


@app.post("/api/vision/analyze/{symbol}")
async def analyze_single_symbol(symbol: str):
    """Analyze a single stock with vision."""
    vision = get_vision_module()
    if not vision:
        return {"error": "Vision module not available"}

    try:
        # Get current price
        current_price = await vision.get_current_price(symbol)
        if not current_price:
            return {"error": f"Could not fetch price for {symbol}"}

        # Run analysis
        analysis = await vision.scan_stock_for_zones(symbol, current_price)

        # Convert to levels format
        levels = vision.convert_to_bill_fanter_format(analysis)

        return {
            "symbol": symbol,
            "current_price": current_price,
            "analysis": analysis,
            "levels": levels
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/vision/monitor")
async def monitor_positions_with_vision():
    """
    Monitor all open positions using 5-minute chart vision analysis.
    Returns exit recommendations for each position.
    """
    vision = get_vision_module()
    if not vision:
        return {"error": "Vision module not available"}

    try:
        # Import position monitor
        import position_monitor as pm
        results = await pm.monitor_all_positions()

        return {
            "status": "success",
            "positions_monitored": len(results),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/vision/status")
async def get_vision_status():
    """Check vision system status and last scan info."""
    vision_file = Path(__file__).parent.parent / "data" / "vision_levels.json"
    monitor_file = Path(__file__).parent.parent / "data" / "position_monitor_log.json"

    status = {
        "vision_available": get_vision_module() is not None,
        "last_scan": None,
        "last_monitor": None,
        "levels_count": 0
    }

    if vision_file.exists():
        with open(vision_file) as f:
            data = json.load(f)
            status["last_scan"] = data.get("scan_time")
            status["levels_count"] = len(data.get("levels", []))

    if monitor_file.exists():
        with open(monitor_file) as f:
            data = json.load(f)
            status["last_monitor"] = data.get("timestamp")

    return status


# ============================================================================
# PUTT INDICATOR ENDPOINTS
# ============================================================================

class PuttAnalysisRequest(BaseModel):
    """Request for Putt Indicator analysis."""
    symbol: str
    direction: str  # "long" or "short"
    zone_type: str  # "demand" or "supply"
    zone_level: float
    base_confidence: float = 50.0


@app.post("/api/putt/analyze")
async def analyze_with_putt(request: PuttAnalysisRequest):
    """
    Analyze a setup using the Putt Indicator.

    The Putt Indicator combines:
    - Your trading intuition (base confidence)
    - RAG database (historical patterns from Bill Fanter)
    - Bill's methodology (insights, zone history)
    """
    try:
        putt = get_putt_indicator()
        context = putt.analyze_setup(
            symbol=request.symbol.upper(),
            direction=request.direction.lower(),
            zone_type=request.zone_type.lower(),
            zone_level=request.zone_level,
            base_confidence=request.base_confidence,
        )

        return {
            "symbol": context.symbol,
            "direction": context.direction,
            "zone_type": context.zone_type,
            "base_confidence": context.base_confidence,
            "putt_adjustment": context.putt_adjustment,
            "final_confidence": context.final_confidence,
            "win_rate": context.win_rate,
            "avg_rr": context.avg_rr,
            "similar_trades": len(context.similar_trades),
            "bill_mentioned": context.bill_mentioned,
            "pattern_recognized": context.pattern_recognized,
            "summary": context.summary,
            "key_insights": context.key_insights,
            "zone_history": [
                {"text": z.get("text", "")[:150], "metadata": z.get("metadata", {})}
                for z in context.zone_history[:3]
            ],
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/putt/context/{symbol}")
async def get_putt_context(symbol: str):
    """
    Get full Putt Indicator context for a symbol.

    Returns all historical data Bill has mentioned about this symbol.
    """
    try:
        putt = get_putt_indicator()

        # Get trading context from RAG
        rag_context = putt.rag.get_trading_context(symbol.upper())

        # Quick validation
        quick = putt.get_quick_validation(symbol.upper(), "long")

        return {
            "symbol": symbol.upper(),
            "zones": [
                {"text": z.get("text", "")[:150], "type": z.get("metadata", {}).get("zone_type")}
                for z in rag_context.get("zones", [])[:5]
            ],
            "signals": [
                {"text": s.get("text", "")[:150], "direction": s.get("metadata", {}).get("direction")}
                for s in rag_context.get("signals", [])[:5]
            ],
            "insights": [
                i.get("text", "")[:200] for i in rag_context.get("insights", [])[:5]
            ],
            "mentions": len(rag_context.get("recent_mentions", [])),
            "validation": quick,
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/putt/stats")
async def get_putt_stats():
    """Get Putt Indicator database statistics."""
    try:
        putt = get_putt_indicator()
        doc_count = putt.rag.get_collection_count()

        return {
            "total_documents": doc_count,
            "status": "active",
            "description": "Putt Indicator: Your Brain + RAG + Bill Fanter Combined",
        }
    except Exception as e:
        return {"error": str(e), "status": "inactive"}


@app.get("/api/risk-indicators")
async def get_risk_indicators():
    """
    Get risk-off indicators that Bill Fanter watches.

    From his videos:
    - Bitcoin < $87k = risk-off signal
    - HOOD < $120 = risk-off signal
    - MSTR < $166 = risk-off signal
    - VIX > 20 = elevated fear
    """
    import aiohttp

    indicators = {
        "bitcoin": {"symbol": "BTC-USD", "threshold": 87000, "direction": "below", "value": None, "signal": None},
        "hood": {"symbol": "HOOD", "threshold": 120, "direction": "below", "value": None, "signal": None},
        "mstr": {"symbol": "MSTR", "threshold": 166, "direction": "below", "value": None, "signal": None},
        "vix": {"symbol": "^VIX", "threshold": 20, "direction": "above", "value": None, "signal": None},
    }

    async with aiohttp.ClientSession() as session:
        for key, ind in indicators.items():
            try:
                symbol = ind["symbol"]
                url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1m&range=1d"
                headers = {"User-Agent": "Mozilla/5.0"}

                async with session.get(url, headers=headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        result = data.get("chart", {}).get("result", [])
                        if result:
                            price = result[0].get("meta", {}).get("regularMarketPrice")
                            if price:
                                indicators[key]["value"] = round(price, 2)

                                # Determine if signal is triggered
                                if ind["direction"] == "below":
                                    indicators[key]["signal"] = price < ind["threshold"]
                                else:  # above
                                    indicators[key]["signal"] = price > ind["threshold"]
            except Exception as e:
                print(f"Error fetching {key}: {e}")

            await asyncio.sleep(0.1)  # Rate limiting

    # Calculate overall risk level
    risk_signals = sum(1 for ind in indicators.values() if ind["signal"] is True)
    risk_level = "HIGH" if risk_signals >= 3 else "ELEVATED" if risk_signals >= 2 else "CAUTION" if risk_signals >= 1 else "NORMAL"

    return {
        "indicators": indicators,
        "risk_signals_active": risk_signals,
        "risk_level": risk_level,
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
