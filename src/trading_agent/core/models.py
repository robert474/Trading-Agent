"""
Core data models for the trading agent.
Implements Bill Fanter's supply/demand zone methodology.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional


class ZoneType(Enum):
    """Type of supply/demand zone."""

    SUPPLY = "supply"
    DEMAND = "demand"


class ZoneFreshness(Enum):
    """Zone freshness status."""

    FRESH = "fresh"  # Never tested
    TESTED = "tested"  # Tested once
    BROKEN = "broken"  # Zone has been violated


class Timeframe(Enum):
    """Chart timeframes."""

    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "D"
    W1 = "W"


class OrderType(Enum):
    """Order types."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order side/direction."""

    BUY = "buy"
    SELL = "sell"
    BUY_TO_OPEN = "buy_to_open"
    SELL_TO_CLOSE = "sell_to_close"
    BUY_TO_CLOSE = "buy_to_close"
    SELL_TO_OPEN = "sell_to_open"


class OrderStatus(Enum):
    """Order status."""

    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class TradeDirection(Enum):
    """Trade direction."""

    LONG = "long"
    SHORT = "short"


class MarketTrend(Enum):
    """Higher timeframe market trend."""

    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class Candle:
    """OHLCV candle data."""

    symbol: str
    timeframe: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int

    @property
    def body_size(self) -> float:
        """Absolute size of candle body."""
        return abs(self.close - self.open)

    @property
    def total_range(self) -> float:
        """Total high-low range."""
        return self.high - self.low

    @property
    def is_bullish(self) -> bool:
        """True if close > open."""
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        """True if close < open."""
        return self.close < self.open

    @property
    def body_percent(self) -> float:
        """Body as percentage of total range."""
        if self.total_range == 0:
            return 0
        return self.body_size / self.total_range


@dataclass
class Zone:
    """
    Base class for supply/demand zones.

    A zone represents an area where there was significant
    buying or selling pressure, causing an imbalance.
    """

    symbol: str
    zone_type: ZoneType
    zone_high: float
    zone_low: float
    timeframe: str
    freshness: ZoneFreshness = ZoneFreshness.FRESH
    quality_score: float = 0.0
    departure_strength: float = 0.0
    candles_in_zone: int = 1
    origin_candle_time: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    broken_at: Optional[datetime] = None
    notes: str = ""
    id: Optional[int] = None

    @property
    def zone_midpoint(self) -> float:
        """Middle of the zone."""
        return (self.zone_high + self.zone_low) / 2

    @property
    def zone_height(self) -> float:
        """Height of the zone."""
        return self.zone_high - self.zone_low

    @property
    def is_valid(self) -> bool:
        """Zone is valid if fresh or tested only once."""
        return self.freshness in [ZoneFreshness.FRESH, ZoneFreshness.TESTED]

    def contains_price(self, price: float) -> bool:
        """Check if price is within the zone."""
        return self.zone_low <= price <= self.zone_high

    def distance_to_price(self, price: float) -> float:
        """
        Calculate distance from price to nearest zone edge.
        Positive = price outside zone, Negative = price inside zone.
        """
        if price > self.zone_high:
            return price - self.zone_high
        elif price < self.zone_low:
            return self.zone_low - price
        else:
            # Inside zone - return negative distance to nearest edge
            return -min(price - self.zone_low, self.zone_high - price)


@dataclass
class SupplyZone(Zone):
    """
    Supply Zone (Resistance).

    Area where selling pressure exceeded buying pressure,
    causing price to drop. When price returns, expect selling again.

    Entry: Sell/short when price approaches from below
    """

    def __post_init__(self):
        self.zone_type = ZoneType.SUPPLY


@dataclass
class DemandZone(Zone):
    """
    Demand Zone (Support).

    Area where buying pressure exceeded selling pressure,
    causing price to rise. When price returns, expect buying again.

    Entry: Buy/long when price approaches from above
    """

    def __post_init__(self):
        self.zone_type = ZoneType.DEMAND


@dataclass
class TradeSignal:
    """
    Generated trade signal from analysis.
    Contains entry, stop, target, and reasoning.
    """

    symbol: str
    direction: TradeDirection
    entry_price: float
    stop_loss: float
    target_price: float
    risk_reward: float
    zone: Optional[Zone] = None
    zone_id: Optional[int] = None
    llm_reasoning: str = ""
    llm_confidence: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    status: str = "pending"
    id: Optional[int] = None

    # Optional recommended option contract
    option_symbol: Optional[str] = None
    option_strike: Optional[float] = None
    option_expiration: Optional[str] = None
    option_type: Optional[str] = None  # 'call' or 'put'
    option_delta: Optional[float] = None
    option_premium: Optional[float] = None

    @property
    def risk_amount(self) -> float:
        """Dollar risk per share."""
        return abs(self.entry_price - self.stop_loss)

    @property
    def reward_amount(self) -> float:
        """Dollar reward per share."""
        return abs(self.target_price - self.entry_price)

    def validate(self) -> tuple[bool, str]:
        """Validate the signal meets minimum criteria."""
        if self.risk_reward < 2.0:
            return False, f"R:R too low: {self.risk_reward:.2f}"

        if self.direction == TradeDirection.LONG:
            if self.stop_loss >= self.entry_price:
                return False, "Stop loss must be below entry for long"
            if self.target_price <= self.entry_price:
                return False, "Target must be above entry for long"
        else:
            if self.stop_loss <= self.entry_price:
                return False, "Stop loss must be above entry for short"
            if self.target_price >= self.entry_price:
                return False, "Target must be below entry for short"

        return True, "OK"


@dataclass
class Order:
    """Order to be submitted to broker."""

    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "day"
    order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_price: Optional[float] = None
    filled_quantity: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    broker_response: Optional[dict] = None

    @property
    def is_option(self) -> bool:
        """Check if this is an options order (OCC format symbols are longer)."""
        return len(self.symbol) > 10

    @property
    def is_filled(self) -> bool:
        """Check if order is completely filled."""
        return self.status == OrderStatus.FILLED

    @property
    def is_active(self) -> bool:
        """Check if order is still active/working."""
        return self.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL]


@dataclass
class Position:
    """Open position being tracked."""

    symbol: str
    quantity: int
    entry_price: float
    direction: TradeDirection
    entry_time: datetime
    stop_loss: float
    target_price: float
    trailing_stop: Optional[float] = None
    current_price: Optional[float] = None
    unrealized_pnl: float = 0.0
    signal_id: Optional[int] = None
    option_expiry: Optional[datetime] = None
    id: Optional[str] = None

    @property
    def cost_basis(self) -> float:
        """Total cost of position."""
        multiplier = 100 if self._is_option else 1
        return self.quantity * self.entry_price * multiplier

    @property
    def _is_option(self) -> bool:
        """Check if this is an options position."""
        return len(self.symbol) > 10

    @property
    def profit_percent(self) -> float:
        """Current profit as percentage."""
        if not self.current_price:
            return 0.0
        if self.direction == TradeDirection.LONG:
            return (self.current_price - self.entry_price) / self.entry_price
        else:
            return (self.entry_price - self.current_price) / self.entry_price

    def update_trailing_stop(self, current_price: float) -> bool:
        """
        Update trailing stop if price has moved favorably.
        Activates at 50% of target, trails at 50% of profit.

        Returns True if trailing stop was updated.
        """
        is_long = self.direction == TradeDirection.LONG

        if is_long:
            profit_pct = (current_price - self.entry_price) / self.entry_price
            target_pct = (self.target_price - self.entry_price) / self.entry_price

            # Activate at 50% of target
            if profit_pct >= target_pct * 0.5:
                # Trail at 50% of current profit
                new_trailing = self.entry_price + (current_price - self.entry_price) * 0.5
                if self.trailing_stop is None or new_trailing > self.trailing_stop:
                    self.trailing_stop = new_trailing
                    return True
        else:
            profit_pct = (self.entry_price - current_price) / self.entry_price
            target_pct = (self.entry_price - self.target_price) / self.entry_price

            if profit_pct >= target_pct * 0.5:
                new_trailing = self.entry_price - (self.entry_price - current_price) * 0.5
                if self.trailing_stop is None or new_trailing < self.trailing_stop:
                    self.trailing_stop = new_trailing
                    return True

        return False


@dataclass
class OptionContract:
    """Options contract details."""

    symbol: str  # OCC symbol
    underlying: str
    option_type: str  # 'call' or 'put'
    strike: float
    expiration: str  # YYYY-MM-DD
    days_to_expiry: int
    bid: float
    ask: float
    last: float
    volume: int
    open_interest: int
    delta: float
    gamma: float
    theta: float
    vega: float
    iv: float  # Implied volatility

    @property
    def mid(self) -> float:
        """Mid price."""
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> float:
        """Bid-ask spread."""
        return self.ask - self.bid

    @property
    def spread_percent(self) -> float:
        """Spread as percentage of mid."""
        if self.mid == 0:
            return float("inf")
        return self.spread / self.mid

    @property
    def is_liquid(self) -> bool:
        """Check if contract meets liquidity requirements."""
        return self.open_interest >= 500 and self.spread_percent < 0.10


@dataclass
class MarketContext:
    """Current market context for analysis."""

    symbol: str
    current_price: float
    htf_trend: MarketTrend
    active_supply_zones: list[SupplyZone] = field(default_factory=list)
    active_demand_zones: list[DemandZone] = field(default_factory=list)
    key_levels: list[float] = field(default_factory=list)
    caution_factors: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TradeResult:
    """Completed trade result."""

    signal_id: Optional[int]
    symbol: str
    option_symbol: Optional[str]
    direction: TradeDirection
    quantity: int
    entry_price: float
    entry_time: datetime
    exit_price: float
    exit_time: datetime
    exit_reason: str  # 'target', 'stop_loss', 'trailing_stop', 'manual', 'expiry'
    pnl: float
    pnl_percent: float
    fees: float = 0.0
    notes: str = ""
    id: Optional[int] = None
