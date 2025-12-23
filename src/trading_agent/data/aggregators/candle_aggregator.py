"""
Candle aggregator for multiple timeframes.
Aggregates tick/trade data into OHLCV candles.
"""

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Callable, Optional

import structlog

from trading_agent.core.models import Candle

logger = structlog.get_logger()


class CandleAggregator:
    """
    Aggregates trades into OHLCV candles for multiple timeframes.

    Supports real-time aggregation and emits completed candles
    via callbacks.
    """

    # Timeframe definitions in minutes
    TIMEFRAME_MINUTES = {
        "1m": 1,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": 60,
        "4h": 240,
        "D": 1440,  # 24 * 60
    }

    def __init__(
        self,
        timeframes: Optional[list[str]] = None,
        on_candle_complete: Optional[Callable[[Candle], None]] = None,
    ):
        """
        Initialize candle aggregator.

        Args:
            timeframes: List of timeframes to aggregate (default: all)
            on_candle_complete: Callback when a candle completes
        """
        self.timeframes = timeframes or list(self.TIMEFRAME_MINUTES.keys())
        self.on_candle_complete = on_candle_complete

        # Current incomplete candles: {(symbol, timeframe): Candle}
        self._current_candles: dict[tuple[str, str], Candle] = {}

        # Completed candles buffer: {symbol: {timeframe: [candles]}}
        self._completed_candles: dict[str, dict[str, list[Candle]]] = defaultdict(
            lambda: defaultdict(list)
        )

        # Maximum completed candles to keep in memory per symbol/timeframe
        self._max_buffer_size = 500

    def add_trade(
        self,
        symbol: str,
        price: float,
        volume: int,
        timestamp: datetime,
    ) -> list[Candle]:
        """
        Add a trade and update all timeframe candles.

        Returns list of any completed candles.
        """
        completed = []

        for tf in self.timeframes:
            candle_start = self._get_candle_start(timestamp, tf)
            key = (symbol, tf)

            if key not in self._current_candles:
                # Start new candle
                self._current_candles[key] = Candle(
                    symbol=symbol,
                    timeframe=tf,
                    timestamp=candle_start,
                    open=price,
                    high=price,
                    low=price,
                    close=price,
                    volume=volume,
                )
            else:
                current = self._current_candles[key]

                # Check if we've moved to a new candle period
                if candle_start > current.timestamp:
                    # Complete the current candle
                    completed.append(current)
                    self._store_completed(current)

                    if self.on_candle_complete:
                        self.on_candle_complete(current)

                    # Start new candle
                    self._current_candles[key] = Candle(
                        symbol=symbol,
                        timeframe=tf,
                        timestamp=candle_start,
                        open=price,
                        high=price,
                        low=price,
                        close=price,
                        volume=volume,
                    )
                else:
                    # Update existing candle
                    current.high = max(current.high, price)
                    current.low = min(current.low, price)
                    current.close = price
                    current.volume += volume

        return completed

    def add_candle(self, candle: Candle) -> list[Candle]:
        """
        Add a pre-formed candle (e.g., from Polygon AM event).
        Aggregates into higher timeframes.

        Returns list of any completed higher timeframe candles.
        """
        completed = []
        symbol = candle.symbol

        # Store the incoming candle
        self._store_completed(candle)

        # Aggregate into higher timeframes
        incoming_minutes = self.TIMEFRAME_MINUTES.get(candle.timeframe, 1)

        for tf in self.timeframes:
            tf_minutes = self.TIMEFRAME_MINUTES[tf]

            # Only aggregate into higher timeframes
            if tf_minutes <= incoming_minutes:
                continue

            candle_start = self._get_candle_start(candle.timestamp, tf)
            key = (symbol, tf)

            if key not in self._current_candles:
                # Start new higher TF candle
                self._current_candles[key] = Candle(
                    symbol=symbol,
                    timeframe=tf,
                    timestamp=candle_start,
                    open=candle.open,
                    high=candle.high,
                    low=candle.low,
                    close=candle.close,
                    volume=candle.volume,
                )
            else:
                current = self._current_candles[key]

                # Check if we've moved to a new candle period
                if candle_start > current.timestamp:
                    # Complete the current candle
                    completed.append(current)
                    self._store_completed(current)

                    if self.on_candle_complete:
                        self.on_candle_complete(current)

                    # Start new candle
                    self._current_candles[key] = Candle(
                        symbol=symbol,
                        timeframe=tf,
                        timestamp=candle_start,
                        open=candle.open,
                        high=candle.high,
                        low=candle.low,
                        close=candle.close,
                        volume=candle.volume,
                    )
                else:
                    # Update existing candle
                    current.high = max(current.high, candle.high)
                    current.low = min(current.low, candle.low)
                    current.close = candle.close
                    current.volume += candle.volume

        return completed

    def _get_candle_start(self, timestamp: datetime, timeframe: str) -> datetime:
        """
        Calculate the start time of the candle for a given timestamp.
        """
        minutes = self.TIMEFRAME_MINUTES.get(timeframe, 1)

        if timeframe == "D":
            # Daily candles start at midnight
            return timestamp.replace(hour=0, minute=0, second=0, microsecond=0)

        # Round down to nearest timeframe interval
        total_minutes = timestamp.hour * 60 + timestamp.minute
        candle_minutes = (total_minutes // minutes) * minutes

        return timestamp.replace(
            hour=candle_minutes // 60,
            minute=candle_minutes % 60,
            second=0,
            microsecond=0,
        )

    def _store_completed(self, candle: Candle) -> None:
        """Store a completed candle in the buffer."""
        buffer = self._completed_candles[candle.symbol][candle.timeframe]
        buffer.append(candle)

        # Trim buffer if too large
        if len(buffer) > self._max_buffer_size:
            self._completed_candles[candle.symbol][candle.timeframe] = buffer[
                -self._max_buffer_size :
            ]

    def get_candles(
        self,
        symbol: str,
        timeframe: str,
        count: int = 100,
        include_current: bool = False,
    ) -> list[Candle]:
        """
        Get recent candles for a symbol/timeframe.

        Args:
            symbol: Stock symbol
            timeframe: Candle timeframe
            count: Number of candles to return
            include_current: Include the incomplete current candle

        Returns:
            List of Candle objects, oldest first
        """
        candles = self._completed_candles[symbol][timeframe][-count:]

        if include_current:
            key = (symbol, timeframe)
            if key in self._current_candles:
                candles = candles + [self._current_candles[key]]

        return candles

    def get_current_candle(self, symbol: str, timeframe: str) -> Optional[Candle]:
        """Get the current incomplete candle."""
        return self._current_candles.get((symbol, timeframe))

    def load_historical(self, candles: list[Candle]) -> None:
        """
        Load historical candles into the buffer.

        Used to initialize with historical data from database or API.
        """
        for candle in candles:
            self._store_completed(candle)

        logger.info(
            "Loaded historical candles",
            count=len(candles),
            symbols=list(set(c.symbol for c in candles)),
        )

    def get_multi_timeframe_candles(
        self,
        symbol: str,
        count: int = 50,
    ) -> dict[str, list[Candle]]:
        """
        Get candles for all timeframes for a symbol.

        Returns dict mapping timeframe -> candles.
        """
        result = {}
        for tf in self.timeframes:
            result[tf] = self.get_candles(symbol, tf, count)
        return result

    def clear(self, symbol: Optional[str] = None) -> None:
        """
        Clear candle data.

        Args:
            symbol: Clear only this symbol, or all if None
        """
        if symbol:
            if symbol in self._completed_candles:
                del self._completed_candles[symbol]
            keys_to_remove = [k for k in self._current_candles if k[0] == symbol]
            for key in keys_to_remove:
                del self._current_candles[key]
        else:
            self._completed_candles.clear()
            self._current_candles.clear()

        logger.info("Cleared candle data", symbol=symbol or "all")
