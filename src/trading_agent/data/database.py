"""
Database manager for PostgreSQL/TimescaleDB operations.
Handles all database interactions for the trading agent.
"""

from datetime import datetime
from typing import Optional

import asyncpg
import structlog

from trading_agent.core.config import settings
from trading_agent.core.models import (
    Candle,
    DemandZone,
    Position,
    SupplyZone,
    TradeResult,
    TradeSignal,
    Zone,
    ZoneFreshness,
    ZoneType,
)

logger = structlog.get_logger()


class DatabaseManager:
    """
    Async database manager for PostgreSQL with TimescaleDB.

    Handles:
    - Candle data storage and retrieval
    - Zone management
    - Trade signals and execution logging
    - Position tracking
    """

    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or settings.database_url
        self._pool: Optional[asyncpg.Pool] = None

    async def connect(self) -> None:
        """Establish database connection pool."""
        logger.info("Connecting to database...")

        self._pool = await asyncpg.create_pool(
            self.database_url,
            min_size=2,
            max_size=10,
            command_timeout=60,
        )

        logger.info("Database connection established")

    async def disconnect(self) -> None:
        """Close database connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("Database connection closed")

    async def execute(self, query: str, *args) -> str:
        """Execute a query and return status."""
        async with self._pool.acquire() as conn:
            return await conn.execute(query, *args)

    async def fetch(self, query: str, *args) -> list:
        """Fetch multiple rows."""
        async with self._pool.acquire() as conn:
            return await conn.fetch(query, *args)

    async def fetchrow(self, query: str, *args) -> Optional[asyncpg.Record]:
        """Fetch a single row."""
        async with self._pool.acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def fetchval(self, query: str, *args):
        """Fetch a single value."""
        async with self._pool.acquire() as conn:
            return await conn.fetchval(query, *args)

    # =========================================================================
    # Candle Operations
    # =========================================================================

    async def save_candles(self, candles: list[Candle]) -> int:
        """
        Bulk save candles to database.
        Uses upsert to handle duplicates.

        Returns number of candles saved.
        """
        if not candles:
            return 0

        query = """
            INSERT INTO candles (time, symbol, timeframe, open, high, low, close, volume)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (time, symbol, timeframe)
            DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume
        """

        async with self._pool.acquire() as conn:
            await conn.executemany(
                query,
                [
                    (
                        c.timestamp,
                        c.symbol,
                        c.timeframe,
                        c.open,
                        c.high,
                        c.low,
                        c.close,
                        c.volume,
                    )
                    for c in candles
                ],
            )

        logger.debug("Saved candles", count=len(candles))
        return len(candles)

    async def get_candles(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 500,
        from_time: Optional[datetime] = None,
        to_time: Optional[datetime] = None,
    ) -> list[Candle]:
        """
        Retrieve candles from database.

        Args:
            symbol: Stock symbol
            timeframe: Candle timeframe
            limit: Max candles to return
            from_time: Start time filter
            to_time: End time filter

        Returns:
            List of Candle objects, oldest first
        """
        conditions = ["symbol = $1", "timeframe = $2"]
        params = [symbol.upper(), timeframe]
        param_idx = 3

        if from_time:
            conditions.append(f"time >= ${param_idx}")
            params.append(from_time)
            param_idx += 1

        if to_time:
            conditions.append(f"time <= ${param_idx}")
            params.append(to_time)
            param_idx += 1

        query = f"""
            SELECT time, symbol, timeframe, open, high, low, close, volume
            FROM candles
            WHERE {' AND '.join(conditions)}
            ORDER BY time DESC
            LIMIT ${param_idx}
        """
        params.append(limit)

        rows = await self.fetch(query, *params)

        candles = [
            Candle(
                symbol=row["symbol"],
                timeframe=row["timeframe"],
                timestamp=row["time"],
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=row["volume"],
            )
            for row in rows
        ]

        # Return oldest first
        return list(reversed(candles))

    # =========================================================================
    # Zone Operations
    # =========================================================================

    async def save_zone(self, zone: Zone) -> int:
        """Save a zone and return its ID."""
        query = """
            INSERT INTO zones (
                symbol, zone_type, zone_high, zone_low, timeframe,
                freshness, departure_strength, candles_in_zone,
                origin_candle_time, notes
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            RETURNING id
        """

        zone_id = await self.fetchval(
            query,
            zone.symbol,
            zone.zone_type.value,
            zone.zone_high,
            zone.zone_low,
            zone.timeframe,
            zone.freshness.value,
            zone.departure_strength,
            zone.candles_in_zone,
            zone.origin_candle_time,
            zone.notes,
        )

        logger.info(
            "Saved zone",
            id=zone_id,
            symbol=zone.symbol,
            type=zone.zone_type.value,
            high=zone.zone_high,
            low=zone.zone_low,
        )

        return zone_id

    async def get_active_zones(
        self,
        symbol: Optional[str] = None,
        zone_type: Optional[ZoneType] = None,
        timeframe: Optional[str] = None,
    ) -> list[Zone]:
        """
        Get active (non-broken) zones.

        Args:
            symbol: Filter by symbol
            zone_type: Filter by supply/demand
            timeframe: Filter by timeframe

        Returns:
            List of Zone objects sorted by quality score
        """
        conditions = ["freshness != 'broken'"]
        params = []
        param_idx = 1

        if symbol:
            conditions.append(f"symbol = ${param_idx}")
            params.append(symbol.upper())
            param_idx += 1

        if zone_type:
            conditions.append(f"zone_type = ${param_idx}")
            params.append(zone_type.value)
            param_idx += 1

        if timeframe:
            conditions.append(f"timeframe = ${param_idx}")
            params.append(timeframe)
            param_idx += 1

        query = f"""
            SELECT id, symbol, zone_type, zone_high, zone_low, timeframe,
                   quality_score, freshness, departure_strength, candles_in_zone,
                   origin_candle_time, created_at, broken_at, notes
            FROM zones
            WHERE {' AND '.join(conditions)}
            ORDER BY quality_score DESC
        """

        rows = await self.fetch(query, *params)

        zones = []
        for row in rows:
            zone_cls = SupplyZone if row["zone_type"] == "supply" else DemandZone
            zone = zone_cls(
                symbol=row["symbol"],
                zone_type=ZoneType(row["zone_type"]),
                zone_high=float(row["zone_high"]),
                zone_low=float(row["zone_low"]),
                timeframe=row["timeframe"],
                freshness=ZoneFreshness(row["freshness"]),
                quality_score=row["quality_score"],
                departure_strength=float(row["departure_strength"]),
                candles_in_zone=row["candles_in_zone"],
                origin_candle_time=row["origin_candle_time"],
                created_at=row["created_at"],
                broken_at=row["broken_at"],
                notes=row["notes"],
                id=row["id"],
            )
            zones.append(zone)

        return zones

    async def mark_zone_tested(self, zone_id: int) -> None:
        """Mark a zone as tested (or broken if already tested)."""
        await self.execute("SELECT mark_zone_tested($1)", zone_id)
        logger.info("Marked zone as tested", zone_id=zone_id)

    async def break_zone(self, zone_id: int) -> None:
        """Mark a zone as broken."""
        await self.execute(
            "UPDATE zones SET freshness = 'broken', broken_at = NOW() WHERE id = $1",
            zone_id,
        )
        logger.info("Marked zone as broken", zone_id=zone_id)

    # =========================================================================
    # Trade Signal Operations
    # =========================================================================

    async def save_signal(self, signal: TradeSignal) -> int:
        """Save a trade signal and return its ID."""
        query = """
            INSERT INTO trade_signals (
                symbol, direction, entry_price, stop_loss, target_price,
                risk_reward, zone_id, llm_reasoning, llm_confidence,
                option_symbol, option_strike, option_expiration,
                option_type, option_delta, option_premium, expires_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
            RETURNING id
        """

        signal_id = await self.fetchval(
            query,
            signal.symbol,
            signal.direction.value,
            signal.entry_price,
            signal.stop_loss,
            signal.target_price,
            signal.risk_reward,
            signal.zone_id,
            signal.llm_reasoning,
            signal.llm_confidence,
            signal.option_symbol,
            signal.option_strike,
            signal.option_expiration,
            signal.option_type,
            signal.option_delta,
            signal.option_premium,
            signal.expires_at,
        )

        logger.info(
            "Saved trade signal",
            id=signal_id,
            symbol=signal.symbol,
            direction=signal.direction.value,
            entry=signal.entry_price,
        )

        return signal_id

    async def get_pending_signals(self, symbol: Optional[str] = None) -> list[dict]:
        """Get pending trade signals."""
        conditions = ["status = 'pending'"]
        params = []

        if symbol:
            conditions.append("symbol = $1")
            params.append(symbol.upper())

        query = f"""
            SELECT * FROM trade_signals
            WHERE {' AND '.join(conditions)}
            ORDER BY created_at DESC
        """

        rows = await self.fetch(query, *params)
        return [dict(row) for row in rows]

    async def update_signal_status(self, signal_id: int, status: str) -> None:
        """Update a signal's status."""
        await self.execute(
            "UPDATE trade_signals SET status = $1 WHERE id = $2",
            status,
            signal_id,
        )

    # =========================================================================
    # Trade Operations
    # =========================================================================

    async def save_trade(self, trade: TradeResult) -> int:
        """Save a completed trade result."""
        query = """
            INSERT INTO trades (
                signal_id, symbol, option_symbol, direction, quantity,
                entry_price, entry_time, exit_price, exit_time,
                exit_reason, pnl, pnl_percent, fees, notes
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
            RETURNING id
        """

        trade_id = await self.fetchval(
            query,
            trade.signal_id,
            trade.symbol,
            trade.option_symbol,
            trade.direction.value,
            trade.quantity,
            trade.entry_price,
            trade.entry_time,
            trade.exit_price,
            trade.exit_time,
            trade.exit_reason,
            trade.pnl,
            trade.pnl_percent,
            trade.fees,
            trade.notes,
        )

        logger.info(
            "Saved trade",
            id=trade_id,
            symbol=trade.symbol,
            pnl=trade.pnl,
            exit_reason=trade.exit_reason,
        )

        return trade_id

    async def get_trade_summary(self, days: int = 30) -> dict:
        """Get trading performance summary."""
        query = """
            SELECT
                COUNT(*) as total_trades,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losses,
                SUM(pnl) as total_pnl,
                AVG(pnl) as avg_pnl,
                MAX(pnl) as best_trade,
                MIN(pnl) as worst_trade
            FROM trades
            WHERE exit_time > NOW() - INTERVAL '%s days'
        """

        row = await self.fetchrow(query % days)

        if not row or row["total_trades"] == 0:
            return {
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0,
                "total_pnl": 0,
                "avg_pnl": 0,
                "best_trade": 0,
                "worst_trade": 0,
            }

        return {
            "total_trades": row["total_trades"],
            "wins": row["wins"],
            "losses": row["losses"],
            "win_rate": row["wins"] / row["total_trades"] * 100 if row["total_trades"] > 0 else 0,
            "total_pnl": float(row["total_pnl"] or 0),
            "avg_pnl": float(row["avg_pnl"] or 0),
            "best_trade": float(row["best_trade"] or 0),
            "worst_trade": float(row["worst_trade"] or 0),
        }

    # =========================================================================
    # Position Operations
    # =========================================================================

    async def save_position(self, position: Position) -> None:
        """Save or update an open position."""
        query = """
            INSERT INTO positions (
                id, symbol, quantity, entry_price, direction,
                entry_time, stop_loss, target_price, trailing_stop,
                signal_id, option_expiry
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            ON CONFLICT (id) DO UPDATE SET
                trailing_stop = EXCLUDED.trailing_stop,
                updated_at = NOW()
        """

        await self.execute(
            query,
            position.id,
            position.symbol,
            position.quantity,
            position.entry_price,
            position.direction.value,
            position.entry_time,
            position.stop_loss,
            position.target_price,
            position.trailing_stop,
            position.signal_id,
            position.option_expiry,
        )

    async def delete_position(self, position_id: str) -> None:
        """Delete a closed position."""
        await self.execute("DELETE FROM positions WHERE id = $1", position_id)

    async def get_open_positions(self) -> list[Position]:
        """Get all open positions."""
        from trading_agent.core.models import TradeDirection

        rows = await self.fetch("SELECT * FROM positions")

        positions = []
        for row in rows:
            position = Position(
                id=row["id"],
                symbol=row["symbol"],
                quantity=row["quantity"],
                entry_price=float(row["entry_price"]),
                direction=TradeDirection(row["direction"]),
                entry_time=row["entry_time"],
                stop_loss=float(row["stop_loss"]),
                target_price=float(row["target_price"]),
                trailing_stop=float(row["trailing_stop"]) if row["trailing_stop"] else None,
                signal_id=row["signal_id"],
                option_expiry=row["option_expiry"],
            )
            positions.append(position)

        return positions

    # =========================================================================
    # Trade Examples (Few-Shot Learning)
    # =========================================================================

    async def save_trade_example(
        self,
        symbol: str,
        setup_type: str,
        setup_description: str,
        entry_reasoning: str,
        chart_context: dict,
        entry_price: float,
        exit_price: float,
        result: str,
        pnl: float,
        lessons: str,
        source: str = "manual",
    ) -> int:
        """Save a trade example for few-shot learning."""
        import json

        query = """
            INSERT INTO trade_examples (
                source, symbol, setup_type, setup_description,
                entry_reasoning, chart_context, entry_price,
                exit_price, result, pnl, lessons
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            RETURNING id
        """

        example_id = await self.fetchval(
            query,
            source,
            symbol,
            setup_type,
            setup_description,
            entry_reasoning,
            json.dumps(chart_context),
            entry_price,
            exit_price,
            result,
            pnl,
            lessons,
        )

        logger.info("Saved trade example", id=example_id, symbol=symbol, result=result)
        return example_id

    async def get_trade_examples(
        self,
        limit: int = 10,
        result_filter: Optional[str] = None,
    ) -> list[dict]:
        """
        Get trade examples for few-shot prompts.

        Args:
            limit: Max examples to return
            result_filter: 'win' or 'loss' to filter

        Returns:
            List of trade example dicts
        """
        import json

        conditions = []
        params = []
        param_idx = 1

        if result_filter:
            conditions.append(f"result = ${param_idx}")
            params.append(result_filter)
            param_idx += 1

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        query = f"""
            SELECT * FROM trade_examples
            {where_clause}
            ORDER BY created_at DESC
            LIMIT ${param_idx}
        """
        params.append(limit)

        rows = await self.fetch(query, *params)

        examples = []
        for row in rows:
            example = dict(row)
            if example.get("chart_context"):
                example["chart_context"] = json.loads(example["chart_context"])
            examples.append(example)

        return examples

    # =========================================================================
    # Audit Logging
    # =========================================================================

    async def log_event(self, event_type: str, event_data: dict) -> None:
        """Log an event to the audit log."""
        import json

        await self.execute(
            "INSERT INTO audit_log (event_type, event_data) VALUES ($1, $2)",
            event_type,
            json.dumps(event_data),
        )
