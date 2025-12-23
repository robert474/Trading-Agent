"""
Position Manager for tracking and managing open positions.

Handles position monitoring, exit conditions, and trailing stops.
"""

from datetime import datetime
from typing import Optional
from uuid import uuid4

import structlog

from trading_agent.core.models import (
    Order,
    OrderSide,
    OrderType,
    Position,
    TradeDirection,
    TradeResult,
    TradeSignal,
)
from trading_agent.execution.order_manager import OrderManager

logger = structlog.get_logger()


class PositionManager:
    """
    Manages open positions and exit logic.

    Implements Bill Fanter's exit rules:
    1. Hard stop loss - non-negotiable
    2. Take profit at target
    3. Trailing stop after 50% of target
    4. Time-based exit for options (avoid theta decay)
    """

    def __init__(self, order_manager: OrderManager):
        self.order_manager = order_manager
        self._positions: dict[str, Position] = {}

    async def open_position(
        self,
        signal: TradeSignal,
        quantity: int,
        actual_entry_price: Optional[float] = None,
    ) -> Optional[Position]:
        """
        Create a new position from a trade signal.

        Args:
            signal: The trade signal that triggered the position
            quantity: Number of shares/contracts
            actual_entry_price: Actual fill price (if different from signal)

        Returns:
            Position object or None if failed
        """
        entry_price = actual_entry_price or signal.entry_price

        position = Position(
            id=str(uuid4()),
            symbol=signal.option_symbol or signal.symbol,
            quantity=quantity,
            entry_price=entry_price,
            direction=signal.direction,
            entry_time=datetime.now(),
            stop_loss=signal.stop_loss,
            target_price=signal.target_price,
            signal_id=signal.id,
        )

        # Set option expiry if this is an options position
        if signal.option_expiration:
            from datetime import datetime as dt
            position.option_expiry = dt.strptime(signal.option_expiration, "%Y-%m-%d")

        self._positions[position.id] = position

        logger.info(
            "Position opened",
            id=position.id,
            symbol=position.symbol,
            direction=position.direction.value,
            entry=entry_price,
            stop=position.stop_loss,
            target=position.target_price,
        )

        return position

    def get_position(self, position_id: str) -> Optional[Position]:
        """Get a position by ID."""
        return self._positions.get(position_id)

    def get_all_positions(self) -> list[Position]:
        """Get all open positions."""
        return list(self._positions.values())

    def get_positions_for_symbol(self, symbol: str) -> list[Position]:
        """Get all positions for a symbol."""
        return [p for p in self._positions.values() if p.symbol == symbol]

    async def update_position_price(
        self,
        position_id: str,
        current_price: float,
    ) -> Optional[dict]:
        """
        Update position with current price and check exit conditions.

        Args:
            position_id: Position ID
            current_price: Current market price

        Returns:
            Exit signal dict if should exit, None otherwise
        """
        position = self._positions.get(position_id)
        if not position:
            return None

        position.current_price = current_price

        # Calculate unrealized P&L
        multiplier = 100 if len(position.symbol) > 10 else 1  # Options = 100 shares
        if position.direction == TradeDirection.LONG:
            position.unrealized_pnl = (current_price - position.entry_price) * position.quantity * multiplier
        else:
            position.unrealized_pnl = (position.entry_price - current_price) * position.quantity * multiplier

        # Check exit conditions
        exit_signal = self._check_exit_conditions(position, current_price)

        if exit_signal:
            return exit_signal

        # Update trailing stop if applicable
        position.update_trailing_stop(current_price)

        return None

    def _check_exit_conditions(
        self,
        position: Position,
        current_price: float,
    ) -> Optional[dict]:
        """
        Check all exit conditions in priority order.

        Exit Conditions (Bill Fanter's rules):
        1. Stop loss hit (HIGHEST PRIORITY - always honor)
        2. Target reached
        3. Trailing stop hit (after partial profit)
        4. Time-based exit (options expiration approaching)
        """
        is_long = position.direction == TradeDirection.LONG
        exit_signal = None

        # 1. STOP LOSS - Non-negotiable
        if is_long and current_price <= position.stop_loss:
            exit_signal = {
                "should_exit": True,
                "reason": "Stop loss hit",
                "exit_type": "stop_loss",
                "exit_price": position.stop_loss,
            }
        elif not is_long and current_price >= position.stop_loss:
            exit_signal = {
                "should_exit": True,
                "reason": "Stop loss hit",
                "exit_type": "stop_loss",
                "exit_price": position.stop_loss,
            }

        # 2. TARGET REACHED
        elif is_long and current_price >= position.target_price:
            exit_signal = {
                "should_exit": True,
                "reason": "Target reached",
                "exit_type": "target",
                "exit_price": current_price,
            }
        elif not is_long and current_price <= position.target_price:
            exit_signal = {
                "should_exit": True,
                "reason": "Target reached",
                "exit_type": "target",
                "exit_price": current_price,
            }

        # 3. TRAILING STOP
        elif position.trailing_stop:
            if is_long and current_price <= position.trailing_stop:
                exit_signal = {
                    "should_exit": True,
                    "reason": "Trailing stop hit",
                    "exit_type": "trailing_stop",
                    "exit_price": current_price,
                }
            elif not is_long and current_price >= position.trailing_stop:
                exit_signal = {
                    "should_exit": True,
                    "reason": "Trailing stop hit",
                    "exit_type": "trailing_stop",
                    "exit_price": current_price,
                }

        # 4. TIME-BASED EXIT (Options expiration)
        if not exit_signal and position.option_expiry:
            days_to_expiry = (position.option_expiry - datetime.now()).days
            if days_to_expiry <= 1:
                exit_signal = {
                    "should_exit": True,
                    "reason": f"Expiration approaching ({days_to_expiry} days)",
                    "exit_type": "time_exit",
                    "exit_price": current_price,
                }

        return exit_signal

    async def close_position(
        self,
        position_id: str,
        exit_price: float,
        exit_reason: str,
        submit_order: bool = True,
    ) -> Optional[TradeResult]:
        """
        Close a position and create trade result.

        Args:
            position_id: Position to close
            exit_price: Exit price
            exit_reason: Reason for exit
            submit_order: Whether to submit exit order to broker

        Returns:
            TradeResult object
        """
        position = self._positions.get(position_id)
        if not position:
            logger.warning("Position not found", position_id=position_id)
            return None

        # Submit exit order if requested
        if submit_order:
            # Determine exit side
            if position.direction == TradeDirection.LONG:
                exit_side = OrderSide.SELL_TO_CLOSE if len(position.symbol) > 10 else OrderSide.SELL
            else:
                exit_side = OrderSide.BUY_TO_CLOSE if len(position.symbol) > 10 else OrderSide.BUY

            exit_order = Order(
                symbol=position.symbol,
                side=exit_side,
                quantity=position.quantity,
                order_type=OrderType.MARKET,
            )

            result = await self.order_manager.submit_order(exit_order)

            if not result.get("success"):
                logger.error(
                    "Failed to submit exit order",
                    position_id=position_id,
                    error=result.get("error"),
                )
                return None

        # Calculate P&L
        multiplier = 100 if len(position.symbol) > 10 else 1
        if position.direction == TradeDirection.LONG:
            pnl = (exit_price - position.entry_price) * position.quantity * multiplier
        else:
            pnl = (position.entry_price - exit_price) * position.quantity * multiplier

        pnl_percent = pnl / (position.entry_price * position.quantity * multiplier) * 100

        # Create trade result
        trade_result = TradeResult(
            signal_id=position.signal_id,
            symbol=position.symbol,
            option_symbol=position.symbol if len(position.symbol) > 10 else None,
            direction=position.direction,
            quantity=position.quantity,
            entry_price=position.entry_price,
            entry_time=position.entry_time,
            exit_price=exit_price,
            exit_time=datetime.now(),
            exit_reason=exit_reason,
            pnl=pnl,
            pnl_percent=pnl_percent,
        )

        # Remove from active positions
        del self._positions[position_id]

        logger.info(
            "Position closed",
            id=position_id,
            symbol=position.symbol,
            pnl=pnl,
            pnl_percent=f"{pnl_percent:.2f}%",
            reason=exit_reason,
        )

        return trade_result

    async def check_all_positions(
        self,
        prices: dict[str, float],
    ) -> list[dict]:
        """
        Check all positions against current prices.

        Args:
            prices: Dict of symbol -> current price

        Returns:
            List of exit signals for positions that need to be closed
        """
        exit_signals = []

        for position_id, position in list(self._positions.items()):
            current_price = prices.get(position.symbol)

            if current_price is None:
                continue

            exit_signal = await self.update_position_price(position_id, current_price)

            if exit_signal:
                exit_signal["position_id"] = position_id
                exit_signal["position"] = position
                exit_signals.append(exit_signal)

        return exit_signals

    def get_total_exposure(self) -> float:
        """Calculate total exposure across all positions."""
        return sum(p.cost_basis for p in self._positions.values())

    def get_position_count(self) -> int:
        """Get number of open positions."""
        return len(self._positions)

    def load_positions(self, positions: list[Position]) -> None:
        """Load positions from database."""
        for position in positions:
            self._positions[position.id] = position

        logger.info("Loaded positions", count=len(positions))
