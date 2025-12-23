"""
Risk Management System.

Enforces position sizing, daily loss limits, and trade validation.
All trades must pass through risk management before execution.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

import structlog

from trading_agent.core.config import settings
from trading_agent.core.models import TradeSignal

logger = structlog.get_logger()


@dataclass
class RiskLimits:
    """Risk management configuration."""

    max_position_size_pct: float = 0.05      # Max 5% of account per trade
    max_daily_loss_pct: float = 0.02         # Stop trading after 2% daily loss
    max_total_exposure_pct: float = 0.25     # Max 25% of account in positions
    max_positions: int = 5                    # Max 5 concurrent positions
    max_loss_per_trade_pct: float = 0.01     # Max 1% loss per trade
    min_risk_reward: float = 2.0              # Minimum 2:1 R:R
    max_options_dte: int = 14                 # Max 14 days to expiration
    min_options_dte: int = 3                  # Min 3 days to expiration


class RiskManager:
    """
    Manages all risk controls for the trading system.

    Responsibilities:
    - Validate trades against risk limits
    - Calculate position sizing
    - Track daily P&L
    - Enforce max positions and exposure
    - Gate all trade entries
    """

    def __init__(
        self,
        account_balance: float,
        limits: Optional[RiskLimits] = None,
    ):
        self.account_balance = account_balance
        self.initial_balance = account_balance
        self.limits = limits or RiskLimits(
            max_position_size_pct=settings.risk_limits.max_position_size_pct,
            max_daily_loss_pct=settings.risk_limits.max_daily_loss_pct,
            max_total_exposure_pct=settings.risk_limits.max_total_exposure_pct,
            max_positions=settings.risk_limits.max_positions,
            max_loss_per_trade_pct=settings.risk_limits.max_loss_per_trade_pct,
            min_risk_reward=settings.risk_limits.min_risk_reward,
            max_options_dte=settings.risk_limits.max_options_dte,
            min_options_dte=settings.risk_limits.min_options_dte,
        )

        # Daily tracking
        self.daily_pnl: float = 0.0
        self.daily_trades: int = 0
        self.daily_wins: int = 0
        self.daily_losses: int = 0
        self._last_reset_date: date = date.today()

        # Position tracking
        self._open_positions: dict[str, dict] = {}
        self._trade_log: list[dict] = []

    def reset_daily_metrics(self) -> None:
        """Reset daily P&L tracking (call at market open)."""
        today = date.today()
        if today > self._last_reset_date:
            logger.info(
                "Daily metrics reset",
                previous_pnl=self.daily_pnl,
                trades=self.daily_trades,
            )
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.daily_wins = 0
            self.daily_losses = 0
            self._last_reset_date = today

    def can_trade(self) -> tuple[bool, str]:
        """
        Check if trading is allowed based on risk limits.

        Returns:
            (allowed, reason)
        """
        self.reset_daily_metrics()

        # Check daily loss limit
        if self.daily_pnl < 0:
            daily_loss_pct = abs(self.daily_pnl) / self.account_balance
            if daily_loss_pct >= self.limits.max_daily_loss_pct:
                return False, f"Daily loss limit reached ({daily_loss_pct:.1%} >= {self.limits.max_daily_loss_pct:.1%})"

        # Check max positions
        if len(self._open_positions) >= self.limits.max_positions:
            return False, f"Max positions reached ({len(self._open_positions)}/{self.limits.max_positions})"

        # Check total exposure
        total_exposure = sum(p.get("cost_basis", 0) for p in self._open_positions.values())
        exposure_pct = total_exposure / self.account_balance
        if exposure_pct >= self.limits.max_total_exposure_pct:
            return False, f"Max exposure reached ({exposure_pct:.1%} >= {self.limits.max_total_exposure_pct:.1%})"

        return True, "OK"

    def validate_trade(
        self,
        signal: TradeSignal,
        option_premium: Optional[float] = None,
    ) -> tuple[bool, str, Optional[dict]]:
        """
        Validate a proposed trade against all risk limits.

        Args:
            signal: Trade signal to validate
            option_premium: Premium per contract if options trade

        Returns:
            (valid, reason, adjusted_trade_params)
        """
        # Check if we can trade at all
        can_trade, reason = self.can_trade()
        if not can_trade:
            return False, reason, None

        # Validate risk/reward
        if signal.risk_reward < self.limits.min_risk_reward:
            return (
                False,
                f"R:R too low ({signal.risk_reward:.1f}:1 < {self.limits.min_risk_reward}:1)",
                None,
            )

        # Validate signal itself
        valid, signal_reason = signal.validate()
        if not valid:
            return False, f"Invalid signal: {signal_reason}", None

        # Check options DTE if applicable
        if signal.option_expiration:
            from datetime import datetime as dt
            exp_date = dt.strptime(signal.option_expiration, "%Y-%m-%d").date()
            dte = (exp_date - date.today()).days

            if dte < self.limits.min_options_dte:
                return False, f"DTE too short ({dte} < {self.limits.min_options_dte})", None
            if dte > self.limits.max_options_dte:
                return False, f"DTE too long ({dte} > {self.limits.max_options_dte})", None

        # Calculate position size
        position_params = self._calculate_position_size(signal, option_premium)

        if position_params["quantity"] < 1:
            return False, "Position size too small after risk adjustment", None

        logger.info(
            "Trade validated",
            symbol=signal.symbol,
            direction=signal.direction.value,
            quantity=position_params["quantity"],
            risk_amount=position_params["risk_amount"],
        )

        return True, "OK", position_params

    def _calculate_position_size(
        self,
        signal: TradeSignal,
        option_premium: Optional[float] = None,
    ) -> dict:
        """
        Calculate position size based on risk parameters.

        Position sizing rules:
        1. Max loss per trade = max_loss_per_trade_pct of account
        2. Position value <= max_position_size_pct of account
        """
        entry = signal.entry_price
        stop = signal.stop_loss
        risk_per_share = abs(entry - stop)

        # Max loss allowed for this trade
        max_loss = self.account_balance * self.limits.max_loss_per_trade_pct

        # Max position value
        max_position_value = self.account_balance * self.limits.max_position_size_pct

        if option_premium:
            # For options, calculate based on contract cost
            contract_cost = option_premium * 100  # Options are 100 shares

            # Max contracts by risk (assuming max loss is full premium)
            max_by_risk = int(max_loss / contract_cost)

            # Max contracts by position size
            max_by_size = int(max_position_value / contract_cost)

            quantity = min(max_by_risk, max_by_size, 10)  # Cap at 10 contracts
            risk_amount = quantity * contract_cost
            position_value = quantity * contract_cost

        else:
            # For stock, calculate based on risk per share
            max_shares_by_risk = int(max_loss / risk_per_share)
            max_shares_by_size = int(max_position_value / entry)

            quantity = min(max_shares_by_risk, max_shares_by_size)
            risk_amount = quantity * risk_per_share
            position_value = quantity * entry

        return {
            "quantity": max(0, quantity),
            "risk_amount": risk_amount,
            "position_value": position_value,
            "risk_per_unit": risk_per_share,
            "max_loss": max_loss,
        }

    def add_position(self, position_id: str, position_data: dict) -> None:
        """Track a new open position."""
        self._open_positions[position_id] = position_data
        logger.debug("Position added to risk tracking", position_id=position_id)

    def remove_position(self, position_id: str) -> None:
        """Remove a closed position."""
        if position_id in self._open_positions:
            del self._open_positions[position_id]
            logger.debug("Position removed from risk tracking", position_id=position_id)

    def record_trade_result(self, trade_id: str, pnl: float) -> None:
        """Record a completed trade result."""
        self.daily_pnl += pnl
        self.daily_trades += 1

        if pnl > 0:
            self.daily_wins += 1
        else:
            self.daily_losses += 1

        # Update account balance
        self.account_balance += pnl

        self._trade_log.append({
            "trade_id": trade_id,
            "pnl": pnl,
            "timestamp": datetime.now(),
            "balance_after": self.account_balance,
        })

        logger.info(
            "Trade result recorded",
            trade_id=trade_id,
            pnl=pnl,
            daily_pnl=self.daily_pnl,
            account_balance=self.account_balance,
        )

    def get_risk_report(self) -> dict:
        """Generate current risk status report."""
        total_exposure = sum(p.get("cost_basis", 0) for p in self._open_positions.values())

        daily_loss_limit = self.limits.max_daily_loss_pct * self.account_balance
        daily_loss_remaining = daily_loss_limit - abs(min(0, self.daily_pnl))

        exposure_limit = self.limits.max_total_exposure_pct * self.account_balance
        exposure_remaining = exposure_limit - total_exposure

        return {
            "account_balance": self.account_balance,
            "initial_balance": self.initial_balance,
            "total_return_pct": (self.account_balance - self.initial_balance) / self.initial_balance * 100,
            "daily_pnl": self.daily_pnl,
            "daily_pnl_pct": self.daily_pnl / self.account_balance * 100,
            "daily_trades": self.daily_trades,
            "daily_wins": self.daily_wins,
            "daily_losses": self.daily_losses,
            "daily_win_rate": self.daily_wins / self.daily_trades * 100 if self.daily_trades > 0 else 0,
            "open_positions": len(self._open_positions),
            "total_exposure": total_exposure,
            "exposure_pct": total_exposure / self.account_balance * 100,
            "can_trade": self.can_trade()[0],
            "limits": {
                "daily_loss_remaining": daily_loss_remaining,
                "positions_remaining": self.limits.max_positions - len(self._open_positions),
                "exposure_remaining": exposure_remaining,
            },
        }

    def update_account_balance(self, new_balance: float) -> None:
        """Update account balance (e.g., from broker sync)."""
        self.account_balance = new_balance
        logger.info("Account balance updated", balance=new_balance)

    def get_max_position_size(self, entry_price: float) -> int:
        """Get maximum position size for a given entry price."""
        max_value = self.account_balance * self.limits.max_position_size_pct
        return int(max_value / entry_price)

    def should_trail_stop(self, profit_pct: float) -> bool:
        """Check if trailing stop should be activated."""
        # Activate trailing stop at 50% of typical target
        # Assuming 2:1 R:R with 1% risk, target is ~2%
        return profit_pct >= 0.01  # 1% profit triggers trailing

    def get_trailing_stop_level(
        self,
        entry_price: float,
        current_price: float,
        direction: str,
        trail_pct: float = 0.5,
    ) -> float:
        """
        Calculate trailing stop level.

        Trails at trail_pct of current profit from entry.
        """
        if direction == "long":
            profit = current_price - entry_price
            return entry_price + (profit * trail_pct)
        else:
            profit = entry_price - current_price
            return entry_price - (profit * trail_pct)
