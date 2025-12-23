"""
Discord Webhook Alerts.

Sends trade notifications, alerts, and daily summaries to Discord.
"""

from datetime import datetime
from typing import Optional

import aiohttp
import structlog

from trading_agent.core.config import settings
from trading_agent.core.models import Position, TradeResult, TradeSignal

logger = structlog.get_logger()


class DiscordAlerts:
    """
    Discord webhook integration for trade alerts.

    Sends:
    - Trade entry signals
    - Trade executions
    - Stop loss / target hits
    - Daily summaries
    - Risk warnings
    """

    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url or settings.discord_webhook_url
        self._enabled = bool(self.webhook_url)

    async def _send_webhook(self, payload: dict) -> bool:
        """Send a message to Discord webhook."""
        if not self._enabled:
            logger.debug("Discord alerts disabled (no webhook URL)")
            return False

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                ) as resp:
                    if resp.status not in [200, 204]:
                        error = await resp.text()
                        logger.error(
                            "Discord webhook failed",
                            status=resp.status,
                            error=error,
                        )
                        return False

                    return True

        except Exception as e:
            logger.error("Discord webhook error", error=str(e))
            return False

    async def send_trade_signal(self, signal: TradeSignal) -> bool:
        """Send a new trade signal alert."""
        direction_emoji = "🟢" if signal.direction.value == "long" else "🔴"
        option_info = ""

        if signal.option_symbol:
            option_info = f"\n**Option**: {signal.option_symbol}"
            if signal.option_strike and signal.option_expiration:
                option_info += f"\n**Strike**: ${signal.option_strike:.2f} ({signal.option_expiration})"
            if signal.option_premium:
                option_info += f"\n**Premium**: ${signal.option_premium:.2f}"

        payload = {
            "embeds": [
                {
                    "title": f"{direction_emoji} New Trade Signal: {signal.symbol}",
                    "color": 0x00FF00 if signal.direction.value == "long" else 0xFF0000,
                    "fields": [
                        {
                            "name": "Direction",
                            "value": signal.direction.value.upper(),
                            "inline": True,
                        },
                        {
                            "name": "Entry",
                            "value": f"${signal.entry_price:.2f}",
                            "inline": True,
                        },
                        {
                            "name": "Stop Loss",
                            "value": f"${signal.stop_loss:.2f}",
                            "inline": True,
                        },
                        {
                            "name": "Target",
                            "value": f"${signal.target_price:.2f}",
                            "inline": True,
                        },
                        {
                            "name": "Risk/Reward",
                            "value": f"{signal.risk_reward:.1f}:1",
                            "inline": True,
                        },
                        {
                            "name": "Confidence",
                            "value": f"{signal.llm_confidence * 100:.0f}%",
                            "inline": True,
                        },
                    ],
                    "description": f"{signal.llm_reasoning[:500] if signal.llm_reasoning else 'No reasoning provided'}{option_info}",
                    "footer": {
                        "text": f"Trading Agent | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ET"
                    },
                }
            ]
        }

        return await self._send_webhook(payload)

    async def send_trade_executed(
        self,
        signal: TradeSignal,
        quantity: int,
        fill_price: float,
    ) -> bool:
        """Send trade execution alert."""
        direction_emoji = "🟢" if signal.direction.value == "long" else "🔴"

        payload = {
            "embeds": [
                {
                    "title": f"{direction_emoji} Trade Executed: {signal.symbol}",
                    "color": 0x0099FF,
                    "fields": [
                        {
                            "name": "Direction",
                            "value": signal.direction.value.upper(),
                            "inline": True,
                        },
                        {
                            "name": "Quantity",
                            "value": str(quantity),
                            "inline": True,
                        },
                        {
                            "name": "Fill Price",
                            "value": f"${fill_price:.2f}",
                            "inline": True,
                        },
                        {
                            "name": "Stop Loss",
                            "value": f"${signal.stop_loss:.2f}",
                            "inline": True,
                        },
                        {
                            "name": "Target",
                            "value": f"${signal.target_price:.2f}",
                            "inline": True,
                        },
                    ],
                    "footer": {
                        "text": f"Trading Agent | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ET"
                    },
                }
            ]
        }

        return await self._send_webhook(payload)

    async def send_trade_closed(self, result: TradeResult) -> bool:
        """Send trade closed alert."""
        pnl_emoji = "✅" if result.pnl > 0 else "❌"
        color = 0x00FF00 if result.pnl > 0 else 0xFF0000

        exit_reason_display = {
            "target": "🎯 Target Reached",
            "stop_loss": "🛑 Stop Loss Hit",
            "trailing_stop": "📈 Trailing Stop",
            "manual": "👤 Manual Exit",
            "expiry": "⏰ Expiration Exit",
        }

        payload = {
            "embeds": [
                {
                    "title": f"{pnl_emoji} Trade Closed: {result.symbol}",
                    "color": color,
                    "fields": [
                        {
                            "name": "Direction",
                            "value": result.direction.value.upper(),
                            "inline": True,
                        },
                        {
                            "name": "Entry",
                            "value": f"${result.entry_price:.2f}",
                            "inline": True,
                        },
                        {
                            "name": "Exit",
                            "value": f"${result.exit_price:.2f}",
                            "inline": True,
                        },
                        {
                            "name": "P&L",
                            "value": f"${result.pnl:+.2f} ({result.pnl_percent:+.2f}%)",
                            "inline": True,
                        },
                        {
                            "name": "Exit Reason",
                            "value": exit_reason_display.get(result.exit_reason, result.exit_reason),
                            "inline": True,
                        },
                        {
                            "name": "Duration",
                            "value": str(result.exit_time - result.entry_time).split(".")[0],
                            "inline": True,
                        },
                    ],
                    "footer": {
                        "text": f"Trading Agent | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ET"
                    },
                }
            ]
        }

        return await self._send_webhook(payload)

    async def send_daily_summary(
        self,
        summary: dict,
    ) -> bool:
        """Send daily performance summary."""
        win_rate = summary.get("daily_win_rate", 0)
        total_pnl = summary.get("daily_pnl", 0)

        pnl_emoji = "📈" if total_pnl >= 0 else "📉"
        color = 0x00FF00 if total_pnl >= 0 else 0xFF0000

        payload = {
            "embeds": [
                {
                    "title": f"{pnl_emoji} Daily Summary - {datetime.now().strftime('%Y-%m-%d')}",
                    "color": color,
                    "fields": [
                        {
                            "name": "Total P&L",
                            "value": f"${total_pnl:+.2f}",
                            "inline": True,
                        },
                        {
                            "name": "Trades",
                            "value": str(summary.get("daily_trades", 0)),
                            "inline": True,
                        },
                        {
                            "name": "Win Rate",
                            "value": f"{win_rate:.1f}%",
                            "inline": True,
                        },
                        {
                            "name": "Wins / Losses",
                            "value": f"{summary.get('daily_wins', 0)} / {summary.get('daily_losses', 0)}",
                            "inline": True,
                        },
                        {
                            "name": "Account Balance",
                            "value": f"${summary.get('account_balance', 0):,.2f}",
                            "inline": True,
                        },
                        {
                            "name": "Open Positions",
                            "value": str(summary.get("open_positions", 0)),
                            "inline": True,
                        },
                    ],
                    "footer": {
                        "text": f"Trading Agent | End of Day Report"
                    },
                }
            ]
        }

        return await self._send_webhook(payload)

    async def send_risk_warning(
        self,
        warning_type: str,
        message: str,
        details: Optional[dict] = None,
    ) -> bool:
        """Send risk warning alert."""
        payload = {
            "embeds": [
                {
                    "title": f"⚠️ Risk Warning: {warning_type}",
                    "color": 0xFFA500,  # Orange
                    "description": message,
                    "fields": [],
                    "footer": {
                        "text": f"Trading Agent | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ET"
                    },
                }
            ]
        }

        if details:
            for key, value in details.items():
                payload["embeds"][0]["fields"].append({
                    "name": key,
                    "value": str(value),
                    "inline": True,
                })

        return await self._send_webhook(payload)

    async def send_system_status(
        self,
        status: str,
        message: str,
    ) -> bool:
        """Send system status update."""
        status_config = {
            "online": ("🟢", 0x00FF00, "System Online"),
            "offline": ("🔴", 0xFF0000, "System Offline"),
            "error": ("❌", 0xFF0000, "System Error"),
            "warning": ("⚠️", 0xFFA500, "System Warning"),
        }

        emoji, color, title = status_config.get(status, ("ℹ️", 0x808080, "System Update"))

        payload = {
            "embeds": [
                {
                    "title": f"{emoji} {title}",
                    "color": color,
                    "description": message,
                    "footer": {
                        "text": f"Trading Agent | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ET"
                    },
                }
            ]
        }

        return await self._send_webhook(payload)

    async def send_position_update(
        self,
        position: Position,
        update_type: str,
    ) -> bool:
        """Send position update (e.g., trailing stop moved)."""
        update_messages = {
            "trailing_activated": "📊 Trailing stop activated",
            "trailing_moved": "📈 Trailing stop moved up",
            "approaching_target": "🎯 Approaching target",
            "approaching_stop": "⚠️ Approaching stop loss",
        }

        payload = {
            "content": (
                f"{update_messages.get(update_type, update_type)}\n"
                f"**{position.symbol}** | Current: ${position.current_price:.2f}\n"
                f"Stop: ${position.stop_loss:.2f} | "
                f"Trailing: ${position.trailing_stop:.2f if position.trailing_stop else 'N/A'} | "
                f"Target: ${position.target_price:.2f}"
            )
        }

        return await self._send_webhook(payload)
