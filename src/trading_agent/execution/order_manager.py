"""
Order Manager for trade execution via Tradier API.

Handles order submission, monitoring, and bracket orders
with stop-loss and take-profit.
"""

import asyncio
from datetime import datetime
from typing import Optional
from uuid import uuid4

import aiohttp
import structlog

from trading_agent.core.config import settings
from trading_agent.core.models import Order, OrderSide, OrderStatus, OrderType

logger = structlog.get_logger()


class OrderManager:
    """
    Execute trades via Tradier API with full order management.

    Supports:
    - Market, limit, stop, and stop-limit orders
    - Bracket orders (entry + stop loss + take profit)
    - OCO (one-cancels-other) orders
    - Order monitoring and status updates
    """

    def __init__(
        self,
        account_id: Optional[str] = None,
        access_token: Optional[str] = None,
        sandbox: Optional[bool] = None,
    ):
        self.account_id = account_id or settings.tradier_account_id
        self.access_token = access_token or settings.tradier_access_token
        self.sandbox = sandbox if sandbox is not None else settings.tradier_sandbox

        self.base_url = (
            "https://sandbox.tradier.com/v1"
            if self.sandbox
            else "https://api.tradier.com/v1"
        )

        self.headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Accept": "application/json",
        }

        # Track open orders
        self._open_orders: dict[str, Order] = {}

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[dict] = None,
        data: Optional[dict] = None,
    ) -> Optional[dict]:
        """Make authenticated request to Tradier API."""
        url = f"{self.base_url}{endpoint}"

        async with aiohttp.ClientSession() as session:
            async with session.request(
                method,
                url,
                headers=self.headers,
                params=params,
                data=data,
            ) as resp:
                response_text = await resp.text()

                if resp.status not in [200, 201]:
                    logger.error(
                        "Tradier API error",
                        status=resp.status,
                        endpoint=endpoint,
                        response=response_text,
                    )
                    return None

                return await resp.json()

    async def submit_order(self, order: Order) -> dict:
        """
        Submit a single order to Tradier.

        Args:
            order: Order object to submit

        Returns:
            Dict with order result including order_id
        """
        # Determine order class (equity or option)
        order_class = "option" if order.is_option else "equity"

        # Build payload
        payload = {
            "class": order_class,
            "symbol": order.symbol,
            "side": order.side.value,
            "quantity": str(order.quantity),
            "type": order.order_type.value,
            "duration": order.time_in_force,
        }

        # Add price for limit orders
        if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
            if order.limit_price:
                payload["price"] = str(order.limit_price)

        # Add stop price for stop orders
        if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT]:
            if order.stop_price:
                payload["stop"] = str(order.stop_price)

        logger.info(
            "Submitting order",
            symbol=order.symbol,
            side=order.side.value,
            quantity=order.quantity,
            type=order.order_type.value,
            limit=order.limit_price,
            stop=order.stop_price,
        )

        result = await self._request(
            "POST",
            f"/accounts/{self.account_id}/orders",
            data=payload,
        )

        if result and "order" in result:
            order.order_id = str(result["order"]["id"])
            order.status = OrderStatus(result["order"]["status"])
            order.created_at = datetime.now()
            order.broker_response = result

            self._open_orders[order.order_id] = order

            logger.info(
                "Order submitted",
                order_id=order.order_id,
                status=order.status.value,
            )

            return {
                "success": True,
                "order_id": order.order_id,
                "status": order.status.value,
                "result": result,
            }

        return {
            "success": False,
            "error": "Failed to submit order",
            "result": result,
        }

    async def submit_bracket_order(
        self,
        entry_order: Order,
        stop_loss_price: float,
        take_profit_price: float,
        wait_for_fill: bool = True,
        fill_timeout: int = 60,
    ) -> dict:
        """
        Submit entry order with attached stop loss and take profit (OCO bracket).

        Args:
            entry_order: The entry order
            stop_loss_price: Stop loss price
            take_profit_price: Take profit price
            wait_for_fill: Wait for entry to fill before placing bracket
            fill_timeout: Timeout in seconds for fill wait

        Returns:
            Dict with entry and bracket order results
        """
        # Submit entry order first
        entry_result = await self.submit_order(entry_order)

        if not entry_result.get("success"):
            return {
                "success": False,
                "error": "Entry order failed",
                "entry_result": entry_result,
            }

        if wait_for_fill:
            # Wait for entry to fill
            filled = await self._wait_for_fill(entry_order.order_id, fill_timeout)

            if not filled:
                # Cancel entry if not filled
                await self.cancel_order(entry_order.order_id)
                return {
                    "success": False,
                    "error": "Entry order not filled within timeout",
                    "entry_result": entry_result,
                }

        # Determine exit side (opposite of entry)
        if entry_order.side in [OrderSide.BUY, OrderSide.BUY_TO_OPEN]:
            exit_side = OrderSide.SELL_TO_CLOSE if entry_order.is_option else OrderSide.SELL
        else:
            exit_side = OrderSide.BUY_TO_CLOSE if entry_order.is_option else OrderSide.BUY

        # Create stop loss order
        stop_order = Order(
            symbol=entry_order.symbol,
            side=exit_side,
            quantity=entry_order.quantity,
            order_type=OrderType.STOP,
            stop_price=stop_loss_price,
            time_in_force="gtc",
        )

        # Create take profit order
        profit_order = Order(
            symbol=entry_order.symbol,
            side=exit_side,
            quantity=entry_order.quantity,
            order_type=OrderType.LIMIT,
            limit_price=take_profit_price,
            time_in_force="gtc",
        )

        # Submit as OCO bracket
        oco_result = await self._submit_oco(stop_order, profit_order)

        return {
            "success": True,
            "entry": entry_result,
            "bracket": oco_result,
            "stop_loss": stop_loss_price,
            "take_profit": take_profit_price,
        }

    async def _submit_oco(self, order1: Order, order2: Order) -> dict:
        """
        Submit One-Cancels-Other order pair.

        When one order fills, the other is automatically cancelled.
        """
        # Build OCO payload
        # Note: Tradier's OCO format may vary - this is a common structure
        payload = {
            "class": "oco",
            "duration": "gtc",
            "symbol[0]": order1.symbol,
            "side[0]": order1.side.value,
            "quantity[0]": str(order1.quantity),
            "type[0]": order1.order_type.value,
            "symbol[1]": order2.symbol,
            "side[1]": order2.side.value,
            "quantity[1]": str(order2.quantity),
            "type[1]": order2.order_type.value,
        }

        # Add prices
        if order1.stop_price:
            payload["stop[0]"] = str(order1.stop_price)
        if order1.limit_price:
            payload["price[0]"] = str(order1.limit_price)
        if order2.stop_price:
            payload["stop[1]"] = str(order2.stop_price)
        if order2.limit_price:
            payload["price[1]"] = str(order2.limit_price)

        result = await self._request(
            "POST",
            f"/accounts/{self.account_id}/orders",
            data=payload,
        )

        if result:
            logger.info("OCO bracket submitted", result=result)
            return {"success": True, "result": result}

        return {"success": False, "error": "OCO submission failed"}

    async def cancel_order(self, order_id: str) -> dict:
        """Cancel an open order."""
        result = await self._request(
            "DELETE",
            f"/accounts/{self.account_id}/orders/{order_id}",
        )

        if order_id in self._open_orders:
            self._open_orders[order_id].status = OrderStatus.CANCELLED

        logger.info("Order cancelled", order_id=order_id)

        return {"success": True, "order_id": order_id, "result": result}

    async def get_order_status(self, order_id: str) -> Optional[dict]:
        """Get current order status."""
        result = await self._request(
            "GET",
            f"/accounts/{self.account_id}/orders/{order_id}",
        )

        if result and "order" in result:
            order_data = result["order"]

            # Update local tracking
            if order_id in self._open_orders:
                self._open_orders[order_id].status = OrderStatus(order_data["status"])
                if order_data.get("avg_fill_price"):
                    self._open_orders[order_id].filled_price = float(order_data["avg_fill_price"])
                if order_data.get("exec_quantity"):
                    self._open_orders[order_id].filled_quantity = int(order_data["exec_quantity"])

            return order_data

        return None

    async def _wait_for_fill(self, order_id: str, timeout: int = 60) -> bool:
        """
        Wait for an order to fill.

        Args:
            order_id: Order ID to monitor
            timeout: Timeout in seconds

        Returns:
            True if filled, False otherwise
        """
        start = datetime.now()

        while (datetime.now() - start).seconds < timeout:
            status = await self.get_order_status(order_id)

            if status:
                order_status = status.get("status", "")

                if order_status == "filled":
                    return True
                elif order_status in ["cancelled", "rejected", "expired"]:
                    return False

            await asyncio.sleep(1)

        return False

    async def get_open_orders(self) -> list[dict]:
        """Get all open orders from the broker."""
        result = await self._request(
            "GET",
            f"/accounts/{self.account_id}/orders",
        )

        if not result:
            return []

        orders = result.get("orders", {}).get("order", [])
        if isinstance(orders, dict):
            orders = [orders]
        if orders is None:
            orders = []

        # Filter to open/pending orders
        open_orders = [
            o for o in orders if o.get("status") in ["open", "pending", "partially_filled"]
        ]

        return open_orders

    async def cancel_all_orders(self, symbol: Optional[str] = None) -> dict:
        """
        Cancel all open orders.

        Args:
            symbol: Optional - only cancel orders for this symbol
        """
        open_orders = await self.get_open_orders()

        if symbol:
            open_orders = [o for o in open_orders if o.get("symbol") == symbol]

        cancelled = []
        for order in open_orders:
            order_id = str(order.get("id"))
            result = await self.cancel_order(order_id)
            if result.get("success"):
                cancelled.append(order_id)

        logger.info("Cancelled orders", count=len(cancelled), symbol=symbol)

        return {"cancelled": cancelled, "count": len(cancelled)}

    def get_tracked_order(self, order_id: str) -> Optional[Order]:
        """Get a locally tracked order by ID."""
        return self._open_orders.get(order_id)

    def clear_filled_orders(self) -> None:
        """Remove filled orders from local tracking."""
        filled = [
            oid for oid, order in self._open_orders.items()
            if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.EXPIRED]
        ]

        for oid in filled:
            del self._open_orders[oid]
