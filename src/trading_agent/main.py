"""
Trading Agent Main Orchestrator.

The main entry point that coordinates all components:
- Data ingestion and aggregation
- Zone detection and LLM analysis
- Trade signal generation and execution
- Position management and risk control
- Monitoring and alerts
"""

import asyncio
import signal
from datetime import datetime, timedelta
from typing import Optional

import structlog

from trading_agent.analysis.llm_analyzer import MarketAnalyzer, TradingAnalysisLLM
from trading_agent.analysis.zone_detector import ZoneDetector
from trading_agent.core.config import settings
from trading_agent.core.models import Order, OrderSide, OrderType, TradeDirection
from trading_agent.data.aggregators.candle_aggregator import CandleAggregator
from trading_agent.data.database import DatabaseManager
from trading_agent.data.providers.polygon import PolygonDataProvider
from trading_agent.data.providers.tradier import TradierDataProvider
from trading_agent.execution.order_manager import OrderManager
from trading_agent.execution.position_manager import PositionManager
from trading_agent.monitoring.discord_alerts import DiscordAlerts
from trading_agent.monitoring.logger import setup_logging
from trading_agent.risk.risk_manager import RiskManager
from trading_agent.utils.helpers import is_market_open, get_next_market_open

logger = structlog.get_logger()


class TradingAgent:
    """
    Main trading agent orchestrator.

    Coordinates all components to:
    1. Ingest real-time market data
    2. Detect supply/demand zones
    3. Analyze setups with LLM
    4. Execute trades with risk management
    5. Monitor positions and send alerts
    """

    def __init__(
        self,
        initial_balance: float = 10000.0,
        paper_trading: bool = True,
    ):
        self.initial_balance = initial_balance
        self.paper_trading = paper_trading
        self._running = False
        self._shutdown_event = asyncio.Event()

        # Initialize components (will be set up in initialize())
        self.db: Optional[DatabaseManager] = None
        self.polygon: Optional[PolygonDataProvider] = None
        self.tradier: Optional[TradierDataProvider] = None
        self.candle_aggregator: Optional[CandleAggregator] = None
        self.zone_detector: Optional[ZoneDetector] = None
        self.llm_analyzer: Optional[TradingAnalysisLLM] = None
        self.market_analyzer: Optional[MarketAnalyzer] = None
        self.order_manager: Optional[OrderManager] = None
        self.position_manager: Optional[PositionManager] = None
        self.risk_manager: Optional[RiskManager] = None
        self.discord: Optional[DiscordAlerts] = None

        # Analysis interval (seconds)
        self.analysis_interval = 60  # Run analysis every minute

    async def initialize(self) -> None:
        """Initialize all components."""
        logger.info("Initializing Trading Agent...")

        # Database
        self.db = DatabaseManager()
        await self.db.connect()

        # Data providers
        self.polygon = PolygonDataProvider()
        self.tradier = TradierDataProvider()

        # Data processing
        self.candle_aggregator = CandleAggregator(
            timeframes=["1m", "5m", "15m", "1h", "4h"],
            on_candle_complete=self._on_candle_complete,
        )

        # Analysis
        self.zone_detector = ZoneDetector()
        self.llm_analyzer = TradingAnalysisLLM()
        self.market_analyzer = MarketAnalyzer(
            zone_detector=self.zone_detector,
            llm_analyzer=self.llm_analyzer,
        )

        # Load few-shot examples from database
        examples = await self.db.get_trade_examples(limit=10)
        self.llm_analyzer.set_examples(examples)

        # Execution
        self.order_manager = OrderManager()
        self.position_manager = PositionManager(self.order_manager)

        # Load existing positions from database
        positions = await self.db.get_open_positions()
        self.position_manager.load_positions(positions)

        # Risk management
        # Get account balance from broker or use initial
        balance = self.initial_balance
        if not self.paper_trading:
            broker_balance = await self.tradier.get_account_balance()
            if broker_balance:
                balance = broker_balance.get("total_equity", self.initial_balance)

        self.risk_manager = RiskManager(account_balance=balance)

        # Monitoring
        self.discord = DiscordAlerts()

        # Register data callback
        self.polygon.add_callback(self._on_market_data)

        logger.info(
            "Trading Agent initialized",
            paper_trading=self.paper_trading,
            balance=balance,
            watchlist=settings.watchlist,
        )

    async def start(self) -> None:
        """Start the trading agent."""
        self._running = True

        # Send startup notification
        await self.discord.send_system_status(
            "online",
            f"Trading Agent started in {'PAPER' if self.paper_trading else 'LIVE'} mode\n"
            f"Balance: ${self.risk_manager.account_balance:,.2f}\n"
            f"Watchlist: {', '.join(settings.watchlist)}",
        )

        # Start background tasks
        tasks = [
            asyncio.create_task(self._market_data_loop()),
            asyncio.create_task(self._analysis_loop()),
            asyncio.create_task(self._position_monitor_loop()),
            asyncio.create_task(self._daily_tasks_loop()),
        ]

        logger.info("Trading Agent started")

        # Wait for shutdown signal
        await self._shutdown_event.wait()

        # Cancel tasks
        for task in tasks:
            task.cancel()

        # Cleanup
        await self.shutdown()

    async def shutdown(self) -> None:
        """Shutdown the trading agent."""
        logger.info("Shutting down Trading Agent...")

        self._running = False

        # Send shutdown notification
        summary = self.risk_manager.get_risk_report()
        await self.discord.send_daily_summary(summary)
        await self.discord.send_system_status("offline", "Trading Agent shutting down")

        # Disconnect from data sources
        if self.polygon:
            await self.polygon.disconnect()

        # Close database
        if self.db:
            await self.db.disconnect()

        logger.info("Trading Agent shutdown complete")

    def request_shutdown(self) -> None:
        """Request graceful shutdown."""
        self._shutdown_event.set()

    # =========================================================================
    # Background Loops
    # =========================================================================

    async def _market_data_loop(self) -> None:
        """Stream market data from Polygon."""
        while self._running:
            try:
                # Wait for market to be open
                if not is_market_open():
                    next_open = get_next_market_open()
                    wait_seconds = (next_open - datetime.now(next_open.tzinfo)).total_seconds()

                    logger.info(
                        "Market closed, waiting for open",
                        next_open=next_open.strftime("%Y-%m-%d %H:%M %Z"),
                        wait_minutes=wait_seconds / 60,
                    )

                    # Wait but check periodically
                    while wait_seconds > 0 and self._running:
                        await asyncio.sleep(min(wait_seconds, 60))
                        wait_seconds -= 60

                    if not self._running:
                        break

                # Connect and subscribe
                await self.polygon.connect()
                await self.polygon.subscribe(settings.watchlist)

                # Stream data
                await self.polygon.start_streaming()

            except Exception as e:
                logger.error("Market data loop error", error=str(e))
                await asyncio.sleep(10)

    async def _analysis_loop(self) -> None:
        """Periodic analysis of watchlist symbols."""
        while self._running:
            try:
                if not is_market_open():
                    await asyncio.sleep(60)
                    continue

                # Can we trade?
                can_trade, reason = self.risk_manager.can_trade()
                if not can_trade:
                    logger.info("Trading disabled", reason=reason)
                    await asyncio.sleep(self.analysis_interval)
                    continue

                # Analyze each symbol in watchlist
                for symbol in settings.watchlist:
                    if not self._running:
                        break

                    await self._analyze_symbol(symbol)

                await asyncio.sleep(self.analysis_interval)

            except Exception as e:
                logger.error("Analysis loop error", error=str(e))
                await asyncio.sleep(30)

    async def _position_monitor_loop(self) -> None:
        """Monitor open positions for exit conditions."""
        while self._running:
            try:
                if not is_market_open():
                    await asyncio.sleep(60)
                    continue

                # Get current prices for all position symbols
                positions = self.position_manager.get_all_positions()
                if not positions:
                    await asyncio.sleep(10)
                    continue

                symbols = list(set(p.symbol for p in positions))
                quotes = await self.tradier.get_quotes(symbols)

                prices = {
                    symbol: quote.get("last", 0)
                    for symbol, quote in quotes.items()
                }

                # Check all positions
                exit_signals = await self.position_manager.check_all_positions(prices)

                # Process exits
                for exit_signal in exit_signals:
                    await self._process_exit(exit_signal)

                await asyncio.sleep(5)  # Check every 5 seconds

            except Exception as e:
                logger.error("Position monitor error", error=str(e))
                await asyncio.sleep(10)

    async def _daily_tasks_loop(self) -> None:
        """Daily tasks like summary reports."""
        last_summary_date = None

        while self._running:
            try:
                now = datetime.now()

                # Send daily summary at 4:30 PM ET
                if now.hour == 16 and now.minute >= 30:
                    today = now.date()
                    if last_summary_date != today:
                        summary = self.risk_manager.get_risk_report()
                        await self.discord.send_daily_summary(summary)
                        last_summary_date = today

                        # Reset daily metrics
                        self.risk_manager.reset_daily_metrics()

                await asyncio.sleep(60)

            except Exception as e:
                logger.error("Daily tasks error", error=str(e))
                await asyncio.sleep(60)

    # =========================================================================
    # Event Handlers
    # =========================================================================

    async def _on_market_data(self, event_type: str, data) -> None:
        """Handle incoming market data."""
        if event_type == "candle":
            # Add to aggregator
            completed = self.candle_aggregator.add_candle(data)

            # Save completed candles to database
            if completed:
                await self.db.save_candles(completed)

        elif event_type == "trade":
            # Update position prices in real-time
            symbol = data.get("symbol")
            price = data.get("price")

            for position in self.position_manager.get_positions_for_symbol(symbol):
                position.current_price = price

    def _on_candle_complete(self, candle) -> None:
        """Handle completed candle (callback from aggregator)."""
        logger.debug(
            "Candle complete",
            symbol=candle.symbol,
            timeframe=candle.timeframe,
            close=candle.close,
        )

    # =========================================================================
    # Analysis & Trading
    # =========================================================================

    async def _analyze_symbol(self, symbol: str) -> None:
        """Run full analysis on a symbol."""
        try:
            # Get multi-timeframe candles
            candles = {}
            for tf in ["5m", "15m", "1h", "4h"]:
                tf_candles = self.candle_aggregator.get_candles(symbol, tf, count=100)
                if tf_candles:
                    candles[tf] = tf_candles

            if not candles:
                # Fetch historical if no candles in aggregator
                hist_candles = await self.polygon.get_historical_candles(
                    symbol, timeframe="15", limit=200
                )
                if hist_candles:
                    self.candle_aggregator.load_historical(hist_candles)
                    candles["15m"] = hist_candles
                else:
                    return

            # Get current price
            quote = await self.tradier.get_quote(symbol)
            if not quote:
                return
            current_price = quote.get("last", 0)

            if current_price <= 0:
                return

            # Get options chain
            options_chain = await self.tradier.get_options_chain(
                symbol,
                min_dte=settings.risk_limits.min_options_dte,
                max_dte=settings.risk_limits.max_options_dte,
            )

            # Run analysis
            analysis = await self.market_analyzer.analyze(
                symbol=symbol,
                candles=candles,
                current_price=current_price,
                options_chain=options_chain,
            )

            # Process trade signal if present
            signal = analysis.get("trade_signal")
            if signal:
                await self._process_signal(signal, options_chain)

        except Exception as e:
            logger.error("Analysis error", symbol=symbol, error=str(e))

    async def _process_signal(self, signal, options_chain) -> None:
        """Process a trade signal through risk management and execution."""
        # Validate through risk manager
        option_premium = signal.option_premium
        valid, reason, params = self.risk_manager.validate_trade(signal, option_premium)

        if not valid:
            logger.info(
                "Signal rejected by risk manager",
                symbol=signal.symbol,
                reason=reason,
            )
            return

        # Send signal alert
        await self.discord.send_trade_signal(signal)

        # Save signal to database
        signal_id = await self.db.save_signal(signal)
        signal.id = signal_id

        # Execute trade
        await self._execute_trade(signal, params)

    async def _execute_trade(self, signal, params: dict) -> None:
        """Execute a validated trade."""
        quantity = params["quantity"]

        # Determine the symbol to trade (option or underlying)
        trade_symbol = signal.option_symbol or signal.symbol

        # Determine order side
        if signal.direction == TradeDirection.LONG:
            side = OrderSide.BUY_TO_OPEN if signal.option_symbol else OrderSide.BUY
        else:
            side = OrderSide.SELL_TO_OPEN if signal.option_symbol else OrderSide.SELL

        # Create entry order
        entry_order = Order(
            symbol=trade_symbol,
            side=side,
            quantity=quantity,
            order_type=OrderType.LIMIT,
            limit_price=signal.entry_price if not signal.option_symbol else signal.option_premium,
        )

        # Submit bracket order
        result = await self.order_manager.submit_bracket_order(
            entry_order=entry_order,
            stop_loss_price=signal.stop_loss,
            take_profit_price=signal.target_price,
        )

        if result.get("success"):
            # Create position
            fill_price = entry_order.filled_price or signal.entry_price
            position = await self.position_manager.open_position(
                signal=signal,
                quantity=quantity,
                actual_entry_price=fill_price,
            )

            if position:
                # Track in risk manager
                self.risk_manager.add_position(position.id, {
                    "cost_basis": position.cost_basis,
                    "symbol": position.symbol,
                })

                # Save to database
                await self.db.save_position(position)

                # Send execution alert
                await self.discord.send_trade_executed(signal, quantity, fill_price)

                # Update signal status
                await self.db.update_signal_status(signal.id, "triggered")

                logger.info(
                    "Trade executed",
                    symbol=signal.symbol,
                    quantity=quantity,
                    fill_price=fill_price,
                )
        else:
            logger.error("Trade execution failed", error=result.get("error"))

    async def _process_exit(self, exit_signal: dict) -> None:
        """Process a position exit."""
        position = exit_signal["position"]
        exit_price = exit_signal["exit_price"]
        exit_reason = exit_signal["exit_type"]

        # Close position
        result = await self.position_manager.close_position(
            position_id=position.id,
            exit_price=exit_price,
            exit_reason=exit_reason,
        )

        if result:
            # Record in risk manager
            self.risk_manager.record_trade_result(position.id, result.pnl)
            self.risk_manager.remove_position(position.id)

            # Save to database
            await self.db.save_trade(result)
            await self.db.delete_position(position.id)

            # Send alert
            await self.discord.send_trade_closed(result)

            logger.info(
                "Position closed",
                symbol=position.symbol,
                pnl=result.pnl,
                reason=exit_reason,
            )


async def main():
    """Main entry point."""
    setup_logging()

    agent = TradingAgent(
        initial_balance=10000.0,
        paper_trading=settings.tradier_sandbox,
    )

    # Handle shutdown signals
    loop = asyncio.get_event_loop()

    def handle_shutdown():
        logger.info("Shutdown signal received")
        agent.request_shutdown()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, handle_shutdown)

    # Initialize and start
    await agent.initialize()
    await agent.start()


if __name__ == "__main__":
    asyncio.run(main())
