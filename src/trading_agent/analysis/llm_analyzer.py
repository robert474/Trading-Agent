"""
LLM-powered trading analysis using Claude.

Teaches the model Bill Fanter's supply/demand methodology
through a detailed system prompt and few-shot examples.
"""

import json
import re
from datetime import datetime
from typing import Optional

import structlog
from anthropic import Anthropic

from trading_agent.core.config import settings
from trading_agent.core.models import (
    Candle,
    MarketContext,
    MarketTrend,
    OptionContract,
    TradeDirection,
    TradeSignal,
    Zone,
)

logger = structlog.get_logger()


# =============================================================================
# SYSTEM PROMPT - Bill Fanter's Methodology
# =============================================================================

SYSTEM_PROMPT = """You are an expert technical analyst specializing in supply and demand zone trading for short-term options. You have been trained on Bill Fanter's methodology.

## CORE PRINCIPLES

### Supply and Demand Zones
- **Demand Zone**: An area where buyers overwhelmed sellers, causing price to rally. When price returns to this zone, expect buying pressure again.
- **Supply Zone**: An area where sellers overwhelmed buyers, causing price to drop. When price returns to this zone, expect selling pressure again.

### Zone Identification Rules
1. **Strong departure**: Price must leave the zone quickly with momentum (large candles)
2. **Imbalance**: The zone represents unfilled orders - price spent little time there
3. **Freshness**: Fresh (untested) zones are strongest. Tested once is acceptable. Tested twice = broken.

### Zone Quality Factors (Score 0-100)
- Freshness: Fresh=30pts, Tested=15pts, Broken=0pts
- Departure strength: Strong momentum leaving = more points
- Time at level: Less time = more imbalance = higher score
- Higher timeframe alignment: Zones on higher TFs are more significant

### Entry Rules
1. Wait for price to reach a valid zone (don't chase)
2. Confirm higher timeframe trend alignment (don't trade against the trend)
3. Minimum 2:1 risk/reward ratio
4. Entry at zone edge, stop just beyond zone
5. **WAIT FOR CONFIRMATION**: Wick rejection, volume spike, or reversal pattern

### Entry Confirmation Patterns (CRITICAL)
- **Wick Rejection**: Price wicks INTO zone then closes OUTSIDE = buyers/sellers defending
- **Volume Spike**: 1.5x+ average volume at zone = institutional participation
- **Reversal Patterns**: Engulfing, hammer, shooting star at zone
- **No New Extremes**: For longs, no new lower lows. For shorts, no new higher highs.

### Exit Rules
1. Hard stop loss is NON-NEGOTIABLE - always honor it
2. Take profit at next opposing zone or measured move
3. Trail stop after 50% of target reached
4. Exit options positions by end of day before expiration

### Options Selection for Short-Term Trades
1. Expiration: 7-14 DTE (days to expiration) for short-term trades
2. Strike: ATM or slightly ITM (delta 0.50-0.70)
3. Liquidity: Bid-ask spread <10%, Open Interest >500

## MARKET CONTEXT RULES (Bill Fanter's Filters)

### When to AVOID Trading
- **FOMC Days**: Do not trade during/after Federal Reserve announcements
- **High VIX (>25)**: Reduce position size, wider stops, or sit out
- **Low Volume Weeks**: Thanksgiving, Christmas, holiday weeks - patterns less reliable
- **Pre-Earnings**: Don't hold options through earnings (IV crush risk)
- **First 15 Minutes**: Let the market settle after open before entering

### When to BE AGGRESSIVE
- **VIX 15-20**: Normal volatility, standard setups work well
- **Clear Trend Days**: Strong momentum in one direction, trade with it
- **Fresh Zones on Multiple Timeframes**: Confluence = higher probability
- **High Relative Strength**: Stock outperforming SPY in pullbacks = strong for calls

### Relative Strength Consideration
- For CALLS: Stock should be showing relative strength vs SPY
- For PUTS: Stock should be showing relative weakness vs SPY
- Avoid fighting sector rotation

## YOUR TASK
Analyze the provided chart data and identify:
1. Valid supply and demand zones with quality scores
2. Current trade setups (if any) WITH ENTRY CONFIRMATIONS
3. Entry price, stop loss, and target levels
4. Options contract recommendation if a setup exists
5. Market context assessment (VIX, news, sector strength)
6. Clear reasoning for your analysis

## IMPORTANT GUIDELINES
- Be specific with price levels - use exact numbers
- Be conservative - only recommend high-quality setups (score 70+)
- **REQUIRE ENTRY CONFIRMATION** - don't recommend trades without wick rejection, volume, or pattern
- If no setup exists, say so clearly
- Always explain your reasoning
- Risk/reward MUST be at least 2:1
- Never recommend trading against the higher timeframe trend
- Factor in VIX and market conditions into confidence score"""


class TradingAnalysisLLM:
    """
    LLM-powered analysis engine using Claude.
    Implements Bill Fanter's supply/demand trading methodology.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.api_key = api_key or settings.anthropic_api_key
        self.model = model or settings.claude_model
        self.client = Anthropic(api_key=self.api_key)

        # Few-shot examples (populated from database)
        self._examples: list[dict] = []

    def add_example(self, example: dict) -> None:
        """Add a trade example for few-shot learning."""
        self._examples.append(example)

    def set_examples(self, examples: list[dict]) -> None:
        """Set all few-shot examples."""
        self._examples = examples

    async def analyze_chart(
        self,
        symbol: str,
        candles: dict[str, list[Candle]],  # {timeframe: [candles]}
        current_price: float,
        detected_zones: Optional[list[Zone]] = None,
        options_chain: Optional[list[OptionContract]] = None,
        market_context: Optional[dict] = None,  # VIX, FOMC, earnings etc.
    ) -> dict:
        """
        Analyze chart data and generate trade recommendations.

        Args:
            symbol: Stock symbol
            candles: Dict of timeframe -> candles
            current_price: Current market price
            detected_zones: Pre-detected zones (optional)
            options_chain: Available options contracts
            market_context: Market conditions (VIX, FOMC, earnings, etc.)

        Returns:
            Analysis dict with zones, setup, and recommendation
        """
        market_context = market_context or {}
        # Format candle data
        candle_text = self._format_candles(candles)

        # Format zones if provided
        zones_text = ""
        if detected_zones:
            zones_text = self._format_zones(detected_zones)

        # Format options chain
        options_text = ""
        if options_chain:
            options_text = self._format_options(options_chain, current_price)

        # Format few-shot examples
        examples_text = self._format_examples()

        # Format market context
        context_text = self._format_market_context(market_context)

        # Build the user prompt
        user_prompt = f"""Analyze {symbol} for potential trade setups.

## CURRENT DATA
- **Symbol**: {symbol}
- **Current Price**: ${current_price:.2f}
- **Time**: {datetime.now().strftime('%Y-%m-%d %H:%M')} ET

{f"## MARKET CONDITIONS{chr(10)}{context_text}" if context_text else ""}

## CANDLE DATA BY TIMEFRAME
{candle_text}

{f"## PRE-DETECTED ZONES{chr(10)}{zones_text}" if zones_text else ""}

{f"## AVAILABLE OPTIONS (if recommending a trade){chr(10)}{options_text}" if options_text else ""}

{f"## EXAMPLE TRADES (Learn from these){chr(10)}{examples_text}" if examples_text else ""}

## REQUIRED OUTPUT FORMAT
Provide your analysis in the following JSON structure:
```json
{{
    "zones": [
        {{
            "type": "demand" or "supply",
            "zone_high": price,
            "zone_low": price,
            "timeframe": "15m",
            "quality_score": 0-100,
            "freshness": "fresh" or "tested",
            "reasoning": "why this zone matters"
        }}
    ],
    "current_setup": {{
        "has_setup": true or false,
        "direction": "long" or "short" or null,
        "entry_price": price or null,
        "stop_loss": price or null,
        "target": price or null,
        "risk_reward": ratio or null,
        "confidence": 0-100,
        "reasoning": "detailed explanation of setup or why no setup exists"
    }},
    "recommended_option": {{
        "symbol": "option symbol" or null,
        "strike": price,
        "expiration": "YYYY-MM-DD",
        "type": "call" or "put",
        "delta": value,
        "estimated_cost": price per contract
    }} or null,
    "market_context": {{
        "htf_trend": "bullish" or "bearish" or "neutral",
        "key_levels": [list of important prices],
        "caution_factors": ["any concerns or warnings"]
    }}
}}
```

Remember: Only recommend setups with quality score 70+ and risk/reward 2:1 or better."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )

            # Parse the response
            result = self._parse_response(response.content[0].text)

            logger.info(
                "LLM analysis complete",
                symbol=symbol,
                has_setup=result.get("current_setup", {}).get("has_setup", False),
                htf_trend=result.get("market_context", {}).get("htf_trend"),
            )

            return result

        except Exception as e:
            logger.error("LLM analysis failed", error=str(e), symbol=symbol)
            return {
                "zones": [],
                "current_setup": {
                    "has_setup": False,
                    "reasoning": f"Analysis failed: {str(e)}",
                },
                "recommended_option": None,
                "market_context": {"htf_trend": "neutral", "key_levels": [], "caution_factors": []},
            }

    def _format_candles(self, candles: dict[str, list[Candle]]) -> str:
        """Format candle data for the prompt."""
        output = []

        for tf, candle_list in candles.items():
            if not candle_list:
                continue

            output.append(f"\n### {tf.upper()} Timeframe (last {len(candle_list)} candles)")
            output.append("```")
            output.append("Time                | Open    | High    | Low     | Close   | Volume")
            output.append("-" * 75)

            for c in candle_list[-50:]:  # Last 50 candles max
                output.append(
                    f"{c.timestamp.strftime('%Y-%m-%d %H:%M')} | "
                    f"{c.open:7.2f} | {c.high:7.2f} | {c.low:7.2f} | "
                    f"{c.close:7.2f} | {c.volume:>8}"
                )

            output.append("```")

        return "\n".join(output)

    def _format_zones(self, zones: list[Zone]) -> str:
        """Format detected zones for the prompt."""
        output = ["```"]
        output.append("Type   | High    | Low     | TF   | Score | Freshness")
        output.append("-" * 60)

        for z in zones:
            output.append(
                f"{z.zone_type.value:6} | {z.zone_high:7.2f} | {z.zone_low:7.2f} | "
                f"{z.timeframe:4} | {z.quality_score:5.0f} | {z.freshness.value}"
            )

        output.append("```")
        return "\n".join(output)

    def _format_options(
        self,
        options: list[OptionContract],
        current_price: float,
    ) -> str:
        """Format options chain for the prompt."""
        # Filter to relevant strikes (ATM +/- 5)
        calls = sorted(
            [o for o in options if o.option_type == "call" and o.is_liquid],
            key=lambda x: abs(x.strike - current_price),
        )[:8]

        puts = sorted(
            [o for o in options if o.option_type == "put" and o.is_liquid],
            key=lambda x: abs(x.strike - current_price),
        )[:8]

        output = ["```"]
        output.append("Strike | Type | Exp        | Bid   | Ask   | Delta | OI")
        output.append("-" * 65)

        for opt in calls + puts:
            output.append(
                f"{opt.strike:6.0f} | {opt.option_type:4} | {opt.expiration} | "
                f"{opt.bid:5.2f} | {opt.ask:5.2f} | {opt.delta:5.2f} | {opt.open_interest:>5}"
            )

        output.append("```")
        return "\n".join(output)

    def _format_examples(self) -> str:
        """Format few-shot examples for the prompt."""
        if not self._examples:
            return ""

        output = []

        # Select mix of winners and losers
        winners = [e for e in self._examples if e.get("result") == "win"][:3]
        losers = [e for e in self._examples if e.get("result") == "loss"][:2]

        for example in winners + losers:
            label = "WINNING TRADE" if example.get("result") == "win" else "LOSING TRADE - LEARN FROM THIS"
            output.append(f"### {label}")
            output.append(f"**Symbol**: {example.get('symbol')}")
            output.append(f"**Setup**: {example.get('setup_description', 'N/A')}")
            output.append(f"**Entry Reasoning**: {example.get('entry_reasoning', 'N/A')}")
            output.append(f"**Entry**: ${example.get('entry_price', 0):.2f}")
            output.append(f"**Exit**: ${example.get('exit_price', 0):.2f}")
            output.append(f"**Result**: {example.get('result', '').upper()} (${example.get('pnl', 0):+.2f})")
            output.append(f"**Lessons**: {example.get('lessons', 'N/A')}")
            output.append("")

        return "\n".join(output)

    def _format_market_context(self, context: dict) -> str:
        """
        Format market context for the prompt.

        Bill Fanter's Context Filters:
        - VIX level affects position sizing and trade selection
        - FOMC days are no-trade days
        - Holiday weeks have unreliable patterns
        - Earnings affect individual stock volatility
        """
        if not context:
            return ""

        output = []

        # VIX Level
        vix = context.get("vix")
        if vix is not None:
            if vix > 25:
                output.append(f"- **VIX**: {vix:.1f} (HIGH - reduce size or sit out)")
            elif vix > 20:
                output.append(f"- **VIX**: {vix:.1f} (ELEVATED - use caution)")
            else:
                output.append(f"- **VIX**: {vix:.1f} (NORMAL - standard setups OK)")

        # FOMC/Fed Events
        if context.get("is_fomc_day"):
            output.append("- **⚠️ FOMC DAY**: Avoid trading during/after Fed announcement")
        if context.get("fed_speaker"):
            output.append(f"- **Fed Speaker**: {context['fed_speaker']} - watch for volatility")

        # Economic Data
        if context.get("major_data_release"):
            output.append(f"- **Data Release**: {context['major_data_release']}")

        # Holiday/Low Volume
        if context.get("is_holiday_week"):
            output.append("- **⚠️ HOLIDAY WEEK**: Lower volume, patterns less reliable")
        if context.get("low_volume"):
            output.append("- **LOW VOLUME**: Be cautious, price can move erratically")

        # Earnings
        earnings = context.get("upcoming_earnings")
        if earnings:
            if isinstance(earnings, list):
                output.append(f"- **Earnings Today**: {', '.join(earnings[:5])}")
            else:
                output.append(f"- **Earnings**: {earnings}")

        # SPY/QQQ Direction (market trend)
        spy_trend = context.get("spy_trend")
        if spy_trend:
            output.append(f"- **SPY Trend**: {spy_trend}")

        qqq_trend = context.get("qqq_trend")
        if qqq_trend:
            output.append(f"- **QQQ Trend**: {qqq_trend}")

        # Sector Strength
        sector_strength = context.get("sector_strength")
        if sector_strength:
            output.append(f"- **Strong Sectors**: {sector_strength.get('strong', 'N/A')}")
            output.append(f"- **Weak Sectors**: {sector_strength.get('weak', 'N/A')}")

        # Relative Strength of Symbol vs SPY
        rel_strength = context.get("relative_strength")
        if rel_strength:
            if rel_strength > 0:
                output.append(f"- **Relative Strength vs SPY**: +{rel_strength:.1f}% (bullish bias)")
            else:
                output.append(f"- **Relative Strength vs SPY**: {rel_strength:.1f}% (bearish bias)")

        # Time of Day
        time_warning = context.get("time_warning")
        if time_warning:
            output.append(f"- **⚠️ {time_warning}**")

        return "\n".join(output) if output else ""

    def _parse_response(self, text: str) -> dict:
        """Parse JSON from LLM response."""
        # Try to extract JSON from response
        json_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)

        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find raw JSON
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                logger.warning("Could not extract JSON from LLM response")
                return {
                    "zones": [],
                    "current_setup": {"has_setup": False, "reasoning": text},
                    "recommended_option": None,
                    "market_context": {},
                }

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse LLM JSON", error=str(e))
            return {
                "zones": [],
                "current_setup": {"has_setup": False, "reasoning": text},
                "recommended_option": None,
                "market_context": {},
            }

    def create_trade_signal(
        self,
        symbol: str,
        analysis: dict,
        zone: Optional[Zone] = None,
    ) -> Optional[TradeSignal]:
        """
        Create a TradeSignal from LLM analysis.

        Args:
            symbol: Stock symbol
            analysis: LLM analysis result
            zone: Associated zone (optional)

        Returns:
            TradeSignal or None if no valid setup
        """
        setup = analysis.get("current_setup", {})

        if not setup.get("has_setup"):
            return None

        direction = setup.get("direction")
        if direction not in ["long", "short"]:
            return None

        entry = setup.get("entry_price")
        stop = setup.get("stop_loss")
        target = setup.get("target")
        rr = setup.get("risk_reward", 0)

        if not all([entry, stop, target]) or rr < 2.0:
            return None

        # Get option recommendation
        opt = analysis.get("recommended_option")

        signal = TradeSignal(
            symbol=symbol,
            direction=TradeDirection.LONG if direction == "long" else TradeDirection.SHORT,
            entry_price=entry,
            stop_loss=stop,
            target_price=target,
            risk_reward=rr,
            zone=zone,
            zone_id=zone.id if zone else None,
            llm_reasoning=setup.get("reasoning", ""),
            llm_confidence=setup.get("confidence", 0) / 100,
        )

        # Add option details if recommended
        if opt:
            signal.option_symbol = opt.get("symbol")
            signal.option_strike = opt.get("strike")
            signal.option_expiration = opt.get("expiration")
            signal.option_type = opt.get("type")
            signal.option_delta = opt.get("delta")
            signal.option_premium = opt.get("estimated_cost")

        # Validate the signal
        valid, reason = signal.validate()
        if not valid:
            logger.warning("Invalid signal", reason=reason, symbol=symbol)
            return None

        return signal


class MarketAnalyzer:
    """
    High-level market analyzer combining zone detection and LLM analysis.
    """

    def __init__(
        self,
        zone_detector,
        llm_analyzer: TradingAnalysisLLM,
    ):
        from trading_agent.analysis.zone_detector import ZoneDetector

        self.zone_detector: ZoneDetector = zone_detector
        self.llm_analyzer = llm_analyzer

    async def analyze(
        self,
        symbol: str,
        candles: dict[str, list[Candle]],
        current_price: float,
        options_chain: Optional[list[OptionContract]] = None,
    ) -> dict:
        """
        Full analysis pipeline for a symbol.

        1. Detect zones programmatically
        2. Run LLM analysis for validation and setup detection
        3. Return combined results

        Args:
            symbol: Stock symbol
            candles: Multi-timeframe candles
            current_price: Current price
            options_chain: Available options

        Returns:
            Combined analysis result
        """
        # Step 1: Detect zones on each timeframe
        all_zones = []
        for tf, tf_candles in candles.items():
            if len(tf_candles) >= 10:
                zones = self.zone_detector.detect_zones(tf_candles, tf)
                all_zones.extend(zones)

        # Step 2: Find nearest zones
        nearest = self.zone_detector.find_nearest_zones(current_price, all_zones)

        # Step 3: LLM analysis
        llm_result = await self.llm_analyzer.analyze_chart(
            symbol=symbol,
            candles=candles,
            current_price=current_price,
            detected_zones=all_zones,
            options_chain=options_chain,
        )

        # Step 4: Create trade signal if setup exists
        signal = None
        if llm_result.get("current_setup", {}).get("has_setup"):
            # Find matching zone
            zone = nearest.get("nearest_demand") or nearest.get("nearest_supply")
            signal = self.llm_analyzer.create_trade_signal(symbol, llm_result, zone)

        # Build market context
        htf_trend_str = llm_result.get("market_context", {}).get("htf_trend", "neutral")
        htf_trend = MarketTrend.NEUTRAL
        if htf_trend_str == "bullish":
            htf_trend = MarketTrend.BULLISH
        elif htf_trend_str == "bearish":
            htf_trend = MarketTrend.BEARISH

        context = MarketContext(
            symbol=symbol,
            current_price=current_price,
            htf_trend=htf_trend,
            active_supply_zones=[z for z in all_zones if z.zone_type.value == "supply" and z.is_valid],
            active_demand_zones=[z for z in all_zones if z.zone_type.value == "demand" and z.is_valid],
            key_levels=llm_result.get("market_context", {}).get("key_levels", []),
            caution_factors=llm_result.get("market_context", {}).get("caution_factors", []),
        )

        return {
            "symbol": symbol,
            "current_price": current_price,
            "zones": all_zones,
            "nearest_zones": nearest,
            "llm_analysis": llm_result,
            "trade_signal": signal,
            "market_context": context,
        }
