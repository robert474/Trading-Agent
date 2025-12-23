#!/usr/bin/env python3
"""
Putt Indicator - The Brain + RAG + Bill Fanter Combined.

Validates and enriches trading setups by querying the RAG database
for historical context, similar trades, and Bill's methodology insights.

Named after the creator who combines:
- Personal trading intuition (the brain)
- RAG database retrieval (historical patterns)
- Bill Fanter's supply/demand methodology
"""

import sys
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.rag_pipeline.rag_database import BillFanterRAG


@dataclass
class PuttContext:
    """Context returned by the Putt Indicator for a setup."""

    symbol: str
    direction: str  # "long" or "short"
    zone_type: str  # "demand" or "supply"

    # Confidence adjustment
    base_confidence: float  # Original confidence (0-100)
    putt_adjustment: float  # RAG-based adjustment (-20 to +20)
    final_confidence: float  # base + adjustment

    # Historical context
    similar_trades: list  # Past trades with this symbol/pattern
    win_rate: Optional[float]  # Historical win rate for similar setups
    avg_rr: Optional[float]  # Average risk/reward from similar trades

    # Bill's insights
    key_insights: list  # Relevant quotes/lessons from Bill
    zone_history: list  # Past zones Bill mentioned for this symbol

    # Validation
    bill_mentioned: bool  # Has Bill ever discussed this symbol?
    pattern_recognized: bool  # Does this match a known Bill pattern?

    # Summary for display
    summary: str  # One-line summary of Putt's assessment


class PuttIndicator:
    """
    The Putt Indicator - validates and enriches setups with RAG context.

    Usage:
        putt = PuttIndicator()
        context = putt.analyze_setup(
            symbol="NVDA",
            direction="long",
            zone_type="demand",
            zone_level=180.0,
            base_confidence=65.0
        )
        print(f"Putt Confidence: {context.final_confidence}")
        print(f"Win Rate: {context.win_rate}")
        print(f"Insights: {context.key_insights}")
    """

    def __init__(self):
        self.rag = BillFanterRAG()
        self._cache = {}  # Simple cache for repeated queries

    def analyze_setup(
        self,
        symbol: str,
        direction: str,
        zone_type: str,
        zone_level: float,
        base_confidence: float = 50.0,
        timeframe: str = "daily",
    ) -> PuttContext:
        """
        Analyze a setup and return enriched context.

        Args:
            symbol: Stock ticker (e.g., "NVDA")
            direction: "long" or "short"
            zone_type: "demand" or "supply"
            zone_level: Price level of the zone
            base_confidence: Starting confidence score (0-100)
            timeframe: Timeframe of the setup

        Returns:
            PuttContext with validation and historical data
        """
        symbol = symbol.upper()

        # Get historical similar trades
        similar_trades = self._find_similar_trades(symbol, direction, zone_type)

        # Calculate win rate from similar trades
        win_rate, avg_rr = self._calculate_historical_stats(similar_trades)

        # Get Bill's insights for this symbol
        key_insights = self._get_key_insights(symbol)

        # Get zone history
        zone_history = self._get_zone_history(symbol, zone_type)

        # Check if Bill has mentioned this symbol
        bill_mentioned = len(similar_trades) > 0 or len(zone_history) > 0

        # Check if pattern is recognized
        pattern_recognized = self._check_pattern_match(symbol, direction, zone_type, zone_level)

        # Calculate Putt adjustment
        putt_adjustment = self._calculate_adjustment(
            win_rate=win_rate,
            bill_mentioned=bill_mentioned,
            pattern_recognized=pattern_recognized,
            similar_count=len(similar_trades),
        )

        # Final confidence (capped 0-100)
        final_confidence = max(0, min(100, base_confidence + putt_adjustment))

        # Generate summary
        summary = self._generate_summary(
            symbol=symbol,
            direction=direction,
            win_rate=win_rate,
            similar_count=len(similar_trades),
            bill_mentioned=bill_mentioned,
            putt_adjustment=putt_adjustment,
        )

        return PuttContext(
            symbol=symbol,
            direction=direction,
            zone_type=zone_type,
            base_confidence=base_confidence,
            putt_adjustment=putt_adjustment,
            final_confidence=final_confidence,
            similar_trades=similar_trades[:5],  # Top 5 most relevant
            win_rate=win_rate,
            avg_rr=avg_rr,
            key_insights=key_insights[:3],  # Top 3 insights
            zone_history=zone_history[:5],  # Top 5 zones
            bill_mentioned=bill_mentioned,
            pattern_recognized=pattern_recognized,
            summary=summary,
        )

    def _find_similar_trades(
        self,
        symbol: str,
        direction: str,
        zone_type: str
    ) -> list:
        """Find similar historical trades from RAG."""
        # Query for trade signals matching this setup
        query = f"{symbol} {direction} {zone_type} zone trade entry"

        results = self.rag.query(
            query_text=query,
            n_results=10,
            doc_type="trade_signal",
        )

        # Also search without symbol filter for pattern matches
        pattern_query = f"{direction} {zone_type} zone entry confirmation"
        pattern_results = self.rag.query(
            query_text=pattern_query,
            n_results=5,
            doc_type="trade_signal",
        )

        # Combine and deduplicate
        all_results = results + pattern_results
        seen = set()
        unique = []
        for r in all_results:
            text = r.get("text", "")[:100]
            if text not in seen:
                seen.add(text)
                unique.append(r)

        return unique

    def _calculate_historical_stats(self, trades: list) -> tuple:
        """Calculate win rate and average R:R from historical trades."""
        if not trades:
            return None, None

        # Extract outcomes from trade metadata
        wins = 0
        total_rr = 0
        rr_count = 0

        for trade in trades:
            meta = trade.get("metadata", {})
            text = trade.get("text", "").lower()

            # Check for win/loss indicators in text
            if "profit" in text or "winner" in text or "hit target" in text:
                wins += 1
            elif "loss" in text or "stopped" in text:
                pass  # Count as loss
            else:
                wins += 0.5  # Unknown = neutral

            # Try to extract R:R if mentioned
            # (This is approximate - real implementation would parse better)
            if "2:1" in text or "2r" in text.replace(" ", ""):
                total_rr += 2.0
                rr_count += 1
            elif "3:1" in text or "3r" in text.replace(" ", ""):
                total_rr += 3.0
                rr_count += 1
            elif "1:1" in text or "1r" in text.replace(" ", ""):
                total_rr += 1.0
                rr_count += 1

        win_rate = (wins / len(trades)) * 100 if trades else None
        avg_rr = total_rr / rr_count if rr_count > 0 else None

        return win_rate, avg_rr

    def _get_key_insights(self, symbol: str) -> list:
        """Get Bill's key insights relevant to this symbol or general methodology."""
        # Symbol-specific insights
        symbol_results = self.rag.query(
            query_text=f"{symbol} trading insight lesson",
            n_results=3,
            doc_type="insight",
        )

        # General methodology insights
        general_results = self.rag.query(
            query_text="supply demand zone entry confirmation patience",
            n_results=3,
            doc_type="insight",
        )

        insights = []
        for r in symbol_results + general_results:
            text = r.get("text", "")
            # Clean up the insight text
            if text.startswith("Trading insight from Bill Fanter: "):
                text = text.replace("Trading insight from Bill Fanter: ", "")
            if text and text not in insights:
                insights.append(text)

        return insights[:5]

    def _get_zone_history(self, symbol: str, zone_type: str) -> list:
        """Get historical zones Bill has mentioned for this symbol."""
        query = f"{symbol} {zone_type} zone level"

        results = self.rag.query(
            query_text=query,
            n_results=10,
            doc_type="zone_level",
        )

        # Filter for this symbol
        zones = []
        for r in results:
            meta = r.get("metadata", {})
            if meta.get("symbol", "").upper() == symbol.upper():
                zones.append(r)

        return zones

    def _check_pattern_match(
        self,
        symbol: str,
        direction: str,
        zone_type: str,
        zone_level: float
    ) -> bool:
        """Check if this setup matches a known Bill Fanter pattern."""
        # Search for transcript mentions of this exact setup
        query = f"{symbol} {zone_level} {zone_type}"

        results = self.rag.query(
            query_text=query,
            n_results=5,
        )

        # Check if any result is highly relevant
        for r in results:
            distance = r.get("distance", 1.0)
            if distance < 0.5:  # High similarity
                return True

            # Check if symbol and zone type match in text
            text = r.get("text", "").lower()
            if symbol.lower() in text and zone_type.lower() in text:
                return True

        return False

    def _calculate_adjustment(
        self,
        win_rate: Optional[float],
        bill_mentioned: bool,
        pattern_recognized: bool,
        similar_count: int,
    ) -> float:
        """
        Calculate confidence adjustment based on RAG data.

        Adjustment range: -20 to +20
        """
        adjustment = 0.0

        # Win rate adjustment (-10 to +10)
        if win_rate is not None:
            if win_rate >= 70:
                adjustment += 10
            elif win_rate >= 60:
                adjustment += 5
            elif win_rate >= 50:
                adjustment += 0
            elif win_rate >= 40:
                adjustment -= 5
            else:
                adjustment -= 10

        # Bill mentioned bonus (+5)
        if bill_mentioned:
            adjustment += 5

        # Pattern recognized bonus (+5)
        if pattern_recognized:
            adjustment += 5

        # Similar trades volume bonus (0 to +5)
        if similar_count >= 5:
            adjustment += 5
        elif similar_count >= 3:
            adjustment += 3
        elif similar_count >= 1:
            adjustment += 1

        # Cap adjustment
        return max(-20, min(20, adjustment))

    def _generate_summary(
        self,
        symbol: str,
        direction: str,
        win_rate: Optional[float],
        similar_count: int,
        bill_mentioned: bool,
        putt_adjustment: float,
    ) -> str:
        """Generate a one-line summary of Putt's assessment."""
        parts = []

        # Confidence direction
        if putt_adjustment > 10:
            parts.append("Strong validation")
        elif putt_adjustment > 0:
            parts.append("Validated")
        elif putt_adjustment < -10:
            parts.append("Caution advised")
        elif putt_adjustment < 0:
            parts.append("Weak validation")
        else:
            parts.append("Neutral")

        # Historical context
        if similar_count > 0:
            if win_rate:
                parts.append(f"{similar_count} similar trades ({win_rate:.0f}% win rate)")
            else:
                parts.append(f"{similar_count} similar trades found")
        else:
            parts.append("No similar trades in database")

        # Bill mention
        if bill_mentioned:
            parts.append("Bill has discussed this symbol")

        return " | ".join(parts)

    def get_quick_validation(self, symbol: str, direction: str) -> dict:
        """
        Quick validation check - lighter weight than full analysis.

        Returns dict with:
        - valid: bool - Should we consider this trade?
        - confidence_boost: float - Adjustment to apply
        - reason: str - Why
        """
        symbol = symbol.upper()

        # Quick search for this symbol
        results = self.rag.query(
            query_text=f"{symbol} {direction}",
            n_results=5,
        )

        if not results:
            return {
                "valid": True,  # No data doesn't mean invalid
                "confidence_boost": 0,
                "reason": "No historical data for this symbol"
            }

        # Check if Bill has traded this direction
        direction_match = False
        for r in results:
            text = r.get("text", "").lower()
            if direction.lower() in text:
                direction_match = True
                break

        if direction_match:
            return {
                "valid": True,
                "confidence_boost": 5,
                "reason": f"Bill has taken {direction} trades on {symbol}"
            }
        else:
            return {
                "valid": True,
                "confidence_boost": 0,
                "reason": f"{symbol} found but no {direction} trades recorded"
            }


def main():
    """Test the Putt Indicator."""
    print("=" * 60)
    print("PUTT INDICATOR TEST")
    print("=" * 60)

    putt = PuttIndicator()

    # Test setups
    test_setups = [
        {"symbol": "NVDA", "direction": "long", "zone_type": "demand", "zone_level": 180.0},
        {"symbol": "TSLA", "direction": "short", "zone_type": "supply", "zone_level": 400.0},
        {"symbol": "AAPL", "direction": "long", "zone_type": "demand", "zone_level": 230.0},
        {"symbol": "SPY", "direction": "long", "zone_type": "demand", "zone_level": 580.0},
    ]

    for setup in test_setups:
        print(f"\n{'='*60}")
        print(f"Analyzing: {setup['symbol']} {setup['direction'].upper()} at {setup['zone_type']} {setup['zone_level']}")
        print("=" * 60)

        context = putt.analyze_setup(
            symbol=setup["symbol"],
            direction=setup["direction"],
            zone_type=setup["zone_type"],
            zone_level=setup["zone_level"],
            base_confidence=60.0,
        )

        print(f"\nBase Confidence:  {context.base_confidence:.0f}")
        print(f"Putt Adjustment:  {context.putt_adjustment:+.0f}")
        print(f"Final Confidence: {context.final_confidence:.0f}")
        print(f"\nWin Rate: {context.win_rate:.0f}%" if context.win_rate else "\nWin Rate: N/A")
        print(f"Similar Trades: {len(context.similar_trades)}")
        print(f"Bill Mentioned: {'Yes' if context.bill_mentioned else 'No'}")
        print(f"Pattern Match: {'Yes' if context.pattern_recognized else 'No'}")

        print(f"\nSummary: {context.summary}")

        if context.key_insights:
            print(f"\nKey Insights:")
            for insight in context.key_insights[:2]:
                print(f"  - {insight[:80]}...")

        if context.zone_history:
            print(f"\nZone History:")
            for zone in context.zone_history[:2]:
                print(f"  - {zone.get('text', '')[:80]}...")


if __name__ == "__main__":
    main()
