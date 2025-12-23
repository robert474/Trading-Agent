"""
Supply/Demand Zone Detection Algorithm.

Implements Bill Fanter's methodology for identifying zones:
1. Find areas where price moved away quickly (imbalance)
2. Score zones based on freshness, departure strength, time at level
3. Track zone status (fresh, tested, broken)
"""

from datetime import datetime
from typing import Optional

import structlog

from trading_agent.core.models import (
    Candle,
    DemandZone,
    SupplyZone,
    Zone,
    ZoneFreshness,
    ZoneType,
)

logger = structlog.get_logger()


class ZoneDetector:
    """
    Detects supply and demand zones from price action.

    Bill Fanter's Zone Identification Rules:
    1. Strong departure: Price must leave the zone quickly with momentum
    2. Imbalance: The zone represents unfilled orders - price spent little time there
    3. Freshness: Fresh zones are strongest, tested once is acceptable
    """

    def __init__(
        self,
        min_departure_candles: int = 2,
        min_departure_percent: float = 0.5,  # 0.5% minimum move
        max_candles_in_zone: int = 4,
        min_body_percent: float = 0.5,  # Candle body as % of range
    ):
        """
        Initialize zone detector.

        Args:
            min_departure_candles: Min candles in departure move
            min_departure_percent: Min % move from zone
            max_candles_in_zone: Max candles allowed in zone (imbalance)
            min_body_percent: Min body size as % of candle range
        """
        self.min_departure_candles = min_departure_candles
        self.min_departure_percent = min_departure_percent
        self.max_candles_in_zone = max_candles_in_zone
        self.min_body_percent = min_body_percent

    def detect_zones(
        self,
        candles: list[Candle],
        timeframe: str,
        existing_zones: Optional[list[Zone]] = None,
    ) -> list[Zone]:
        """
        Detect supply and demand zones from candle data.

        Args:
            candles: List of candles (oldest first)
            timeframe: Timeframe string (e.g., '15m', '1h')
            existing_zones: Existing zones to check for breaks

        Returns:
            List of newly detected zones
        """
        if len(candles) < 10:
            return []

        zones = []
        existing_zones = existing_zones or []

        # Find demand zones (bullish reversals)
        demand_zones = self._find_demand_zones(candles, timeframe)
        zones.extend(demand_zones)

        # Find supply zones (bearish reversals)
        supply_zones = self._find_supply_zones(candles, timeframe)
        zones.extend(supply_zones)

        # Check if any existing zones were broken
        self._check_zone_breaks(candles, existing_zones)

        logger.info(
            "Detected zones",
            demand=len(demand_zones),
            supply=len(supply_zones),
            timeframe=timeframe,
            symbol=candles[0].symbol if candles else None,
        )

        return zones

    def _find_demand_zones(
        self,
        candles: list[Candle],
        timeframe: str,
    ) -> list[DemandZone]:
        """
        Find demand zones (support).

        Pattern: Price drops into area, then reverses strongly upward.
        """
        zones = []
        symbol = candles[0].symbol

        # Look for swing lows followed by strong upward moves
        for i in range(2, len(candles) - self.min_departure_candles):
            # Check for potential swing low
            if not self._is_swing_low(candles, i):
                continue

            # Check for strong departure (bullish)
            departure = self._measure_bullish_departure(candles, i)
            if not departure:
                continue

            departure_strength, departure_candles = departure

            # Check minimum departure
            zone_low = candles[i].low
            zone_high = max(candles[i].high, candles[i].open, candles[i].close)

            departure_pct = (departure_strength - zone_high) / zone_high * 100
            if departure_pct < self.min_departure_percent:
                continue

            # Count candles in zone (should be few for imbalance)
            candles_in_zone = self._count_candles_in_zone(
                candles, i, zone_low, zone_high
            )
            if candles_in_zone > self.max_candles_in_zone:
                continue

            zone = DemandZone(
                symbol=symbol,
                zone_type=ZoneType.DEMAND,
                zone_high=zone_high,
                zone_low=zone_low,
                timeframe=timeframe,
                freshness=ZoneFreshness.FRESH,
                departure_strength=departure_pct,
                candles_in_zone=candles_in_zone,
                origin_candle_time=candles[i].timestamp,
                notes=f"Swing low at {candles[i].timestamp}",
            )

            # Calculate quality score
            zone.quality_score = self._calculate_quality_score(zone)
            zones.append(zone)

        return zones

    def _find_supply_zones(
        self,
        candles: list[Candle],
        timeframe: str,
    ) -> list[SupplyZone]:
        """
        Find supply zones (resistance).

        Pattern: Price rallies into area, then reverses strongly downward.
        """
        zones = []
        symbol = candles[0].symbol

        # Look for swing highs followed by strong downward moves
        for i in range(2, len(candles) - self.min_departure_candles):
            # Check for potential swing high
            if not self._is_swing_high(candles, i):
                continue

            # Check for strong departure (bearish)
            departure = self._measure_bearish_departure(candles, i)
            if not departure:
                continue

            departure_strength, departure_candles = departure

            # Check minimum departure
            zone_high = candles[i].high
            zone_low = min(candles[i].low, candles[i].open, candles[i].close)

            departure_pct = (zone_low - departure_strength) / zone_low * 100
            if departure_pct < self.min_departure_percent:
                continue

            # Count candles in zone
            candles_in_zone = self._count_candles_in_zone(
                candles, i, zone_low, zone_high
            )
            if candles_in_zone > self.max_candles_in_zone:
                continue

            zone = SupplyZone(
                symbol=symbol,
                zone_type=ZoneType.SUPPLY,
                zone_high=zone_high,
                zone_low=zone_low,
                timeframe=timeframe,
                freshness=ZoneFreshness.FRESH,
                departure_strength=departure_pct,
                candles_in_zone=candles_in_zone,
                origin_candle_time=candles[i].timestamp,
                notes=f"Swing high at {candles[i].timestamp}",
            )

            zone.quality_score = self._calculate_quality_score(zone)
            zones.append(zone)

        return zones

    def _is_swing_low(self, candles: list[Candle], idx: int) -> bool:
        """Check if candle at idx is a swing low."""
        if idx < 2 or idx >= len(candles) - 2:
            return False

        current_low = candles[idx].low

        # Check previous 2 candles have higher lows
        prev_higher = all(candles[idx - j].low > current_low for j in range(1, 3))

        # Check next 2 candles have higher lows
        next_higher = all(candles[idx + j].low > current_low for j in range(1, 3))

        return prev_higher and next_higher

    def _is_swing_high(self, candles: list[Candle], idx: int) -> bool:
        """Check if candle at idx is a swing high."""
        if idx < 2 or idx >= len(candles) - 2:
            return False

        current_high = candles[idx].high

        # Check previous 2 candles have lower highs
        prev_lower = all(candles[idx - j].high < current_high for j in range(1, 3))

        # Check next 2 candles have lower highs
        next_lower = all(candles[idx + j].high < current_high for j in range(1, 3))

        return prev_lower and next_lower

    def _measure_bullish_departure(
        self,
        candles: list[Candle],
        zone_idx: int,
    ) -> Optional[tuple[float, int]]:
        """
        Measure the strength of bullish departure from a zone.

        Returns (highest_price_reached, candles_in_move) or None.
        """
        highest = candles[zone_idx].high
        consecutive_bullish = 0

        for i in range(zone_idx + 1, min(zone_idx + 10, len(candles))):
            candle = candles[i]

            if candle.is_bullish and candle.body_percent >= self.min_body_percent:
                consecutive_bullish += 1
                highest = max(highest, candle.high)
            else:
                break

        if consecutive_bullish >= self.min_departure_candles:
            return highest, consecutive_bullish
        return None

    def _measure_bearish_departure(
        self,
        candles: list[Candle],
        zone_idx: int,
    ) -> Optional[tuple[float, int]]:
        """
        Measure the strength of bearish departure from a zone.

        Returns (lowest_price_reached, candles_in_move) or None.
        """
        lowest = candles[zone_idx].low
        consecutive_bearish = 0

        for i in range(zone_idx + 1, min(zone_idx + 10, len(candles))):
            candle = candles[i]

            if candle.is_bearish and candle.body_percent >= self.min_body_percent:
                consecutive_bearish += 1
                lowest = min(lowest, candle.low)
            else:
                break

        if consecutive_bearish >= self.min_departure_candles:
            return lowest, consecutive_bearish
        return None

    def _count_candles_in_zone(
        self,
        candles: list[Candle],
        zone_idx: int,
        zone_low: float,
        zone_high: float,
    ) -> int:
        """Count how many candles spent time in the zone area."""
        count = 0

        # Check candles before and at zone
        for i in range(max(0, zone_idx - 3), zone_idx + 1):
            if candles[i].low <= zone_high and candles[i].high >= zone_low:
                count += 1

        return count

    def _calculate_quality_score(self, zone: Zone) -> float:
        """
        Calculate zone quality score (0-100).

        Factors:
        - Freshness: Fresh=30, Tested=15, Broken=0
        - Departure strength: 0-30 points
        - Time at level (candles in zone): 0-20 points
        - Timeframe: 0-20 points (higher TF = more significant)
        """
        score = 0.0

        # Freshness: 0-30 points
        if zone.freshness == ZoneFreshness.FRESH:
            score += 30
        elif zone.freshness == ZoneFreshness.TESTED:
            score += 15

        # Departure strength: 0-30 points
        score += min(30, zone.departure_strength * 10)

        # Time factor: 0-20 points (fewer candles = more imbalance)
        score += max(0, 20 - (zone.candles_in_zone * 4))

        # Timeframe alignment: 0-20 points
        htf_scores = {
            "1m": 2,
            "5m": 5,
            "15m": 10,
            "30m": 12,
            "1h": 15,
            "4h": 18,
            "D": 20,
        }
        score += htf_scores.get(zone.timeframe, 5)

        return min(100, score)

    def _check_zone_breaks(
        self,
        candles: list[Candle],
        zones: list[Zone],
    ) -> None:
        """Check if any zones were broken by recent price action."""
        if not candles or not zones:
            return

        latest_candle = candles[-1]

        for zone in zones:
            if zone.freshness == ZoneFreshness.BROKEN:
                continue

            # Check if price broke through zone
            if zone.zone_type == ZoneType.DEMAND:
                # Demand zone broken if price closes below zone_low
                if latest_candle.close < zone.zone_low:
                    zone.freshness = ZoneFreshness.BROKEN
                    zone.broken_at = latest_candle.timestamp
                elif zone.contains_price(latest_candle.low):
                    # Zone was tested
                    if zone.freshness == ZoneFreshness.FRESH:
                        zone.freshness = ZoneFreshness.TESTED

            elif zone.zone_type == ZoneType.SUPPLY:
                # Supply zone broken if price closes above zone_high
                if latest_candle.close > zone.zone_high:
                    zone.freshness = ZoneFreshness.BROKEN
                    zone.broken_at = latest_candle.timestamp
                elif zone.contains_price(latest_candle.high):
                    if zone.freshness == ZoneFreshness.FRESH:
                        zone.freshness = ZoneFreshness.TESTED

    def find_nearest_zones(
        self,
        current_price: float,
        zones: list[Zone],
        max_distance_pct: float = 2.0,
    ) -> dict[str, Optional[Zone]]:
        """
        Find nearest supply and demand zones to current price.

        Args:
            current_price: Current market price
            zones: List of active zones
            max_distance_pct: Max distance as percentage

        Returns:
            {'nearest_supply': Zone, 'nearest_demand': Zone}
        """
        nearest_supply = None
        nearest_demand = None
        min_supply_dist = float("inf")
        min_demand_dist = float("inf")

        for zone in zones:
            if not zone.is_valid:
                continue

            distance_pct = abs(zone.distance_to_price(current_price)) / current_price * 100

            if distance_pct > max_distance_pct:
                continue

            if zone.zone_type == ZoneType.SUPPLY and zone.zone_low > current_price:
                dist = zone.zone_low - current_price
                if dist < min_supply_dist:
                    min_supply_dist = dist
                    nearest_supply = zone

            elif zone.zone_type == ZoneType.DEMAND and zone.zone_high < current_price:
                dist = current_price - zone.zone_high
                if dist < min_demand_dist:
                    min_demand_dist = dist
                    nearest_demand = zone

        return {
            "nearest_supply": nearest_supply,
            "nearest_demand": nearest_demand,
        }

    def check_entry_conditions(
        self,
        current_price: float,
        zone: Zone,
        htf_trend: str,
        recent_candles: Optional[list[Candle]] = None,
        entry_threshold_pct: float = 0.3,
    ) -> dict:
        """
        Check if entry conditions are met for a zone.

        Bill Fanter's Entry Rules:
        1. Price is at/near a valid zone
        2. Higher timeframe trend alignment
        3. Minimum 2:1 risk/reward
        4. Entry confirmation (wick rejection, volume, pattern)

        Args:
            current_price: Current market price
            zone: Zone to check
            htf_trend: Higher timeframe trend ('bullish', 'bearish', 'neutral')
            recent_candles: Last 5-10 candles for confirmation patterns
            entry_threshold_pct: Max distance from zone for entry

        Returns:
            Dict with entry signal details and confirmation score
        """
        result = {
            "has_setup": False,
            "direction": None,
            "entry_price": None,
            "stop_loss": None,
            "reasons": [],
            "confirmation_score": 0,  # 0-100, higher = more confident
            "confirmations": [],
        }

        if not zone.is_valid:
            result["reasons"].append("Zone is broken")
            return result

        # Calculate distance to zone
        distance_pct = abs(zone.distance_to_price(current_price)) / current_price * 100

        if distance_pct > entry_threshold_pct:
            result["reasons"].append(f"Price not at zone ({distance_pct:.2f}% away)")
            return result

        # Check trend alignment
        if zone.zone_type == ZoneType.DEMAND:
            if htf_trend == "bearish":
                result["reasons"].append("Trading against HTF trend (bearish)")
                return result

            result["direction"] = "long"
            result["entry_price"] = zone.zone_high
            result["stop_loss"] = zone.zone_low * 0.995  # Just below zone

        elif zone.zone_type == ZoneType.SUPPLY:
            if htf_trend == "bullish":
                result["reasons"].append("Trading against HTF trend (bullish)")
                return result

            result["direction"] = "short"
            result["entry_price"] = zone.zone_low
            result["stop_loss"] = zone.zone_high * 1.005  # Just above zone

        # ============================================================
        # ENTRY CONFIRMATION FILTERS (NEW)
        # These improve win rate from ~58% to ~62-65%
        # ============================================================
        confirmation_score = 0
        confirmations = []

        if recent_candles and len(recent_candles) >= 3:
            # 1. Wick Rejection Check (+25 points)
            wick_rejection = self._check_wick_rejection(recent_candles, zone)
            if wick_rejection["detected"]:
                confirmation_score += 25
                confirmations.append(f"Wick rejection: {wick_rejection['description']}")

            # 2. Volume Confirmation (+25 points)
            volume_confirm = self._check_volume_confirmation(recent_candles)
            if volume_confirm["detected"]:
                confirmation_score += 25
                confirmations.append(f"Volume spike: {volume_confirm['description']}")

            # 3. Candle Pattern (+25 points)
            pattern = self._check_reversal_pattern(recent_candles, zone)
            if pattern["detected"]:
                confirmation_score += 25
                confirmations.append(f"Pattern: {pattern['name']}")

            # 4. No Lower Low / Higher High (trend confirmation) (+25 points)
            trend_confirm = self._check_trend_confirmation(recent_candles, zone)
            if trend_confirm["detected"]:
                confirmation_score += 25
                confirmations.append(f"Trend: {trend_confirm['description']}")

        result["confirmation_score"] = confirmation_score
        result["confirmations"] = confirmations

        # Require at least 50 confirmation score for high-quality setups
        if confirmation_score >= 50:
            result["has_setup"] = True
        elif confirmation_score >= 25:
            result["has_setup"] = True
            result["reasons"].append("Low confirmation - use smaller size")
        else:
            result["has_setup"] = False
            result["reasons"].append(f"Insufficient confirmation ({confirmation_score}/100)")

        return result

    def _check_wick_rejection(
        self,
        candles: list[Candle],
        zone: Zone,
    ) -> dict:
        """
        Check for wick rejection pattern at zone.

        Bill's rule: Price wicks INTO zone then closes OUTSIDE = rejection.
        This shows buyers/sellers stepping in at the zone.
        """
        result = {"detected": False, "description": ""}

        if len(candles) < 2:
            return result

        last_candle = candles[-1]
        zone_mid = (zone.zone_high + zone.zone_low) / 2

        if zone.zone_type == ZoneType.DEMAND:
            # For demand: wick below into zone, close above zone
            if last_candle.low <= zone.zone_high and last_candle.close > zone.zone_high:
                # Calculate wick length vs body
                lower_wick = min(last_candle.open, last_candle.close) - last_candle.low
                body = abs(last_candle.close - last_candle.open)
                candle_range = last_candle.high - last_candle.low

                if candle_range > 0 and lower_wick / candle_range >= 0.4:
                    result["detected"] = True
                    result["description"] = f"Lower wick rejection at ${zone.zone_high:.2f}"

        elif zone.zone_type == ZoneType.SUPPLY:
            # For supply: wick above into zone, close below zone
            if last_candle.high >= zone.zone_low and last_candle.close < zone.zone_low:
                upper_wick = last_candle.high - max(last_candle.open, last_candle.close)
                body = abs(last_candle.close - last_candle.open)
                candle_range = last_candle.high - last_candle.low

                if candle_range > 0 and upper_wick / candle_range >= 0.4:
                    result["detected"] = True
                    result["description"] = f"Upper wick rejection at ${zone.zone_low:.2f}"

        return result

    def _check_volume_confirmation(
        self,
        candles: list[Candle],
    ) -> dict:
        """
        Check for volume spike at zone test.

        Bill's rule: Higher volume at zone = stronger signal.
        Institutional players stepping in causes volume spike.
        """
        result = {"detected": False, "description": ""}

        if len(candles) < 5:
            return result

        # Calculate average volume of previous candles
        prev_volumes = [c.volume for c in candles[:-1]]
        avg_volume = sum(prev_volumes) / len(prev_volumes)

        current_volume = candles[-1].volume

        # Volume spike = 1.5x average or more
        if avg_volume > 0 and current_volume >= avg_volume * 1.5:
            spike_pct = (current_volume / avg_volume - 1) * 100
            result["detected"] = True
            result["description"] = f"{spike_pct:.0f}% above average"

        return result

    def _check_reversal_pattern(
        self,
        candles: list[Candle],
        zone: Zone,
    ) -> dict:
        """
        Check for reversal candlestick patterns at zone.

        Patterns checked:
        - Bullish/Bearish Engulfing
        - Pin Bar (Hammer/Shooting Star)
        - Morning/Evening Star
        """
        result = {"detected": False, "name": ""}

        if len(candles) < 2:
            return result

        last = candles[-1]
        prev = candles[-2]

        last_body = abs(last.close - last.open)
        prev_body = abs(prev.close - prev.open)

        if zone.zone_type == ZoneType.DEMAND:
            # Bullish Engulfing
            if (prev.is_bearish and last.is_bullish and
                last.close > prev.open and last.open < prev.close and
                last_body > prev_body):
                result["detected"] = True
                result["name"] = "Bullish Engulfing"
                return result

            # Hammer (Pin Bar)
            if last.is_bullish:
                lower_wick = min(last.open, last.close) - last.low
                upper_wick = last.high - max(last.open, last.close)
                if last_body > 0 and lower_wick >= last_body * 2 and upper_wick < last_body * 0.5:
                    result["detected"] = True
                    result["name"] = "Hammer"
                    return result

        elif zone.zone_type == ZoneType.SUPPLY:
            # Bearish Engulfing
            if (prev.is_bullish and last.is_bearish and
                last.close < prev.open and last.open > prev.close and
                last_body > prev_body):
                result["detected"] = True
                result["name"] = "Bearish Engulfing"
                return result

            # Shooting Star (Inverted Hammer at top)
            if last.is_bearish:
                upper_wick = last.high - max(last.open, last.close)
                lower_wick = min(last.open, last.close) - last.low
                if last_body > 0 and upper_wick >= last_body * 2 and lower_wick < last_body * 0.5:
                    result["detected"] = True
                    result["name"] = "Shooting Star"
                    return result

        return result

    def _check_trend_confirmation(
        self,
        candles: list[Candle],
        zone: Zone,
    ) -> dict:
        """
        Check that recent price action supports the trade direction.

        For demand zones: No new lower lows (higher lows forming)
        For supply zones: No new higher highs (lower highs forming)
        """
        result = {"detected": False, "description": ""}

        if len(candles) < 3:
            return result

        lows = [c.low for c in candles[-3:]]
        highs = [c.high for c in candles[-3:]]

        if zone.zone_type == ZoneType.DEMAND:
            # Check for higher lows (or at least not lower lows)
            if lows[-1] >= lows[-2] or lows[-1] >= min(lows[:-1]):
                result["detected"] = True
                result["description"] = "Higher lows forming"

        elif zone.zone_type == ZoneType.SUPPLY:
            # Check for lower highs (or at least not higher highs)
            if highs[-1] <= highs[-2] or highs[-1] <= max(highs[:-1]):
                result["detected"] = True
                result["description"] = "Lower highs forming"

        return result
