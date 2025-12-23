"""
Utility helper functions.
"""

from datetime import datetime, time, timedelta
from typing import Optional
import pytz

from trading_agent.core.config import settings


def get_market_timezone():
    """Get the market timezone (default: America/New_York)."""
    return pytz.timezone(settings.timezone)


def is_market_open(check_time: Optional[datetime] = None) -> bool:
    """
    Check if the US stock market is currently open.

    Args:
        check_time: Time to check (default: now)

    Returns:
        True if market is open
    """
    tz = get_market_timezone()

    if check_time is None:
        check_time = datetime.now(tz)
    elif check_time.tzinfo is None:
        check_time = tz.localize(check_time)

    # Check if weekend
    if check_time.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return False

    # Parse market hours from settings
    market_open = time(
        int(settings.trading_hours.start.split(":")[0]),
        int(settings.trading_hours.start.split(":")[1]),
    )
    market_close = time(
        int(settings.trading_hours.end.split(":")[0]),
        int(settings.trading_hours.end.split(":")[1]),
    )

    current_time = check_time.time()

    return market_open <= current_time <= market_close


def get_next_market_open(from_time: Optional[datetime] = None) -> datetime:
    """
    Get the next market open time.

    Args:
        from_time: Starting time (default: now)

    Returns:
        Next market open datetime
    """
    tz = get_market_timezone()

    if from_time is None:
        from_time = datetime.now(tz)
    elif from_time.tzinfo is None:
        from_time = tz.localize(from_time)

    market_open_hour = int(settings.trading_hours.start.split(":")[0])
    market_open_minute = int(settings.trading_hours.start.split(":")[1])

    # Start with today's market open
    next_open = from_time.replace(
        hour=market_open_hour,
        minute=market_open_minute,
        second=0,
        microsecond=0,
    )

    # If we're past today's open, move to tomorrow
    if next_open <= from_time:
        next_open += timedelta(days=1)

    # Skip weekends
    while next_open.weekday() >= 5:
        next_open += timedelta(days=1)

    return next_open


def is_premarket(check_time: Optional[datetime] = None) -> bool:
    """Check if we're in pre-market hours (4:00 AM - 9:30 AM ET)."""
    tz = get_market_timezone()

    if check_time is None:
        check_time = datetime.now(tz)
    elif check_time.tzinfo is None:
        check_time = tz.localize(check_time)

    if check_time.weekday() >= 5:
        return False

    premarket_start = time(4, 0)
    market_open = time(9, 30)

    current_time = check_time.time()

    return premarket_start <= current_time < market_open


def is_after_hours(check_time: Optional[datetime] = None) -> bool:
    """Check if we're in after-hours trading (4:00 PM - 8:00 PM ET)."""
    tz = get_market_timezone()

    if check_time is None:
        check_time = datetime.now(tz)
    elif check_time.tzinfo is None:
        check_time = tz.localize(check_time)

    if check_time.weekday() >= 5:
        return False

    market_close = time(16, 0)
    after_hours_end = time(20, 0)

    current_time = check_time.time()

    return market_close < current_time <= after_hours_end


def format_currency(amount: float, include_sign: bool = False) -> str:
    """Format a number as currency."""
    if include_sign:
        return f"${amount:+,.2f}"
    return f"${amount:,.2f}"


def format_percent(value: float, include_sign: bool = True) -> str:
    """Format a number as percentage."""
    if include_sign:
        return f"{value:+.2f}%"
    return f"{value:.2f}%"


def calculate_risk_reward(
    entry: float,
    stop: float,
    target: float,
    direction: str = "long",
) -> float:
    """
    Calculate risk/reward ratio.

    Args:
        entry: Entry price
        stop: Stop loss price
        target: Target price
        direction: 'long' or 'short'

    Returns:
        Risk/reward ratio
    """
    if direction == "long":
        risk = entry - stop
        reward = target - entry
    else:
        risk = stop - entry
        reward = entry - target

    if risk <= 0:
        return 0.0

    return reward / risk


def occ_to_details(occ_symbol: str) -> dict:
    """
    Parse an OCC options symbol into components.

    OCC format: SYMBOL + YYMMDD + C/P + STRIKE (8 digits, 3 decimal places implied)
    Example: AAPL231215C00185000 = AAPL $185 Call expiring 12/15/2023

    Returns:
        Dict with underlying, expiration, type, strike
    """
    if len(occ_symbol) < 15:
        return {}

    # Find where the date starts (6 digits after symbol)
    # Symbol can be 1-6 characters
    for i in range(1, 7):
        potential_date = occ_symbol[i:i+6]
        if potential_date.isdigit():
            underlying = occ_symbol[:i]
            date_str = potential_date
            option_type = occ_symbol[i+6]
            strike_str = occ_symbol[i+7:]
            break
    else:
        return {}

    try:
        expiration = datetime.strptime(date_str, "%y%m%d").strftime("%Y-%m-%d")
        strike = float(strike_str) / 1000

        return {
            "underlying": underlying,
            "expiration": expiration,
            "option_type": "call" if option_type.upper() == "C" else "put",
            "strike": strike,
        }
    except (ValueError, IndexError):
        return {}


def details_to_occ(
    underlying: str,
    expiration: str,
    option_type: str,
    strike: float,
) -> str:
    """
    Create an OCC options symbol from components.

    Args:
        underlying: Stock symbol
        expiration: Expiration date (YYYY-MM-DD)
        option_type: 'call' or 'put'
        strike: Strike price

    Returns:
        OCC format symbol
    """
    # Parse expiration
    exp_date = datetime.strptime(expiration, "%Y-%m-%d")
    date_str = exp_date.strftime("%y%m%d")

    # Option type
    type_char = "C" if option_type.lower() == "call" else "P"

    # Strike (multiply by 1000, pad to 8 digits)
    strike_str = f"{int(strike * 1000):08d}"

    return f"{underlying.upper()}{date_str}{type_char}{strike_str}"
