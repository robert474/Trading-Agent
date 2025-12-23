"""
Discord Trade Parser.

Parser for Bill Fanter's Discord trade alerts.
Use this to extract trade setups for few-shot learning.

NOTE: Only use with your own subscription data.
"""

import re
from datetime import datetime
from typing import Optional

import structlog

logger = structlog.get_logger()


class DiscordTradeParser:
    """
    Parse trade alerts from Discord messages.

    Example formats from Bill Fanter's Discord:
    - "AAPL 12/15 $185c @ $3.50 - demand zone at 184.50"
    - "NVDA puts $480p 12/20 entry $4.20 stop $3.00 target $8.00"
    - "SPY call 580c 12/18 entry 2.50"
    """

    def __init__(self):
        self.patterns = {
            "symbol": r"^([A-Z]{1,5})\b",
            "option_contract": r"(\d{1,2}/\d{1,2})\s*\$?(\d+(?:\.\d+)?)(c|p)",
            "option_alt": r"\$?(\d+(?:\.\d+)?)(c|p)\s*(\d{1,2}/\d{1,2})",
            "entry": r"(?:@|entry|at)\s*\$?(\d+(?:\.\d+)?)",
            "stop": r"(?:stop|sl)\s*\$?(\d+(?:\.\d+)?)",
            "target": r"(?:target|tp|pt)\s*\$?(\d+(?:\.\d+)?)",
            "zone": r"(supply|demand)\s*zone\s*(?:at)?\s*\$?(\d+(?:\.\d+)?)",
            "price_only": r"\$(\d+(?:\.\d+)?)",
        }

    def parse_alert(self, message: str) -> Optional[dict]:
        """
        Parse a Discord trade alert message.

        Args:
            message: Raw Discord message text

        Returns:
            Dict with parsed trade details or None if unparseable
        """
        original = message
        message = message.lower().strip()

        result = {
            "raw_message": original,
            "parsed_at": datetime.now(),
        }

        # Extract symbol (should be at the beginning)
        symbol_match = re.search(self.patterns["symbol"], original.upper())
        if symbol_match:
            result["symbol"] = symbol_match.group(1)
        else:
            # Try to find any uppercase word that could be a symbol
            words = original.upper().split()
            for word in words:
                if re.match(r"^[A-Z]{1,5}$", word):
                    result["symbol"] = word
                    break

        if "symbol" not in result:
            return None

        # Extract option details (try multiple formats)
        option_match = re.search(self.patterns["option_contract"], message)
        if option_match:
            result["expiration"] = option_match.group(1)
            result["strike"] = float(option_match.group(2))
            result["option_type"] = "call" if option_match.group(3) == "c" else "put"
        else:
            # Try alternate format
            option_alt = re.search(self.patterns["option_alt"], message)
            if option_alt:
                result["strike"] = float(option_alt.group(1))
                result["option_type"] = "call" if option_alt.group(2) == "c" else "put"
                result["expiration"] = option_alt.group(3)

        # Check for calls/puts keywords
        if "option_type" not in result:
            if "call" in message or "calls" in message:
                result["option_type"] = "call"
            elif "put" in message or "puts" in message:
                result["option_type"] = "put"

        # Extract prices
        entry_match = re.search(self.patterns["entry"], message)
        if entry_match:
            result["entry_price"] = float(entry_match.group(1))

        stop_match = re.search(self.patterns["stop"], message)
        if stop_match:
            result["stop_loss"] = float(stop_match.group(1))

        target_match = re.search(self.patterns["target"], message)
        if target_match:
            result["target"] = float(target_match.group(1))

        # Extract zone info
        zone_match = re.search(self.patterns["zone"], message)
        if zone_match:
            result["zone_type"] = zone_match.group(1)
            result["zone_level"] = float(zone_match.group(2))

        # If no entry price found, look for @ symbol or first price
        if "entry_price" not in result:
            at_match = re.search(r"@\s*\$?(\d+(?:\.\d+)?)", message)
            if at_match:
                result["entry_price"] = float(at_match.group(1))

        # Determine direction
        if result.get("option_type") == "call":
            result["direction"] = "long"
        elif result.get("option_type") == "put":
            result["direction"] = "short"
        elif result.get("zone_type") == "demand":
            result["direction"] = "long"
        elif result.get("zone_type") == "supply":
            result["direction"] = "short"

        logger.debug("Parsed Discord alert", result=result)

        return result

    def parse_outcome(
        self,
        entry_message: dict,
        exit_message: str,
    ) -> Optional[dict]:
        """
        Parse an exit/outcome message and combine with entry.

        Args:
            entry_message: Previously parsed entry message
            exit_message: Exit message text

        Returns:
            Combined trade with outcome
        """
        exit_lower = exit_message.lower()

        result = entry_message.copy()
        result["exit_message"] = exit_message

        # Look for exit price
        exit_patterns = [
            r"(?:exit|sold|out|closed)\s*(?:at)?\s*\$?(\d+(?:\.\d+)?)",
            r"\$(\d+(?:\.\d+)?)\s*(?:profit|gain|loss)",
        ]

        for pattern in exit_patterns:
            match = re.search(pattern, exit_lower)
            if match:
                result["exit_price"] = float(match.group(1))
                break

        # Determine result
        if "win" in exit_lower or "profit" in exit_lower or "gain" in exit_lower:
            result["result"] = "win"
        elif "loss" in exit_lower or "stop" in exit_lower or "stopped" in exit_lower:
            result["result"] = "loss"

        # Calculate P&L if we have prices
        if "exit_price" in result and "entry_price" in result:
            entry = result["entry_price"]
            exit_price = result["exit_price"]

            if result.get("direction") == "long":
                result["pnl_percent"] = (exit_price - entry) / entry * 100
            else:
                result["pnl_percent"] = (entry - exit_price) / entry * 100

            if result["pnl_percent"] > 0:
                result["result"] = "win"
            else:
                result["result"] = "loss"

        return result

    def batch_parse(self, messages: list[str]) -> list[dict]:
        """
        Parse multiple Discord messages.

        Args:
            messages: List of message texts

        Returns:
            List of successfully parsed messages
        """
        results = []
        for msg in messages:
            parsed = self.parse_alert(msg)
            if parsed and "symbol" in parsed:
                results.append(parsed)

        logger.info("Batch parsed messages", total=len(messages), parsed=len(results))
        return results
