"""
Trade Signal Extractor using LLM.

Extracts trading signals and setups from Bill Fanter's
video transcripts for few-shot learning.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import structlog
from anthropic import Anthropic

from trading_agent.core.config import settings

logger = structlog.get_logger()


# Prompt for extracting trade signals from transcripts
EXTRACTION_PROMPT = """You are an expert at analyzing trading education content. Analyze the following transcript from Bill Fanter's trading video and extract any trade signals, setups, or educational examples mentioned.

## TRANSCRIPT
{transcript}

## VIDEO INFO
- Title: {title}
- Published: {published}

## WHAT TO EXTRACT

Look for:
1. **Specific Trade Calls** - Any mentioned entries with ticker, direction, strike, expiration
2. **Zone Levels** - Supply/demand zones with specific price levels
3. **Trade Outcomes** - Results of previous trades (wins/losses with prices)
4. **Educational Setups** - Examples used to teach concepts
5. **Key Levels** - Important support/resistance prices mentioned
6. **Market Context** - Overall market sentiment or trend discussed

## OUTPUT FORMAT
Return a JSON object with the following structure:
```json
{{
    "trade_signals": [
        {{
            "symbol": "AAPL",
            "direction": "long" or "short",
            "entry_type": "call" or "put" or "stock",
            "strike": 185.0,
            "expiration": "12/15",
            "entry_price": 3.50,
            "stop_loss": 2.80,
            "target": 5.50,
            "zone_type": "demand" or "supply",
            "zone_level": 184.50,
            "reasoning": "Fresh demand zone, HTF bullish",
            "confidence": "high" or "medium" or "low",
            "timestamp_hint": "around 5 minutes in"
        }}
    ],
    "zone_levels": [
        {{
            "symbol": "NVDA",
            "zone_type": "supply",
            "zone_high": 485.0,
            "zone_low": 480.0,
            "timeframe": "15m",
            "freshness": "fresh" or "tested",
            "context": "Major resistance from previous week"
        }}
    ],
    "trade_outcomes": [
        {{
            "symbol": "META",
            "direction": "long",
            "entry_price": 4.20,
            "exit_price": 7.80,
            "result": "win",
            "pnl_percent": 85.7,
            "lessons": "Patience at zone paid off"
        }}
    ],
    "key_insights": [
        "Wait for price to come to you at zones",
        "Don't trade against the higher timeframe trend",
        "Fresh zones are stronger than tested zones"
    ],
    "market_context": {{
        "overall_sentiment": "bullish" or "bearish" or "neutral",
        "key_levels_mentioned": ["SPY 580", "QQQ 500"],
        "notable_events": ["FOMC meeting", "earnings season"]
    }},
    "educational_value": "high" or "medium" or "low",
    "summary": "Brief 2-3 sentence summary of the video content"
}}
```

If no trade signals or zones are mentioned, return empty arrays but still extract any insights.
Only include information that is EXPLICITLY mentioned in the transcript - do not infer or guess.
"""


class SignalExtractor:
    """
    Extract trading signals from video transcripts using LLM.

    Uses Claude to analyze transcripts and extract:
    - Trade calls (ticker, direction, prices)
    - Supply/demand zone levels
    - Trade outcomes for few-shot learning
    - Key educational insights
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize signal extractor.

        Args:
            api_key: Anthropic API key
            model: Claude model to use
            output_dir: Directory to save extracted signals
        """
        self.api_key = api_key or settings.anthropic_api_key
        self.model = model or settings.claude_model
        self.client = Anthropic(api_key=self.api_key)
        self.output_dir = output_dir or Path("data/transcriber/signals")

        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def extract_signals(
        self,
        transcript: str,
        video_info: Optional[dict] = None,
        video_id: Optional[str] = None,
    ) -> Optional[dict]:
        """
        Extract trading signals from a transcript.

        Args:
            transcript: Full transcript text
            video_info: Video metadata (title, published date, etc.)
            video_id: Video ID for saving output

        Returns:
            Extracted signals dict
        """
        video_info = video_info or {}

        # Truncate very long transcripts (keep first ~15000 chars)
        if len(transcript) > 15000:
            transcript = transcript[:15000] + "\n\n[TRANSCRIPT TRUNCATED]"

        prompt = EXTRACTION_PROMPT.format(
            transcript=transcript,
            title=video_info.get("title", "Unknown"),
            published=video_info.get("published", "Unknown"),
        )

        logger.info("Extracting signals from transcript", video_id=video_id)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )

            response_text = response.content[0].text

            # Parse JSON from response
            result = self._parse_response(response_text)

            if result:
                # Add metadata
                result["video_id"] = video_id
                result["video_info"] = video_info
                result["extracted_at"] = datetime.now().isoformat()

                # Save to file
                if video_id:
                    output_path = self.output_dir / f"{video_id}_signals.json"
                    with open(output_path, "w") as f:
                        json.dump(result, f, indent=2)
                    logger.info("Signals saved", path=str(output_path))

                # Log summary
                logger.info(
                    "Extraction complete",
                    video_id=video_id,
                    signals=len(result.get("trade_signals", [])),
                    zones=len(result.get("zone_levels", [])),
                    outcomes=len(result.get("trade_outcomes", [])),
                )

            return result

        except Exception as e:
            logger.error("Signal extraction error", video_id=video_id, error=str(e))
            return None

    def _parse_response(self, text: str) -> Optional[dict]:
        """Parse JSON from LLM response."""
        # Try to find JSON block
        json_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)

        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find raw JSON
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                logger.warning("Could not find JSON in response")
                return None

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse JSON", error=str(e))
            # Try to fix common JSON issues
            try:
                # Remove trailing commas before } or ]
                fixed = re.sub(r',\s*([\}\]])', r'\1', json_str)
                # Try again
                return json.loads(fixed)
            except:
                # Save raw response for debugging
                debug_path = self.output_dir / "last_failed_response.txt"
                debug_path.write_text(json_str)
                logger.warning("Saved failed response to", path=str(debug_path))
                return None

    def load_signals(self, video_id: str) -> Optional[dict]:
        """Load previously extracted signals."""
        signal_path = self.output_dir / f"{video_id}_signals.json"

        if not signal_path.exists():
            return None

        try:
            with open(signal_path) as f:
                return json.load(f)
        except Exception as e:
            logger.error("Error loading signals", video_id=video_id, error=str(e))
            return None

    def convert_to_trade_examples(
        self,
        signals: dict,
    ) -> list[dict]:
        """
        Convert extracted signals to trade example format
        for few-shot learning in the trading bot.

        Returns list of examples ready for the database.
        """
        examples = []
        video_info = signals.get("video_info", {})

        # Convert trade outcomes to examples
        for outcome in signals.get("trade_outcomes", []):
            example = {
                "source": "youtube",
                "symbol": outcome.get("symbol"),
                "setup_type": outcome.get("direction"),
                "setup_description": f"{outcome.get('symbol')} {outcome.get('direction')} trade",
                "entry_reasoning": outcome.get("lessons", "From Bill Fanter video"),
                "chart_context": {
                    "video_id": signals.get("video_id"),
                    "video_title": video_info.get("title"),
                },
                "entry_price": outcome.get("entry_price"),
                "exit_price": outcome.get("exit_price"),
                "result": outcome.get("result"),
                "pnl_percent": outcome.get("pnl_percent"),
                "lessons": outcome.get("lessons"),
            }
            examples.append(example)

        # Convert trade signals with enough detail
        for signal in signals.get("trade_signals", []):
            if signal.get("entry_price") and signal.get("reasoning"):
                example = {
                    "source": "youtube",
                    "symbol": signal.get("symbol"),
                    "setup_type": f"{signal.get('zone_type', 'zone')} {signal.get('direction')}",
                    "setup_description": (
                        f"{signal.get('symbol')} {signal.get('direction')} at "
                        f"{signal.get('zone_type', '')} zone {signal.get('zone_level', '')}"
                    ),
                    "entry_reasoning": signal.get("reasoning"),
                    "chart_context": {
                        "video_id": signals.get("video_id"),
                        "zone_type": signal.get("zone_type"),
                        "zone_level": signal.get("zone_level"),
                        "strike": signal.get("strike"),
                        "expiration": signal.get("expiration"),
                    },
                    "entry_price": signal.get("entry_price"),
                    "exit_price": signal.get("target"),  # Target as potential exit
                    "result": None,  # Unknown until trade completes
                    "lessons": f"Confidence: {signal.get('confidence', 'unknown')}",
                }
                examples.append(example)

        return examples

    def get_all_signals(self) -> list[dict]:
        """Load all extracted signals from output directory."""
        all_signals = []

        for signal_file in self.output_dir.glob("*_signals.json"):
            try:
                with open(signal_file) as f:
                    signals = json.load(f)
                    all_signals.append(signals)
            except Exception as e:
                logger.warning(f"Error loading {signal_file}: {e}")

        return all_signals

    def get_all_trade_examples(self) -> list[dict]:
        """Get all trade examples from all extracted signals."""
        all_examples = []

        for signals in self.get_all_signals():
            examples = self.convert_to_trade_examples(signals)
            all_examples.extend(examples)

        return all_examples
