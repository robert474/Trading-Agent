"""
Audio Transcription using OpenAI Whisper.

Supports both local Whisper and OpenAI's API.
"""

import asyncio
import json
from pathlib import Path
from typing import Optional

import structlog

logger = structlog.get_logger()


class Transcriber:
    """
    Transcribe audio files using OpenAI Whisper.

    Supports:
    - Local Whisper model (free, requires GPU for speed)
    - OpenAI Whisper API (paid, faster)

    Install local Whisper:
        pip install openai-whisper

    For faster local transcription:
        pip install whisper-faster
    """

    def __init__(
        self,
        use_api: bool = False,
        api_key: Optional[str] = None,
        model: str = "base",  # tiny, base, small, medium, large
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize transcriber.

        Args:
            use_api: Use OpenAI API instead of local model
            api_key: OpenAI API key (required if use_api=True)
            model: Whisper model size (for local)
            output_dir: Directory to save transcripts
        """
        self.use_api = use_api
        self.api_key = api_key
        self.model_name = model
        self.output_dir = output_dir or Path("data/transcriber/transcripts")

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Local model (loaded lazily)
        self._model = None

    def _load_local_model(self):
        """Load local Whisper model."""
        if self._model is not None:
            return

        try:
            import whisper
            logger.info("Loading Whisper model", model=self.model_name)
            self._model = whisper.load_model(self.model_name)
            logger.info("Whisper model loaded")
        except ImportError:
            logger.error(
                "Whisper not installed. Install with: pip install openai-whisper"
            )
            raise

    async def transcribe(
        self,
        audio_path: Path,
        video_id: Optional[str] = None,
        language: str = "en",
    ) -> Optional[dict]:
        """
        Transcribe an audio file.

        Args:
            audio_path: Path to audio file
            video_id: Optional ID for output filename
            language: Language code

        Returns:
            Dict with text, segments, and metadata
        """
        if not audio_path.exists():
            logger.error("Audio file not found", path=str(audio_path))
            return None

        logger.info("Transcribing audio", path=str(audio_path))

        if self.use_api:
            result = await self._transcribe_api(audio_path, language)
        else:
            result = await self._transcribe_local(audio_path, language)

        if result:
            # Save transcript
            output_name = video_id or audio_path.stem
            transcript_path = self.output_dir / f"{output_name}.json"

            with open(transcript_path, "w") as f:
                json.dump(result, f, indent=2, default=str)

            # Also save plain text version
            text_path = self.output_dir / f"{output_name}.txt"
            with open(text_path, "w") as f:
                f.write(result.get("text", ""))

            logger.info(
                "Transcript saved",
                json_path=str(transcript_path),
                text_path=str(text_path),
            )

        return result

    async def _transcribe_local(
        self,
        audio_path: Path,
        language: str,
    ) -> Optional[dict]:
        """Transcribe using local Whisper model."""
        try:
            # Load model if needed
            self._load_local_model()

            # Run transcription in thread pool (CPU intensive)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self._model.transcribe(
                    str(audio_path),
                    language=language,
                    verbose=False,
                ),
            )

            return {
                "text": result["text"],
                "segments": result.get("segments", []),
                "language": result.get("language"),
                "audio_path": str(audio_path),
            }

        except Exception as e:
            logger.error("Local transcription error", error=str(e))
            return None

    async def _transcribe_api(
        self,
        audio_path: Path,
        language: str,
    ) -> Optional[dict]:
        """Transcribe using OpenAI Whisper API."""
        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.api_key)

            with open(audio_path, "rb") as audio_file:
                # Run in executor since it's blocking
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        language=language,
                        response_format="verbose_json",
                    ),
                )

            return {
                "text": result.text,
                "segments": getattr(result, "segments", []),
                "language": language,
                "audio_path": str(audio_path),
            }

        except ImportError:
            logger.error("OpenAI not installed. Install with: pip install openai")
            return None

        except Exception as e:
            logger.error("API transcription error", error=str(e))
            return None

    async def transcribe_with_timestamps(
        self,
        audio_path: Path,
        video_id: Optional[str] = None,
    ) -> Optional[list[dict]]:
        """
        Transcribe with word-level timestamps.

        Returns list of segments with start/end times.
        """
        result = await self.transcribe(audio_path, video_id)

        if not result:
            return None

        segments = result.get("segments", [])

        # Format segments
        formatted = []
        for seg in segments:
            formatted.append({
                "start": seg.get("start", 0),
                "end": seg.get("end", 0),
                "text": seg.get("text", "").strip(),
            })

        return formatted

    def load_transcript(self, video_id: str) -> Optional[dict]:
        """Load a previously saved transcript."""
        transcript_path = self.output_dir / f"{video_id}.json"

        if not transcript_path.exists():
            return None

        try:
            with open(transcript_path) as f:
                return json.load(f)
        except Exception as e:
            logger.error("Error loading transcript", video_id=video_id, error=str(e))
            return None

    def load_transcript_text(self, video_id: str) -> Optional[str]:
        """Load plain text transcript."""
        text_path = self.output_dir / f"{video_id}.txt"

        if not text_path.exists():
            # Try JSON
            transcript = self.load_transcript(video_id)
            if transcript:
                return transcript.get("text")
            return None

        try:
            with open(text_path) as f:
                return f.read()
        except Exception as e:
            logger.error("Error loading transcript text", error=str(e))
            return None
