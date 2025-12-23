#!/usr/bin/env python3
"""
Whisper Transcription for RAG Pipeline.

Wraps the existing transcriber for the RAG pipeline.
"""

import asyncio
import json
from pathlib import Path
from typing import Optional
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from trading_agent.transcriber.transcriber import Transcriber

DATA_DIR = Path(__file__).parent.parent.parent / "data"
AUDIO_DIR = DATA_DIR / "transcriber" / "audio"
TRANSCRIPTS_DIR = DATA_DIR / "transcriber" / "transcripts"


class WhisperTranscriber:
    """
    Whisper transcription wrapper for RAG pipeline.

    Uses the existing Transcriber class but provides
    a simpler interface for the pipeline.
    """

    def __init__(
        self,
        use_api: bool = False,
        model: str = "base",
        audio_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ):
        self.audio_dir = audio_dir or AUDIO_DIR
        self.output_dir = output_dir or TRANSCRIPTS_DIR

        self.transcriber = Transcriber(
            use_api=use_api,
            model=model,
            output_dir=self.output_dir,
        )

    def get_audio_path(self, video_id: str) -> Optional[Path]:
        """Find audio file for a video ID."""
        for ext in [".mp3", ".m4a", ".wav"]:
            path = self.audio_dir / f"{video_id}{ext}"
            if path.exists():
                return path
        return None

    def is_transcribed(self, video_id: str) -> bool:
        """Check if video is already transcribed."""
        json_path = self.output_dir / f"{video_id}.json"
        txt_path = self.output_dir / f"{video_id}.txt"
        return json_path.exists() or txt_path.exists()

    async def transcribe(self, video_id: str) -> Optional[dict]:
        """
        Transcribe a video by ID.

        Returns transcript dict with segments.
        """
        if self.is_transcribed(video_id):
            print(f"Already transcribed: {video_id}")
            return self.load_transcript(video_id)

        audio_path = self.get_audio_path(video_id)
        if not audio_path:
            print(f"Audio not found for {video_id}")
            return None

        print(f"Transcribing {video_id}...")
        result = await self.transcriber.transcribe(audio_path, video_id)

        return result

    def load_transcript(self, video_id: str) -> Optional[dict]:
        """Load existing transcript."""
        json_path = self.output_dir / f"{video_id}.json"
        if json_path.exists():
            with open(json_path) as f:
                return json.load(f)

        txt_path = self.output_dir / f"{video_id}.txt"
        if txt_path.exists():
            with open(txt_path) as f:
                return {"text": f.read(), "segments": []}

        return None

    def get_segments(self, video_id: str) -> list[dict]:
        """Get transcript segments with timestamps."""
        transcript = self.load_transcript(video_id)
        if transcript:
            return transcript.get("segments", [])
        return []

    async def transcribe_all_pending(self) -> list[str]:
        """
        Transcribe all audio files that haven't been transcribed.

        Returns list of video IDs transcribed.
        """
        transcribed = []

        for audio_file in self.audio_dir.glob("*"):
            if audio_file.suffix in [".mp3", ".m4a", ".wav"]:
                video_id = audio_file.stem

                if not self.is_transcribed(video_id):
                    result = await self.transcribe(video_id)
                    if result:
                        transcribed.append(video_id)

        return transcribed

    def get_all_video_ids(self) -> list[str]:
        """Get all video IDs with transcripts."""
        video_ids = set()

        for path in self.output_dir.glob("*.json"):
            video_ids.add(path.stem)

        for path in self.output_dir.glob("*.txt"):
            video_ids.add(path.stem)

        return sorted(video_ids)


async def main():
    """Test transcription."""
    transcriber = WhisperTranscriber()

    # List available audio
    audio_files = list(AUDIO_DIR.glob("*.mp3"))
    print(f"Found {len(audio_files)} audio files")

    # List existing transcripts
    existing = transcriber.get_all_video_ids()
    print(f"Existing transcripts: {len(existing)}")

    # Find pending
    pending = []
    for audio_file in audio_files:
        video_id = audio_file.stem
        if not transcriber.is_transcribed(video_id):
            pending.append(video_id)

    print(f"Pending transcription: {len(pending)}")

    if pending:
        print(f"\nPending videos: {pending[:5]}...")


if __name__ == "__main__":
    asyncio.run(main())
