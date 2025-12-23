#!/usr/bin/env python3
"""
Frame Extraction for Bill Fanter Videos.

Extracts key frames from videos using:
1. Scene change detection (new charts appearing)
2. Regular interval sampling
3. Timestamp-aligned extraction (from transcript)

Requires: ffmpeg, opencv-python
"""

import asyncio
import json
import subprocess
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional
import hashlib

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "rag_pipeline"
FRAMES_DIR = DATA_DIR / "frames"
VIDEOS_DIR = DATA_DIR / "videos"


class FrameExtractor:
    """Extract frames from trading videos for vision analysis."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or FRAMES_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Check for ffmpeg
        if not shutil.which("ffmpeg"):
            raise RuntimeError("ffmpeg not found. Install with: brew install ffmpeg")

    def get_video_path(self, video_id: str) -> Optional[Path]:
        """Find video file for a video ID."""
        # Check multiple locations
        search_dirs = [
            VIDEOS_DIR,
            DATA_DIR.parent / "transcriber" / "downloads",
        ]

        for search_dir in search_dirs:
            for ext in [".mp4", ".webm", ".mkv"]:
                path = search_dir / f"{video_id}{ext}"
                if path.exists():
                    return path

        return None

    def get_frames_dir(self, video_id: str) -> Path:
        """Get output directory for video frames."""
        frames_dir = self.output_dir / video_id
        frames_dir.mkdir(parents=True, exist_ok=True)
        return frames_dir

    async def extract_at_timestamps(
        self,
        video_id: str,
        timestamps: list[float],
        video_path: Optional[Path] = None,
    ) -> list[dict]:
        """
        Extract frames at specific timestamps.

        Args:
            video_id: Video ID
            timestamps: List of timestamps in seconds
            video_path: Optional path to video file

        Returns:
            List of extracted frame info dicts
        """
        video_path = video_path or self.get_video_path(video_id)
        if not video_path or not video_path.exists():
            print(f"Video not found for {video_id}")
            return []

        frames_dir = self.get_frames_dir(video_id)
        extracted = []

        for ts in timestamps:
            # Format timestamp for filename
            ts_str = f"{int(ts):05d}"
            output_path = frames_dir / f"frame_{ts_str}.jpg"

            # Skip if already extracted
            if output_path.exists():
                extracted.append({
                    "video_id": video_id,
                    "timestamp": ts,
                    "path": str(output_path),
                    "cached": True,
                })
                continue

            # Extract frame using ffmpeg
            cmd = [
                "ffmpeg",
                "-ss", str(ts),
                "-i", str(video_path),
                "-vframes", "1",
                "-q:v", "2",  # High quality JPEG
                "-y",  # Overwrite
                str(output_path),
            ]

            try:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await process.communicate()

                if output_path.exists():
                    extracted.append({
                        "video_id": video_id,
                        "timestamp": ts,
                        "path": str(output_path),
                        "cached": False,
                    })

            except Exception as e:
                print(f"Error extracting frame at {ts}: {e}")

        return extracted

    async def extract_scene_changes(
        self,
        video_id: str,
        video_path: Optional[Path] = None,
        threshold: float = 0.3,
        min_interval: float = 5.0,
    ) -> list[dict]:
        """
        Extract frames at scene changes (new charts appearing).

        Uses ffmpeg's scene detection filter.

        Args:
            video_id: Video ID
            video_path: Optional path to video file
            threshold: Scene change threshold (0-1, lower = more sensitive)
            min_interval: Minimum seconds between extracted frames

        Returns:
            List of extracted frame info dicts
        """
        video_path = video_path or self.get_video_path(video_id)
        if not video_path or not video_path.exists():
            print(f"Video not found for {video_id}")
            return []

        frames_dir = self.get_frames_dir(video_id)

        # First, detect scene changes
        print(f"Detecting scene changes in {video_id}...")

        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-vf", f"select='gt(scene,{threshold})',showinfo",
            "-f", "null",
            "-",
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            # Parse stderr for timestamps (ffmpeg outputs to stderr)
            output = stderr.decode()
            timestamps = []

            import re
            for match in re.finditer(r"pts_time:(\d+\.?\d*)", output):
                ts = float(match.group(1))
                # Apply minimum interval filter
                if not timestamps or (ts - timestamps[-1]) >= min_interval:
                    timestamps.append(ts)

            print(f"Found {len(timestamps)} scene changes")

            # Extract frames at detected timestamps
            return await self.extract_at_timestamps(video_id, timestamps, video_path)

        except Exception as e:
            print(f"Error detecting scene changes: {e}")
            return []

    async def extract_regular_interval(
        self,
        video_id: str,
        video_path: Optional[Path] = None,
        interval: float = 30.0,
        max_frames: int = 100,
    ) -> list[dict]:
        """
        Extract frames at regular intervals.

        Args:
            video_id: Video ID
            video_path: Optional path to video file
            interval: Seconds between frames
            max_frames: Maximum number of frames to extract

        Returns:
            List of extracted frame info dicts
        """
        video_path = video_path or self.get_video_path(video_id)
        if not video_path or not video_path.exists():
            print(f"Video not found for {video_id}")
            return []

        # Get video duration
        duration = await self._get_duration(video_path)
        if not duration:
            return []

        # Generate timestamps
        timestamps = []
        ts = 0.0
        while ts < duration and len(timestamps) < max_frames:
            timestamps.append(ts)
            ts += interval

        print(f"Extracting {len(timestamps)} frames at {interval}s intervals")

        return await self.extract_at_timestamps(video_id, timestamps, video_path)

    async def extract_from_transcript(
        self,
        video_id: str,
        transcript_segments: list[dict],
        keywords: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        Extract frames at timestamps where specific keywords are mentioned.

        Args:
            video_id: Video ID
            transcript_segments: List of transcript segments with start/end times
            keywords: Keywords to look for (stock tickers, "chart", "zone", etc.)

        Returns:
            List of extracted frame info with matched text
        """
        if keywords is None:
            keywords = [
                # Stock tickers
                "SPY", "QQQ", "AAPL", "TSLA", "NVDA", "AMD", "META", "GOOGL",
                "AMZN", "MSFT", "AVGO", "ORCL", "PLTR", "NKE", "SBUX",
                # Trading terms
                "chart", "zone", "supply", "demand", "breakout", "support",
                "resistance", "level", "target", "stop", "entry",
            ]

        # Find segments with keywords
        matched_timestamps = []
        for seg in transcript_segments:
            text = seg.get("text", "").upper()
            start = seg.get("start", 0)

            for kw in keywords:
                if kw.upper() in text:
                    matched_timestamps.append({
                        "timestamp": start,
                        "keyword": kw,
                        "text": seg.get("text", ""),
                    })
                    break  # Only match once per segment

        if not matched_timestamps:
            print(f"No keyword matches found in transcript")
            return []

        # Deduplicate timestamps within 3 seconds
        deduped = []
        for match in sorted(matched_timestamps, key=lambda x: x["timestamp"]):
            if not deduped or (match["timestamp"] - deduped[-1]["timestamp"]) > 3:
                deduped.append(match)

        print(f"Found {len(deduped)} keyword matches")

        # Extract frames
        timestamps = [m["timestamp"] for m in deduped]
        frames = await self.extract_at_timestamps(video_id, timestamps)

        # Merge with keyword info
        for i, frame in enumerate(frames):
            if i < len(deduped):
                frame["keyword"] = deduped[i]["keyword"]
                frame["transcript_text"] = deduped[i]["text"]

        return frames

    async def _get_duration(self, video_path: Path) -> Optional[float]:
        """Get video duration in seconds."""
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                return float(stdout.decode().strip())

        except Exception as e:
            print(f"Error getting duration: {e}")

        return None

    async def extract_all_methods(
        self,
        video_id: str,
        transcript_segments: Optional[list[dict]] = None,
    ) -> dict:
        """
        Run all extraction methods and combine results.

        Returns dict with frames from each method.
        """
        results = {
            "video_id": video_id,
            "extracted_at": datetime.now().isoformat(),
            "scene_changes": [],
            "regular_interval": [],
            "keyword_matches": [],
        }

        # Scene change detection
        print(f"\n[1/3] Scene change detection...")
        results["scene_changes"] = await self.extract_scene_changes(video_id)

        # Regular interval (every 30 seconds)
        print(f"\n[2/3] Regular interval extraction...")
        results["regular_interval"] = await self.extract_regular_interval(
            video_id, interval=30.0
        )

        # Keyword-based (if transcript available)
        if transcript_segments:
            print(f"\n[3/3] Keyword-based extraction...")
            results["keyword_matches"] = await self.extract_from_transcript(
                video_id, transcript_segments
            )

        # Save extraction manifest
        manifest_path = self.get_frames_dir(video_id) / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(results, f, indent=2)

        total = (
            len(results["scene_changes"]) +
            len(results["regular_interval"]) +
            len(results["keyword_matches"])
        )
        print(f"\nTotal frames extracted: {total}")

        return results


async def main():
    """Test frame extraction."""
    extractor = FrameExtractor()

    # Test with a video we have
    test_video = "9xhP60b25Rk"

    # Check if video exists
    video_path = extractor.get_video_path(test_video)
    if not video_path:
        print(f"Video {test_video} not found. Download it first.")
        return

    # Load transcript if available
    transcript_path = Path("data/transcriber/transcripts") / f"{test_video}.json"
    transcript_segments = None
    if transcript_path.exists():
        with open(transcript_path) as f:
            data = json.load(f)
            transcript_segments = data.get("segments", [])

    # Run extraction
    results = await extractor.extract_all_methods(test_video, transcript_segments)
    print(f"\nResults saved to: {extractor.get_frames_dir(test_video)}")


if __name__ == "__main__":
    asyncio.run(main())
