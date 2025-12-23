#!/usr/bin/env python3
"""
Video Downloader using yt-dlp.

Downloads YouTube videos for processing through the RAG pipeline.
"""

import asyncio
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional
import shutil

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "rag_pipeline"
VIDEOS_DIR = DATA_DIR / "videos"
AUDIO_DIR = DATA_DIR / "audio"
METADATA_DIR = DATA_DIR / "metadata"


class VideoDownloader:
    """Download YouTube videos using yt-dlp."""

    def __init__(self):
        # Ensure directories exist
        VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
        AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        METADATA_DIR.mkdir(parents=True, exist_ok=True)

        # Check for yt-dlp
        if not shutil.which("yt-dlp"):
            raise RuntimeError(
                "yt-dlp not found. Install with: brew install yt-dlp"
            )

    def get_video_path(self, video_id: str) -> Path:
        """Get path where video would be saved."""
        return VIDEOS_DIR / f"{video_id}.mp4"

    def get_audio_path(self, video_id: str) -> Path:
        """Get path where audio would be saved."""
        return AUDIO_DIR / f"{video_id}.mp3"

    def get_metadata_path(self, video_id: str) -> Path:
        """Get path where metadata would be saved."""
        return METADATA_DIR / f"{video_id}.json"

    def is_downloaded(self, video_id: str) -> bool:
        """Check if video is already downloaded."""
        return self.get_video_path(video_id).exists()

    def is_audio_extracted(self, video_id: str) -> bool:
        """Check if audio is already extracted."""
        return self.get_audio_path(video_id).exists()

    async def download_video(
        self,
        video_id: str,
        url: Optional[str] = None,
        quality: str = "720p",
    ) -> dict:
        """
        Download a YouTube video.

        Args:
            video_id: YouTube video ID
            url: Full YouTube URL (optional, constructed from video_id if not provided)
            quality: Video quality (360p, 480p, 720p, 1080p)

        Returns:
            dict with download status and paths
        """
        if url is None:
            url = f"https://www.youtube.com/watch?v={video_id}"

        video_path = self.get_video_path(video_id)
        metadata_path = self.get_metadata_path(video_id)

        # Skip if already downloaded
        if video_path.exists():
            print(f"Video {video_id} already downloaded")
            return {
                "status": "exists",
                "video_path": str(video_path),
                "video_id": video_id,
            }

        print(f"Downloading video {video_id}...")

        # Quality format string
        quality_map = {
            "360p": "bestvideo[height<=360]+bestaudio/best[height<=360]",
            "480p": "bestvideo[height<=480]+bestaudio/best[height<=480]",
            "720p": "bestvideo[height<=720]+bestaudio/best[height<=720]",
            "1080p": "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
        }
        format_str = quality_map.get(quality, quality_map["720p"])

        # yt-dlp command
        cmd = [
            "yt-dlp",
            "-f", format_str,
            "--merge-output-format", "mp4",
            "-o", str(video_path),
            "--write-info-json",  # Save metadata
            "--no-playlist",
            url,
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                return {
                    "status": "error",
                    "error": error_msg,
                    "video_id": video_id,
                }

            # Move the info.json to our metadata directory
            info_json = VIDEOS_DIR / f"{video_id}.info.json"
            if info_json.exists():
                shutil.move(str(info_json), str(metadata_path))

            return {
                "status": "success",
                "video_path": str(video_path),
                "metadata_path": str(metadata_path) if metadata_path.exists() else None,
                "video_id": video_id,
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "video_id": video_id,
            }

    async def extract_audio(self, video_id: str) -> dict:
        """
        Extract audio from video for transcription.

        Uses yt-dlp to extract just the audio track as MP3.
        """
        audio_path = self.get_audio_path(video_id)

        # Skip if already extracted
        if audio_path.exists():
            print(f"Audio for {video_id} already extracted")
            return {
                "status": "exists",
                "audio_path": str(audio_path),
                "video_id": video_id,
            }

        url = f"https://www.youtube.com/watch?v={video_id}"

        print(f"Extracting audio from {video_id}...")

        cmd = [
            "yt-dlp",
            "-x",  # Extract audio
            "--audio-format", "mp3",
            "--audio-quality", "0",  # Best quality
            "-o", str(audio_path),
            "--no-playlist",
            url,
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                return {
                    "status": "error",
                    "error": error_msg,
                    "video_id": video_id,
                }

            return {
                "status": "success",
                "audio_path": str(audio_path),
                "video_id": video_id,
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "video_id": video_id,
            }

    async def download_with_audio(
        self,
        video_id: str,
        url: Optional[str] = None,
        quality: str = "720p",
    ) -> dict:
        """Download video and extract audio in one go."""
        # Download video
        video_result = await self.download_video(video_id, url, quality)
        if video_result["status"] == "error":
            return video_result

        # Extract audio
        audio_result = await self.extract_audio(video_id)

        return {
            "status": "success" if audio_result["status"] != "error" else "partial",
            "video_path": video_result.get("video_path"),
            "audio_path": audio_result.get("audio_path"),
            "metadata_path": video_result.get("metadata_path"),
            "video_id": video_id,
        }

    def get_video_metadata(self, video_id: str) -> Optional[dict]:
        """Load video metadata from disk."""
        metadata_path = self.get_metadata_path(video_id)
        if metadata_path.exists():
            with open(metadata_path) as f:
                return json.load(f)
        return None

    def get_video_duration(self, video_id: str) -> Optional[float]:
        """Get video duration in seconds."""
        metadata = self.get_video_metadata(video_id)
        if metadata:
            return metadata.get("duration")
        return None


async def main():
    """Test the video downloader."""
    downloader = VideoDownloader()

    # Test with a short Bill Fanter video
    test_video_id = "MKHF4qzPj3g"  # Example video ID

    print(f"Testing download for video: {test_video_id}")

    # Check if already downloaded
    if downloader.is_downloaded(test_video_id):
        print("Video already downloaded!")
    else:
        result = await downloader.download_with_audio(test_video_id)
        print(f"Result: {json.dumps(result, indent=2)}")


if __name__ == "__main__":
    asyncio.run(main())
