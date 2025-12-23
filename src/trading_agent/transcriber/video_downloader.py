"""
Video Downloader using yt-dlp.

Downloads YouTube videos and extracts audio for transcription.
"""

import asyncio
import subprocess
from pathlib import Path
from typing import Optional

import structlog

logger = structlog.get_logger()


class VideoDownloader:
    """
    Download YouTube videos and extract audio using yt-dlp.

    Requires yt-dlp to be installed:
        pip install yt-dlp
    or
        brew install yt-dlp
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        audio_format: str = "mp3",
        audio_quality: str = "192",
    ):
        """
        Initialize video downloader.

        Args:
            output_dir: Directory to save downloaded files
            audio_format: Audio format (mp3, wav, m4a)
            audio_quality: Audio bitrate (128, 192, 320)
        """
        self.output_dir = output_dir or Path("data/transcriber/downloads")
        self.audio_format = audio_format
        self.audio_quality = audio_quality

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def download_audio(
        self,
        video_url: str,
        video_id: Optional[str] = None,
    ) -> Optional[Path]:
        """
        Download video and extract audio.

        Args:
            video_url: YouTube video URL
            video_id: Optional video ID for filename

        Returns:
            Path to downloaded audio file or None if failed
        """
        # Determine output filename
        if video_id:
            output_template = str(self.output_dir / f"{video_id}.%(ext)s")
        else:
            output_template = str(self.output_dir / "%(id)s.%(ext)s")

        # yt-dlp command
        cmd = [
            "yt-dlp",
            "--extract-audio",
            "--audio-format", self.audio_format,
            "--audio-quality", self.audio_quality,
            "--output", output_template,
            "--no-playlist",
            "--quiet",
            "--no-warnings",
            video_url,
        ]

        logger.info("Downloading audio", url=video_url, video_id=video_id)

        try:
            # Run yt-dlp
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                logger.error(
                    "Download failed",
                    url=video_url,
                    stderr=stderr.decode() if stderr else "Unknown error",
                )
                return None

            # Find the downloaded file
            if video_id:
                audio_path = self.output_dir / f"{video_id}.{self.audio_format}"
            else:
                # Try to find by pattern
                audio_files = list(self.output_dir.glob(f"*.{self.audio_format}"))
                if audio_files:
                    audio_path = max(audio_files, key=lambda p: p.stat().st_mtime)
                else:
                    logger.error("Could not find downloaded file")
                    return None

            if audio_path.exists():
                logger.info(
                    "Audio downloaded",
                    path=str(audio_path),
                    size_mb=audio_path.stat().st_size / 1024 / 1024,
                )
                return audio_path

            return None

        except FileNotFoundError:
            logger.error(
                "yt-dlp not found. Install with: pip install yt-dlp"
            )
            return None

        except Exception as e:
            logger.error("Download error", url=video_url, error=str(e))
            return None

    async def download_video(
        self,
        video_url: str,
        video_id: Optional[str] = None,
        quality: str = "best",
    ) -> Optional[Path]:
        """
        Download full video (not just audio).

        Args:
            video_url: YouTube video URL
            video_id: Optional video ID for filename
            quality: Video quality (best, worst, 720p, etc.)

        Returns:
            Path to downloaded video file or None if failed
        """
        if video_id:
            output_template = str(self.output_dir / f"{video_id}.%(ext)s")
        else:
            output_template = str(self.output_dir / "%(id)s.%(ext)s")

        cmd = [
            "yt-dlp",
            "--format", quality,
            "--output", output_template,
            "--no-playlist",
            "--quiet",
            "--no-warnings",
            video_url,
        ]

        logger.info("Downloading video", url=video_url, quality=quality)

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                logger.error(
                    "Video download failed",
                    url=video_url,
                    stderr=stderr.decode() if stderr else "Unknown error",
                )
                return None

            # Find downloaded file
            video_files = list(self.output_dir.glob(f"{video_id}.*" if video_id else "*"))
            video_files = [f for f in video_files if f.suffix in [".mp4", ".webm", ".mkv"]]

            if video_files:
                video_path = max(video_files, key=lambda p: p.stat().st_mtime)
                logger.info("Video downloaded", path=str(video_path))
                return video_path

            return None

        except Exception as e:
            logger.error("Video download error", url=video_url, error=str(e))
            return None

    async def get_video_info(self, video_url: str) -> Optional[dict]:
        """
        Get video metadata without downloading.

        Returns:
            Dict with title, duration, description, etc.
        """
        cmd = [
            "yt-dlp",
            "--dump-json",
            "--no-download",
            "--no-playlist",
            video_url,
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                return None

            import json
            info = json.loads(stdout.decode())

            return {
                "video_id": info.get("id"),
                "title": info.get("title"),
                "description": info.get("description"),
                "duration": info.get("duration"),
                "duration_string": info.get("duration_string"),
                "upload_date": info.get("upload_date"),
                "uploader": info.get("uploader"),
                "channel_id": info.get("channel_id"),
                "view_count": info.get("view_count"),
                "like_count": info.get("like_count"),
                "thumbnail": info.get("thumbnail"),
            }

        except Exception as e:
            logger.error("Error getting video info", url=video_url, error=str(e))
            return None

    def cleanup(self, video_id: str) -> None:
        """Remove downloaded files for a video."""
        for ext in [self.audio_format, "mp4", "webm", "mkv", "part"]:
            path = self.output_dir / f"{video_id}.{ext}"
            if path.exists():
                path.unlink()
                logger.debug("Cleaned up file", path=str(path))
