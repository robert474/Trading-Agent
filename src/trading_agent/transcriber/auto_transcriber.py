"""
Auto-Transcriber - Main orchestrator for automatic video transcription.

Monitors YouTube RSS feeds, downloads new videos, transcribes them,
and extracts trading signals automatically.
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional

import structlog

from trading_agent.core.config import settings
from trading_agent.data.database import DatabaseManager

from .rss_monitor import RSSMonitor
from .video_downloader import VideoDownloader
from .transcriber import Transcriber
from .signal_extractor import SignalExtractor

logger = structlog.get_logger()


class AutoTranscriber:
    """
    Automatic video transcription pipeline.

    Flow:
    1. Monitor YouTube RSS feeds for new videos
    2. Download audio from new videos
    3. Transcribe audio using Whisper
    4. Extract trading signals using Claude
    5. Save to database for few-shot learning
    """

    def __init__(
        self,
        channel_ids: list[str],
        data_dir: Optional[Path] = None,
        check_interval: int = 3600,  # 1 hour
        use_api_transcription: bool = False,
        openai_api_key: Optional[str] = None,
        whisper_model: str = "base",
    ):
        """
        Initialize auto-transcriber.

        Args:
            channel_ids: YouTube channel IDs to monitor
            data_dir: Base directory for all data
            check_interval: Seconds between RSS checks
            use_api_transcription: Use OpenAI API for Whisper
            openai_api_key: OpenAI API key (if using API)
            whisper_model: Local Whisper model size
        """
        self.data_dir = data_dir or Path("data/transcriber")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.rss_monitor = RSSMonitor(
            channel_ids=channel_ids,
            data_dir=self.data_dir,
            check_interval=check_interval,
        )

        self.downloader = VideoDownloader(
            output_dir=self.data_dir / "downloads",
        )

        self.transcriber = Transcriber(
            use_api=use_api_transcription,
            api_key=openai_api_key,
            model=whisper_model,
            output_dir=self.data_dir / "transcripts",
        )

        self.signal_extractor = SignalExtractor(
            output_dir=self.data_dir / "signals",
        )

        # Database manager (optional, for saving to trading bot DB)
        self.db: Optional[DatabaseManager] = None

        # Register callback for new videos
        self.rss_monitor.on_new_video(self._process_new_video)

        # Track processing status
        self._processing: set[str] = set()

    async def initialize_db(self) -> None:
        """Initialize database connection for saving examples."""
        self.db = DatabaseManager()
        await self.db.connect()

    async def _process_new_video(self, video: dict) -> None:
        """
        Process a newly discovered video.

        Called automatically when RSS monitor finds new videos.
        """
        video_id = video["video_id"]

        # Skip if already processing
        if video_id in self._processing:
            return

        self._processing.add(video_id)

        try:
            logger.info(
                "Processing new video",
                video_id=video_id,
                title=video["title"],
                channel=video.get("channel_name"),
            )

            # Step 1: Download audio
            audio_path = await self.downloader.download_audio(
                video_url=video["url"],
                video_id=video_id,
            )

            if not audio_path:
                logger.error("Failed to download audio", video_id=video_id)
                return

            # Step 2: Transcribe
            transcript = await self.transcriber.transcribe(
                audio_path=audio_path,
                video_id=video_id,
            )

            if not transcript:
                logger.error("Failed to transcribe", video_id=video_id)
                return

            # Step 3: Extract signals
            signals = await self.signal_extractor.extract_signals(
                transcript=transcript["text"],
                video_info=video,
                video_id=video_id,
            )

            # Step 4: Save to database if available
            if signals and self.db:
                examples = self.signal_extractor.convert_to_trade_examples(signals)
                for example in examples:
                    if example.get("symbol") and example.get("entry_price"):
                        await self.db.save_trade_example(
                            symbol=example["symbol"],
                            setup_type=example.get("setup_type", ""),
                            setup_description=example.get("setup_description", ""),
                            entry_reasoning=example.get("entry_reasoning", ""),
                            chart_context=example.get("chart_context", {}),
                            entry_price=example.get("entry_price", 0),
                            exit_price=example.get("exit_price", 0),
                            result=example.get("result", ""),
                            pnl=0,  # Will be filled in later if known
                            lessons=example.get("lessons", ""),
                            source="youtube",
                        )

                logger.info(
                    "Saved trade examples to database",
                    video_id=video_id,
                    count=len(examples),
                )

            # Mark as processed
            self.rss_monitor.mark_processed(video_id)

            # Cleanup audio file (keep transcript)
            self.downloader.cleanup(video_id)

            logger.info(
                "Video processing complete",
                video_id=video_id,
                title=video["title"],
                signals=len(signals.get("trade_signals", [])) if signals else 0,
            )

        except Exception as e:
            logger.error("Error processing video", video_id=video_id, error=str(e))

        finally:
            self._processing.discard(video_id)

    async def process_video_url(self, url: str) -> Optional[dict]:
        """
        Process a single video URL manually.

        Args:
            url: YouTube video URL

        Returns:
            Extracted signals dict
        """
        # Get video info
        video_info = await self.downloader.get_video_info(url)

        if not video_info:
            logger.error("Could not get video info", url=url)
            return None

        video_id = video_info["video_id"]

        # Check if already processed
        existing = self.signal_extractor.load_signals(video_id)
        if existing:
            logger.info("Video already processed", video_id=video_id)
            return existing

        # Process like a new video
        video = {
            "video_id": video_id,
            "title": video_info.get("title", ""),
            "url": url,
            "published": video_info.get("upload_date"),
            "channel_name": video_info.get("uploader"),
        }

        await self._process_new_video(video)

        # Return the extracted signals
        return self.signal_extractor.load_signals(video_id)

    async def start(self) -> None:
        """Start the auto-transcriber monitoring loop."""
        logger.info("Starting Auto-Transcriber")

        # Check for any missed videos first
        new_videos = await self.rss_monitor.check_for_new_videos()

        if new_videos:
            logger.info(f"Found {len(new_videos)} unprocessed videos")

        # Start monitoring loop
        await self.rss_monitor.start_monitoring()

    async def process_backlog(self, limit: int = 10) -> None:
        """
        Process backlog of unprocessed videos.

        Args:
            limit: Maximum number of videos to process
        """
        logger.info("Processing video backlog", limit=limit)

        new_videos = await self.rss_monitor.check_for_new_videos()

        for i, video in enumerate(new_videos[:limit]):
            logger.info(f"Processing backlog video {i+1}/{min(limit, len(new_videos))}")
            await self._process_new_video(video)
            # Small delay between videos
            await asyncio.sleep(5)

    def get_all_examples(self) -> list[dict]:
        """Get all extracted trade examples."""
        return self.signal_extractor.get_all_trade_examples()


# ============================================================================
# CLI Script
# ============================================================================

async def main():
    """CLI entry point for auto-transcriber."""
    import argparse

    parser = argparse.ArgumentParser(description="Auto-transcribe YouTube trading videos")

    parser.add_argument(
        "--channel-id",
        action="append",
        dest="channel_ids",
        help="YouTube channel ID to monitor (can specify multiple)",
    )
    parser.add_argument(
        "--channel-url",
        help="YouTube channel URL (will extract channel ID)",
    )
    parser.add_argument(
        "--video-url",
        help="Process a single video URL",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=3600,
        help="Check interval in seconds (default: 3600)",
    )
    parser.add_argument(
        "--whisper-model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base)",
    )
    parser.add_argument(
        "--use-api",
        action="store_true",
        help="Use OpenAI Whisper API instead of local model",
    )
    parser.add_argument(
        "--backlog",
        type=int,
        help="Process N backlog videos then exit",
    )
    parser.add_argument(
        "--find-channel-id",
        help="Find channel ID from URL and exit",
    )

    args = parser.parse_args()

    from trading_agent.monitoring.logger import setup_logging
    setup_logging(level="INFO")

    # Just find channel ID
    if args.find_channel_id:
        channel_id = await RSSMonitor.find_channel_id(args.find_channel_id)
        if channel_id:
            print(f"\nChannel ID: {channel_id}")
            print(f"RSS Feed URL: {RSSMonitor.get_channel_rss_url(channel_id)}")
        else:
            print("Could not find channel ID")
        return

    # Build channel ID list
    channel_ids = args.channel_ids or []

    if args.channel_url:
        channel_id = await RSSMonitor.find_channel_id(args.channel_url)
        if channel_id:
            channel_ids.append(channel_id)
            logger.info(f"Found channel ID: {channel_id}")
        else:
            logger.error("Could not find channel ID from URL")
            return

    # Process single video
    if args.video_url:
        transcriber = AutoTranscriber(
            channel_ids=[],
            whisper_model=args.whisper_model,
            use_api_transcription=args.use_api,
        )

        signals = await transcriber.process_video_url(args.video_url)

        if signals:
            import json
            print("\n" + "=" * 60)
            print("EXTRACTED SIGNALS")
            print("=" * 60)
            print(json.dumps(signals, indent=2, default=str))
        return

    # Need at least one channel for monitoring
    if not channel_ids:
        print("Error: No channel IDs specified")
        print("Use --channel-id or --channel-url to specify channels to monitor")
        print("\nTo find a channel ID, use:")
        print("  python -m trading_agent.transcriber.auto_transcriber --find-channel-id 'https://youtube.com/@channelname'")
        return

    # Create transcriber
    transcriber = AutoTranscriber(
        channel_ids=channel_ids,
        check_interval=args.interval,
        whisper_model=args.whisper_model,
        use_api_transcription=args.use_api,
    )

    # Process backlog only
    if args.backlog:
        await transcriber.process_backlog(limit=args.backlog)
        return

    # Start monitoring
    print(f"\nMonitoring {len(channel_ids)} channel(s)")
    print(f"Check interval: {args.interval} seconds")
    print("Press Ctrl+C to stop\n")

    try:
        await transcriber.start()
    except KeyboardInterrupt:
        print("\nStopping...")


if __name__ == "__main__":
    asyncio.run(main())
