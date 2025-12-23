#!/usr/bin/env python3
"""
Full RAG Pipeline Orchestrator.

Runs the complete pipeline:
1. Monitor RSS for new videos
2. Download video and extract audio
3. Transcribe with timestamps
4. Extract trading signals
5. Extract key frames
6. Align frames with transcript
7. Index into RAG database
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Optional
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_pipeline.monitor_rss import YouTubeRSSMonitor
from rag_pipeline.download_video import VideoDownloader
from rag_pipeline.transcribe import WhisperTranscriber
from rag_pipeline.extract_frames import FrameExtractor
from rag_pipeline.align_data import TranscriptFrameAligner
from rag_pipeline.rag_database import BillFanterRAG

# Import signal extractor from trading_agent
from trading_agent.transcriber.signal_extractor import SignalExtractor

DATA_DIR = Path(__file__).parent.parent.parent / "data"
RAG_DIR = DATA_DIR / "rag_pipeline"
PIPELINE_LOG = RAG_DIR / "pipeline_log.json"


class RAGPipeline:
    """
    Complete RAG pipeline for Bill Fanter trading methodology.

    Orchestrates all components to process new videos and
    add them to the RAG database.
    """

    def __init__(
        self,
        download_videos: bool = True,
        extract_frames: bool = True,
        use_whisper_api: bool = False,
    ):
        self.download_videos = download_videos
        self.extract_frames = extract_frames

        # Initialize components
        self.rss_monitor = YouTubeRSSMonitor()
        self.downloader = VideoDownloader()
        self.transcriber = WhisperTranscriber(use_api=use_whisper_api)
        self.signal_extractor = SignalExtractor()
        self.frame_extractor = FrameExtractor()
        self.aligner = TranscriptFrameAligner()
        self.rag_db = BillFanterRAG()

        # Pipeline state
        self.log = self._load_log()

    def _load_log(self) -> dict:
        """Load pipeline processing log."""
        if PIPELINE_LOG.exists():
            with open(PIPELINE_LOG) as f:
                return json.load(f)
        return {
            "processed_videos": [],
            "last_run": None,
            "stats": {},
        }

    def _save_log(self):
        """Save pipeline log."""
        PIPELINE_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(PIPELINE_LOG, "w") as f:
            json.dump(self.log, f, indent=2)

    async def process_video(self, video_id: str, video_info: Optional[dict] = None) -> dict:
        """
        Process a single video through the full pipeline.

        Args:
            video_id: YouTube video ID
            video_info: Optional video metadata

        Returns:
            Processing result dict
        """
        result = {
            "video_id": video_id,
            "started_at": datetime.now().isoformat(),
            "steps": {},
            "success": False,
        }

        video_info = video_info or {}
        url = video_info.get("url") or f"https://www.youtube.com/watch?v={video_id}"

        print(f"\n{'='*60}")
        print(f"Processing: {video_id}")
        print(f"Title: {video_info.get('title', 'Unknown')}")
        print(f"{'='*60}\n")

        # Step 1: Download audio
        print("[1/6] Downloading audio...")
        try:
            audio_result = await self.downloader.extract_audio(video_id)
            result["steps"]["download"] = {"success": audio_result is not None}
        except Exception as e:
            result["steps"]["download"] = {"success": False, "error": str(e)}
            print(f"  Error: {e}")

        # Step 2: Transcribe
        print("[2/6] Transcribing...")
        try:
            transcript = await self.transcriber.transcribe(video_id)
            result["steps"]["transcribe"] = {
                "success": transcript is not None,
                "segments": len(transcript.get("segments", [])) if transcript else 0,
            }
        except Exception as e:
            result["steps"]["transcribe"] = {"success": False, "error": str(e)}
            transcript = None
            print(f"  Error: {e}")

        # Step 3: Extract signals
        print("[3/6] Extracting signals...")
        try:
            if transcript:
                signals = await self.signal_extractor.extract_signals(
                    transcript.get("text", ""),
                    video_info,
                    video_id,
                )
                result["steps"]["signals"] = {
                    "success": signals is not None,
                    "trade_signals": len(signals.get("trade_signals", [])) if signals else 0,
                    "zones": len(signals.get("zone_levels", [])) if signals else 0,
                }
            else:
                result["steps"]["signals"] = {"success": False, "error": "No transcript"}
        except Exception as e:
            result["steps"]["signals"] = {"success": False, "error": str(e)}
            print(f"  Error: {e}")

        # Step 4: Download video and extract frames (optional)
        if self.download_videos and self.extract_frames:
            print("[4/6] Downloading video and extracting frames...")
            try:
                # Download video
                video_result = await self.downloader.download_video(video_id, url)

                if video_result:
                    # Extract frames
                    segments = transcript.get("segments", []) if transcript else []
                    frames = await self.frame_extractor.extract_all_methods(
                        video_id, segments
                    )
                    result["steps"]["frames"] = {
                        "success": True,
                        "scene_changes": len(frames.get("scene_changes", [])),
                        "regular": len(frames.get("regular_interval", [])),
                        "keyword": len(frames.get("keyword_matches", [])),
                    }
                else:
                    result["steps"]["frames"] = {"success": False, "error": "Video download failed"}
            except Exception as e:
                result["steps"]["frames"] = {"success": False, "error": str(e)}
                print(f"  Error: {e}")
        else:
            print("[4/6] Skipping video/frame extraction")
            result["steps"]["frames"] = {"skipped": True}

        # Step 5: Align data
        print("[5/6] Aligning transcript and frames...")
        try:
            aligned = self.aligner.align_video(video_id)
            result["steps"]["align"] = {
                "success": aligned is not None,
                "pairs": len(aligned.get("pairs", [])) if aligned else 0,
            }
        except Exception as e:
            result["steps"]["align"] = {"success": False, "error": str(e)}
            print(f"  Error: {e}")

        # Step 6: Index into RAG database
        print("[6/6] Indexing into RAG database...")
        try:
            transcript_count = self.rag_db.add_transcript_segments(video_id)
            signal_count = self.rag_db.add_trading_signals(video_id)
            frame_count = self.rag_db.add_aligned_frames(video_id)

            result["steps"]["index"] = {
                "success": True,
                "transcripts": transcript_count,
                "signals": signal_count,
                "frames": frame_count,
            }
        except Exception as e:
            result["steps"]["index"] = {"success": False, "error": str(e)}
            print(f"  Error: {e}")

        # Determine overall success
        required_steps = ["download", "transcribe", "signals", "index"]
        result["success"] = all(
            result["steps"].get(step, {}).get("success", False)
            for step in required_steps
        )

        result["completed_at"] = datetime.now().isoformat()

        # Update log
        if result["success"]:
            if video_id not in self.log["processed_videos"]:
                self.log["processed_videos"].append(video_id)
            self._save_log()

        print(f"\n{'='*60}")
        print(f"Result: {'SUCCESS' if result['success'] else 'PARTIAL/FAILED'}")
        print(f"{'='*60}\n")

        return result

    async def process_new_videos(self, limit: int = 10) -> list[dict]:
        """
        Check for and process new videos.

        Args:
            limit: Maximum number of videos to process

        Returns:
            List of processing results
        """
        print("Checking for new videos...")

        # Get new videos from RSS
        new_videos = await self.rss_monitor.check_for_new_videos()

        # Filter to trading-related videos
        trading_videos = [
            v for v in new_videos
            if self.rss_monitor.is_trading_related(v)
        ]

        print(f"Found {len(trading_videos)} new trading videos")

        results = []
        for video in trading_videos[:limit]:
            result = await self.process_video(
                video["video_id"],
                video,
            )
            results.append(result)

            # Mark as processed in RSS monitor
            if result["success"]:
                self.rss_monitor.mark_as_processed(video["video_id"])

        return results

    async def reprocess_existing(self, video_ids: Optional[list[str]] = None) -> list[dict]:
        """
        Reprocess existing videos (useful for re-indexing).

        Args:
            video_ids: List of video IDs to reprocess, or None for all

        Returns:
            List of processing results
        """
        if video_ids is None:
            # Get all videos with audio
            audio_dir = DATA_DIR / "transcriber" / "audio"
            video_ids = [p.stem for p in audio_dir.glob("*.mp3")]

        print(f"Reprocessing {len(video_ids)} videos...")

        results = []
        for video_id in video_ids:
            result = await self.process_video(video_id)
            results.append(result)

        return results

    async def run_continuous(self, check_interval: int = 3600):
        """
        Run continuous monitoring and processing.

        Args:
            check_interval: Seconds between RSS checks
        """
        print(f"\nStarting continuous pipeline (checking every {check_interval}s)...\n")

        while True:
            try:
                await self.process_new_videos()
            except Exception as e:
                print(f"Pipeline error: {e}")

            print(f"\nSleeping for {check_interval} seconds...")
            await asyncio.sleep(check_interval)

    def get_stats(self) -> dict:
        """Get pipeline statistics."""
        return {
            "processed_videos": len(self.log.get("processed_videos", [])),
            "last_run": self.log.get("last_run"),
            "rag_documents": self.rag_db.get_collection_count(),
            "transcripts": len(list((DATA_DIR / "transcriber" / "transcripts").glob("*.json"))),
            "signals": len(list((DATA_DIR / "transcriber" / "signals").glob("*_signals.json"))),
        }


async def main():
    """Run the pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="Bill Fanter RAG Pipeline")
    parser.add_argument(
        "--mode",
        choices=["new", "reprocess", "continuous", "stats"],
        default="stats",
        help="Pipeline mode",
    )
    parser.add_argument(
        "--video-id",
        help="Process specific video ID",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Max videos to process",
    )
    parser.add_argument(
        "--skip-frames",
        action="store_true",
        help="Skip video download and frame extraction",
    )

    args = parser.parse_args()

    pipeline = RAGPipeline(
        download_videos=not args.skip_frames,
        extract_frames=not args.skip_frames,
    )

    if args.mode == "stats":
        stats = pipeline.get_stats()
        print("\n=== Pipeline Stats ===")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    elif args.mode == "new":
        if args.video_id:
            result = await pipeline.process_video(args.video_id)
            print(json.dumps(result, indent=2))
        else:
            results = await pipeline.process_new_videos(limit=args.limit)
            print(f"\nProcessed {len(results)} videos")

    elif args.mode == "reprocess":
        if args.video_id:
            result = await pipeline.process_video(args.video_id)
            print(json.dumps(result, indent=2))
        else:
            results = await pipeline.reprocess_existing()
            print(f"\nReprocessed {len(results)} videos")

    elif args.mode == "continuous":
        await pipeline.run_continuous()


if __name__ == "__main__":
    asyncio.run(main())
