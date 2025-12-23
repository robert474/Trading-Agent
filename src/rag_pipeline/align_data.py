#!/usr/bin/env python3
"""
Transcript-Frame Alignment for RAG Pipeline.

Aligns transcript segments with extracted frames to create
training pairs for vision+text understanding.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).parent.parent.parent / "data"
RAG_DIR = DATA_DIR / "rag_pipeline"
TRANSCRIPTS_DIR = DATA_DIR / "transcriber" / "transcripts"
SIGNALS_DIR = DATA_DIR / "transcriber" / "signals"
FRAMES_DIR = RAG_DIR / "frames"
ALIGNED_DIR = RAG_DIR / "aligned"


class TranscriptFrameAligner:
    """
    Align transcript segments with extracted video frames.

    Creates training pairs of (frame_image, context_text, trading_signals)
    for the RAG pipeline.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or ALIGNED_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_transcript(self, video_id: str) -> Optional[dict]:
        """Load transcript with segments."""
        json_path = TRANSCRIPTS_DIR / f"{video_id}.json"
        if json_path.exists():
            with open(json_path) as f:
                return json.load(f)

        # Try loading just text
        txt_path = TRANSCRIPTS_DIR / f"{video_id}.txt"
        if txt_path.exists():
            with open(txt_path) as f:
                return {"text": f.read(), "segments": []}

        return None

    def load_signals(self, video_id: str) -> Optional[dict]:
        """Load extracted trading signals."""
        signal_path = SIGNALS_DIR / f"{video_id}_signals.json"
        if signal_path.exists():
            with open(signal_path) as f:
                return json.load(f)
        return None

    def load_frame_manifest(self, video_id: str) -> Optional[dict]:
        """Load frame extraction manifest."""
        manifest_path = FRAMES_DIR / video_id / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                return json.load(f)
        return None

    def get_context_for_timestamp(
        self,
        timestamp: float,
        segments: list[dict],
        window_seconds: float = 15.0,
    ) -> str:
        """
        Get transcript text surrounding a timestamp.

        Args:
            timestamp: Target timestamp in seconds
            segments: Transcript segments with start/end times
            window_seconds: Seconds of context before/after

        Returns:
            Combined text from segments in the window
        """
        context_parts = []

        for seg in segments:
            seg_start = seg.get("start", 0)
            seg_end = seg.get("end", seg_start + 5)

            # Check if segment overlaps with our window
            window_start = timestamp - window_seconds
            window_end = timestamp + window_seconds

            if seg_end >= window_start and seg_start <= window_end:
                context_parts.append(seg.get("text", "").strip())

        return " ".join(context_parts)

    def find_relevant_signals(
        self,
        context_text: str,
        signals: dict,
    ) -> list[dict]:
        """
        Find trading signals mentioned in the context.

        Args:
            context_text: Transcript text around the frame
            signals: Extracted signals for the video

        Returns:
            List of relevant signals
        """
        relevant = []
        context_upper = context_text.upper()

        # Check trade signals
        for signal in signals.get("trade_signals", []):
            symbol = signal.get("symbol", "")
            if symbol.upper() in context_upper:
                relevant.append({
                    "type": "trade_signal",
                    **signal,
                })

        # Check zone levels
        for zone in signals.get("zone_levels", []):
            symbol = zone.get("symbol", "")
            if symbol.upper() in context_upper:
                relevant.append({
                    "type": "zone_level",
                    **zone,
                })

        return relevant

    def align_video(self, video_id: str) -> Optional[dict]:
        """
        Align all data for a single video.

        Returns dict with aligned frame-context pairs.
        """
        # Load all data sources
        transcript = self.load_transcript(video_id)
        signals = self.load_signals(video_id)
        manifest = self.load_frame_manifest(video_id)

        if not transcript:
            print(f"No transcript found for {video_id}")
            return None

        segments = transcript.get("segments", [])

        # Collect all frames
        all_frames = []
        if manifest:
            all_frames.extend(manifest.get("scene_changes", []))
            all_frames.extend(manifest.get("regular_interval", []))
            all_frames.extend(manifest.get("keyword_matches", []))

        if not all_frames:
            print(f"No frames found for {video_id}")
            return None

        # Deduplicate frames by timestamp (within 1 second)
        unique_frames = []
        seen_timestamps = set()
        for frame in sorted(all_frames, key=lambda x: x.get("timestamp", 0)):
            ts = round(frame.get("timestamp", 0))
            if ts not in seen_timestamps:
                seen_timestamps.add(ts)
                unique_frames.append(frame)

        print(f"Aligning {len(unique_frames)} frames for {video_id}...")

        # Create aligned pairs
        aligned_pairs = []
        for frame in unique_frames:
            timestamp = frame.get("timestamp", 0)

            # Get surrounding context
            context = self.get_context_for_timestamp(timestamp, segments)

            # Find relevant signals
            relevant_signals = []
            if signals:
                relevant_signals = self.find_relevant_signals(context, signals)

            pair = {
                "video_id": video_id,
                "timestamp": timestamp,
                "frame_path": frame.get("path"),
                "context_text": context,
                "keyword": frame.get("keyword"),
                "transcript_text": frame.get("transcript_text"),
                "relevant_signals": relevant_signals,
                "extraction_method": self._get_extraction_method(frame, manifest),
            }

            aligned_pairs.append(pair)

        # Build result
        result = {
            "video_id": video_id,
            "aligned_at": datetime.now().isoformat(),
            "total_frames": len(aligned_pairs),
            "video_info": signals.get("video_info", {}) if signals else {},
            "market_context": signals.get("market_context", {}) if signals else {},
            "key_insights": signals.get("key_insights", []) if signals else [],
            "pairs": aligned_pairs,
        }

        # Save alignment
        output_path = self.output_dir / f"{video_id}_aligned.json"
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

        print(f"Saved alignment to {output_path}")

        return result

    def _get_extraction_method(self, frame: dict, manifest: Optional[dict]) -> str:
        """Determine which method extracted this frame."""
        if not manifest:
            return "unknown"

        if frame in manifest.get("scene_changes", []):
            return "scene_change"
        elif frame in manifest.get("regular_interval", []):
            return "regular_interval"
        elif frame in manifest.get("keyword_matches", []):
            return "keyword_match"
        return "unknown"

    def align_all_videos(self) -> list[dict]:
        """Align all videos with transcripts and frames."""
        results = []

        # Find all videos with both transcripts and frames
        transcript_ids = set()
        for path in TRANSCRIPTS_DIR.glob("*.json"):
            transcript_ids.add(path.stem)
        for path in TRANSCRIPTS_DIR.glob("*.txt"):
            transcript_ids.add(path.stem)

        frame_ids = set()
        for path in FRAMES_DIR.iterdir():
            if path.is_dir():
                frame_ids.add(path.name)

        # Align videos that have both
        common_ids = transcript_ids & frame_ids
        print(f"Found {len(common_ids)} videos with both transcripts and frames")

        for video_id in sorted(common_ids):
            result = self.align_video(video_id)
            if result:
                results.append(result)

        return results

    def get_all_aligned_pairs(self) -> list[dict]:
        """Load all aligned pairs from all videos."""
        all_pairs = []

        for path in self.output_dir.glob("*_aligned.json"):
            with open(path) as f:
                data = json.load(f)
                all_pairs.extend(data.get("pairs", []))

        return all_pairs

    def export_for_training(self, output_path: Optional[Path] = None) -> Path:
        """
        Export aligned data in a format suitable for RAG training.

        Creates a JSONL file with one record per frame-context pair.
        """
        output_path = output_path or self.output_dir / "training_data.jsonl"

        pairs = self.get_all_aligned_pairs()
        print(f"Exporting {len(pairs)} training pairs...")

        with open(output_path, "w") as f:
            for pair in pairs:
                # Create training record
                record = {
                    "id": f"{pair['video_id']}_{int(pair['timestamp'])}",
                    "video_id": pair["video_id"],
                    "timestamp": pair["timestamp"],
                    "frame_path": pair["frame_path"],
                    "text": pair["context_text"],
                    "signals": pair.get("relevant_signals", []),
                    "metadata": {
                        "keyword": pair.get("keyword"),
                        "method": pair.get("extraction_method"),
                    },
                }
                f.write(json.dumps(record) + "\n")

        print(f"Exported to {output_path}")
        return output_path


def main():
    """Test alignment."""
    aligner = TranscriptFrameAligner()

    # Check what we have
    print("Checking available data...")

    transcript_count = len(list(TRANSCRIPTS_DIR.glob("*.json")))
    signal_count = len(list(SIGNALS_DIR.glob("*_signals.json")))
    frame_dirs = [p for p in FRAMES_DIR.iterdir() if p.is_dir()]

    print(f"  Transcripts: {transcript_count}")
    print(f"  Signal files: {signal_count}")
    print(f"  Frame directories: {len(frame_dirs)}")

    if frame_dirs:
        # Align available videos
        results = aligner.align_all_videos()
        print(f"\nAligned {len(results)} videos")

        # Export for training
        aligner.export_for_training()
    else:
        print("\nNo frames extracted yet. Run extract_frames.py first.")


if __name__ == "__main__":
    main()
