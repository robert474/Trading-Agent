"""
RAG Pipeline for Bill Fanter Trading Methodology

This pipeline:
1. Monitors Bill Fanter's YouTube channel for new videos
2. Downloads and transcribes videos with timestamps
3. Extracts trading signals and levels using Claude
4. Extracts frames at key moments (scene changes, chart displays)
5. Aligns transcripts to frames for training data
6. Stores everything in ChromaDB for RAG retrieval

Components:
- YouTubeRSSMonitor: Monitors for new videos
- VideoDownloader: Downloads videos via yt-dlp
- WhisperTranscriber: Transcribes audio with timestamps
- FrameExtractor: Extracts key frames from videos
- TranscriptFrameAligner: Aligns frames with transcript context
- BillFanterRAG: Vector database for semantic search
- RAGPipeline: Full orchestration pipeline

Usage:
    from rag_pipeline import RAGPipeline

    pipeline = RAGPipeline()
    await pipeline.process_new_videos()

    # Or query the RAG database directly
    from rag_pipeline import BillFanterRAG

    rag = BillFanterRAG()
    results = rag.query("supply and demand zone trading")
"""

from .monitor_rss import YouTubeRSSMonitor
from .download_video import VideoDownloader
from .transcribe import WhisperTranscriber
from .extract_frames import FrameExtractor
from .align_data import TranscriptFrameAligner
from .rag_database import BillFanterRAG
from .pipeline import RAGPipeline

__all__ = [
    "YouTubeRSSMonitor",
    "VideoDownloader",
    "WhisperTranscriber",
    "FrameExtractor",
    "TranscriptFrameAligner",
    "BillFanterRAG",
    "RAGPipeline",
]
