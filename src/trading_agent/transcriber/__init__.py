"""
YouTube Auto-Transcriber for Bill Fanter's Trading Videos.

Monitors RSS feeds, downloads new videos, transcribes them,
and extracts trading signals for few-shot learning.
"""

from .rss_monitor import RSSMonitor
from .video_downloader import VideoDownloader
from .transcriber import Transcriber
from .signal_extractor import SignalExtractor
from .auto_transcriber import AutoTranscriber

__all__ = [
    "RSSMonitor",
    "VideoDownloader",
    "Transcriber",
    "SignalExtractor",
    "AutoTranscriber",
]
