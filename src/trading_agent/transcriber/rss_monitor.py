"""
RSS Feed Monitor for YouTube Channels.

Monitors YouTube RSS feeds for new videos and tracks
which videos have been processed.
"""

import asyncio
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree

import aiohttp
import structlog

logger = structlog.get_logger()


class RSSMonitor:
    """
    Monitor YouTube RSS feeds for new videos.

    YouTube RSS feed format:
    https://www.youtube.com/feeds/videos.xml?channel_id=CHANNEL_ID

    To find channel ID:
    1. Go to the YouTube channel
    2. View page source (Ctrl+U)
    3. Search for "channel_id" or "externalId"
    """

    # YouTube RSS feed base URL
    YOUTUBE_RSS_BASE = "https://www.youtube.com/feeds/videos.xml"

    def __init__(
        self,
        channel_ids: list[str],
        data_dir: Optional[Path] = None,
        check_interval: int = 3600,  # 1 hour
    ):
        """
        Initialize RSS monitor.

        Args:
            channel_ids: List of YouTube channel IDs to monitor
            data_dir: Directory to store processed video tracking
            check_interval: Seconds between feed checks
        """
        self.channel_ids = channel_ids
        self.data_dir = data_dir or Path("data/transcriber")
        self.check_interval = check_interval

        # Ensure data directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Track processed videos
        self._processed_file = self.data_dir / "processed_videos.json"
        self._processed: set[str] = self._load_processed()

        # Callbacks for new videos
        self._callbacks: list = []

    def _load_processed(self) -> set[str]:
        """Load set of processed video IDs."""
        if self._processed_file.exists():
            try:
                with open(self._processed_file) as f:
                    data = json.load(f)
                    return set(data.get("processed", []))
            except Exception as e:
                logger.warning("Failed to load processed videos", error=str(e))
        return set()

    def _save_processed(self) -> None:
        """Save processed video IDs."""
        try:
            with open(self._processed_file, "w") as f:
                json.dump({"processed": list(self._processed)}, f, indent=2)
        except Exception as e:
            logger.error("Failed to save processed videos", error=str(e))

    def mark_processed(self, video_id: str) -> None:
        """Mark a video as processed."""
        self._processed.add(video_id)
        self._save_processed()

    def is_processed(self, video_id: str) -> bool:
        """Check if video has been processed."""
        return video_id in self._processed

    def on_new_video(self, callback) -> None:
        """Register callback for new videos."""
        self._callbacks.append(callback)

    async def fetch_feed(self, channel_id: str) -> list[dict]:
        """
        Fetch and parse YouTube RSS feed for a channel.

        Returns list of video entries with:
        - video_id
        - title
        - published
        - url
        - channel_id
        - channel_name
        """
        url = f"{self.YOUTUBE_RSS_BASE}?channel_id={channel_id}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=30) as resp:
                    if resp.status != 200:
                        logger.error(
                            "Failed to fetch RSS feed",
                            channel_id=channel_id,
                            status=resp.status,
                        )
                        return []

                    content = await resp.text()

            # Parse XML
            root = ElementTree.fromstring(content)

            # XML namespaces used by YouTube RSS
            ns = {
                "atom": "http://www.w3.org/2005/Atom",
                "yt": "http://www.youtube.com/xml/schemas/2015",
                "media": "http://search.yahoo.com/mrss/",
            }

            videos = []

            # Get channel name from feed title
            channel_name = ""
            title_elem = root.find("atom:title", ns)
            if title_elem is not None and title_elem.text:
                channel_name = title_elem.text

            # Parse entries
            for entry in root.findall("atom:entry", ns):
                video_id_elem = entry.find("yt:videoId", ns)
                title_elem = entry.find("atom:title", ns)
                published_elem = entry.find("atom:published", ns)
                link_elem = entry.find("atom:link", ns)

                if video_id_elem is None or video_id_elem.text is None:
                    continue

                video_id = video_id_elem.text
                title = title_elem.text if title_elem is not None else "Untitled"
                published = published_elem.text if published_elem is not None else ""
                url = link_elem.get("href", "") if link_elem is not None else ""

                # Parse published date
                pub_date = None
                if published:
                    try:
                        pub_date = datetime.fromisoformat(published.replace("Z", "+00:00"))
                    except ValueError:
                        pass

                videos.append({
                    "video_id": video_id,
                    "title": title,
                    "published": pub_date,
                    "published_str": published,
                    "url": url or f"https://www.youtube.com/watch?v={video_id}",
                    "channel_id": channel_id,
                    "channel_name": channel_name,
                })

            logger.info(
                "Fetched RSS feed",
                channel_id=channel_id,
                channel_name=channel_name,
                videos=len(videos),
            )

            return videos

        except Exception as e:
            logger.error("Error fetching RSS feed", channel_id=channel_id, error=str(e))
            return []

    async def check_for_new_videos(self) -> list[dict]:
        """
        Check all channels for new videos.

        Returns list of new (unprocessed) videos.
        """
        all_new = []

        for channel_id in self.channel_ids:
            videos = await self.fetch_feed(channel_id)

            for video in videos:
                if not self.is_processed(video["video_id"]):
                    all_new.append(video)

                    # Trigger callbacks
                    for callback in self._callbacks:
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                await callback(video)
                            else:
                                callback(video)
                        except Exception as e:
                            logger.error(
                                "Callback error",
                                video_id=video["video_id"],
                                error=str(e),
                            )

        if all_new:
            logger.info("Found new videos", count=len(all_new))

        return all_new

    async def start_monitoring(self) -> None:
        """Start continuous monitoring loop."""
        logger.info(
            "Starting RSS monitoring",
            channels=len(self.channel_ids),
            interval=self.check_interval,
        )

        while True:
            try:
                await self.check_for_new_videos()
            except Exception as e:
                logger.error("Monitoring error", error=str(e))

            await asyncio.sleep(self.check_interval)

    @staticmethod
    def get_channel_rss_url(channel_id: str) -> str:
        """Get RSS feed URL for a channel ID."""
        return f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"

    @staticmethod
    async def find_channel_id(channel_url: str) -> Optional[str]:
        """
        Extract channel ID from a YouTube channel URL.

        Works with:
        - https://www.youtube.com/channel/UC...
        - https://www.youtube.com/@username
        - https://www.youtube.com/c/channelname
        """
        try:
            # If URL already contains channel ID
            if "/channel/" in channel_url:
                parts = channel_url.split("/channel/")
                if len(parts) > 1:
                    return parts[1].split("/")[0].split("?")[0]

            # Otherwise fetch the page and extract channel ID
            async with aiohttp.ClientSession() as session:
                async with session.get(channel_url, timeout=30) as resp:
                    if resp.status != 200:
                        return None

                    html = await resp.text()

            # Look for channel ID in page source
            import re

            # Try different patterns
            patterns = [
                r'"channelId":"([^"]+)"',
                r'"externalId":"([^"]+)"',
                r'channel_id=([^"&]+)',
                r'data-channel-external-id="([^"]+)"',
            ]

            for pattern in patterns:
                match = re.search(pattern, html)
                if match:
                    return match.group(1)

            return None

        except Exception as e:
            logger.error("Error finding channel ID", url=channel_url, error=str(e))
            return None
