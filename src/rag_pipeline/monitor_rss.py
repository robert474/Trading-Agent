#!/usr/bin/env python3
"""
RSS Feed Monitor for Bill Fanter's YouTube Channel.

Monitors the channel RSS feed for new videos and triggers the pipeline.
"""

import asyncio
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional
import aiohttp
import xml.etree.ElementTree as ET

# Bill Fanter's YouTube Channel ID
BILL_FANTER_CHANNEL_ID = "UC_A7K2b3qkYHvYGmJrMCPNQ"
RSS_FEED_URL = f"https://www.youtube.com/feeds/videos.xml?channel_id={BILL_FANTER_CHANNEL_ID}"

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "rag_pipeline"
PROCESSED_VIDEOS_FILE = DATA_DIR / "processed_videos.json"


class YouTubeRSSMonitor:
    """Monitor YouTube RSS feed for new Bill Fanter videos."""

    def __init__(self, channel_id: str = BILL_FANTER_CHANNEL_ID):
        self.channel_id = channel_id
        self.rss_url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
        self.processed_videos = self._load_processed_videos()

        # Ensure data directory exists
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    def _load_processed_videos(self) -> dict:
        """Load list of already processed video IDs."""
        if PROCESSED_VIDEOS_FILE.exists():
            with open(PROCESSED_VIDEOS_FILE) as f:
                return json.load(f)
        return {"processed": [], "last_check": None}

    def _save_processed_videos(self):
        """Save processed videos list."""
        PROCESSED_VIDEOS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(PROCESSED_VIDEOS_FILE, "w") as f:
            json.dump(self.processed_videos, f, indent=2)

    async def fetch_feed(self) -> list[dict]:
        """Fetch and parse the RSS feed."""
        async with aiohttp.ClientSession() as session:
            async with session.get(self.rss_url) as resp:
                if resp.status != 200:
                    raise Exception(f"Failed to fetch RSS feed: {resp.status}")

                xml_content = await resp.text()

        # Parse XML
        root = ET.fromstring(xml_content)

        # Define namespaces
        namespaces = {
            "atom": "http://www.w3.org/2005/Atom",
            "yt": "http://www.youtube.com/xml/schemas/2015",
            "media": "http://search.yahoo.com/mrss/",
        }

        videos = []
        for entry in root.findall("atom:entry", namespaces):
            video_id = entry.find("yt:videoId", namespaces)
            title = entry.find("atom:title", namespaces)
            published = entry.find("atom:published", namespaces)
            link = entry.find("atom:link", namespaces)

            # Get thumbnail from media:group
            media_group = entry.find("media:group", namespaces)
            thumbnail_url = None
            description = None
            if media_group is not None:
                thumbnail = media_group.find("media:thumbnail", namespaces)
                if thumbnail is not None:
                    thumbnail_url = thumbnail.get("url")
                desc = media_group.find("media:description", namespaces)
                if desc is not None:
                    description = desc.text

            if video_id is not None:
                videos.append({
                    "video_id": video_id.text,
                    "title": title.text if title is not None else "",
                    "published": published.text if published is not None else "",
                    "url": f"https://www.youtube.com/watch?v={video_id.text}",
                    "thumbnail": thumbnail_url,
                    "description": description,
                })

        return videos

    async def check_for_new_videos(self) -> list[dict]:
        """Check for new videos that haven't been processed."""
        videos = await self.fetch_feed()
        new_videos = []

        for video in videos:
            if video["video_id"] not in self.processed_videos["processed"]:
                new_videos.append(video)

        # Update last check time
        self.processed_videos["last_check"] = datetime.now().isoformat()
        self._save_processed_videos()

        return new_videos

    def mark_as_processed(self, video_id: str):
        """Mark a video as processed."""
        if video_id not in self.processed_videos["processed"]:
            self.processed_videos["processed"].append(video_id)
            self._save_processed_videos()

    def is_trading_related(self, video: dict) -> bool:
        """
        Check if video is trading-related (not a livestream replay or off-topic).

        Bill Fanter's trading videos typically have titles like:
        - "Weekly Watchlist"
        - "Market Analysis"
        - "Supply and Demand"
        - Stock ticker mentions (AAPL, TSLA, SPY, etc.)
        """
        title = video.get("title", "").lower()
        description = (video.get("description") or "").lower()

        # Keywords that indicate trading content
        trading_keywords = [
            "watchlist", "weekly", "market", "stocks", "trading",
            "supply", "demand", "breakout", "spy", "qqq", "nasdaq",
            "s&p", "bulls", "bears", "calls", "puts", "options",
            "resistance", "support", "levels", "chart", "technical",
        ]

        # Check title and description for trading keywords
        combined = f"{title} {description}"
        for keyword in trading_keywords:
            if keyword in combined:
                return True

        # Check for stock tickers (2-5 uppercase letters)
        import re
        tickers = re.findall(r'\b[A-Z]{2,5}\b', video.get("title", ""))
        common_tickers = {"AAPL", "TSLA", "NVDA", "AMD", "META", "GOOGL", "AMZN", "MSFT", "SPY", "QQQ"}
        if any(t in common_tickers for t in tickers):
            return True

        return False

    async def get_recent_trading_videos(self, limit: int = 10) -> list[dict]:
        """Get recent trading-related videos."""
        videos = await self.fetch_feed()
        trading_videos = [v for v in videos if self.is_trading_related(v)]
        return trading_videos[:limit]


async def main():
    """Test the RSS monitor."""
    monitor = YouTubeRSSMonitor()

    print("Fetching Bill Fanter's YouTube feed...")
    videos = await monitor.fetch_feed()

    print(f"\nFound {len(videos)} videos in feed:")
    for video in videos[:5]:
        print(f"\n  {video['title']}")
        print(f"  ID: {video['video_id']}")
        print(f"  Published: {video['published']}")
        print(f"  Trading related: {monitor.is_trading_related(video)}")

    print("\n\nChecking for new (unprocessed) videos...")
    new_videos = await monitor.check_for_new_videos()
    print(f"New videos to process: {len(new_videos)}")


if __name__ == "__main__":
    asyncio.run(main())
