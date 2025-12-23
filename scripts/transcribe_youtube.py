#!/usr/bin/env python3
"""
Quick-start script for the YouTube Auto-Transcriber.

Usage:
    # Find channel ID from URL
    python scripts/transcribe_youtube.py --find-channel "https://youtube.com/@BillFanterOptions"

    # Transcribe a single video
    python scripts/transcribe_youtube.py --video "https://www.youtube.com/watch?v=VIDEO_ID"

    # Monitor a channel for new videos
    python scripts/transcribe_youtube.py --channel-id "UC..." --monitor

    # Process backlog of videos
    python scripts/transcribe_youtube.py --channel-id "UC..." --backlog 10
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

import argparse


async def find_channel_id(url: str):
    """Find and display channel ID from URL."""
    from trading_agent.transcriber.rss_monitor import RSSMonitor

    print(f"\nSearching for channel ID in: {url}")
    channel_id = await RSSMonitor.find_channel_id(url)

    if channel_id:
        print(f"\n✅ Found Channel ID: {channel_id}")
        print(f"\n📡 RSS Feed URL:")
        print(f"   {RSSMonitor.get_channel_rss_url(channel_id)}")
        print(f"\n🚀 To monitor this channel:")
        print(f"   python scripts/transcribe_youtube.py --channel-id {channel_id} --monitor")
    else:
        print("\n❌ Could not find channel ID")
        print("   Try visiting the channel and looking for 'channel/' in the URL")


async def transcribe_video(url: str, whisper_model: str = "base"):
    """Transcribe a single video."""
    from trading_agent.transcriber.auto_transcriber import AutoTranscriber
    import json

    print(f"\n🎬 Processing video: {url}")

    transcriber = AutoTranscriber(
        channel_ids=[],
        whisper_model=whisper_model,
    )

    signals = await transcriber.process_video_url(url)

    if signals:
        print("\n" + "=" * 60)
        print("📊 EXTRACTED TRADING SIGNALS")
        print("=" * 60)

        # Summary
        print(f"\n📝 Summary: {signals.get('summary', 'N/A')}")
        print(f"📈 Educational Value: {signals.get('educational_value', 'N/A')}")

        # Trade signals
        trade_signals = signals.get("trade_signals", [])
        if trade_signals:
            print(f"\n🎯 Trade Signals ({len(trade_signals)}):")
            for i, sig in enumerate(trade_signals, 1):
                print(f"\n  {i}. {sig.get('symbol', 'N/A')} - {sig.get('direction', 'N/A').upper()}")
                if sig.get('strike'):
                    print(f"     Strike: ${sig.get('strike')} {sig.get('entry_type', '')} exp {sig.get('expiration', 'N/A')}")
                if sig.get('entry_price'):
                    print(f"     Entry: ${sig.get('entry_price')} | Stop: ${sig.get('stop_loss', 'N/A')} | Target: ${sig.get('target', 'N/A')}")
                if sig.get('zone_type'):
                    print(f"     Zone: {sig.get('zone_type')} at ${sig.get('zone_level', 'N/A')}")
                print(f"     Reasoning: {sig.get('reasoning', 'N/A')}")

        # Zone levels
        zones = signals.get("zone_levels", [])
        if zones:
            print(f"\n📍 Zone Levels ({len(zones)}):")
            for zone in zones:
                print(f"  • {zone.get('symbol')} {zone.get('zone_type')}: ${zone.get('zone_low', 0):.2f} - ${zone.get('zone_high', 0):.2f} ({zone.get('timeframe', '')})")

        # Trade outcomes
        outcomes = signals.get("trade_outcomes", [])
        if outcomes:
            print(f"\n📈 Trade Outcomes ({len(outcomes)}):")
            for out in outcomes:
                emoji = "✅" if out.get("result") == "win" else "❌"
                print(f"  {emoji} {out.get('symbol')} {out.get('direction')}: ${out.get('entry_price', 0):.2f} → ${out.get('exit_price', 0):.2f} ({out.get('pnl_percent', 0):.1f}%)")

        # Key insights
        insights = signals.get("key_insights", [])
        if insights:
            print(f"\n💡 Key Insights:")
            for insight in insights:
                print(f"  • {insight}")

        # Save full JSON
        output_path = Path("data/transcriber/signals") / f"{signals.get('video_id', 'output')}_signals.json"
        print(f"\n📁 Full JSON saved to: {output_path}")

    else:
        print("\n❌ Failed to process video")


async def monitor_channel(channel_id: str, interval: int = 3600, whisper_model: str = "base"):
    """Monitor a channel for new videos."""
    from trading_agent.transcriber.auto_transcriber import AutoTranscriber

    print(f"\n📡 Starting channel monitor")
    print(f"   Channel ID: {channel_id}")
    print(f"   Check interval: {interval} seconds")
    print(f"   Whisper model: {whisper_model}")
    print(f"\n   Press Ctrl+C to stop\n")

    transcriber = AutoTranscriber(
        channel_ids=[channel_id],
        check_interval=interval,
        whisper_model=whisper_model,
    )

    try:
        await transcriber.start()
    except KeyboardInterrupt:
        print("\n\n👋 Stopping monitor...")


async def process_backlog(channel_id: str, limit: int, whisper_model: str = "base"):
    """Process backlog of videos."""
    from trading_agent.transcriber.auto_transcriber import AutoTranscriber

    print(f"\n📚 Processing backlog")
    print(f"   Channel ID: {channel_id}")
    print(f"   Limit: {limit} videos")
    print(f"   Whisper model: {whisper_model}\n")

    transcriber = AutoTranscriber(
        channel_ids=[channel_id],
        whisper_model=whisper_model,
    )

    await transcriber.process_backlog(limit=limit)
    print("\n✅ Backlog processing complete!")

    # Show summary of extracted examples
    examples = transcriber.get_all_examples()
    if examples:
        print(f"\n📊 Total trade examples extracted: {len(examples)}")


async def main():
    parser = argparse.ArgumentParser(
        description="YouTube Auto-Transcriber for Bill Fanter's Trading Videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Find channel ID
  python scripts/transcribe_youtube.py --find-channel "https://youtube.com/@ChannelName"

  # Transcribe single video
  python scripts/transcribe_youtube.py --video "https://youtube.com/watch?v=VIDEO_ID"

  # Monitor channel
  python scripts/transcribe_youtube.py --channel-id "UCxxxxxxx" --monitor

  # Process backlog
  python scripts/transcribe_youtube.py --channel-id "UCxxxxxxx" --backlog 5
        """
    )

    parser.add_argument(
        "--find-channel",
        metavar="URL",
        help="Find channel ID from YouTube channel URL",
    )
    parser.add_argument(
        "--video",
        metavar="URL",
        help="Transcribe a single YouTube video",
    )
    parser.add_argument(
        "--channel-id",
        metavar="ID",
        help="YouTube channel ID to monitor",
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Start monitoring mode (use with --channel-id)",
    )
    parser.add_argument(
        "--backlog",
        type=int,
        metavar="N",
        help="Process N backlog videos (use with --channel-id)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=3600,
        help="Check interval in seconds for monitor mode (default: 3600)",
    )
    parser.add_argument(
        "--whisper-model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base)",
    )

    args = parser.parse_args()

    # Setup logging
    from trading_agent.monitoring.logger import setup_logging
    setup_logging(level="INFO")

    # Route to appropriate function
    if args.find_channel:
        await find_channel_id(args.find_channel)

    elif args.video:
        await transcribe_video(args.video, args.whisper_model)

    elif args.channel_id:
        if args.monitor:
            await monitor_channel(args.channel_id, args.interval, args.whisper_model)
        elif args.backlog:
            await process_backlog(args.channel_id, args.backlog, args.whisper_model)
        else:
            print("Error: Use --monitor or --backlog with --channel-id")
            parser.print_help()

    else:
        parser.print_help()
        print("\n💡 Quick start:")
        print("   1. Find Bill Fanter's channel ID:")
        print('      python scripts/transcribe_youtube.py --find-channel "https://youtube.com/@BillFanterOptions"')
        print("\n   2. Monitor for new videos:")
        print('      python scripts/transcribe_youtube.py --channel-id "UC..." --monitor')


if __name__ == "__main__":
    asyncio.run(main())
