#!/bin/bash
# Monitor Bill Fanter's YouTube channel for new videos
# Automatically transcribes and extracts trading signals
#
# Run with: ./scripts/monitor_bill_fanter.sh
# Or add to crontab for automatic checking

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( dirname "$SCRIPT_DIR" )"

cd "$PROJECT_DIR"

# Bill Fanter's channel ID
CHANNEL_ID="UCIisJusP9c2idAqKZGhdX-g"

echo "================================================"
echo "  BILL FANTER VIDEO MONITOR"
echo "  Channel: $CHANNEL_ID"
echo "  Time: $(date)"
echo "================================================"
echo ""

# Check for new videos and transcribe
python3 scripts/transcribe_youtube.py --channel-id "$CHANNEL_ID" --monitor --interval 300

echo ""
echo "Monitor stopped at $(date)"
