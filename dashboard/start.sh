#!/bin/bash
# Start the Trading Dashboard
# Usage: ./dashboard/start.sh

echo "================================================"
echo "  BILL FANTER TRADING DASHBOARD"
echo "================================================"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( dirname "$SCRIPT_DIR" )"

# Check for required dependencies
echo "[1/3] Checking dependencies..."

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required"
    exit 1
fi

# Check Node
if ! command -v npm &> /dev/null; then
    echo "Error: npm is required"
    exit 1
fi

# Install Python dependencies if needed
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo "Installing FastAPI..."
    pip3 install fastapi uvicorn aiohttp
fi

echo "  Dependencies OK"
echo ""

# Start API server
echo "[2/3] Starting API server on http://localhost:8000..."
cd "$PROJECT_DIR"
python3 -m uvicorn dashboard.api:app --host 0.0.0.0 --port 8000 &
API_PID=$!

# Wait for API to start
sleep 2

# Start frontend
echo "[3/3] Starting frontend on http://localhost:3000..."
cd "$SCRIPT_DIR/frontend"
npm run dev &
FRONTEND_PID=$!

echo ""
echo "================================================"
echo "  Dashboard is running!"
echo "  "
echo "  Frontend: http://localhost:3000"
echo "  API:      http://localhost:8000"
echo "  API Docs: http://localhost:8000/docs"
echo "  "
echo "  Press Ctrl+C to stop"
echo "================================================"

# Handle shutdown
cleanup() {
    echo ""
    echo "Shutting down..."
    kill $API_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

# Wait for processes
wait
