#!/bin/bash
# Start the trading agent

set -e

echo "Starting Trading Agent..."

# Check for .env file
if [ ! -f .env ]; then
    echo "Error: .env file not found. Copy .env.example to .env and fill in your values."
    exit 1
fi

# Load environment variables
export $(grep -v '^#' .env | xargs)

# Check required variables
required_vars=("POLYGON_API_KEY" "TRADIER_ACCOUNT_ID" "TRADIER_ACCESS_TOKEN" "ANTHROPIC_API_KEY" "DB_PASSWORD")
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "Error: $var is not set in .env"
        exit 1
    fi
done

# Start with docker-compose
docker-compose up -d

echo "Trading Agent started!"
echo ""
echo "Services:"
echo "  - Trading Agent: Running"
echo "  - Database: localhost:5432"
echo "  - Redis: localhost:6379"
echo ""
echo "To view logs: docker-compose logs -f trading-agent"
echo "To stop: docker-compose down"
