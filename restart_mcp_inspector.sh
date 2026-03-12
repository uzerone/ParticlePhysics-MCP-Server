#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="$ROOT_DIR/mcp_config.json"

echo "Ensuring MCP Inspector ports are free (6277 proxy, 6274 UI)..."
set +e
for PORT in 6277 6274; do
  PORT_PIDS=$(lsof -ti tcp:$PORT -sTCP:LISTEN 2>/dev/null)
  if [ -n "$PORT_PIDS" ]; then
    echo "Stopping processes on port $PORT: $PORT_PIDS"
    echo "$PORT_PIDS" | xargs -I {} kill -TERM {} 2>/dev/null || true
    sleep 0.5
    PORT_PIDS=$(lsof -ti tcp:$PORT -sTCP:LISTEN 2>/dev/null)
    if [ -n "$PORT_PIDS" ]; then
      echo "Force stopping remaining processes on port $PORT: $PORT_PIDS"
      echo "$PORT_PIDS" | xargs -I {} kill -KILL {} 2>/dev/null || true
    fi
  fi
done
set -e

echo "Restarting MCP Inspector... (local server)"
echo "Using config: $CONFIG (server: particlephysics)"

export DANGEROUSLY_OMIT_AUTH=true

if ! command -v npx >/dev/null 2>&1; then
  echo "Error: npx not found. Please install Node.js (which includes npx)."
  exit 1
fi

echo "Starting MCP inspector..."
echo "Open http://localhost:5173 in your browser if it does not open automatically."
npx --yes @modelcontextprotocol/inspector --config "$CONFIG" --open


