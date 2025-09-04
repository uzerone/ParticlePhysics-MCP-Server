#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="$ROOT_DIR/mcp_config.json"

echo "Killing processes bound to 127.0.0.1 TCP/UDP ports..."
set +e
PIDS=$(lsof -iTCP@127.0.0.1 -iUDP@127.0.0.1 -sTCP:LISTEN -t 2>/dev/null)
if [ -n "$PIDS" ]; then
  echo "Killing PIDs: $PIDS"
  echo "$PIDS" | xargs -I {} kill -9 {} 2>/dev/null || true
fi

# Ensure MCP Inspector proxy and UI ports are free (6277 proxy, 6274 UI)
for PORT in 6277 6274; do
  PORT_PIDS=$(lsof -ti tcp:$PORT -sTCP:LISTEN 2>/dev/null)
  if [ -n "$PORT_PIDS" ]; then
    echo "Killing processes on port $PORT: $PORT_PIDS"
    echo "$PORT_PIDS" | xargs -I {} kill -9 {} 2>/dev/null || true
  fi
done
set -e

echo "Restarting MCP Inspector... (local server)"
echo "Using config: $CONFIG (server: particlephysics)"

export DANGEROUSLY_OMIT_AUTH=true

echo "Starting MCP inspector..."
npx --yes @modelcontextprotocol/inspector --config "$CONFIG" --open


