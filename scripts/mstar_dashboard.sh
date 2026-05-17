#!/usr/bin/env bash
# MSTAR Pro Dashboard launcher
# Ensures the dashboard server is running on port 18792

PYTHON="C:/Users/41228/AppData/Local/Python/pythoncore-3.14-64/python.exe"
AGENT_DIR="C:/Users/41228/AppData/Local/hermes/hermes-agent"
SERVER_SCRIPT="$AGENT_DIR/mstar_core/observability/dashboard_server.py"
LOG_DIR="$AGENT_DIR/logs"
PID_FILE="$AGENT_DIR/mstar_dashboard.pid"

mkdir -p "$LOG_DIR"

# Check if already running
if curl -s -o /dev/null -w "%{http_code}" http://localhost:18792/health --connect-timeout 2 2>/dev/null | grep -q "200"; then
    echo "[$(date)] MSTAR Dashboard already running on port 18792"
    exit 0
fi

# Kill stale pid
if [ -f "$PID_FILE" ]; then
    stale_pid=$(cat "$PID_FILE")
    kill "$stale_pid" 2>/dev/null
fi

echo "[$(date)] Starting MSTAR Dashboard..."
cd "$AGENT_DIR"
nohup "$PYTHON" "$SERVER_SCRIPT" >> "$LOG_DIR/mstar_dashboard.log" 2>&1 &
echo $! > "$PID_FILE"
echo "[$(date)] MSTAR Dashboard started with PID $(cat $PID_FILE)"
