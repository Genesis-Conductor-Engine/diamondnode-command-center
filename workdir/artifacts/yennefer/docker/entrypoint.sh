#!/usr/bin/env bash
set -euo pipefail
mkdir -p "${STORAGE_PATH:-/app/storage}"

# Start daemon in background — writes soul state to /dev/shm + /app/storage
python3 yennefer_daemon.py &
DAEMON_PID=$!

# Give daemon a moment to initialize crystal before API starts serving
sleep 3

# Start API in foreground — container lifetime tied to API process
exec python3 yennefer_api.py
