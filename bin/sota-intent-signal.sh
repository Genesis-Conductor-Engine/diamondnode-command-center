#!/usr/bin/env bash
# Poll intent signals for Fable 5 RTMP (heuristic fast path; vibe/xAI on interval)
set -euo pipefail

export PATH="/home/diamondnode/bin:/home/diamondnode/venv312/bin:/usr/local/bin:/usr/bin:/bin"
source /home/diamondnode/load-env.sh 2>/dev/null || true

INTERVAL="${SOTA_INTENT_INTERVAL:-30}"
VIBE_EVERY="${SOTA_VIBE_INTENT_EVERY:-10}"  # every N cycles try vibe/xAI
STATE_DIR="/tmp/sota-livestream"
cycle=0

/home/diamondnode/bin/sota-intent-context.sh

echo "[$(date -Iseconds)] sota-intent-signal started (interval ${INTERVAL}s, vibe every ${VIBE_EVERY} cycles)" >&2

while true; do
  cycle=$((cycle + 1))
  /home/diamondnode/bin/sota-intent-context.sh

  if [ -n "${XAI_API_KEY:-}" ] || [ $((cycle % VIBE_EVERY)) -eq 0 ]; then
    /home/diamondnode/yennefer_venv/bin/python /home/diamondnode/bin/sota-intent-signal.py >/dev/null 2>&1 \
      || /home/diamondnode/venv312/bin/python /home/diamondnode/bin/sota-intent-signal.py >/dev/null 2>&1 \
      || python3 /home/diamondnode/bin/sota-intent-signal.py >/dev/null 2>&1 \
      || true
  else
    # Fast heuristic only (no vibe subprocess)
    python3 - <<'PY' || true
import json, subprocess
subprocess.run(["/home/diamondnode/bin/sota-intent-context.sh"], check=False)
from pathlib import Path
import importlib.util
spec = importlib.util.spec_from_file_location("sis", "/home/diamondnode/bin/sota-intent-signal.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
ctx = mod.load_context()
sig = mod.classify_heuristic(ctx)
mod.write_signal(sig)
PY
  fi

  if [ -f "${STATE_DIR}/intent-signal.json" ]; then
    sig=$(python3 -c 'import json;print(json.load(open("/tmp/sota-livestream/intent-signal.json")).get("signal","?"))' 2>/dev/null || echo "?")
    echo "[$(date -Iseconds)] intent signal -> ${sig}" >&2
  fi

  /home/diamondnode/venv312/bin/python /home/diamondnode/bin/coalition-beacon.py >/dev/null 2>&1 || true
  /home/diamondnode/venv312/bin/python /home/diamondnode/bin/mic-drop-contracts.py >/dev/null 2>&1 || true
  /home/diamondnode/venv312/bin/python /home/diamondnode/bin/dunk-tank.py >/dev/null 2>&1 || true

  # Thermo evolutionary epoch every 8 cycles (~4 min @ 30s)
  if [ $((cycle % 8)) -eq 0 ]; then
    /home/diamondnode/thermodynamic-daemon/run_epoch.sh chr17:41234470:A>G base --no-alphagenome >/dev/null 2>&1 || true
  fi

  if [ $((cycle % 4)) -eq 0 ]; then
    /home/diamondnode/bin/coalition-kv-push.sh >/dev/null 2>&1 || true
  fi

  sleep "$INTERVAL"
done