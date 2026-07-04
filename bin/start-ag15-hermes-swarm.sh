#!/usr/bin/env bash
# Pin AG15 diamondnodebot Hermes swarm — openFDA substrate research + double-loop verify
set -euo pipefail

export PATH="/home/diamondnode/bin:/home/diamondnode/venv312/bin:$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin"
source /home/diamondnode/load-env.sh 2>/dev/null || true

SWARM_DIR="/home/diamondnode/genesis_conductor_engine/swarm"
LOG_DIR="/tmp/ag15-research/logs"
mkdir -p "$LOG_DIR"

echo "[$(date -Iseconds)] AG15 openFDA research pull..." >&2
/home/diamondnode/venv312/bin/python /home/diamondnode/thermodynamic-daemon/ag15_openfda_research.py \
  | tee "$LOG_DIR/openfda-$(date +%s).json" >&2

echo "[$(date -Iseconds)] AG15 double-loop verification..." >&2
/home/diamondnode/venv312/bin/python /home/diamondnode/thermodynamic-daemon/ag15_double_loop_verifier.py \
  | tee "$LOG_DIR/verify-$(date +%s).json" >&2

# Pin 3 Hermes workers (memory bus — no Redis required)
for i in 01 02 03; do
  id="diamondnodebot_ag15_hermes_${i}"
  if pgrep -f "run_agent_worker.py --agent-type hermes --agent-id ${id}" >/dev/null 2>&1; then
    echo "[ag15] ${id} already running" >&2
    continue
  fi
  nohup python3 "$SWARM_DIR/run_agent_worker.py" \
    --agent-type hermes \
    --agent-id "$id" \
    --backend memory \
    >> "$LOG_DIR/${id}.log" 2>&1 &
  echo "[ag15] started ${id} pid=$!" >&2
done

echo "[$(date -Iseconds)] AG15 swarm pinned. Endpoints:" >&2
python3 -c 'import json;print(json.dumps(json.load(open("/home/diamondnode/genesis_conductor_engine/swarm/ag15_diamondnodebot_swarm.json"))["endpoints"],indent=2))' >&2