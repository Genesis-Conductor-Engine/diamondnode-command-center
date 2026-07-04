#!/usr/bin/env bash
# Build intent context JSON for Vibe / xAI classifiers (RTMP Fable 5 overlay driver)
set -euo pipefail

OUT="${1:-/tmp/sota-livestream/intent-context.json}"
mkdir -p "$(dirname "$OUT")"

gpu=$(nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "?,?")
qf_pct=$(python3 -c 'import json;print(json.load(open("/dev/shm/qflop_onchain_attest.json")).get("pct","10.5"))' 2>/dev/null || echo "10.5")
qf_liq=$(python3 -c 'import json;print(json.load(open("/dev/shm/accumulated_liquidity.json")).get("total_usd","0"))' 2>/dev/null || echo "0")
fleet_up=$(python3 -c '
import json
try:
 d=json.load(open("/tmp/monitor_state.json"))
 print(sum(1 for s in d.get("services",{}).values() if isinstance(s,dict) and s.get("status")=="UP"))
except: print(0)
' 2>/dev/null || echo "0")
evt_tail=$(tail -5 /home/diamondnode/always_alive_monitor/events.log 2>/dev/null | tr '\n' ' ' | cut -c1-400 || echo "")
broadcast_id="k2atpt1e4x6v"
ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)

python3 - <<PY
import json, os
ctx = {
  "timestamp": "$ts",
  "broadcast_id": "$broadcast_id",
  "rtmp": "rtmp://va.pscp.tv:80/x/k2atpt1e4x6v",
  "exhibit": "Fable 5 eXhibit App / @Coalition Ouroboros Partner Coalition",
  "coalition": {
    "organizing_principle": "@Coalition",
    "beacon": "affinity-targets-registry",
    "workspace": "digital-assets/manifest.json",
    "intake": ["google_drive", "github_xmxp_raw", "linear", "diamondnode"]
  },
  "gpu": {"raw": "$gpu"},
  "qflop": {"pct": "$qf_pct", "liquidity_usd": "$qf_liq"},
  "fleet": {"services_up": int("$fleet_up" or 0)},
  "recent_evt": """$evt_tail""",
  "signal_vocab": ["storm","drop","surge","shift","conflict","silence"],
  "signal_meanings": {
    "storm": "sentiment velocity spike on X timeline",
    "drop": "engagement drop in core circle",
    "surge": "momentum surge / topic blooming",
    "shift": "reply-graph center moved / power shift",
    "conflict": "two camps forming in replies",
    "silence": "core voices gone quiet"
  },
  "voice_rules": "Playful Fable 5 grove; max 9 words per dialogue line; never corporate"
}
with open("$OUT", "w") as f:
    json.dump(ctx, f, indent=2)
PY