#!/usr/bin/env bash
# Build intent context JSON for Vibe / xAI classifiers (RTMP Fable 5 overlay driver)
set -euo pipefail

OUT="${1:-/tmp/sota-livestream/intent-context.json}"
mkdir -p "$(dirname "$OUT")"

# nvidia-smi writes its driver-missing banner to STDOUT, not stderr, and `head`
# is the last stage of the pipe so its exit status masks nvidia-smi's — the
# `||` fallback below never fired. The banner ("...it couldn't communicate...")
# then reached the generator heredoc, where the apostrophe terminated the Python
# string literal early and the service crashlooped on SyntaxError every 8s.
# Validate the shape instead of trusting the exit status.
gpu=$(nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 || true)
if ! printf '%s' "$gpu" | grep -qE '^[0-9]+, *[0-9]+, *[0-9]+$'; then
  gpu="unavailable"
fi

# qflop_onchain_attest.json / accumulated_liquidity.json don't exist on this box (verified 2026-08-02) --
# every read from them silently fell back to the "10.5" / "0" defaults below. Read the files that are
# actually live (liquidity_flywheel.json, wqflop_liquidity.json) first; keep the old paths as a
# last-resort fallback in case they reappear.
read -r qf_pct qf_liq qf_phase qf_goal_met <<<"$(python3 -c '
import json

def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None

flywheel = load("/dev/shm/liquidity_flywheel.json")
wqflop = load("/dev/shm/wqflop_liquidity.json")
attest = load("/dev/shm/qflop_onchain_attest.json")
accum = load("/dev/shm/accumulated_liquidity.json")

pct, liq, phase, goal_met = "10.5", "0", "?", "?"
if flywheel:
    pct = str(flywheel.get("progress_pct", pct))
    liq = str(flywheel.get("liquidity_usd", liq))
    phase = str(flywheel.get("phase", phase))
    goal_met = str(flywheel.get("goal_met", goal_met))
elif wqflop:
    liq = str(wqflop.get("liquidity_usd", liq))
    goal_met = str(wqflop.get("goal_met", goal_met))
elif attest:
    pct = str(attest.get("pct", pct))
if accum:
    liq = str(accum.get("total_usd", liq))

print(pct, liq, phase, goal_met)
' 2>/dev/null || echo "10.5 0 ? ?")"

lp_share=$(python3 -c 'import json;print(json.load(open("/dev/shm/lp_dashboard.json")).get("pool_share_pct","?"))' 2>/dev/null || echo "?")
lp_status=$(python3 -c 'import json;print(json.load(open("/dev/shm/lp_dashboard.json")).get("status","?"))' 2>/dev/null || echo "?")

fleet_up=$(python3 -c '
import json
try:
 d=json.load(open("/tmp/monitor_state.json"))
 print(sum(1 for s in d.get("services",{}).values() if isinstance(s,dict) and s.get("status")=="UP"))
except: print(0)
' 2>/dev/null || echo "0")
evt_tail=$(tail -5 /home/diamondnode/always_alive_monitor/events.log 2>/dev/null | tr '\n' ' ' | cut -c1-400 || echo "")

# Read the X observer pulse cache that sota-livestream-updater.sh already refreshes every 5s --
# do not re-fetch here (avoids a second, uncoordinated call path against X API rate limits).
read -r x_source x_posts x_likes x_replies x_retweets x_impr x_activity <<<"$(python3 -c '
import json
try:
    d = json.load(open("/tmp/sota-livestream/x-observer-pulse.json"))
    m = d.get("aggregate_metrics", {})
    print(d.get("source", "?"), d.get("post_count", 0), m.get("like_count", 0),
          m.get("reply_count", 0), m.get("retweet_count", 0), m.get("impression_count", 0),
          d.get("observer_activity", 0))
except Exception:
    print("unavailable 0 0 0 0 0 0")
' 2>/dev/null || echo "unavailable 0 0 0 0 0 0")"

broadcast_id="k2atpt1e4x6v"
ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)

GPU_RAW="$gpu" python3 - <<PY
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
  # Read through the environment rather than interpolated into this source.
  # The shape check above already constrains it, but this field is the one fed
  # from an external tool's free text, so it must not be able to reach the
  # Python parser at all.
  "gpu": {"raw": os.environ.get("GPU_RAW", "unavailable")},
  "qflop": {"pct": "$qf_pct", "liquidity_usd": "$qf_liq", "phase": "$qf_phase", "goal_met": "$qf_goal_met"},
  "lp": {"pool_share_pct": "$lp_share", "status": "$lp_status"},
  "fleet": {"services_up": int("$fleet_up" or 0)},
  "x_signal": {
    "source": "$x_source",
    "post_count": int("$x_posts" or 0),
    "like_count": int("$x_likes" or 0),
    "reply_count": int("$x_replies" or 0),
    "retweet_count": int("$x_retweets" or 0),
    "impression_count": int("$x_impr" or 0),
    "observer_activity": float("$x_activity" or 0)
  },
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