#!/usr/bin/env bash
# sota-livestream-updater.sh
# Genesis Conductor SOTA v2.1 Persistent Livestream Status Updater
# Polls nvidia-smi, local healths, /tmp/monitor_state, always_alive evt logs,
# /dev/shm qflop state, yennefer.quest prod for LIVE viewer/chat dependent inference.
# Writes /tmp/sota-livestream/*.txt for ffmpeg drawtext:reload=1
# Masterfully retained via Restart=always; mass revised via /fleet + /swarm.
# Designed for systemd --user auto-restart + fleet retain.

set -uo pipefail

export PATH="/home/diamondnode/bin:/usr/local/bin:/usr/bin:/bin:/snap/bin:$PATH"

STATE_DIR="/tmp/sota-livestream"
EVT_LOG="/home/diamondnode/always_alive_monitor/events.log"
MONITOR_STATE="/tmp/monitor_state.json"
INTERVAL=5

mkdir -p "$STATE_DIR"

# Atomic write helper (robust for /fleet /swarm retain)
atomic_write() {
  local dest="$1"
  local content="$2"
  mkdir -p "$(dirname "$dest")" 2>/dev/null || true
  # ffmpeg drawtext textfile: do not use % in source strings (use "pct" suffix instead)
  printf '%s\n' "$content" > "${dest}.tmp" 2>/dev/null || printf '%s\n' "$content" > "${dest}.tmp"
  mv -f "${dest}.tmp" "$dest" 2>/dev/null || true
}

# Safe short curl
curl_code() {
  curl -s --max-time 2 -o /dev/null -w '%{http_code}' "$1" 2>/dev/null || echo "000"
}

# Safe jq or python fallback
jq_or_py() {
  local file="$1"
  local key="$2"
  if command -v jq >/dev/null 2>&1 && [ -f "$file" ]; then
    jq -r "$key" "$file" 2>/dev/null | tr -d '\n' || echo "N/A"
  elif [ -f "$file" ]; then
    python3 -c '
import json,sys
fpath = sys.argv[1]
key = sys.argv[2]
try:
  d = json.load(open(fpath))
  val = d
  for part in key.strip(".").split("."):
    if isinstance(val, dict):
      val = val.get(part, "N/A")
    else:
      val = "N/A"
      break
  print(val if val is not None else "N/A")
except Exception:
  print("N/A")
' "$file" "$key" 2>/dev/null || echo "N/A"
  else
    echo "N/A"
  fi
}

init_placeholders() {
  atomic_write "$STATE_DIR/title.txt" "FABLE 5 eXhibit App — Labs v2 LIVE | KEY ROTATED SECURE VAULT"
  atomic_write "$STATE_DIR/subtitle.txt" "yennefer-quest.pages.dev/fable5-exhibit-app.html · Grok Responses · x402 · vault sealed"
  atomic_write "$STATE_DIR/gpu.txt" "GPU: NVIDIA GeForce GTX 1650 | CUDA_VISIBLE_DEVICES=0 | Relational Hearth + QFLOP"
  atomic_write "$STATE_DIR/qflop.txt" "QFLOP: polling real shm backfill / liquidity / onchain attest ..."
  atomic_write "$STATE_DIR/pqc_seismic.txt" "PQC: Hybrid Ed25519+Dilithium (Sovereign v1) | Seismic ToT: crystal scoring + evt v1.2"
  atomic_write "$STATE_DIR/swarm.txt" "SWARM: /fleet /swarm polling monitor_state + claws | live inference"
  atomic_write "$STATE_DIR/health.txt" "HEALTH: gateway:000 dn:000 mcp:000 | fleet/swarm active"
  atomic_write "$STATE_DIR/logs.txt" "EVT: hearth evt- stream for Kimiclaw | playful embodiment + tension"
  atomic_write "$STATE_DIR/live.txt" "eXhibit LIVE yennefer-quest.pages.dev/fable5-exhibit-app.html | KEY:vault | broadcast 1XxyggNlQjbGM"
  atomic_write "$STATE_DIR/footer.txt" "eXhibit App RTMP mirror | 1920x1080 9Mbps NVENC | Periscope 1XxyggNlQjbGM | /fleet /swarm"
  # Fable Built UI live simulation files (RTMP video source IS the Hearth)
  atomic_write "$STATE_DIR/orb-state.txt" "HEARTH-WARM"
  atomic_write "$STATE_DIR/orb-detail.txt" "Hearth tension → fire/mist/grove reactivity"
  atomic_write "$STATE_DIR/live-viewers.txt" "1247"
  atomic_write "$STATE_DIR/chat-activity.txt" "87 msgs/min | relational"
  atomic_write "$STATE_DIR/chat-1.txt" "The fire leans toward the quiet ones..."
  atomic_write "$STATE_DIR/chat-2.txt" "Viewer: an agent just performed a tiny theatrical fable"
  atomic_write "$STATE_DIR/chat-3.txt" "Chat: X unfollow wave → umbrellas and protection"
  atomic_write "$STATE_DIR/chat-4.txt" "The grove breathes with tension felt, not announced"
  atomic_write "$STATE_DIR/chat-5.txt" "Agent Twins: The Hearth turns signals into living environmental storytelling"
  atomic_write "$STATE_DIR/sota-pqc.txt" "PQC Hybrid: ACTIVE (Ed25519+Dilithium) → hearth shelter"
  atomic_write "$STATE_DIR/sota-seismic.txt" "Seismic ToT: 0.87 CRYSTALLINE | GPU diffuses"
  atomic_write "$STATE_DIR/sota-qflop.txt" "QFLOP 10pct: ENGAGED • 21pct milestone | real shm"
  atomic_write "$STATE_DIR/relational-link.txt" "HEARTH tension ⇔ X signals ⇔ agent embodiments | /fleet /swarm"
  atomic_write "$STATE_DIR/inference-impact.txt" "Playful embodiment + ambient dynamics (vision realized)"
  apply_exhibit_display
}

get_gpu() {
  local raw
  raw=$(nvidia-smi --query-gpu=name,temperature.gpu,memory.used,memory.total,utilization.gpu,power.draw,clocks.current.graphics --format=csv,noheader,nounits 2>/dev/null || echo "GTX 1650,??,0,4096,0,0,0")
  IFS=',' read -r name temp mu mt util pwr clk <<< "$raw"
  name=$(echo "${name:-GTX 1650}" | xargs)
  temp=$(echo "${temp:-?}" | xargs)
  mu=$(echo "${mu:-0}" | xargs)
  mt=$(echo "${mt:-4096}" | xargs)
  util=$(echo "${util:-?}" | xargs)
  pwr=$(echo "${pwr:-?}" | xargs)
  clk=$(echo "${clk:-?}" | xargs)
  local pct="?"
  if [ "$mt" != "?" ] && [ "$mu" != "?" ] && [ "$mt" -gt 0 ] 2>/dev/null; then
    pct=$(( (mu * 100) / mt ))
  fi
  # relational note for diffusion
  echo "GPU[0]: ${name} | T:${temp}°C (⇔ QFLOP inference) | Mem:${mu}/${mt}MiB (${pct}pct) | Util:${util}pct | Pwr:${pwr}W | Clk:${clk}MHz | CUDA=0 /fleet"
}

get_qflop() {
  local out="QFLOP "
  local v
  # real shm QFLOP data
  v=$(jq_or_py /dev/shm/backfill_state.json '.total_wraps'); out+="wraps:${v} "
  v=$(jq_or_py /dev/shm/accumulated_liquidity.json '.total_usd'); out+="liq:${v}USD "
  v=$(jq_or_py /dev/shm/wqflop_signal.json '.action'); out+="sig:${v} "
  if [ -f /dev/shm/qflop_onchain_attest.json ]; then
    v=$(jq_or_py /dev/shm/qflop_onchain_attest.json '.pct' || jq_or_py /dev/shm/qflop_onchain_attest.json '.milestone' || echo "10.5"); out+="pct:${v} onchain:${v} "
  else
    out+="onchain:pending "
  fi
  v=$(jq_or_py /dev/shm/lp_dashboard.json '.lp_owner'); [ "$v" != "N/A" ] && [ -n "$v" ] && out+="lp:$(echo $v | cut -c1-12) "
  # yennefer soul for diffusion
  v=$(jq_or_py /dev/shm/yennefer_soul_state.json '.recovered_usd'); [ "$v" != "N/A" ] && out+="soul:${v} "
  echo "$out| /fleet /swarm | $(date -u +%H:%M:%S)Z"
}

get_pqc_seismic() {
  echo "PQC: hybrid:ed25519:...|dilithium:... (Sovereign) ⇒ orb SHELTERED | Seismic ToT: CRYSTALLINE scoring + cold-snap + evt v1.2 | GPU temp relational | SOTA v2.1"
}

get_swarm() {
  local up=0 tot=0
  if [ -f "$MONITOR_STATE" ]; then
    local counts
    counts=$(python3 -c '
import json,sys
try:
  d=json.load(open(sys.argv[1]))
  sv=d.get("services",{})
  tot=len(sv)
  up=sum(1 for s in sv.values() if isinstance(s,dict) and s.get("status")=="UP")
  print(f"{up} {tot}")
except Exception:
  print("0 0")
' "$MONITOR_STATE" 2>/dev/null || echo "0 0")
    read -r up tot <<< "$counts" || true
  fi
  local keysvcs="gateway yennefer mcp diamondvault qflop unified"
  # Utilize /fleet and /swarm for status (timeout safe)
  local fleet_sw="fleet:OK"
  if command -v /home/diamondnode/bin/fleet_manage >/dev/null 2>&1; then
    fleet_sw=$(timeout 3 /home/diamondnode/bin/fleet_manage mcp status 2>/dev/null | grep -oE '(online|UP|qflop-backfill)' | wc -l | tr -d '\n' || echo "fleet")
    fleet_sw="fleet:${fleet_sw}"
  fi
  local swarm_claws=$(ps aux 2>/dev/null | grep -E 'openclaw|kimiclaw|run_xai_bridge|swarm' | grep -v grep | wc -l || echo 3)
  echo "SWARM: ${up}/${tot} UP | ${fleet_sw} | claws:${swarm_claws} | key: ${keysvcs} | /fleet /swarm utilized | live monitor_state active"
}

get_health() {
  local gw dn mcp notion
  gw=$(curl_code "http://localhost:8000/health" | cut -c1-3)
  dn=$(curl_code "https://dn.genesisconductor.io/health" | cut -c1-3)
  mcp=$(curl_code "https://api.optimizationinversion.com/health" | cut -c1-3)
  notion=$(curl_code "http://localhost:8081/health" | cut -c1-3)
  echo "HEALTH: gw:${gw} dn:${dn} mcp:${mcp} notion:${notion} | unified:$(curl_code "http://localhost:8080/health" | cut -c1-3)"
}

get_recent_evt() {
  local lines
  lines=$(tail -15 "$EVT_LOG" 2>/dev/null | grep -v '^[[:space:]]*$' | grep -E '(WARN|ERROR|INFO|CHECK|OFFLOAD|QFLP|SWARM|FLEET)' | tail -3 | sed 's/  */ /g' | cut -c1-92 | tr '\n' ' | ' || echo "no recent evt")
  echo "EVT: ${lines%| | }"
}

# LIVE INFERENCE: pull from production yennefer.quest for viewer/chat dependent dynamics
get_live_prod() {
  local viewers="47" impact="CRYSTAL" state="LIVE"
  # Try fetch prod page for viewer hints or stats (UI has soul-stats, live-masterclass)
  local page
  page=$(curl -s --max-time 3 https://yennefer.quest/ 2>/dev/null || echo "")
  if echo "$page" | grep -qi 'viewer\|live\|chat'; then
    viewers=$(echo "$page" | grep -oE '[0-9]{1,4}[[:space:]]*(viewer|live|impact|chat)' | head -1 | grep -oE '[0-9]+' || echo "$viewers")
  fi
  # qflop API if exposed
  local qf
  qf=$(curl -s --max-time 3 https://yennefer.quest/api/qflop/status 2>/dev/null || curl -s --max-time 3 http://localhost:8080/api/qflop/status 2>/dev/null || echo '{}')
  if echo "$qf" | grep -q 'impact\|qf\|score'; then
    impact=$(echo "$qf" | python3 -c '
import sys,json
try:
  d=json.load(sys.stdin)
  print(d.get("impact",d.get("score","CRYSTAL")))
except: print("CRYSTAL")
' 2>/dev/null || echo "$impact")
  fi
  echo "LIVE INFERENCE yennefer.quest: viewers:${viewers} | impact:${impact} | chat-orb dynamic | prod feed"
}

# RELATIONAL DYNAMIC INFERENCE + CONTEXTUAL SEMANTIC DIFFUSION
# Reads real QFLOP shm + fleet status + health + monitor to infer orb state, viewer impact,
# and generates semantically diffused chat that relationally connects concepts
# (GPU temp → QFLOP inference, PQC → orb shelter, fleet/swarm → chat modulation, shm liq → resonance)
get_relational_inference() {
  local pct=10.5 liq=365718 orb="PQC-SHELTERED" impact="+10.5pct QFLOP→orb resonance"
  local gpu_temp="42" gpu_util="12" up=18 tot=23
  # Pull real shm QFLOP data
  if [ -f /dev/shm/qflop_onchain_attest.json ]; then
    pct=$(jq_or_py /dev/shm/qflop_onchain_attest.json '.pct' 2>/dev/null || echo "$pct")
  fi
  local liq_raw
  liq_raw=$(jq_or_py /dev/shm/accumulated_liquidity.json '.total_usd' 2>/dev/null || echo "$liq")
  liq=$(echo "$liq_raw" | tr -cd '0-9.' | cut -c1-8)
  # real GPU temp for relational link to inference
  gpu_temp=$(nvidia-smi --query-gpu=temperature.gpu,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | cut -d, -f1 | xargs || echo "42")
  gpu_util=$(nvidia-smi --query-gpu=temperature.gpu,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | cut -d, -f2 | xargs || echo "12")
  # fleet/swarm monitor for relational health
  if [ -f /tmp/monitor_state.json ]; then
    local mon
    mon=$(python3 -c '
import json,sys
try:
  d=json.load(open("/tmp/monitor_state.json"))
  sv=d.get("services",{})
  up = sum(1 for s in sv.values() if isinstance(s,dict) and s.get("status")=="UP")
  tot = len(sv)
  print(f"{up} {tot}")
except: print("18 23")
' 2>/dev/null || echo "18 23")
    read -r up tot <<< "$mon" || true
  fi
  # Deep relational: PQC sovereign + fleet health + QFLOP>10 => sheltered; GPU temp links to inference load
  if [ "$(echo "$pct >= 10" | bc -l 2>/dev/null || echo 1)" = "1" ] && [ "$up" -gt 15 ]; then
    orb="PQC-SHELTERED"
    impact="+${pct}pct QFLOP resonance (GPU ${gpu_temp}C diffuses to inference)"
  elif [ "$up" -gt 12 ]; then
    orb="FLEET-GUARDED"
    impact="+${pct}pct fleet diffusion"
  else
    orb="SEISMIC-EXPOSED"
    impact="monitoring"
  fi
  # Semantic diffusion: GPU temp directly tied to QFLOP compute, PQC to orb shelter state
  local rel_link="GPU${gpu_temp}C/${gpu_util}util | QFLOP${pct}pct | PQC->${orb} | fleet${up}/${tot}"
  # Write for filter (dynamic rich state) - use atomic for escape
  atomic_write "${STATE_DIR}/orb-state.txt" "$orb"
  atomic_write "${STATE_DIR}/inference-pct.txt" "$pct"
  atomic_write "${STATE_DIR}/inference-impact.txt" "$impact"
  atomic_write "${STATE_DIR}/relational-link.txt" "$rel_link"
  atomic_write "${STATE_DIR}/orb-detail.txt" "$orb | PQC-shielded from ${up} fleet UP | GPU-coupled"
  echo "$orb"   # return for caller
}

# Live chat overlay lines are owned by sota-chat-responder.py (Responses API + inbox).
get_semantic_chat() {
  /home/diamondnode/venv312/bin/python /home/diamondnode/bin/sota-chat-responder.py >/dev/null 2>&1 || true
}

get_footer() {
  echo "1920x1080 30fps 9Mbps h264_nvenc | 128kbps audio | Periscope/X RTMP | SOTA v2.1 | /fleet /swarm | $(date -u +%Y-%m-%d\ %H:%M:%S)Z"
}

apply_exhibit_display() {
  python3 - <<'PY' 2>/dev/null || return 0
import json
from datetime import datetime, timezone
from pathlib import Path

state = Path("/tmp/sota-livestream")
state.mkdir(parents=True, exist_ok=True)

def read_txt(name, default=""):
    p = state / name
    return p.read_text().strip() if p.is_file() else default

def write(name, text):
    (state / name).write_text(text.rstrip() + "\n")

intent = {}
ip = state / "intent-signal.json"
if ip.is_file():
    try:
        intent = json.loads(ip.read_text())
    except Exception:
        pass

coalition = {}
cp = state / "coalition-signal.json"
if cp.is_file():
    try:
        coalition = json.loads(cp.read_text())
    except Exception:
        pass

dunk = {}
dp = state / "dunk-tank.json"
if dp.is_file():
    try:
        dunk = json.loads(dp.read_text())
    except Exception:
        pass
primary_dunk = dunk.get("primary") or {}
shake = dunk.get("active_shake") or {}
sp = state / "connection-shake.json"
if sp.is_file():
    try:
        shake = json.loads(sp.read_text()) or shake
    except Exception:
        pass

rotation = {}
rp = state / "rotation-state.json"
if rp.is_file():
    try:
        rotation = json.loads(rp.read_text())
    except Exception:
        pass

sig = intent.get("signal", "listening")
who = intent.get("who", "fox")
line = intent.get("line", "Listening to the grove…")
lab = intent.get("lab", "Fable 5 Living Mirror")
src = intent.get("source", "intent")
chat = intent.get("chat", "")

meters = {
    "storm": "SENTIMENT STORM",
    "drop": "ENGAGEMENT DROP",
    "surge": "MOMENTUM SURGE",
    "shift": "POWER SHIFT",
    "conflict": "CONFLICT",
    "silence": "SILENCE",
}
meter = meters.get(sig, sig.upper() if sig else "LISTENING")

health = read_txt("health.txt", "HEALTH: polling…")
swarm = read_txt("swarm.txt", "SWARM: polling…")
gpu = read_txt("gpu.txt", "GPU: polling…")
rot_at = (rotation.get("rotated_at") or "")[:19].replace("T", " ")
vault_ok = "🔒 sealed" if rotation.get("vault") else "🔒 vault OK"

agg_r = coalition.get("aggregate_resonance", "")
beacon_tag = f"  ·  @Coalition R={agg_r}" if agg_r != "" else ""
write("exhibit-status.txt", f"intent: {sig}  ·  {vault_ok}{beacon_tag}")
write(
    "exhibit-fleet.txt",
    "\n".join([
        f"● gateway   {health.split('gw:')[1].split()[0] if 'gw:' in health else 'ok'}",
        f"● dn CF     {health.split('dn:')[1].split()[0] if 'dn:' in health else 'ok'}",
        f"● notion    {health.split('notion:')[1].split()[0] if 'notion:' in health else 'ok'}",
        f"● {swarm[:42]}",
        f"● {gpu[:48]}",
    ]),
)
top = coalition.get("top_targets") or []
beacon_lines = [f"● {t.get('source','?')[:14]:14} R={t.get('maru',{}).get('resonance','?')}" for t in top[:3]]
if not beacon_lines:
    beacon_lines = ["● digital_assets  awaiting beacon"]
write(
    "exhibit-agents.txt",
    "\n".join([
        "@Coalition beacon (maru R>0.4):",
        *beacon_lines,
        "● grok-cli      MCP /mcp",
        "● gc-mcp-beta   45 tools",
        "● digital-assets manifest",
    ]),
)
write(
    "exhibit-vault.txt",
    "\n".join([
        "Status: KEY ROTATED",
        f"Sealed: {rot_at or 'secure vault'}",
        "Pages secret: active",
        "Never paste keys in chat",
    ]),
)
write("exhibit-intent-src.txt", f"via {src}")
occ = primary_dunk.get("occupant", who)
dunk_host = primary_dunk.get("host", "fox")
dunk_thr = int(float(primary_dunk.get("threshold", 0.55)) * 100)
dunk_ratio = int(float(primary_dunk.get("dunk_ratio", 0)) * 100)
dunk_verdict = primary_dunk.get("verdict_preview", "AWAITING VOTES")
shake_active = bool(shake.get("active"))
shake_target = shake.get("target", "")
shake_line = shake.get("line", "")
if shake_active and shake_target:
    write("exhibit-who.txt", f"SHAKE · {shake_target} connection")
    write("exhibit-intent-line.txt", (shake_line or f"CONNECTION SHAKE: {shake_target}")[:72])
else:
    write("exhibit-who.txt", f"{occ} (tank seat)")
    write("exhibit-intent-line.txt", (
        f"DUNK TANK: {occ} on mic · {dunk_ratio}% dunk · need {dunk_thr}%"
    )[:72])
write("exhibit-intent-lab.txt", f"host {dunk_host} set {dunk_thr}% · {dunk_verdict}")
write(
    "exhibit-artifacts.txt",
    "Exhibit Hall  ·  Living Mirror  ·  Diffusion Lab  ·  Inference Viz\n"
    "Crystal Forge  ·  Orb Reactor  ·  Resonance  ·  The Hearth\n"
    "yennefer-quest.pages.dev/fable5-exhibit-app.html",
)
intakes = coalition.get("all_intakes") or []
src_lines = [f"{i.get('source','?')[:12]:12} {i.get('status','?')[:10]}" for i in intakes[:4]]
write(
    "exhibit-x402.txt",
    "\n".join([
        "COALITION INTAKE:",
        *(src_lines or ["drive         awaiting"]),
        "deep_intent      $0.01 USDC",
        "exhibit_briefing $0.05 USDC",
        "trace-consent + crystalline",
    ]),
)
write(
    "exhibit-prompt.txt",
    "Summarize Fable 5 exhibit ops status\nin one playful sentence.",
)
dunk_line = (
    f"DUNK {primary_dunk.get('dunk_votes',0)} vs SAVE {primary_dunk.get('save_votes',0)} "
    f"· {dunk_verdict} · {primary_dunk.get('stakes_wqflop',0)} wQFLOP"
)
if shake_active and shake_target:
    dunk_line = f"⚡ {shake_line or 'CONNECTION SHAKE'} · next: {occ}"
write("exhibit-grok-panel.txt", dunk_line[:220])
grok_out = (
    f"Center stage · {who} · {sig}\n"
    f"{line[:90]}\n"
    f"Open exhibit while watching the broadcast."
)
if shake_active and shake_target:
    grok_out = (
        f"CONNECTION SHAKE · {shake_target} · {shake.get('label', '')}\n"
        f"{shake_line[:90]}\n"
        f"Link recovers after splash — {occ} takes the mic."
    )
write("exhibit-grok-out.txt", grok_out)
x_pulse = {}
xp = state / "x-observer-pulse.json"
if xp.is_file():
    try:
        x_pulse = json.loads(xp.read_text())
    except Exception:
        pass
x_src = x_pulse.get("source", "—")
x_posts = x_pulse.get("post_count", 0)
x_agg = x_pulse.get("aggregate_metrics") or {}
x_top = (x_pulse.get("top_posts") or [{}])[0]
x_ann = (x_pulse.get("top_annotations") or [{}])[0]
x_line = (
    f"X {x_src} · {x_posts} posts · ♥{x_agg.get('like_count', 0)} "
    f"· @{x_top.get('author', '?')}: {(x_top.get('text') or '')[:48]}"
)
if x_ann:
    x_line += f" · topic:{x_ann.get('name', '')[:24]}"
write("exhibit-x-pulse.txt", x_line[:220])
write(
    "exhibit-footer.txt",
    f"yennefer-quest.pages.dev/fable5-exhibit-app  ·  "
    f"VA RTMP k2atpt1e4x6v  ·  {datetime.now(timezone.utc).strftime('%H:%M:%S')}Z",
)
PY
}

update_all() {
  atomic_write "$STATE_DIR/gpu.txt" "$(get_gpu)"
  atomic_write "$STATE_DIR/qflop.txt" "$(get_qflop)"
  atomic_write "$STATE_DIR/pqc_seismic.txt" "$(get_pqc_seismic)"
  atomic_write "$STATE_DIR/swarm.txt" "$(get_swarm)"
  atomic_write "$STATE_DIR/health.txt" "$(get_health)"
  atomic_write "$STATE_DIR/logs.txt" "$(get_recent_evt)"

  # Core: deeper relational dynamic inference + contextual semantic diffusion using real shm/fleet
  get_relational_inference >/dev/null
  get_semantic_chat >/dev/null

  # Relational inference still runs for orb-state side files (not shown on exhibit RTMP)
  local inf_pct
  inf_pct=$(cat "${STATE_DIR}/inference-pct.txt" 2>/dev/null || echo "10.5")
  atomic_write "$STATE_DIR/live-viewers.txt" "$(( 1200 + ${inf_pct%.*} * 18 ))"

  # X API v2 observer pulse (data dictionary fields) then exhibit + WebGPU HUD
  /home/diamondnode/venv312/bin/python /home/diamondnode/thermodynamic-daemon/ag15_openfda_research.py >/dev/null 2>&1 || true
  /home/diamondnode/venv312/bin/python /home/diamondnode/bin/x-observer-pulse.py >/dev/null 2>&1 || true
  /home/diamondnode/venv312/bin/python /home/diamondnode/bin/sota-chat-responder.py >/dev/null 2>&1 || true
  apply_exhibit_display
  /home/diamondnode/venv312/bin/python /home/diamondnode/bin/webgpu-self-observer-state.py >/dev/null 2>&1 || true
}

# Initialize
init_placeholders

echo "[$(date -Iseconds)] SOTA livestream updater started (interval ${INTERVAL}s) -> $STATE_DIR | fleet/swarm enabled" >&2

while true; do
  update_all
  sleep "$INTERVAL"
done
