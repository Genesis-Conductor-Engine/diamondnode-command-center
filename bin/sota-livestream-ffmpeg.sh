#!/bin/bash
# sota-livestream-ffmpeg.sh
# Pure ffmpeg livestream for Fable 5 SOTA Masterclass
# THE RTMP IS OF THE FABLE BUILT UI: full visual recreation of yennefer-quest public/index.html
# (hero orb + rings + stats + #live-masterclass live chat + SOTA modules)
# Viewer/chat dependent dynamics via live text files (orb state, chat lines, viewers, modules)
# + real time fleet data overlays (GPU/QFLOP/PQC)
# Exact settings: 1920x1080 30fps 9Mbps h264_nvenc, 128kbps AAC
# RTMP: rtmp://ca.pscp.tv:80/x/j64ragsivuvm
# Uses CUDA_VISIBLE_DEVICES=0

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export PATH="/home/diamondnode/bin:/usr/local/bin:/usr/bin:/bin:/snap/bin:/usr/local/cuda/bin:$PATH"

STATE_DIR="/tmp/sota-livestream"
mkdir -p "$STATE_DIR"

# Robust ffmpeg path (may be installed by user / nvidia / package at runtime)
FFMPEG_BIN="$(command -v ffmpeg 2>/dev/null || echo /usr/bin/ffmpeg)"
if [ ! -x "$FFMPEG_BIN" ]; then
  # fallback candidates
  for cand in /usr/bin/ffmpeg /usr/local/bin/ffmpeg /opt/ffmpeg/bin/ffmpeg; do
    if [ -x "$cand" ]; then FFMPEG_BIN="$cand"; break; fi
  done
fi

FONT_REG="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_MONO="/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"

# Fable Built UI is the video source: we render the actual page layout (orb + live-masterclass)
# with live updating chat, viewers, orb state, SOTA modules + real fleet telemetry.
# ENHANCED: deeper relational dynamic inference + contextual semantic diffusion
# Uses real shm QFLOP, fleet status, health. Chat/orb link GPU temp to QFLOP, PQC to orb shelter.
# /fleet and /swarm utilized in state.
VIDEO_INPUT="color=c=#0a0a1a:s=1920x1080:r=30"
AUDIO_INPUT="anullsrc=cl=stereo:r=44100"

# Source VA encoder config (override with legacy CA key if needed)
if [ -f /home/diamondnode/bin/sota-rtmp.env ]; then
  # shellcheck disable=SC1091
  source /home/diamondnode/bin/sota-rtmp.env
  RTMP_URL="${SOTA_RTMP_URL:-rtmp://va.pscp.tv:80/x/k2atpt1e4x6v}"
else
  RTMP_URL="rtmp://va.pscp.tv:80/x/k2atpt1e4x6v"
fi

echo "[$(date -Iseconds)] Starting SOTA v2.1 livestream ffmpeg (eXhibit App UI panels, CUDA=0, NVENC) via $FFMPEG_BIN -> $RTMP_URL" >&2

# Reconnect flags for indefinite runtime against transient RTMP drops
FFMPEG_RECONNECT=(
  -reconnect 1
  -reconnect_at_eof 1
  -reconnect_streamed 1
  -reconnect_delay_max 5
)

# Build the Fable 5 SOTA Masterclass UI live in the video (matches public/index.html hero + #live-masterclass)
# - Top: YENNEFER nav + title + subtitle
# - Center hero: simulated orb (big symbol + rings) + rich orb-state + orb-detail (PQC shelter)
# - Live-masterclass: viewers + 5 chat lines (semantic rich relational) + SOTA modules
# - Overlays: GPU (with QFLOP link), qflop shm, swarm (/fleet), health, relational-link, footer
# All dynamic via :reload=1 textfiles written by updater (real data + fleet live)
FILTER_SRC="/home/diamondnode/bin/sota-exhibit-ui-filter.txt"
[ -f "$FILTER_SRC" ] || FILTER_SRC="/home/diamondnode/bin/sota-fable-ui-filter.txt"
cp "$FILTER_SRC" "${STATE_DIR}/fable-ui-filter.txt"

# Pre-seed exhibit panel textfiles so ffmpeg drawtext never fails on cold start
for seed in exhibit-status exhibit-fleet exhibit-agents exhibit-vault exhibit-intent-src \
  exhibit-who exhibit-intent-line exhibit-intent-lab exhibit-artifacts exhibit-x402 \
  exhibit-prompt exhibit-grok-panel exhibit-grok-out exhibit-footer; do
  [ -s "${STATE_DIR}/${seed}.txt" ] || printf 'Loading…\n' > "${STATE_DIR}/${seed}.txt"
done
exec "$FFMPEG_BIN" -hide_banner -nostats -loglevel warning \
  -re \
  -f lavfi -i "$VIDEO_INPUT" \
  -f lavfi -i "$AUDIO_INPUT" \
  -filter_complex_script "${STATE_DIR}/fable-ui-filter.txt" \
  -map "[vout]" -map 1:a \
  -c:v h264_nvenc \
    -preset p4 -tune ll \
    -b:v 9000k -maxrate 9000k -bufsize 18000k \
    -g 60 -keyint_min 60 -r 30 -s 1920x1080 \
    -pix_fmt yuv420p -profile:v high \
  -c:a aac -b:a 128k -ar 44100 -ac 2 \
  "${FFMPEG_RECONNECT[@]}" \
  -f flv "$RTMP_URL"
