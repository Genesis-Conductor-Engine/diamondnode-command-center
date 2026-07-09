#!/usr/bin/env bash
# WebGPU self-vs-observers → VA RTMP (1920×1080 30fps 9Mbps NVENC)
set -euo pipefail

source /home/diamondnode/bin/sota-rtmp.env
source /home/diamondnode/bin/gpu-env.sh 2>/dev/null || export CUDA_VISIBLE_DEVICES=0
export PATH="/home/diamondnode/bin:/home/diamondnode/venv312/bin:/usr/local/bin:/usr/bin:/bin:$PATH"

FFMPEG_BIN="$(command -v ffmpeg 2>/dev/null || echo /usr/bin/ffmpeg)"
CAPTURE_DIR="/home/diamondnode/bin/webgpu-stream"
STATE_DIR="/tmp/sota-livestream"
mkdir -p "$STATE_DIR"

# Ensure state + page server
if ! pgrep -f "webgpu-stream-server.py" >/dev/null 2>&1; then
  nohup /home/diamondnode/venv312/bin/python /home/diamondnode/bin/webgpu-stream-server.py \
    >/tmp/sota-livestream/webgpu-server.log 2>&1 &
  sleep 1
fi

/home/diamondnode/venv312/bin/python /home/diamondnode/bin/webgpu-self-observer-state.py >/dev/null 2>&1 || true

if [ ! -d "$CAPTURE_DIR/node_modules/playwright" ]; then
  echo "[webgpu-stream] installing playwright..." >&2
  (cd "$CAPTURE_DIR" && npm install --omit=dev 2>&1 | tail -3)
  (cd "$CAPTURE_DIR" && npx playwright install chromium 2>&1 | tail -5) || true
fi

echo "[$(date -Iseconds)] WebGPU stream → $SOTA_RTMP_URL (${SOTA_STREAM_WIDTH}x${SOTA_STREAM_HEIGHT} @ ${SOTA_STREAM_FPS}fps ${SOTA_VIDEO_BITRATE})" >&2

# Playwright WebGPU screencast → MJPEG pipe → NVENC → RTMP
exec "$FFMPEG_BIN" -hide_banner -nostats -loglevel warning \
  -f lavfi -re -i "anullsrc=cl=stereo:r=44100" \
  -f image2pipe -framerate "$SOTA_STREAM_FPS" -i <(
    node "$CAPTURE_DIR/capture.mjs" 2>/tmp/sota-livestream/webgpu-capture.log
  ) \
  -map 1:v -map 0:a \
  -c:v h264_nvenc -preset p4 -tune ll \
  -b:v "$SOTA_VIDEO_BITRATE" -maxrate "$SOTA_VIDEO_BITRATE" -bufsize 18000k \
  -g 60 -keyint_min 60 -r "$SOTA_STREAM_FPS" -s "${SOTA_STREAM_WIDTH}x${SOTA_STREAM_HEIGHT}" \
  -pix_fmt yuv420p -profile:v high \
  -c:a aac -b:a "$SOTA_AUDIO_BITRATE" -ar 44100 -ac 2 \
  -reconnect 1 -reconnect_at_eof 1 -reconnect_streamed 1 -reconnect_delay_max 5 \
  -f flv "$SOTA_RTMP_URL"