#!/usr/bin/env bash
# Push coalition beacon + crystalline pass to gc-mcp-beta KV via memory_write
set -euo pipefail

MCP_URL="${GC_MCP_BETA_URL:-https://gc-mcp-beta.iholt.workers.dev/mcp}"
BEACON="${1:-/tmp/sota-livestream/coalition-signal.json}"
CRYST="${2:-$HOME/digital-assets/crystalline/latest.json}"

mcp_write() {
  local key="$1" name="$2" body_file="$3"
  local body tags
  body=$(python3 -c 'import json,sys; print(json.dumps(open(sys.argv[1]).read()))' "$body_file")
  tags='["coalition","beacon","@Coalition"]'
  curl -s -X POST "$MCP_URL" \
    -H 'Content-Type: application/json' \
    -d "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/call\",\"params\":{\"name\":\"memory_write\",\"arguments\":{\"key\":\"$key\",\"name\":\"$name\",\"type\":\"project\",\"tags\":$tags,\"body\":$body}}}" \
    | python3 -c 'import sys,json; r=json.load(sys.stdin); print(r.get("result",r))'
}

/home/diamondnode/venv312/bin/python /home/diamondnode/bin/coalition-beacon.py >/dev/null
/home/diamondnode/venv312/bin/python /home/diamondnode/bin/coalition-crystalline-pass.py

echo ">>> KV push: coalition/beacon/latest"
mcp_write "coalition/beacon/latest" "Coalition Beacon Latest" "$BEACON"

if [ -f "$CRYST" ]; then
  echo ">>> KV push: coalition/crystalline/latest"
  mcp_write "coalition/crystalline/latest" "Coalition Crystalline Pass" "$CRYST"
fi

MIC="/tmp/sota-livestream/mic-drop.json"
if [ -f "$MIC" ]; then
  echo ">>> KV push: coalition/mic-drop/latest"
  mcp_write "coalition/mic-drop/latest" "MIC DROP Contracts" "$MIC"
  cp -f "$MIC" "$HOME/digital-assets/Coalition/Exhibit/Fable5/mic-drop-latest.json" 2>/dev/null || true
fi

DUNK="/tmp/sota-livestream/dunk-tank.json"
if [ -f "$DUNK" ]; then
  echo ">>> KV push: coalition/dunk-tank/latest"
  mcp_write "coalition/dunk-tank/latest" "Dunk Tank Vote-Out" "$DUNK"
fi

EPOCH="/tmp/thermo-epoch/epoch_latest.json"
if [ -f "$EPOCH" ]; then
  echo ">>> KV push: coalition/thermo-epoch/latest"
  mcp_write "coalition/thermo-epoch/latest" "Thermo Opux Epoch" "$EPOCH"
fi

ATTEST="/tmp/thermo-epoch/attestations/attestation_latest.json"
if [ -f "$ATTEST" ]; then
  echo ">>> KV push: coalition/thermo-attestation/latest"
  mcp_write "coalition/thermo-attestation/latest" "Thermo Attestation Witness" "$ATTEST"
fi

echo ">>> Done"