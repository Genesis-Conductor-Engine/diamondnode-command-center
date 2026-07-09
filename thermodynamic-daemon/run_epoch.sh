#!/usr/bin/env bash
# Run one evolutionary epoch: variance → Opux HyperNEAT → attestation → Alchemy story
set -euo pipefail
source ~/load-env.sh 2>/dev/null || true
cd "$(dirname "$0")"
VARIANT="${1:-chr17:41234470:A>G}"
CHAIN="${2:-base}"
EXTRA=()
[[ "${3:-}" == "--no-alphagenome" ]] && EXTRA+=(--no-alphagenome)
exec ~/venv312/bin/python epoch_orchestrator.py --variant "$VARIANT" --chain "$CHAIN" "${EXTRA[@]}"