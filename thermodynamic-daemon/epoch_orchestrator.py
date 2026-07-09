#!/usr/bin/env python3
"""Epoch orchestrator — AlphaGenome variance → Opux epoch → attestation → story log.

Pipeline (Goal 1 guides; Goals 2–3 are inevitable milestones):
  1. Variance from AlphaGenome discovery scan OR synthetic fallback
  2. Thermo daemon NVML snapshot
  3. evolutionary_epoch.run_opux_epoch
  4. alchemy_story_logger.append_story_log
  5. Solidity-ready attestation witness JSON

AlphaGenome: requires ALPHAGENOME_API_KEY in ~/.env (uv run when available).
License: https://deepmind.google.com/science/alphagenome/ (see LICENSE_NOTIFICATION.txt)
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from alchemy_story_logger import append_story_log, push_to_alchemy
from evolutionary_epoch import run_opux_epoch

THERMO_URL = "http://127.0.0.1:9100/state"
ATTESTATION_DIR = Path("/tmp/thermo-epoch/attestations")
ATTESTATION_DIR.mkdir(parents=True, exist_ok=True)
ATTESTATION_LATEST = ATTESTATION_DIR / "attestation_latest.json"

SKILL_DIR = Path.home() / ".agents/skills/alphagenome-single-variant-analysis"
DEFAULT_VARIANT = "chr17:41234470:A>G"  # BRCA1-region demo locus


def fetch_thermo_state() -> dict:
    try:
        with urllib.request.urlopen(THERMO_URL, timeout=3) as r:
            return json.loads(r.read().decode())
    except Exception:
        return {"temperature_c": 55.0, "vram": {"used_pct": 30.0}, "source": "fallback"}


def _synthetic_variance(variant_id: str) -> float:
    import hashlib
    h = hashlib.sha256(variant_id.encode()).digest()
    return round((h[0] / 255.0) * 0.08 + 0.01, 6)


def alphagenome_variance(variant_id: str) -> tuple[float, dict]:
    """Run AlphaGenome discovery scan via uv; return variance of quantile scores."""
    scan_script = SKILL_DIR / "scripts" / "_epoch_variance_scan.py"
    try:
        proc = subprocess.run(
            ["uv", "run", "--project", str(SKILL_DIR), str(scan_script), variant_id],
            capture_output=True,
            text=True,
            timeout=180,
            cwd=str(SKILL_DIR),
        )
        if proc.returncode != 0:
            return _synthetic_variance(variant_id), {"source": "synthetic", "reason": "alphagenome_failed"}
        data = json.loads(proc.stdout.strip().split("\n")[-1])
        return float(data.get("variance", _synthetic_variance(variant_id))), data
    except Exception as e:
        return _synthetic_variance(variant_id), {"source": "synthetic", "reason": str(e)}


def build_attestation_witness(epoch: dict, story: dict) -> dict:
    """Solidity ThermoInformationAttestation-ready witness."""
    witness = {
        "type": "THERMO_ATTESTATION_WITNESS",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "epoch_id": epoch["epoch_id"],
        "work_hash": epoch["work_hash"],
        "reducibility_score": epoch["reducibility_score"],
        "field_hash": epoch["field_hash"],
        "variant_id": epoch["variant_id"],
        "variance": epoch["variance"],
        "story_id": story["story_id"],
        "story_root": story["story_root"],
        "device_attestation_hmac": story["device_attestation"]["device_attestation_hmac"],
        "contract": "ThermoInformationAttestation",
        "chain": story.get("chain", "base"),
        "three_goals": epoch.get("three_goals"),
        "sol_calldata_hint": {
            "function": "attestEpochWork",
            "args": [
                f"bytes32:{epoch['work_hash']}",
                f"uint256:{int(epoch['reducibility_score'] * 1e6)}",
                f"bytes32:{story['story_root']}",
            ],
        },
        "provenance": "thermodynamic-daemon:epoch_orchestrator",
    }
    ATTESTATION_LATEST.write_text(json.dumps(witness, indent=2) + "\n")
    return witness


def run_pipeline(variant_id: str = DEFAULT_VARIANT, chain: str = "base", use_alphagenome: bool = True) -> dict:
    thermo = fetch_thermo_state()

    ag_meta: dict = {"source": "synthetic"}
    if use_alphagenome:
        variance, ag_meta = alphagenome_variance(variant_id)
    else:
        variance = _synthetic_variance(variant_id)

    epoch = run_opux_epoch(variance, variant_id, thermo)
    epoch["alphagenome"] = ag_meta
    story = append_story_log(epoch, chain=chain)
    push_result = push_to_alchemy(story)
    witness = build_attestation_witness(epoch, story)

    return {
        "ok": True,
        "epoch_id": epoch["epoch_id"],
        "work_hash": epoch["work_hash"],
        "reducibility_score": epoch["reducibility_score"],
        "variance": variance,
        "variance_source": ag_meta.get("source", "synthetic"),
        "story_id": story["story_id"],
        "attestation": witness,
        "alchemy_push": push_result,
        "three_goals": epoch["three_goals"],
        "substrate": epoch["substrate"],
    }


def main():
    p = argparse.ArgumentParser(description="Thermo evolutionary epoch orchestrator")
    p.add_argument("--variant", default=DEFAULT_VARIANT)
    p.add_argument("--chain", default="base", choices=["base", "polygon"])
    p.add_argument("--no-alphagenome", action="store_true")
    args = p.parse_args()
    result = run_pipeline(args.variant, args.chain, use_alphagenome=not args.no_alphagenome)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()