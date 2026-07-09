#!/usr/bin/env python3
"""Hermes task handler for AG15 diamondnodebot pinned swarm."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

DAEMON_DIR = Path.home() / "thermodynamic-daemon"
SWARM_CFG = Path(__file__).resolve().parent / "ag15_diamondnodebot_swarm.json"


def handle_task(task: dict) -> dict:
    action = task.get("action", "status")
    if action == "openfda_pull":
        proc = subprocess.run(
            [sys.executable, str(DAEMON_DIR / "ag15_openfda_research.py")],
            capture_output=True, text=True, timeout=120,
        )
        return {"ok": proc.returncode == 0, "stdout": proc.stdout[-500:], "action": action}
    if action == "double_loop_verify":
        proc = subprocess.run(
            [sys.executable, str(DAEMON_DIR / "ag15_double_loop_verifier.py")],
            capture_output=True, text=True, timeout=90,
        )
        return {"ok": proc.returncode == 0, "stdout": proc.stdout[-500:], "action": action}
    if action == "full_pipeline":
        subprocess.run([sys.executable, str(DAEMON_DIR / "ag15_openfda_research.py")], check=False)
        proc = subprocess.run(
            [sys.executable, str(DAEMON_DIR / "ag15_double_loop_verifier.py")],
            capture_output=True, text=True, timeout=90,
        )
        return {"ok": proc.returncode == 0, "stdout": proc.stdout[-500:], "action": action}
    manifest = Path("/tmp/ag15-research/manifest.json")
    verification = Path("/tmp/ag15-research/verification/latest.json")
    return {
        "ok": True,
        "action": "status",
        "swarm": json.loads(SWARM_CFG.read_text()) if SWARM_CFG.is_file() else {},
        "manifest_exists": manifest.is_file(),
        "verification_exists": verification.is_file(),
    }


def main():
    task = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {"action": "status"}
    print(json.dumps(handle_task(task)))


if __name__ == "__main__":
    main()