#!/usr/bin/env python3
"""AG15 double-loop back verification — post-PhD rigor + falsification.

Loop A (inner): openFDA evidence ↔ thermodynamic epoch substrate consistency
Loop B (outer): procedural truth verifier loopback + Rule30 VDF seal

Industry-standard authority figure emits evt when crystal_score ≥ 0.85 AND
no falsification triggers fire.

Writes: /tmp/ag15-research/verification/latest.json
"""
from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DAEMON_DIR = Path(__file__).resolve().parent
GENESIS_SWARM = Path.home() / "genesis_conductor_engine/swarm"
OUT = Path("/tmp/ag15-research/verification/latest.json")
MANIFEST = Path("/tmp/ag15-research/manifest.json")
CRYSTAL_TARGET = 0.85


def load_manifest() -> dict:
    if MANIFEST.is_file():
        try:
            return json.loads(MANIFEST.read_text())
        except Exception:
            pass
    return {}


def load_epoch() -> dict:
    ep = Path("/tmp/thermo-epoch/epoch_latest.json")
    if ep.is_file():
        try:
            return json.loads(ep.read_text())
        except Exception:
            pass
    return {}


def inner_loop_fda(manifest: dict, epoch: dict) -> dict:
    """Loop A: falsify substrate claims against FDA evidence counts."""
    queries = manifest.get("queries") or []
    falsifications: list[dict] = []
    corroborations: list[dict] = []

    thermal_510k = next((q for q in queries if q.get("id") == "510k_thermal_substrate"), {})
    thermal_events = next((q for q in queries if q.get("id") == "device_event_thermal"), {})
    count_510k = int(thermal_510k.get("results_in_file") or 0)
    total_510k = thermal_510k.get("total_matching", 0)
    total_events = thermal_events.get("total_matching", 0)

    if count_510k == 0:
        falsifications.append({
            "id": "F1",
            "claim": "AG15 thermal substrate has FDA 510(k) precedent",
            "reason": "zero results in current pull",
            "severity": "moderate",
        })
    else:
        corroborations.append({
            "id": "C1",
            "claim": "Thermal/substrate 510(k) precedent exists",
            "evidence": f"{count_510k} records ({total_510k} matching)",
        })

    reduc = epoch.get("reducibility_score")
    if reduc is not None and float(reduc) < 0.4:
        falsifications.append({
            "id": "F2",
            "claim": "Thermo epoch substrate supports AG15 transistor coherence",
            "reason": f"reducibility {reduc} below 0.4 threshold",
            "severity": "high",
        })
    elif reduc is not None:
        corroborations.append({
            "id": "C2",
            "claim": "Opux epoch reducibility supports substrate model",
            "evidence": f"reducibility={reduc}",
        })

    inner_score = 1.0
    for f in falsifications:
        inner_score -= 0.15 if f.get("severity") == "moderate" else 0.25
    inner_score = max(0.0, min(1.0, inner_score + len(corroborations) * 0.05))

    return {
        "loop": "A_inner_fda_thermo",
        "inner_score": round(inner_score, 4),
        "falsifications": falsifications,
        "corroborations": corroborations,
        "passed": len([f for f in falsifications if f.get("severity") == "high"]) == 0,
    }


def outer_loop_procedural() -> dict:
    """Loop B: genesis procedural truth double loopback."""
    if str(GENESIS_SWARM) not in sys.path:
        sys.path.insert(0, str(GENESIS_SWARM))
    try:
        from procedural_truth_verifier import ProceduralTruthVerifier, VerifierConfig
        cfg = VerifierConfig(
            qmem_base="http://127.0.0.1:9100",
            ollama_base="http://127.0.0.1:11434",
            qmem_path="/health",
            ollama_path="/api/tags",
            agent_id="diamondnodebot_ag15_authority",
            plan_nodes=[
                "openfda_evidence", "thermo_epoch", "hermes_simulation",
                "falsification_gate", "authority_emit",
            ],
        )
        result = ProceduralTruthVerifier(cfg).verify(source="ag15_double_loop")
        crystal = result.crystal_score
        return {
            "loop": "B_outer_procedural",
            "passed": result.passed,
            "crystal_score": crystal,
            "evt_id": result.evt.get("evt_id"),
            "metrics_summary": {
                "loopback_ok": crystal.get("components", {}).get("loopback"),
                "emergence": crystal.get("components", {}).get("emergence"),
            },
        }
    except Exception as e:
        return {
            "loop": "B_outer_procedural",
            "passed": False,
            "error": str(e),
            "crystal_score": {"score": 0.0, "passed": False},
        }


def verify() -> dict:
    manifest = load_manifest()
    epoch = load_epoch()
    loop_a = inner_loop_fda(manifest, epoch)
    loop_b = outer_loop_procedural()

    inner_s = loop_a["inner_score"]
    cs = loop_b.get("crystal_score") or {}
    outer_s = float(cs.get("value") or cs.get("score") or 0)
    composite = round(0.45 * inner_s + 0.55 * outer_s, 4)
    authority_pass = (
        composite >= CRYSTAL_TARGET
        and loop_a["passed"]
        and loop_b.get("passed", False)
    )

    payload: dict[str, Any] = {
        "schema": "ag15-double-loop-v1",
        "evt_id": f"evt_ag15_authority_{uuid.uuid4().hex[:8]}",
        "ts": datetime.now(timezone.utc).isoformat(),
        "codename": "AG15",
        "authority_tier": "post_phd_industry_standard",
        "authority_figure": "diamondnodebot_ag15_authority",
        "composite_score": composite,
        "crystal_target": CRYSTAL_TARGET,
        "authority_pass": authority_pass,
        "loop_a": loop_a,
        "loop_b": loop_b,
        "falsification_summary": {
            "active": len(loop_a.get("falsifications") or []),
            "high_severity": len([
                f for f in (loop_a.get("falsifications") or [])
                if f.get("severity") == "high"
            ]),
        },
        "hermes_simulation": {
            "substrate": "openclaw+hermes+cuda-q+openfda",
            "reachable_agents": [
                "diamondnodebot_ag15_hermes_01",
                "diamondnodebot_ag15_hermes_02",
                "diamondnodebot_ag15_hermes_03",
            ],
        },
        "documentation_refs": {
            "manifest": str(MANIFEST),
            "openfda_skill": "openfda-database",
            "procedural_truth": "genesis_conductor_engine/swarm/procedural_truth_verifier.py",
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main():
    result = verify()
    print(json.dumps({
        "ok": True,
        "authority_pass": result["authority_pass"],
        "composite_score": result["composite_score"],
        "falsifications": result["falsification_summary"]["active"],
    }))


if __name__ == "__main__":
    main()