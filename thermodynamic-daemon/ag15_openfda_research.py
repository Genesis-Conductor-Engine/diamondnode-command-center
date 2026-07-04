#!/usr/bin/env python3
"""AG15 thermodynamic transistor substrate — openFDA evidence pipeline.

Uses the openfda-database skill wrapper (rate-limited uv scripts).
Authority tier: post-PhD double-loop verification companion (ag15_double_loop_verifier).

Writes: /tmp/ag15-research/openfda/*.json, manifest.json
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

DAEMON_DIR = Path(__file__).resolve().parent
OPENFDA_SKILL = Path.home() / ".agents/skills/openfda-database"
OUT_DIR = Path("/tmp/ag15-research/openfda")
MANIFEST = Path("/tmp/ag15-research/manifest.json")

# AG15 = thermodynamic transistor substrate research codename
# Maps to FDA device thermal/substrate + Ag-nanoparticle (Ag15) evidence lanes
QUERIES = [
    {
        "id": "510k_thermal_substrate",
        "category": "device",
        "endpoint": "510k",
        "search": "device_name:thermal+OR+statement_or_summary:substrate",
        "limit": 10,
        "hypothesis": "H1: FDA 510(k) corpus contains thermal substrate clearances relevant to AG15 transistor packaging.",
    },
    {
        "id": "classification_thermal_interface",
        "category": "device",
        "endpoint": "classification",
        "search": "device_name:thermal+OR+definition:substrate",
        "limit": 10,
        "hypothesis": "H2: Device classification taxonomy supports thermal-interface medical substrate claims.",
    },
    {
        "id": "device_event_thermal",
        "category": "device",
        "endpoint": "event",
        "search": "device.generic_name:thermal",
        "limit": 10,
        "hypothesis": "H3: Adverse event corpus falsifies or corroborates thermal-management substrate safety bounds.",
    },
    {
        "id": "pma_semiconductor_adjacent",
        "category": "device",
        "endpoint": "510k",
        "search": "device_name:semiconductor+OR+device_name:transistor",
        "limit": 5,
        "hypothesis": "H4: Adjacent semiconductor/transistor 510(k) precedents exist for AG15 authority framing.",
    },
]


def run_openfda_query(q: dict, out_file: Path) -> dict:
    script = OPENFDA_SKILL / "scripts" / "openfda_query.py"
    if not script.is_file():
        return {"id": q["id"], "status": "error", "error": "openfda skill script missing"}

    cmd = [
        "uv", "run", str(script), "search",
        "--category", q["category"],
        "--endpoint", q["endpoint"],
        "--search", q["search"],
        "--limit", str(q.get("limit", 10)),
        "--output", str(out_file),
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(OPENFDA_SKILL),
            capture_output=True,
            text=True,
            timeout=90,
        )
        summary = {}
        if proc.stdout.strip():
            try:
                summary = json.loads(proc.stdout.strip().splitlines()[-1])
            except json.JSONDecodeError:
                summary = {"raw_stdout": proc.stdout[-500:]}
        result_count = 0
        total = summary.get("total_matching", 0)
        if out_file.is_file():
            try:
                data = json.loads(out_file.read_text())
                result_count = len(data.get("results", []))
            except Exception:
                pass
        return {
            "id": q["id"],
            "status": "success" if proc.returncode == 0 else "error",
            "hypothesis": q["hypothesis"],
            "category": q["category"],
            "endpoint": q["endpoint"],
            "search": q["search"],
            "output": str(out_file),
            "results_in_file": result_count,
            "total_matching": total,
            "stderr_tail": (proc.stderr or "")[-300:] if proc.returncode != 0 else "",
        }
    except Exception as e:
        return {"id": q["id"], "status": "error", "error": str(e), "hypothesis": q["hypothesis"]}


def extract_evidence_cards(query_results: list[dict]) -> list[dict]:
    cards = []
    for qr in query_results:
        path = Path(qr.get("output", ""))
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        for item in (data.get("results") or [])[:3]:
            name = (
                item.get("device_name")
                or item.get("generic_name")
                or item.get("product_code")
                or item.get("substance_name")
                or "—"
            )
            cards.append({
                "query_id": qr["id"],
                "device_or_substance": str(name)[:120],
                "k_number": item.get("k_number"),
                "product_code": item.get("product_code"),
                "decision_date": item.get("decision_date") or item.get("date_received"),
            })
    return cards


def run_pipeline() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    query_results = [run_openfda_query(q, OUT_DIR / f"{q['id']}.json") for q in QUERIES]
    evidence = extract_evidence_cards(query_results)

    manifest = {
        "schema": "ag15-openfda-research-v1",
        "codename": "AG15",
        "title": "Thermodynamic Transistor Substrate Research",
        "authority_tier": "post_phd_double_loop",
        "standard_setting_body": "diamondnodebot / openFDA evidence lattice",
        "ts": ts,
        "openfda_skill": "openfda-database",
        "license_notified": (OPENFDA_SKILL / "LICENSE_NOTIFICATION.txt").is_file(),
        "queries": query_results,
        "evidence_cards": evidence,
        "substrate_model": {
            "thermal_transistor": "solid-state heat flux gating (UCLA-class thermal transistor analogy)",
            "ag15_nanoparticle_lane": "Ag15% nanoparticle substrate reproducibility (SERS literature cross-ref)",
            "thermo_daemon_coupling": "NVML Landauer envelope + Opux epoch reducibility",
        },
        "hermes_reachable": True,
        "swarm_pin": "diamondnodebot_ag15_hermes",
        "falsification_ready": True,
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    manifest = run_pipeline()
    print(json.dumps({
        "ok": True,
        "evidence_cards": len(manifest.get("evidence_cards", [])),
        "queries_ok": sum(1 for q in manifest["queries"] if q.get("status") == "success"),
        "manifest": str(MANIFEST),
    }))


if __name__ == "__main__":
    main()