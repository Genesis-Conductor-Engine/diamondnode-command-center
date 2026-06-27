#!/usr/bin/env python3
"""
Rψ Validation Engine Pipeline Executor

Executes the daily Rψ Validation Engine pipeline tasks:
1. rpsi_black_hole_lift_v1
2. rpsi_weak_field_refined
3. rpsi_stability
4. rpsi_sparc_ingest
5. rpsi_full_report

Usage:
    python3 execute_rpsi_pipeline.py --all
    python3 execute_rpsi_pipeline.py --tasks rpsi_black_hole_lift_v1,rpsi_weak_field_refined
"""

import asyncio
import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from pathlib import Path
import uuid
import urllib.request
import threading

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

RPSI_SCHEDULE_ID = os.environ.get('RPSI_SCHEDULE_ID', 'daily_rpsi_1am_est')
ARTIFACT_DIR = Path(os.environ.get('RPSI_ARTIFACT_DIR',
    '/home/diamondnode/genesis_conductor_engine/swarm/remote/rpsi-artifacts/daily'))
LOG_DIR = Path(os.environ.get('RPSI_LOG_DIR',
    '/home/diamondnode/genesis_conductor_engine/swarm/logs/rpsi'))
SCRIPTS_DIR = Path('/home/diamondnode/rpsi-validation-engine')
SWARM_DIR    = Path('/home/diamondnode/genesis_conductor_engine/swarm')
DN_INGEST_URL = os.environ.get('DN_INGEST_URL',
    'https://dn.genesisconductor.io/api/rpsi/ingest')

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Thermo-attestation — lazy import so pipeline works even if swarm dir unavailable
try:
    import sys as _sys
    if str(SWARM_DIR) not in _sys.path:
        _sys.path.insert(0, str(SWARM_DIR))
    from thermo_attestation_live_driver import generate_live_thermo_evt
    from thermo_attestation_primitive import ThermoAttestationPrimitive
    from procedural_truth_verifier import ProceduralTruthVerifier, VerifierConfig
    _THERMO_AVAILABLE = True
    _PTV_AVAILABLE = True
except ImportError:
    _THERMO_AVAILABLE = False
    _PTV_AVAILABLE = False

def _post_to_dn_bg(evt: Dict[str, Any]) -> None:
    """Fire-and-forget POST to dn.genesisconductor.io — runs in background thread."""
    try:
        data = json.dumps(evt).encode()
        req  = urllib.request.Request(
            DN_INGEST_URL,
            data=data,
            headers={"Content-Type": "application/json",
                     "User-Agent": "rpsi-pipeline/1.0"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            logger.debug(f"DN ingest OK: {resp.status} {evt.get('record_type','?')}")
    except Exception as exc:
        logger.debug(f"DN ingest skipped ({type(exc).__name__}): {exc}")


def _run_physics_script(script_name: str, timeout: int = 300) -> Dict[str, Any]:
    """Run a physics script, capture JSON stdout, return parsed result."""
    script = SCRIPTS_DIR / script_name
    proc = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True, text=True, timeout=timeout
    )
    if proc.returncode not in (0, 1):  # 1 = crystal fail, not a crash
        raise RuntimeError(f"{script_name} exited {proc.returncode}: {proc.stderr[:500]}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{script_name} produced invalid JSON: {exc}\n{proc.stdout[:300]}")


class RpsiTask:
    def __init__(self, name, description, task_type, flow_step,
                 timeout_minutes=30, retry_attempts=3, dependencies=None):
        self.name = name
        self.description = description
        self.task_type = task_type
        self.flow_step = flow_step
        self.timeout_minutes = timeout_minutes
        self.retry_attempts = retry_attempts
        self.dependencies = dependencies or []
        self.status = "pending"
        self.result = None
        self.error = None
        self.start_time = None
        self.end_time = None
        self.attempts = 0

    async def execute(self) -> Dict[str, Any]:
        raise NotImplementedError

    def _wrap(self, physics_result: Dict) -> Dict[str, Any]:
        """Wrap a physics script result in the standard task envelope."""
        cs = physics_result.get("crystal_score", {})
        passed = cs.get("passed", False)
        return {
            "success": passed,
            "message": f"{self.name} {'passed' if passed else 'failed'} crystal score",
            "crystal_score": cs,
            "physics_evt_id": physics_result.get("evt_id"),
            "timestamp": physics_result.get("timestamp"),
        }

    def _envelope(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "task_name": self.name,
            "result": self.result,
            "error": self.error,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "attempts": self.attempts,
        }

    async def _execute_script(self, script_name: str) -> Dict[str, Any]:
        self.status = "running"
        self.start_time = datetime.now(timezone.utc).isoformat()
        self.attempts += 1
        try:
            logger.info(f"Running {script_name} ...")
            raw = await asyncio.get_event_loop().run_in_executor(
                None, _run_physics_script, script_name, self.timeout_minutes * 60
            )
            self.result = self._wrap(raw)
            self.status = "completed"
            passed = self.result.get("success", False)
            level = logger.info if passed else logger.warning
            level(f"{'PASS' if passed else 'FAIL'} {self.name}: {self.result['message']}")
            # Live feed to dn.genesisconductor.io
            _dn_evt = {**raw, "pipeline_task": self.name, "pipeline_step": self.flow_step}
            _t = threading.Thread(target=_post_to_dn_bg, args=(_dn_evt,), daemon=False)
            _t.start()
            if hasattr(self, "_dn_threads"): self._dn_threads.append(_t)
        except Exception as exc:
            self.error = str(exc)
            self.status = "failed"
            logger.error(f"ERROR {self.name}: {exc}")
        self.end_time = datetime.now(timezone.utc).isoformat()
        return self._envelope()


class RpsiBlackHoleLiftV1(RpsiTask):
    def __init__(self):
        super().__init__("rpsi_black_hole_lift_v1",
                         "Near-horizon Schwarzschild R(ψ) scalar field",
                         "rpsi.validation", 1, timeout_minutes=30,
                         retry_attempts=3, dependencies=[])

    async def execute(self):
        return await self._execute_script("black_hole_lift.py")


class RpsiWeakFieldRefined(RpsiTask):
    def __init__(self):
        super().__init__("rpsi_weak_field_refined",
                         "Weak-field BVP — linearised R(ψ) operator",
                         "rpsi.validation", 2, timeout_minutes=25,
                         retry_attempts=3, dependencies=["rpsi_black_hole_lift_v1"])

    async def execute(self):
        return await self._execute_script("rpsi_refined_operator.py")


class RpsiStability(RpsiTask):
    def __init__(self):
        super().__init__("rpsi_stability",
                         "Eigenvalue stability of linearised R(ψ) operator",
                         "rpsi.validation", 3, timeout_minutes=20,
                         retry_attempts=3, dependencies=["rpsi_weak_field_refined"])

    async def execute(self):
        return await self._execute_script("rpsi_stability.py")


class RpsiSparcIngest(RpsiTask):
    def __init__(self):
        super().__init__("rpsi_sparc_ingest",
                         "SPARC rotation-curve ingest — R(ψ) deep-field limit",
                         "rpsi.ingestion", 4, timeout_minutes=15,
                         retry_attempts=2, dependencies=["rpsi_stability"])

    async def execute(self):
        return await self._execute_script("sparc_diamondnode_ingest.py")


class RpsiFullReport(RpsiTask):
    def __init__(self):
        super().__init__("rpsi_full_report",
                         "Aggregate crystal scores → markdown pipeline report",
                         "rpsi.reporting", 5, timeout_minutes=10,
                         retry_attempts=2, dependencies=["rpsi_sparc_ingest"])

    async def execute(self) -> Dict[str, Any]:
        self.status = "running"
        self.start_time = datetime.now(timezone.utc).isoformat()
        self.attempts += 1
        try:
            self.result = await asyncio.get_event_loop().run_in_executor(
                None, self._build_report)
            self.status = "completed"
            logger.info(f"Report written: {self.result.get('report_path')}")
            # Post report EVT to DN live feed
            _report_evt = {
                "evt_id": f"evt_rpsi_report_{uuid.uuid4().hex[:8]}",
                "record_type": "rpsi_full_report_result",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "pipeline_score": self.result.get("pipeline_score"),
                "evts_processed": self.result.get("evts_processed"),
                "report_path": self.result.get("report_path"),
                "apex_goal": "R(\u03c8) : \u0394g_\u03bc\u03bd^(obs) \u2192 0",
            }
            _t = threading.Thread(target=_post_to_dn_bg, args=(_report_evt,), daemon=False)
            _t.start()
            if hasattr(self, "_dn_threads"): self._dn_threads.append(_t)
        except Exception as exc:
            self.error = str(exc)
            self.status = "failed"
            logger.error(f"ERROR {self.name}: {exc}")
        self.end_time = datetime.now(timezone.utc).isoformat()
        return self._envelope()

    def _build_report(self) -> Dict[str, Any]:
        evts = sorted(ARTIFACT_DIR.glob("evt_*.json"), key=lambda p: p.stat().st_mtime)
        # Deduplicate: keep only the most recent EVT per record_type
        seen: dict = {}
        for p in evts:
            try:
                data = json.loads(p.read_text())
                rt = data.get("record_type", "")
                if rt and data.get("crystal_score"):
                    seen[rt] = data   # later files overwrite earlier ones
            except Exception:
                continue

        sections = []
        total_weight = 0.0
        weighted_score = 0.0

        for data in seen.values():
            try:
                cs = data.get("crystal_score", {})
                if not cs:
                    continue
                val   = cs.get("value", cs.get("stability_index", 0))
                tgt   = cs.get("target", 1)
                w     = cs.get("weight", 0)
                ok    = cs.get("passed", False)
                total_weight += w
                weighted_score += w * (1.0 if ok else 0.0)
                sections.append(
                    f"| {data.get('record_type','?'):<40} | {val:.4f} | {tgt} | {w} | {'PASS' if ok else 'FAIL'} |"
                )
            except Exception:
                continue

        pipeline_score = weighted_score / total_weight if total_weight else 0.0
        date_str = datetime.now().strftime('%Y-%m-%d')
        report_lines = [
            f"# Rψ Validation Engine — Daily Report {date_str}",
            f"**Pipeline score**: {pipeline_score:.3f}  (weighted crystal pass fraction)",
            "",
            "| Task | Value | Target | Weight | Status |",
            "|------|-------|--------|--------|--------|",
        ] + sections + [
            "",
            f"*Generated {datetime.now(timezone.utc).isoformat()} UTC*",
            f"apex_goal: R(ψ) : Δg_μν^(obs) → 0",
        ]

        report_path = ARTIFACT_DIR / f"rpsi_report_{date_str}.md"
        report_path.write_text("\n".join(report_lines))

        json_path = ARTIFACT_DIR / f"rpsi_report_{date_str}.json"
        report_data = {
            "report_date": date_str,
            "pipeline_score": pipeline_score,
            "evts_processed": len(sections),
            "report_path": str(report_path),
        }
        json_path.write_text(json.dumps(report_data, indent=2))
        return {"success": True, "pipeline_score": pipeline_score,
                "report_path": str(report_path), "evts_processed": len(sections)}



class RpsiBlackHoleThermodynamics(RpsiTask):
    def __init__(self):
        super().__init__("rpsi_bh_thermodynamics",
                         "R(psi) Black Hole Thermodynamics Phase II (Hayward)",
                         "rpsi.validation", 6, timeout_minutes=10,
                         retry_attempts=2, dependencies=["rpsi_full_report"])

    async def execute(self):
        return await self._execute_script("rpsi_blackhole_thermodynamics.py")


class RpsiThermoAttestation(RpsiTask):
    """Hardware attestation task — captures live RAPL telemetry to prove computation occurred."""

    def __init__(self):
        super().__init__("rpsi_thermo_attestation",
                         "Live RAPL hardware attestation — Landauer efficiency ZK fingerprint",
                         "rpsi.attestation", 7, timeout_minutes=5,
                         retry_attempts=2, dependencies=["rpsi_bh_thermodynamics"])

    async def execute(self) -> Dict[str, Any]:
        self.status = "running"
        self.start_time = datetime.now(timezone.utc).isoformat()
        self.attempts += 1
        try:
            if not _THERMO_AVAILABLE:
                raise RuntimeError("thermo_attestation modules not available in SWARM_DIR")
            primitive = ThermoAttestationPrimitive(
                node_id="diamondnode",
                version="0.2",
                landauer_target_efficiency=0.55,  # realistic for idle CPU
            )
            # Run blocking 0.8s two-sample RAPL window in executor
            evt = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: generate_live_thermo_evt(
                    primitive,
                    source="rapl",
                    seismic_run_id=f"rpsi_{self.start_time[:10]}",
                    agent_id="rpsi_pipeline",
                )
            )
            eff    = evt.get("landauer", {}).get("efficiency", 0.0)
            meets  = evt.get("landauer", {}).get("meets_target", False)
            driver = evt.get("telemetry", {}).get("driver", "unknown")
            fp     = evt.get("landmark", {}).get("fingerprint_sha256", "")[:16]
            self.result = {
                "success": meets,
                "message": f"thermo_attestation driver={driver} eff={eff:.4f} fp={fp}…",
                "crystal_score": {
                    "metric": "landauer_efficiency",
                    "value": eff,
                    "target": 0.55,
                    "passed": meets,
                    "weight": 0.1,
                    "driver": driver,
                    "fingerprint_sha256": evt.get("landmark", {}).get("fingerprint_sha256", ""),
                },
                "physics_evt_id": evt.get("evt_id"),
                "timestamp": evt.get("timestamp"),
            }
            self.status = "completed"
            level = logger.info if meets else logger.warning
            level(f"{'PASS' if meets else 'FAIL'} {self.name}: {self.result['message']}")
            # Post full thermo EVT to DN
            _t = threading.Thread(target=_post_to_dn_bg, args=(evt,), daemon=False)
            _t.start()
            if hasattr(self, "_dn_threads"): self._dn_threads.append(_t)
        except Exception as exc:
            self.error = str(exc)
            self.status = "failed"
            logger.error(f"ERROR {self.name}: {exc}")
        self.end_time = datetime.now(timezone.utc).isoformat()
        return self._envelope()

class RpsiProceduralTruth(RpsiTask):
    """Double loopback procedural truth verification — q-mem vs ollama eigen seal."""

    def __init__(self):
        super().__init__("rpsi_procedural_truth",
                         "Procedural truth verifier — Rule 30 VDF + eigen loopback",
                         "rpsi.verification", 8, timeout_minutes=5,
                         retry_attempts=2, dependencies=["rpsi_thermo_attestation"])

    async def execute(self) -> Dict[str, Any]:
        self.status = "running"
        self.start_time = datetime.now(timezone.utc).isoformat()
        self.attempts += 1
        try:
            if not _PTV_AVAILABLE:
                raise RuntimeError("procedural_truth_verifier not available in SWARM_DIR")
            cfg = VerifierConfig(
                qmem_base=os.environ.get("PTV_QMEM_BASE", "http://127.0.0.1:8082"),
                ollama_base=os.environ.get("PTV_OLLAMA_BASE", "http://127.0.0.1:11434"),
                agent_id="rpsi_pipeline",
            )
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: ProceduralTruthVerifier(cfg).verify(
                    source="rpsi_pipeline",
                    seismic_run_id=f"rpsi_{self.start_time[:10]}",
                ),
            )
            evt = result.evt
            cs = result.crystal_score
            passed = result.passed
            evt_path = ARTIFACT_DIR / f"{evt['evt_id']}.json"
            evt_path.write_text(json.dumps(evt, indent=2))
            self.result = {
                "success": passed,
                "message": (
                    f"procedural_truth score={cs.get('value', 0):.4f} "
                    f"delta_eigen={result.metrics.get('loopback_delta_eigen', 0):.4f}"
                ),
                "crystal_score": cs,
                "physics_evt_id": evt.get("evt_id"),
                "timestamp": evt.get("timestamp"),
                "artifact_path": str(evt_path),
            }
            self.status = "completed"
            (logger.info if passed else logger.warning)(
                f"{'PASS' if passed else 'FAIL'} {self.name}: {self.result['message']}")
            _t = threading.Thread(target=_post_to_dn_bg, args=(evt,), daemon=False)
            _t.start()
            if hasattr(self, "_dn_threads"):
                self._dn_threads.append(_t)
        except Exception as exc:
            self.error = str(exc)
            self.status = "failed"
            logger.error(f"ERROR {self.name}: {exc}")
        self.end_time = datetime.now(timezone.utc).isoformat()
        return self._envelope()


TASK_REGISTRY = {
    "rpsi_black_hole_lift_v1": RpsiBlackHoleLiftV1,
    "rpsi_weak_field_refined":  RpsiWeakFieldRefined,
    "rpsi_stability":           RpsiStability,
    "rpsi_sparc_ingest":        RpsiSparcIngest,
    "rpsi_full_report":         RpsiFullReport,
    "rpsi_bh_thermodynamics":  RpsiBlackHoleThermodynamics,
    "rpsi_thermo_attestation": RpsiThermoAttestation,
    "rpsi_procedural_truth":   RpsiProceduralTruth,
}


class RpsiPipelineExecutor:
    def __init__(self, schedule_id=RPSI_SCHEDULE_ID):
        self.schedule_id = schedule_id
        self.tasks: Dict[str, RpsiTask] = {}
        self.results: Dict[str, Dict] = {}
        self.start_time = None
        self.end_time = None
        self.status = "pending"
        self._dn_threads: list = []

    def add_task(self, name: str):
        if name in TASK_REGISTRY:
            self.tasks[name] = TASK_REGISTRY[name]()
        else:
            logger.warning(f"Unknown task: {name}")

    def _execution_order(self):
        return sorted(self.tasks.keys(), key=lambda n: self.tasks[n].flow_step)

    def _summary(self):
        completed = sum(1 for r in self.results.values() if r.get("status") == "completed")
        failed    = sum(1 for r in self.results.values() if r.get("status") == "failed")
        return {"total": len(self.tasks), "completed": completed, "failed": failed,
                "success_rate": completed / len(self.tasks) if self.tasks else 0}

    async def execute(self):
        self.start_time = datetime.now(timezone.utc).isoformat()
        self.status = "running"
        logger.info(f"Starting Rψ Validation Pipeline: {self.schedule_id}")

        for name in self._execution_order():
            task = self.tasks[name]
            logger.info(f"--> {name} (step {task.flow_step})")
            task._dn_threads = self._dn_threads
            result = await task.execute()
            self.results[name] = result
            if result.get("status") == "failed":
                logger.error(f"Pipeline halted at: {name}")
                self.status = "failed"
                break

        if self.status == "running":
            self.status = "completed"

        self.end_time = datetime.now(timezone.utc).isoformat()
        summary = self._summary()
        logger.info(f"Pipeline {self.status}: {summary}")
        # Join all per-task DN poster threads (max 10s)
        for _t in getattr(self, "_dn_threads", []):
            _t.join(timeout=10)
        # Post pipeline completion EVT to DN
        _pipeline_evt = {
            "evt_id": f"evt_pipeline_{uuid.uuid4().hex[:8]}",
            "record_type": "rpsi_pipeline_run",
            "timestamp": self.end_time,
            "schedule_id": self.schedule_id,
            "status": self.status,
            "summary": summary,
            "apex_goal": "R(ψ) : Δg_μν^(obs) → 0",
        }
        _pt = threading.Thread(target=_post_to_dn_bg, args=(_pipeline_evt,), daemon=False)
        _pt.start(); _pt.join(timeout=10)
        return {
            "schedule_id": self.schedule_id,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "tasks": self.results,
            "summary": summary,
        }


async def main():
    parser = argparse.ArgumentParser(description='Rψ Validation Engine Pipeline Executor')
    parser.add_argument('--schedule-id', default=RPSI_SCHEDULE_ID)
    parser.add_argument('--tasks', default=None,
                        help='Comma-separated list of tasks')
    parser.add_argument('--all', action='store_true', help='Execute all tasks')
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG','INFO','WARNING','ERROR'])
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    executor = RpsiPipelineExecutor(schedule_id=args.schedule_id)

    if args.tasks:
        task_list = [t.strip() for t in args.tasks.split(',')]
    elif args.all:
        task_list = list(TASK_REGISTRY.keys())
    else:
        task_list = list(TASK_REGISTRY.keys())  # default: all

    for t in task_list:
        executor.add_task(t)

    if not executor.tasks:
        logger.error("No tasks to execute!")
        sys.exit(1)

    try:
        result = await executor.execute()
        sys.exit(0 if result['status'] == 'completed' else 1)
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted")
        sys.exit(130)
    except Exception as exc:
        logger.error(f"Pipeline error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
