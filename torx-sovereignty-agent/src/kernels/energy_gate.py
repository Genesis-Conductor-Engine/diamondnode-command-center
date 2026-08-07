"""Thermodynamic energy gate for the thrml sampling stage.

The node's Thermodynamic Daemon (``127.0.0.1:9100/state``) is the governor for
GPU work: a load negotiates its power budget before drawing any. The thrml
consensus sampler is such a load, so it asks the same question the existing
``thrml_ebm_sampler`` asks before it touches the GPU.

This matters to the model card, not just to the hardware. ``runtime.critical_path``
excludes nothing about consensus — group-intent aggregation *is* on the critical
path — so the gate here never blocks the decision. It only chooses **where** the
sampling runs: inside the governor envelope it may use the GPU, outside it it
falls back to CPU or to the exact/mean-field baseline. A closed gate degrades
throughput; it never degrades the answer's contract.

Failure is failsafe-closed: no telemetry at all means CPU only, matching
``thrml_ebm_sampler.energy_gate``.
"""

from __future__ import annotations

import json
import os
import subprocess
import urllib.request
from dataclasses import dataclass
from typing import Any

DAEMON_STATE_URL = os.environ.get(
    "THERMO_DAEMON_URL", "http://127.0.0.1:9100/state"
)

# Same thresholds the existing sampler uses; well under the 89.6 C hardware
# threshold on this node's GTX 1650.
GATE_TEMP_C = 75.0
GATE_VRAM_PCT = 40.0
GATE_UTIL_PCT = 25.0

_TIMEOUT_S = 3.0


@dataclass(frozen=True, slots=True)
class ThermoState:
    source: str
    temp_c: float | None
    vram_pct: float | None
    util_pct: float | None
    power_w: float | None
    vram_gate: float
    util_gate: float

    def to_json(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "temp_c": self.temp_c,
            "vram_pct": self.vram_pct,
            "util_pct": self.util_pct,
            "power_w": self.power_w,
            "vram_gate": self.vram_gate,
            "util_gate": self.util_gate,
        }


@dataclass(frozen=True, slots=True)
class GateDecision:
    gpu_ok: bool
    reason: str
    thermo: ThermoState | None

    def to_json(self) -> dict[str, Any]:
        return {
            "gpu_ok": self.gpu_ok,
            "reason": self.reason,
            "thermo": self.thermo.to_json() if self.thermo else None,
        }


def read_thermo_state() -> ThermoState | None:
    """Daemon first (it owns the governor envelope), then ``nvidia-smi``."""
    try:
        with urllib.request.urlopen(DAEMON_STATE_URL, timeout=_TIMEOUT_S) as resp:
            d = json.loads(resp.read())
        thresholds = d.get("governor_envelope", {}).get("thresholds", {})
        return ThermoState(
            source="thermo-daemon",
            temp_c=d.get("temperature_c"),
            vram_pct=d.get("vram", {}).get("used_pct"),
            util_pct=d.get("utilization", {}).get("gpu_pct"),
            power_w=d.get("power_limit_w"),
            vram_gate=float(thresholds.get("vram_pct", GATE_VRAM_PCT)),
            util_gate=float(thresholds.get("gpu_util", GATE_UTIL_PCT)),
        )
    except Exception:
        pass
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu,memory.used,memory.total,"
                "utilization.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip().split(", ")
        temp, used, total, util, power = (float(x) for x in out)
        return ThermoState(
            source="nvidia-smi",
            temp_c=temp,
            vram_pct=100.0 * used / total if total else None,
            util_pct=util,
            power_w=power,
            vram_gate=GATE_VRAM_PCT,
            util_gate=GATE_UTIL_PCT,
        )
    except Exception:
        return None


def energy_gate(thermo: ThermoState | None = None) -> GateDecision:
    """May the sampler draw GPU power right now?"""
    if os.environ.get("TORX_FORCE_CPU") == "1":
        return GateDecision(False, "TORX_FORCE_CPU=1", thermo)
    state = thermo if thermo is not None else read_thermo_state()
    if state is None:
        return GateDecision(
            False, "no telemetry (daemon down, nvidia-smi failed) — CPU only", None
        )
    # A *successful* read carrying no usable readings is not the same as a
    # failed read, and it is the more dangerous case: the daemon answers 200
    # with every field null when NVML cannot attach (driver not loaded), so the
    # per-threshold checks below would each skip their None and fall through to
    # "within envelope" — opening the gate on a GPU we know nothing about.
    # Absent telemetry fails closed, exactly as a failed read does.
    if state.temp_c is None and state.vram_pct is None and state.util_pct is None:
        return GateDecision(
            False,
            f"telemetry present but empty (source={state.source}) — "
            "no usable GPU readings, CPU only",
            state,
        )
    if state.temp_c is not None and state.temp_c >= GATE_TEMP_C:
        return GateDecision(
            False, f"temp {state.temp_c}C >= {GATE_TEMP_C}C", state
        )
    if state.vram_pct is not None and state.vram_pct >= state.vram_gate:
        return GateDecision(
            False, f"vram {state.vram_pct:.1f}% >= {state.vram_gate}%", state
        )
    if state.util_pct is not None and state.util_pct >= state.util_gate:
        return GateDecision(
            False, f"gpu util {state.util_pct}% >= {state.util_gate}%", state
        )
    return GateDecision(True, "within governor envelope", state)
