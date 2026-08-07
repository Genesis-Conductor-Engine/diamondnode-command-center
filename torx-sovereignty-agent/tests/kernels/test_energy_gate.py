"""The thermodynamic energy gate fails closed.

The gate decides whether the consensus sampler may draw GPU power. Every test
here monkeypatches :func:`read_thermo_state`: the daemon is not guaranteed to be
running, and a test that silently passed because the daemon happened to be up
would be worthless.

:func:`test_present_but_empty_telemetry_fails_closed` covers a real defect found
against the live daemon. When NVML cannot attach, the daemon answers HTTP 200
with every field ``null`` — a *successful* read carrying no readings. The
per-threshold checks each skip their ``None`` and the gate fell through to
"within envelope", opening on a GPU nothing was known about.
"""

from __future__ import annotations

import pytest

# The package re-exports a *function* named ``energy_gate``, which shadows the
# submodule of the same name — ``from src.kernels import energy_gate`` yields the
# function. Reach the module explicitly so monkeypatching targets the right
# object.
import src.kernels.energy_gate as eg
from src.kernels.energy_gate import (
    GATE_TEMP_C,
    GATE_UTIL_PCT,
    GATE_VRAM_PCT,
    ThermoState,
    energy_gate,
)


def state(temp=45.0, vram=10.0, util=5.0, source="thermo-daemon"):
    return ThermoState(
        source=source,
        temp_c=temp,
        vram_pct=vram,
        util_pct=util,
        power_w=50.0,
        vram_gate=GATE_VRAM_PCT,
        util_gate=GATE_UTIL_PCT,
    )


@pytest.fixture(autouse=True)
def no_force_cpu(monkeypatch):
    monkeypatch.delenv("TORX_FORCE_CPU", raising=False)


# --------------------------------------------------------------------------
# open
# --------------------------------------------------------------------------


def test_gate_opens_inside_the_governor_envelope():
    decision = energy_gate(state())
    assert decision.gpu_ok is True
    assert "within governor envelope" in decision.reason


# --------------------------------------------------------------------------
# closed on breach
# --------------------------------------------------------------------------


def test_gate_closes_on_temperature_breach():
    decision = energy_gate(state(temp=GATE_TEMP_C))
    assert decision.gpu_ok is False
    assert "temp" in decision.reason


def test_gate_closes_well_below_the_hardware_threshold():
    """89.6 C is the hardware throttle; the gate must act far earlier."""
    assert GATE_TEMP_C < 89.6
    assert energy_gate(state(temp=76.0)).gpu_ok is False


def test_gate_closes_on_vram_breach():
    decision = energy_gate(state(vram=GATE_VRAM_PCT + 1))
    assert decision.gpu_ok is False
    assert "vram" in decision.reason


def test_gate_closes_on_utilisation_breach():
    decision = energy_gate(state(util=GATE_UTIL_PCT + 1))
    assert decision.gpu_ok is False
    assert "util" in decision.reason


def test_daemon_supplied_thresholds_override_the_defaults():
    """The daemon owns the governor envelope, so its thresholds win."""
    tight = ThermoState(
        source="thermo-daemon",
        temp_c=45.0,
        vram_pct=15.0,
        util_pct=5.0,
        power_w=50.0,
        vram_gate=10.0,  # tighter than the 40% default
        util_gate=GATE_UTIL_PCT,
    )
    assert energy_gate(tight).gpu_ok is False


# --------------------------------------------------------------------------
# failsafe
# --------------------------------------------------------------------------


def test_no_telemetry_at_all_fails_closed(monkeypatch):
    monkeypatch.setattr(eg, "read_thermo_state", lambda: None)
    decision = energy_gate()
    assert decision.gpu_ok is False
    assert "no telemetry" in decision.reason
    assert decision.thermo is None


def test_present_but_empty_telemetry_fails_closed():
    """A 200 response with all-null fields must not open the gate.

    This is the shape the thermodynamic daemon returns when the NVIDIA driver is
    not loaded. Treating it as "nothing exceeded a threshold" would open the
    gate on an unknown GPU — the one inference absent telemetry must never
    support.
    """
    empty = ThermoState(
        source="thermo-daemon",
        temp_c=None,
        vram_pct=None,
        util_pct=None,
        power_w=None,
        vram_gate=GATE_VRAM_PCT,
        util_gate=GATE_UTIL_PCT,
    )
    decision = energy_gate(empty)
    assert decision.gpu_ok is False
    assert "empty" in decision.reason
    assert decision.thermo is empty


def test_partial_telemetry_is_still_evaluated():
    """One missing field is not the same as no telemetry.

    Temperature alone is enough to make a decision; discarding a usable reading
    would degrade further than necessary.
    """
    partial = ThermoState(
        source="nvidia-smi",
        temp_c=85.0,
        vram_pct=None,
        util_pct=None,
        power_w=None,
        vram_gate=GATE_VRAM_PCT,
        util_gate=GATE_UTIL_PCT,
    )
    decision = energy_gate(partial)
    assert decision.gpu_ok is False
    assert "temp" in decision.reason

    cool_partial = ThermoState(
        source="nvidia-smi",
        temp_c=40.0,
        vram_pct=None,
        util_pct=None,
        power_w=None,
        vram_gate=GATE_VRAM_PCT,
        util_gate=GATE_UTIL_PCT,
    )
    assert energy_gate(cool_partial).gpu_ok is True


def test_force_cpu_env_var_closes_the_gate(monkeypatch):
    monkeypatch.setenv("TORX_FORCE_CPU", "1")
    decision = energy_gate(state())
    assert decision.gpu_ok is False
    assert "TORX_FORCE_CPU" in decision.reason


def test_unreachable_daemon_does_not_raise(monkeypatch):
    """A dead daemon degrades the gate; it must not propagate an exception."""
    monkeypatch.setattr(eg, "DAEMON_STATE_URL", "http://127.0.0.1:1/state")

    def no_smi(*a, **k):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(eg.subprocess, "run", no_smi)
    assert eg.read_thermo_state() is None
    assert energy_gate().gpu_ok is False


# --------------------------------------------------------------------------
# serialisation
# --------------------------------------------------------------------------


def test_decision_serialises_for_the_audit_record():
    doc = energy_gate(state()).to_json()
    assert doc["gpu_ok"] is True
    assert doc["thermo"]["source"] == "thermo-daemon"
    assert doc["thermo"]["vram_gate"] == GATE_VRAM_PCT


def test_decision_serialises_when_there_is_no_telemetry(monkeypatch):
    monkeypatch.setattr(eg, "read_thermo_state", lambda: None)
    doc = energy_gate().to_json()
    assert doc["gpu_ok"] is False
    assert doc["thermo"] is None
