"""thrml-backed kernels: group consensus and the thermodynamic energy gate.

The consensus sampler is a GPU load like any other on this node, so it asks the
Thermodynamic Daemon for a power budget before drawing — the same contract the
existing ``thrml_ebm_sampler`` honours.
"""

from .energy_gate import GateDecision, ThermoState, energy_gate, read_thermo_state
from .thrml_consensus import (
    ConsensusResult,
    DissentRecord,
    MemberPosition,
    aggregate_consensus,
    build_ising,
    engine_label,
    greedy_coloring,
    ising_energy,
    thrml_available,
    weighted_average,
)

__all__ = [
    "ConsensusResult",
    "DissentRecord",
    "GateDecision",
    "MemberPosition",
    "ThermoState",
    "aggregate_consensus",
    "build_ising",
    "energy_gate",
    "engine_label",
    "greedy_coloring",
    "ising_energy",
    "read_thermo_state",
    "thrml_available",
    "weighted_average",
]
