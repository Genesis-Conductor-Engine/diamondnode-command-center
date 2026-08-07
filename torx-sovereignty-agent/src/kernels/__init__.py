"""thrml-backed kernels: group consensus and the thermodynamic energy gate.

The consensus sampler is a GPU load like any other on this node, so it asks the
Thermodynamic Daemon for a power budget before drawing — the same contract the
existing ``thrml_ebm_sampler`` honours.

Note the deliberate rename: the gate *function* is re-exported as
``evaluate_energy_gate``, not ``energy_gate``. A re-export under the submodule's
own name shadows the submodule, so ``import src.kernels.energy_gate`` would bind
the function instead of the module and the module would be unreachable by normal
import syntax. Import the function from its module when you want its own name:
``from src.kernels.energy_gate import energy_gate``.
"""

from . import energy_gate, thrml_consensus
from .energy_gate import GateDecision, ThermoState, read_thermo_state
from .energy_gate import energy_gate as evaluate_energy_gate
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
    "evaluate_energy_gate",
    "greedy_coloring",
    "ising_energy",
    "read_thermo_state",
    "thrml_available",
    "thrml_consensus",
    "weighted_average",
]
