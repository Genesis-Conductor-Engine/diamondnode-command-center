"""TORX state and decision layer.

Named ``torx_layer`` rather than ``torx`` on purpose: ``torx`` is the import name
of the installed Extropic distribution (``extro-torx``) that this package binds
to, and a local package of that name on ``sys.path`` would shadow it — every
kernel call would silently fall back to the pure-Python baseline.
"""

from .circuits import (
    DecisionInputs,
    DecisionMarginals,
    backend_status,
    bridge_candidate_pdit,
    evaluate_categorical,
    evaluate_decision,
    tension_class_pdit,
    torx_available,
    veto_is_structural,
)
from .kernels import (
    BridgeConstraints,
    TensionEstimate,
    bridge_selection,
    diffuse_prior,
    intent_update,
    tension_classification,
    tension_gradient,
)
from .state import (
    BRIDGE_STRATEGIES,
    RESOLUTION_STATES,
    TENSION_CLASSES,
    TENSION_DIMENSIONS,
    PBit,
    PDit,
    PMode,
    TorxDecisionState,
)

__all__ = [
    "BRIDGE_STRATEGIES",
    "BridgeConstraints",
    "DecisionInputs",
    "DecisionMarginals",
    "PBit",
    "PDit",
    "PMode",
    "RESOLUTION_STATES",
    "TENSION_CLASSES",
    "TENSION_DIMENSIONS",
    "TensionEstimate",
    "TorxDecisionState",
    "backend_status",
    "bridge_candidate_pdit",
    "bridge_selection",
    "diffuse_prior",
    "evaluate_categorical",
    "evaluate_decision",
    "intent_update",
    "tension_class_pdit",
    "tension_classification",
    "tension_gradient",
    "torx_available",
    "veto_is_structural",
]
