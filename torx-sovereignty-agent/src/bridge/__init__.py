"""Bridge engine: sovereignty-guarded tension reduction over group intents.

Exposes the strategy catalogue (:data:`BRIDGE_STRATEGIES`), the simulation
record (:class:`BridgeSimulation`), the sovereignty guard
(:class:`SovereigntyGuard`), and the lifecycle engine (:class:`BridgeEngine`).
"""

from src.bridge.engine import (
    CODE_AUTHORIZATION_DENIED,
    CODE_BOUNDARY_VIOLATION,
    CODE_BRIDGE_INCREASES_TENSION,
    CODE_INSUFFICIENT_CONFIDENCE,
    CODE_STALE_INTENT_REVISION,
    DEFAULT_MINIMUM_CONFIDENCE,
    MAX_CANDIDATES_PER_CYCLE,
    BridgeEngine,
    BridgeRejected,
    RankedCandidate,
)
from src.bridge.simulator import (
    BRIDGE_STRATEGIES,
    BridgeSimulation,
    BridgeSimulator,
    ParticipantEstimate,
)
from src.bridge.sovereignty import (
    CLASS_GROUP_RULE,
    CLASS_HARD_BOUNDARY,
    CLASS_INFERRED,
    CLASS_TENANT_POLICY,
    CLASS_TOOL_AUTHORIZATION,
    EFFECT_ADVISORY,
    EFFECT_CANNOT_OVERRIDE,
    EFFECT_CAPS_AUTONOMOUS,
    EFFECT_GOVERNS_ORDINARY,
    HARD_BOUNDARY_CATEGORIES,
    SovereigntyGuard,
    SovereigntyVerdict,
    evaluate_proposal,
)

__all__ = [
    "BRIDGE_STRATEGIES",
    "CODE_AUTHORIZATION_DENIED",
    "CODE_BOUNDARY_VIOLATION",
    "CODE_BRIDGE_INCREASES_TENSION",
    "CODE_INSUFFICIENT_CONFIDENCE",
    "CODE_STALE_INTENT_REVISION",
    "DEFAULT_MINIMUM_CONFIDENCE",
    "MAX_CANDIDATES_PER_CYCLE",
    "BridgeEngine",
    "BridgeRejected",
    "BridgeSimulation",
    "BridgeSimulator",
    "ParticipantEstimate",
    "RankedCandidate",
    "SovereigntyGuard",
    "SovereigntyVerdict",
    "CLASS_GROUP_RULE",
    "CLASS_HARD_BOUNDARY",
    "CLASS_INFERRED",
    "CLASS_TENANT_POLICY",
    "CLASS_TOOL_AUTHORIZATION",
    "EFFECT_ADVISORY",
    "EFFECT_CANNOT_OVERRIDE",
    "EFFECT_CAPS_AUTONOMOUS",
    "EFFECT_GOVERNS_ORDINARY",
    "HARD_BOUNDARY_CATEGORIES",
    "evaluate_proposal",
]
