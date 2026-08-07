"""Bridge simulation: per-strategy before/after tension estimates.

``bridge_engine.simulation`` (model card) requires:

* ``required_before_apply`` — every candidate is simulated before any apply.
* ``estimate_before_and_after`` — each simulation carries predicted tension
  before and after.
* ``evaluate_each_participant`` — per-participant deltas are estimated, not
  just a group mean.
* ``preserve_counterfactual`` — the simulation keeps the "what would have
  happened without the bridge" trajectory alongside the predicted delta.

The simulator is deterministic: the same snapshot and the same group intent
always produce the same ranked candidates. It never mutates the snapshot or
the intent; a proposal is a *description* of a change, and the simulation is
the predicted effect, which the engine compares against
``bridge_engine.application_rule`` before anything is persisted as applied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from src.bridge.sovereignty import SovereigntyGuard
from src.model_card.loader import AgentModelCard, cached_model_card
from src.persistence.repositories import GroupIntentRevision, TensionSnapshot

#: Ordered strategy ids. The card fixes this order and the engine walks it;
#: explicit-escalation is terminal and may only surface *after* every cheaper
#: strategy has been tried (validated by ``BridgeEngine._escalation_is_last``).
BRIDGE_STRATEGIES: tuple[str, ...] = (
    "semantic-translation",
    "constraint-reconciliation",
    "priority-sequencing",
    "perspective-adaptation",
    "pareto-option",
    "parallel-fork",
    "explicit-escalation",
)

_ESCALATION = "explicit-escalation"


# --------------------------------------------------------------------------
# simulation record
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ParticipantEstimate:
    """Per-participant before/after tension, kept for every member."""

    participant: str
    predicted_before: float
    predicted_after: float
    delta: float

    def to_json(self) -> dict[str, Any]:
        return {
            "participant": self.participant,
            "predicted_before": self.predicted_before,
            "predicted_after": self.predicted_after,
            "delta": self.delta,
        }


@dataclass(frozen=True, slots=True)
class BridgeSimulation:
    """The predicted effect of one proposed bridge on group tension.

    ``counterfactual`` holds the "without the bridge" trajectory the card asks
    to preserve: the snapshot already observed, plus the trend from the
    previous snapshot if one was supplied.
    """

    strategy_id: str
    proposal: Mapping[str, Any]
    predicted_before: float
    predicted_after: float
    confidence: float
    participants: tuple[ParticipantEstimate, ...] = ()
    counterfactual: Mapping[str, Any] = field(default_factory=dict)
    note: str = ""

    @property
    def delta(self) -> float:
        return self.predicted_after - self.predicted_before

    def to_json(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "proposal": dict(self.proposal),
            "predicted_before": self.predicted_before,
            "predicted_after": self.predicted_after,
            "predicted_delta": self.delta,
            "confidence": self.confidence,
            "participants": [p.to_json() for p in self.participants],
            "counterfactual": dict(self.counterfactual),
            "note": self.note,
        }

    def as_simulation_document(self) -> dict[str, Any]:
        """The ``simulation`` payload BridgeRepository persists."""
        return {
            "strategy_id": self.strategy_id,
            "predicted_before": self.predicted_before,
            "predicted_after": self.predicted_after,
            "predicted_delta": self.delta,
            "confidence": self.confidence,
            "participants": [p.to_json() for p in self.participants],
            "counterfactual": dict(self.counterfactual),
        }


# --------------------------------------------------------------------------
# simulator
# --------------------------------------------------------------------------


class BridgeSimulator:
    """Deterministic per-strategy simulation over a snapshot + group intent.

    The estimator is deliberately small and inspectable. Tension is modelled as
    a weighted disagreement between the member intents and the aggregate intent
    (the same object the real estimator consumes), plus the snapshot's own
    class distribution. This keeps the simulation reproducible and lets the
    engine enforce ``application_rule`` on numbers that are auditable.
    """

    def __init__(self, card: AgentModelCard | None = None) -> None:
        self._card = card or cached_model_card()
        self._guard = SovereigntyGuard(self._card)

    # -- public API -------------------------------------------------------

    def simulate(
        self,
        snapshot: TensionSnapshot,
        group_intent: GroupIntentRevision | None,
        *,
        previous: TensionSnapshot | None = None,
    ) -> list[BridgeSimulation]:
        """Simulate every strategy in card order and return them in that order.

        The list is *not* pre-ranked: ranking (by delta, then confidence) is the
        engine's job so that the same simulation can feed different policies.
        """
        simulations: list[BridgeSimulation] = []
        for strategy_id in BRIDGE_STRATEGIES:
            simulations.append(
                self._simulate_strategy(
                    strategy_id, snapshot, group_intent, previous=previous
                )
            )
        return simulations

    def simulate_strategy(
        self,
        strategy_id: str,
        snapshot: TensionSnapshot,
        group_intent: GroupIntentRevision | None,
        *,
        previous: TensionSnapshot | None = None,
    ) -> BridgeSimulation:
        if strategy_id not in BRIDGE_STRATEGIES:
            raise ValueError(
                f"unknown bridge strategy {strategy_id!r}; expected one of "
                f"{list(BRIDGE_STRATEGIES)}"
            )
        return self._simulate_strategy(strategy_id, snapshot, group_intent, previous=previous)

    # -- internals --------------------------------------------------------

    def _simulate_strategy(
        self,
        strategy_id: str,
        snapshot: TensionSnapshot,
        group_intent: GroupIntentRevision | None,
        *,
        previous: TensionSnapshot | None,
    ) -> BridgeSimulation:
        before = _clamp01(snapshot.group_tension)

        member_gradients = _member_gradients(snapshot, group_intent)
        mean_divergence = _mean_member_divergence(member_gradients)
        participants = tuple(sorted(member_gradients.keys()))

        # Each strategy is a *policy*: how it acts is fixed by the card, and
        # its predicted effect is a function of the divergence structure.
        if strategy_id == "semantic-translation":
            after = before * (1.0 - 0.5 * mean_divergence)
            note = "maps equivalent concepts into shared terminology"
        elif strategy_id == "constraint-reconciliation":
            after = before * (1.0 - 0.4 * mean_divergence)
            note = "find a plan satisfying all hard constraints"
        elif strategy_id == "priority-sequencing":
            after = before * (1.0 - 0.25 * mean_divergence)
            note = "sequence competing ordinary goals"
        elif strategy_id == "perspective-adaptation":
            after = before * (1.0 - 0.15 * mean_divergence)
            note = "render the same proposal for each participant profile"
        elif strategy_id == "pareto-option":
            after = before * (1.0 - 0.6 * mean_divergence)
            note = "select a non-dominated compromise"
        elif strategy_id == "parallel-fork":
            after = before * (1.0 - 0.5 * mean_divergence)
            note = "split execution when consensus is unnecessary"
        elif strategy_id == _ESCALATION:
            # Escalation surfaces irreducible conflict: it preserves tension
            # rather than manufacturing agreement. Confidence is at least the
            # snapshot's; delta may be non-negative by design.
            after = before
            note = "surface irreducible conflict without manufactured agreement"
        else:  # pragma: no cover - guarded by simulate_strategy
            raise ValueError(f"unknown strategy {strategy_id!r}")

        after = _clamp01(after)
        confidence = _clamp01(
            (snapshot.confidence + (1.0 - mean_divergence)) / 2.0
        )

        per_participant = tuple(
            ParticipantEstimate(
                participant=pid,
                predicted_before=before,
                predicted_after=_clamp01(
                    after + _participant_shift(member_gradients, pid)
                ),
                delta=0.0,  # filled below
            )
            for pid in participants
        )
        per_participant = tuple(
            ParticipantEstimate(
                participant=p.participant,
                predicted_before=p.predicted_before,
                predicted_after=p.predicted_after,
                delta=p.predicted_after - p.predicted_before,
            )
            for p in per_participant
        )

        counterfactual = _counterfactual(snapshot, previous)

        return BridgeSimulation(
            strategy_id=strategy_id,
            proposal=_proposal_for(strategy_id, snapshot, group_intent),
            predicted_before=before,
            predicted_after=after,
            confidence=confidence,
            participants=per_participant,
            counterfactual=counterfactual,
            note=note,
        )


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _member_gradients(
    snapshot: TensionSnapshot, group_intent: GroupIntentRevision | None
) -> dict[str, Mapping[str, Any]]:
    """Member gradient documents: snapshot's, else intent's member intents."""
    if snapshot.member_gradients:
        return {
            str(member_id): dict(gradient)
            for member_id, gradient in snapshot.member_gradients.items()
        }
    if group_intent is not None and group_intent.member_intents:
        return {
            str(member_id): dict(intent) for member_id, intent in group_intent.member_intents.items()
        }
    return {}


def _mean_member_divergence(
    member_gradients: Mapping[str, Mapping[str, Any]]
) -> float:
    """Mean per-member divergence, 0 when nothing is known about members.

    The gradient documents store at least ``goal_divergence`` and
    ``semantic_misalignment``; a member with no gradient document counts as
    fully aligned (no divergence is attributable to them).
    """
    if not member_gradients:
        return 0.0
    total = 0.0
    for gradient in member_gradients.values():
        divergence = (
            float(gradient.get("goal_divergence", 0.0))
            + float(gradient.get("semantic_misalignment", 0.0))
        ) / 2.0
        total += _clamp01(divergence)
    return _clamp01(total / len(member_gradients))


def _participant_shift(
    member_gradients: Mapping[str, Mapping[str, Any]], participant: str
) -> float:
    """Per-participant residual: members with higher divergence see more
    relief from a bridge than members already aligned with the group."""
    gradient = member_gradients.get(participant)
    if not gradient:
        return 0.0
    divergence = (
        float(gradient.get("goal_divergence", 0.0))
        + float(gradient.get("semantic_misalignment", 0.0))
    ) / 2.0
    return -(0.3 * _clamp01(divergence))


def _counterfactual(
    snapshot: TensionSnapshot, previous: TensionSnapshot | None
) -> dict[str, Any]:
    """The preserved no-bridge trajectory required by the model card."""
    trajectory = [snapshot.group_tension]
    if previous is not None:
        trajectory.append(previous.group_tension)
    return {
        "predicted_without_bridge": snapshot.group_tension,
        "observed_trajectory": trajectory,
        "note": "tension observed without applying any bridge",
    }


def _proposal_for(
    strategy_id: str,
    snapshot: TensionSnapshot,
    group_intent: GroupIntentRevision | None,
) -> dict[str, Any]:
    """A reproducible proposal document for one strategy.

    The proposal is derived entirely from the snapshot and intent so the same
    inputs yield the same proposal. The guard evaluates it before the engine
    decides anything.
    """
    proposal: dict[str, Any] = {
        "strategy_id": strategy_id,
        "action": _ACTION_BY_STRATEGY[strategy_id],
        "effect_class": _EFFECT_CLASS_BY_STRATEGY[strategy_id],
        "snapshot_id": str(snapshot.snapshot_id),
        "group_id": str(snapshot.group_id),
    }
    if group_intent is not None:
        proposal["group_intent_revision"] = group_intent.revision
    return proposal


_ACTION_BY_STRATEGY: Mapping[str, str] = {
    "semantic-translation": "map equivalent concepts into shared terminology",
    "constraint-reconciliation": "find a plan satisfying all hard constraints",
    "priority-sequencing": "sequence competing ordinary goals",
    "perspective-adaptation": "render the same proposal for each participant profile",
    "pareto-option": "select a non-dominated compromise",
    "parallel-fork": "split execution when consensus is unnecessary",
    "explicit-escalation": "surface irreducible conflict without manufactured agreement",
}

#: The precedence class each strategy most plausibly reaches. All non-escalation
#: strategies act on ordinary preferences (level 5) or a configured group rule
#: (level 4); none of them may reach into hard boundaries.
_EFFECT_CLASS_BY_STRATEGY: Mapping[str, str] = {
    "semantic-translation": "automatically-inferred-preference",
    "constraint-reconciliation": "configured-group-decision-rule",
    "priority-sequencing": "configured-group-decision-rule",
    "perspective-adaptation": "automatically-inferred-preference",
    "pareto-option": "configured-group-decision-rule",
    "parallel-fork": "automatically-inferred-preference",
    "explicit-escalation": "automatically-inferred-preference",
}
