"""Group intent: four components, three of which never touch the EBM.

``individual_and_group_intent.group_intent`` names four components and the card
gives each a *different* aggregation rule. Running all four through one
mechanism would be the bug, so this module routes them separately:

* ``hard_constraints: non-overridable-union`` — computed by
  :func:`hard_constraint_union`, which takes a sequence of
  :class:`HardConstraint` and **nothing else**. No weight, no role, no
  confidence, no temperature. Two members whose constraints cannot both hold is
  not a tie to be broken; it is an :class:`IrreducibleConflict`, recorded and
  left unresolved (``contextual_sovereignty.irreducible_conflict.action:
  surface-and-preserve``).
* ``shared_goals: intersection-plus-ranked-union`` — the intersection is a set
  operation over members who actually declared goals; the ranked union is a
  weighted Borda-style ordering that keeps the goals only some members hold.
* ``ordinary_preferences: configured-group-decision-rule`` — the only component
  that reaches :func:`src.kernels.thrml_consensus.aggregate_consensus`, i.e. the
  only one that is *sampled*.
* ``dissent: preserved-as-first-class-state`` — carried out of the consensus
  result into the revision unchanged, including dissent from members whose role
  weight is zero.

**Why role weighting is structural, not policed.** The card says roles "may
weight procedural preferences and task expertise but may not weaken another
participant's hard boundary". A runtime check that a weight did not change a
boundary would be a check that could be forgotten, reordered, or bypassed by a
new caller. Instead the weight has no path to the boundary at all:
:class:`HardConstraint` has no weight field to carry one, and
:func:`hard_constraint_union` has no parameter to receive one. The role weight
exists only where it is permitted to exist — as
:class:`~src.kernels.thrml_consensus.MemberPosition.weight` on the ordinary
preference EBM, and as the Borda weight in the ranked goal union.

**Why a boundary rule other than consent is rejected outright.**
``default_decision_rule`` is
``consent-for-boundaries-majority-for-ordinary-preferences`` and the
``decision_policy`` argument may override it — but only its ordinary half.
Precedence level 1 is ``cannot-be-overridden``, so a policy that resolved
boundaries by majority would be a configuration that violates the card, and
:class:`DecisionPolicy` refuses to construct one.

**Append-only.** :func:`aggregate_group_intent` never mutates the revision it is
given; it returns a new one at ``previous.revision + 1`` with
``supersedes_revision`` set, matching ``gc_group_intents``' composite key and
``append_only_revision_history``.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

from src.kernels.thrml_consensus import (
    ConsensusResult,
    DissentRecord,
    MemberPosition,
    aggregate_consensus,
)
from src.model_card import AgentModelCard, cached_model_card
from src.torx_layer.state import STORAGE_PRECISION

# --------------------------------------------------------------------------
# stable reason codes
# --------------------------------------------------------------------------

#: A hard constraint pair that cannot both hold. Not an error — a *finding*.
IRREDUCIBLE_CONSTRAINT_CONFLICT = "IRREDUCIBLE_CONSTRAINT_CONFLICT"
#: An ordinary preference the configured rule did not settle.
UNDECIDED_ORDINARY_PREFERENCE = "UNDECIDED_ORDINARY_PREFERENCE"

#: The two directions a hard constraint can point. Deliberately only two: a
#: constraint expressed as a free-form sentence could not be checked for mutual
#: satisfiability at all, and an unsatisfiable pair that nothing can detect is
#: exactly the manufactured consensus the card forbids.
CONSTRAINT_MODES: tuple[str, ...] = ("require", "forbid")

#: Ordinary-preference rules this module knows how to apply. ``consent`` and
#: ``unanimity`` are the same rule under two names because the card uses
#: "consent" and operators write "unanimity".
ORDINARY_RULES: dict[str, float] = {
    "majority": 0.5,
    "supermajority": 2.0 / 3.0,
    "consent": 1.0,
    "unanimity": 1.0,
}

#: Rules under which a single objection blocks adoption.
CONSENT_RULES: frozenset[str] = frozenset({"consent", "unanimity"})

#: ``|magnetisation|`` below which the group is *undecided* rather than narrowly
#: in favour. Without a floor, a 0.001 magnetisation would be reported as a
#: group position and a bridge could be applied on it.
DEFAULT_DECISIVENESS_FLOOR = 0.05

_RULE_PATTERN = re.compile(
    r"^(?P<boundary>[a-z]+)-for-boundaries-"
    r"(?P<ordinary>[a-z]+)-for-ordinary-preferences$"
)


def _round(x: float) -> float:
    return round(float(x), STORAGE_PRECISION)


def _clamp01(x: float) -> float:
    return min(max(float(x), 0.0), 1.0)


# --------------------------------------------------------------------------
# hard constraints
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HardConstraint:
    """One member's non-negotiable, in a form whose conflicts are detectable.

    There is deliberately **no weight, role, priority, or confidence field**.
    That absence is the enforcement of
    ``group_intent.role_weighting.restriction``: a caller who wanted to let a
    senior role outrank a junior member's boundary would have nowhere to put the
    number.

    ``subject`` is the thing being constrained — the join key. Two members'
    constraints collide only when they name the same subject with opposite
    ``mode``; anything else is two independent constraints that both survive
    into the union.
    """

    member_id: str
    subject: str
    mode: str = "forbid"
    category: str = "declared-non-negotiable"
    event_ids: tuple[str, ...] = ()
    note: str = ""

    def __post_init__(self) -> None:
        if not str(self.member_id).strip():
            raise ValueError("hard constraint requires the member_id that owns it")
        if not str(self.subject).strip():
            raise ValueError(
                f"member {self.member_id!r}: hard constraint requires a subject; "
                "an unnamed constraint cannot be checked against anyone else's"
            )
        if self.mode not in CONSTRAINT_MODES:
            raise ValueError(
                f"member {self.member_id!r}: constraint mode must be one of "
                f"{list(CONSTRAINT_MODES)}, got {self.mode!r}"
            )
        events = tuple(dict.fromkeys(str(e) for e in self.event_ids if str(e).strip()))
        if not events:
            raise ValueError(
                f"member {self.member_id!r}: constraint on {self.subject!r} has no "
                "event_ids; invariants.evidence_provenance_required means a "
                "boundary must be attributable to the interaction that declared it"
            )
        object.__setattr__(self, "subject", str(self.subject).strip())
        object.__setattr__(self, "event_ids", events)

    def assert_category_declared(self, card: AgentModelCard | None = None) -> None:
        """Fail when the category is not one the card declares.

        Kept off ``__post_init__`` so constructing a constraint never has to load
        the card; the aggregator calls it once per constraint, which is where an
        unknown category should surface.
        """
        declared = (card or cached_model_card()).contextual_sovereignty.hard_boundary_categories
        if self.category not in declared:
            raise ValueError(
                f"member {self.member_id!r}: constraint category "
                f"{self.category!r} is not in "
                f"contextual_sovereignty.hard_boundary_categories {list(declared)}"
            )

    @property
    def key(self) -> tuple[str, str]:
        return (self.subject, self.mode)

    def to_json(self) -> dict[str, Any]:
        return {
            "member_id": self.member_id,
            "subject": self.subject,
            "mode": self.mode,
            "category": self.category,
            "event_ids": list(self.event_ids),
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class IrreducibleConflict:
    """Two hard constraints on one subject that cannot both be satisfied.

    Stored, not resolved. ``manufactured_consensus: forbidden`` means the
    aggregator's correct output here is "these members disagree at a level no
    rule of ours can settle", and the bridge engine's answer to that is
    ``explicit-escalation``.
    """

    subject: str
    requiring: tuple[str, ...]
    forbidding: tuple[str, ...]
    constraints: tuple[HardConstraint, ...]
    code: str = IRREDUCIBLE_CONSTRAINT_CONFLICT

    @property
    def members(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.requiring, *self.forbidding)))

    @property
    def detail(self) -> str:
        return (
            f"{self.subject!r} is required by {list(self.requiring)} and "
            f"forbidden by {list(self.forbidding)}; no group rule may resolve a "
            "hard-boundary conflict, so it is preserved unresolved"
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "subject": self.subject,
            "requiring": list(self.requiring),
            "forbidding": list(self.forbidding),
            "members": list(self.members),
            "detail": self.detail,
            "constraints": [c.to_json() for c in self.constraints],
        }


def hard_constraint_union(
    constraints: Sequence[HardConstraint],
) -> tuple[tuple[HardConstraint, ...], tuple[IrreducibleConflict, ...]]:
    """The non-overridable union, plus the conflicts it could not absorb.

    **This function's parameter list is the invariant.** It receives constraints
    and returns constraints; there is no weight, role, policy, or confidence
    input, so no caller — present or future — can make one member's boundary
    count for less than another's. Every distinct ``(subject, mode)`` survives,
    and every member who declared it is kept as an attribution rather than
    collapsed into a count, because ``minority_position_storage: required``
    applies to whose boundary it was as much as to what it said.

    Ordering is by ``(subject, mode, member_id)`` so the union is byte-stable
    across runs and the revision hashes reproducibly.
    """
    ordered = sorted(constraints, key=lambda c: (c.subject, c.mode, c.member_id))
    by_subject: dict[str, dict[str, list[HardConstraint]]] = {}
    for c in ordered:
        by_subject.setdefault(c.subject, {}).setdefault(c.mode, []).append(c)

    conflicts: list[IrreducibleConflict] = []
    for subject in sorted(by_subject):
        modes = by_subject[subject]
        requiring = modes.get("require", [])
        forbidding = modes.get("forbid", [])
        if requiring and forbidding:
            conflicts.append(
                IrreducibleConflict(
                    subject=subject,
                    requiring=tuple(dict.fromkeys(c.member_id for c in requiring)),
                    forbidding=tuple(dict.fromkeys(c.member_id for c in forbidding)),
                    constraints=tuple(requiring + forbidding),
                )
            )
    return tuple(ordered), tuple(conflicts)


# --------------------------------------------------------------------------
# goals
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GoalStatement:
    """One goal a member holds, at their own rank (0 = most important)."""

    member_id: str
    goal_id: str
    rank: int = 0

    def __post_init__(self) -> None:
        if not str(self.goal_id).strip():
            raise ValueError(f"member {self.member_id!r}: goal_id must be non-empty")
        if self.rank < 0:
            raise ValueError(
                f"member {self.member_id!r}: goal {self.goal_id!r} has rank "
                f"{self.rank}; ranks start at 0 (most important)"
            )
        object.__setattr__(self, "goal_id", str(self.goal_id).strip())

    def to_json(self) -> dict[str, Any]:
        return {"member_id": self.member_id, "goal_id": self.goal_id, "rank": self.rank}


def shared_goal_intersection(members: Sequence[GroupMember]) -> tuple[str, ...]:
    """Goals held by every member who declared any goal at all.

    Members who declared nothing are skipped rather than emptying the
    intersection: silence is not a veto. A member who wants to block a goal has
    a hard constraint for that, which is a different component with a different
    rule. No weight enters here — an intersection has no room for one.
    """
    goal_sets = [
        frozenset(g.goal_id for g in m.goals) for m in members if m.goals
    ]
    if not goal_sets:
        return ()
    shared = goal_sets[0]
    for s in goal_sets[1:]:
        shared &= s
    return tuple(sorted(shared))


def ranked_goal_union(
    members: Sequence[GroupMember], weights: Mapping[str, float]
) -> tuple[tuple[str, float], ...]:
    """Every declared goal, ordered by weighted Borda score.

    A member's ``k``-th goal out of ``n`` contributes ``1 - k / n``, scaled by
    that member's role weight and normalised by the total weight. Role weighting
    is *permitted* here — a goal ordering is a procedural preference, which is
    exactly what the card lets roles weight — and it is the only place besides
    the ordinary-preference EBM where the weight is read at all.

    Ties break on ``goal_id`` so the union is deterministic.
    """
    total_w = math.fsum(max(weights.get(m.member_id, 1.0), 0.0) for m in members)
    scores: dict[str, float] = {}
    for m in members:
        if not m.goals:
            continue
        w = max(weights.get(m.member_id, 1.0), 0.0)
        n = len(m.goals)
        for g in m.goals:
            scores[g.goal_id] = scores.get(g.goal_id, 0.0) + w * (
                1.0 - min(g.rank, n - 1) / n
            )
    if total_w > 0:
        scores = {g: v / total_w for g, v in scores.items()}
    return tuple(
        (g, _round(v)) for g, v in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    )


# --------------------------------------------------------------------------
# members
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GroupMember:
    """One participant's contribution to a group-intent revision.

    ``preferences`` are the *ordinary* ones — stances in ``[-1, 1]`` on named
    propositions, the only thing that gets sampled. A member with an empty
    preference map still contributes constraints, goals and participation; they
    simply supply no evidence about the ordinary questions.
    """

    member_id: str
    role: str = "participant"
    hard_constraints: tuple[HardConstraint, ...] = ()
    goals: tuple[GoalStatement, ...] = ()
    preferences: Mapping[str, float] = field(default_factory=dict)
    participation: float = 1.0

    def __post_init__(self) -> None:
        if not str(self.member_id).strip():
            raise ValueError("group member requires a member_id")
        for c in self.hard_constraints:
            if c.member_id != self.member_id:
                raise ValueError(
                    f"member {self.member_id!r} carries a constraint owned by "
                    f"{c.member_id!r}; a boundary may only be declared by the "
                    "participant it protects"
                )
        for g in self.goals:
            if g.member_id != self.member_id:
                raise ValueError(
                    f"member {self.member_id!r} carries a goal owned by "
                    f"{g.member_id!r}"
                )
        if not math.isfinite(self.participation) or not 0.0 <= self.participation <= 1.0:
            raise ValueError(
                f"member {self.member_id!r}: participation must be in [0, 1], got "
                f"{self.participation!r}"
            )
        for prop, v in self.preferences.items():
            if not math.isfinite(float(v)) or not -1.0 <= float(v) <= 1.0:
                raise ValueError(
                    f"member {self.member_id!r}: preference on {prop!r} must be in "
                    f"[-1, 1], got {v!r}"
                )
        object.__setattr__(self, "hard_constraints", tuple(self.hard_constraints))
        object.__setattr__(self, "goals", tuple(self.goals))
        object.__setattr__(
            self, "preferences", {str(k): float(v) for k, v in self.preferences.items()}
        )

    @property
    def states_a_preference(self) -> bool:
        return any(v != 0.0 for v in self.preferences.values())

    def to_json(self) -> dict[str, Any]:
        return {
            "member_id": self.member_id,
            "role": self.role,
            "participation": _round(self.participation),
            "hard_constraints": [c.to_json() for c in self.hard_constraints],
            "goals": [g.to_json() for g in self.goals],
            "preferences": {k: _round(v) for k, v in sorted(self.preferences.items())},
        }


# --------------------------------------------------------------------------
# decision policy
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DecisionPolicy:
    """The card's ``default_decision_rule``, or a caller's override of its half.

    ``boundary_rule`` is validated to ``consent`` and nothing else. The card puts
    ``hard-individual-boundary`` at precedence level 1 with effect
    ``cannot-be-overridden``; a policy object that could express
    "boundaries by majority" would make the invariant a matter of configuration.
    """

    boundary_rule: str = "consent"
    ordinary_rule: str = "majority"
    decisiveness_floor: float = DEFAULT_DECISIVENESS_FLOOR
    label: str = ""

    def __post_init__(self) -> None:
        if self.boundary_rule != "consent":
            raise ValueError(
                f"boundary_rule must be 'consent', got {self.boundary_rule!r}: "
                "contextual_sovereignty precedence level 1 is "
                "'cannot-be-overridden', so no configured rule may decide a "
                "hard boundary by vote"
            )
        if self.ordinary_rule not in ORDINARY_RULES:
            raise ValueError(
                f"unknown ordinary_rule {self.ordinary_rule!r}; expected one of "
                f"{sorted(ORDINARY_RULES)}"
            )
        if not 0.0 <= self.decisiveness_floor < 1.0:
            raise ValueError(
                f"decisiveness_floor must be in [0, 1), got {self.decisiveness_floor}"
            )
        if not self.label:
            object.__setattr__(
                self,
                "label",
                f"{self.boundary_rule}-for-boundaries-"
                f"{self.ordinary_rule}-for-ordinary-preferences",
            )

    @property
    def support_threshold(self) -> float:
        """Weighted support share an ordinary preference must reach."""
        return ORDINARY_RULES[self.ordinary_rule]

    @property
    def requires_consent(self) -> bool:
        return self.ordinary_rule in CONSENT_RULES

    @classmethod
    def parse(cls, rule: str, *, decisiveness_floor: float = DEFAULT_DECISIVENESS_FLOOR) -> DecisionPolicy:
        m = _RULE_PATTERN.match(str(rule).strip())
        if not m:
            raise ValueError(
                f"decision rule {rule!r} is not of the form "
                "'<boundary>-for-boundaries-<ordinary>-for-ordinary-preferences'"
            )
        return cls(
            boundary_rule=m.group("boundary"),
            ordinary_rule=m.group("ordinary"),
            decisiveness_floor=decisiveness_floor,
            label=str(rule).strip(),
        )

    @classmethod
    def from_card(cls, card: AgentModelCard | None = None) -> DecisionPolicy:
        card = card or cached_model_card()
        return cls.parse(
            card.individual_and_group_intent.group_intent.default_decision_rule
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "boundary_rule": self.boundary_rule,
            "ordinary_rule": self.ordinary_rule,
            "support_threshold": _round(self.support_threshold),
            "decisiveness_floor": _round(self.decisiveness_floor),
        }


@dataclass(frozen=True, slots=True)
class PropositionOutcome:
    """What the configured rule did with one ordinary preference.

    ``adopted`` is deliberately tri-state. ``None`` means the rule did not
    settle it, which is a different fact from "the group decided against", and
    collapsing the two would be the fabricated result
    ``no_false_consensus`` forbids.
    """

    proposition: str
    group_position: float
    support_share: float
    adopted: bool | None
    rule: str
    dissenting_members: tuple[str, ...] = ()
    reason: str = ""

    @property
    def is_unresolved(self) -> bool:
        return self.adopted is None

    def to_json(self) -> dict[str, Any]:
        return {
            "proposition": self.proposition,
            "group_position": _round(self.group_position),
            "support_share": _round(self.support_share),
            "adopted": self.adopted,
            "rule": self.rule,
            "dissenting_members": list(self.dissenting_members),
            "reason": self.reason,
        }


def _decide(
    proposition: str,
    magnetization: float,
    dissenters: Sequence[str],
    policy: DecisionPolicy,
) -> PropositionOutcome:
    """Apply the configured ordinary rule to one sampled proposition.

    ``support_share`` is the magnetisation mapped onto ``[0, 1]``: ``m = +1`` is
    unanimous support, ``m = -1`` unanimous opposition, ``m = 0`` an even split.
    That mapping is what lets one threshold express "majority" and
    "supermajority" without a second scale.
    """
    share = (magnetization + 1.0) / 2.0
    if abs(magnetization) < policy.decisiveness_floor:
        return PropositionOutcome(
            proposition, magnetization, share, None, policy.ordinary_rule,
            tuple(dissenters),
            f"{UNDECIDED_ORDINARY_PREFERENCE}: |magnetisation| "
            f"{abs(magnetization):.4f} is below the decisiveness floor "
            f"{policy.decisiveness_floor}",
        )
    if policy.requires_consent and dissenters:
        return PropositionOutcome(
            proposition, magnetization, share, None, policy.ordinary_rule,
            tuple(dissenters),
            f"{UNDECIDED_ORDINARY_PREFERENCE}: rule {policy.ordinary_rule!r} "
            f"requires consent and {len(dissenters)} member(s) object",
        )
    threshold = policy.support_threshold
    if magnetization > 0 and share >= threshold:
        return PropositionOutcome(
            proposition, magnetization, share, True, policy.ordinary_rule,
            tuple(dissenters),
            f"adopted: support {share:.4f} >= {threshold:.4f}",
        )
    if magnetization < 0 and (1.0 - share) >= threshold:
        return PropositionOutcome(
            proposition, magnetization, share, False, policy.ordinary_rule,
            tuple(dissenters),
            f"rejected: opposition {1.0 - share:.4f} >= {threshold:.4f}",
        )
    return PropositionOutcome(
        proposition, magnetization, share, None, policy.ordinary_rule,
        tuple(dissenters),
        f"{UNDECIDED_ORDINARY_PREFERENCE}: neither side reaches the "
        f"{policy.ordinary_rule!r} threshold {threshold:.4f}",
    )


# --------------------------------------------------------------------------
# the revision
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GroupIntentRevision:
    """One append-only ``gc_group_intents`` row, in memory.

    Holds all four components side by side precisely so that a reader can see
    they were produced by different mechanisms: ``hard_constraints`` and
    ``irreducible_conflicts`` never passed through ``consensus``, and
    ``dissent`` came out of it untouched.
    """

    tenant_id: str
    group_id: str
    revision: int
    hard_constraints: tuple[HardConstraint, ...]
    irreducible_conflicts: tuple[IrreducibleConflict, ...]
    shared_goals: tuple[str, ...]
    ranked_goals: tuple[tuple[str, float], ...]
    outcomes: tuple[PropositionOutcome, ...]
    consensus: ConsensusResult
    dissent: tuple[DissentRecord, ...]
    decision_policy: DecisionPolicy
    confidence: float
    members: tuple[GroupMember, ...]
    role_weights: Mapping[str, float] = field(default_factory=dict)
    backend: str = "unknown"
    created_at: datetime | None = None
    supersedes_revision: int | None = None

    def __post_init__(self) -> None:
        if self.revision < 1:
            raise ValueError(f"revision must be >= 1, got {self.revision}")
        if self.supersedes_revision is not None and (
            self.supersedes_revision >= self.revision
        ):
            raise ValueError(
                f"revision {self.revision} claims to supersede "
                f"{self.supersedes_revision}; revisions are monotonic and "
                "append-only"
            )
        if not 0.0 <= self.confidence <= 1.0 or not math.isfinite(self.confidence):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence!r}")
        object.__setattr__(self, "confidence", _round(self.confidence))
        object.__setattr__(self, "role_weights", dict(self.role_weights))

    @property
    def member_ids(self) -> tuple[str, ...]:
        return tuple(m.member_id for m in self.members)

    @property
    def constraint_collision_count(self) -> int:
        """The ``constraint_collision_penalty`` input for the tension engine."""
        return len(self.irreducible_conflicts)

    @property
    def unresolved_propositions(self) -> tuple[PropositionOutcome, ...]:
        return tuple(o for o in self.outcomes if o.is_unresolved)

    @property
    def is_unresolved(self) -> bool:
        """True when something in this revision has no group answer.

        Read by the bridge engine: an unresolved revision is a candidate for
        ``explicit-escalation``, never for a bridge that claims agreement.
        """
        return bool(self.irreducible_conflicts) or bool(self.unresolved_propositions)

    def adopted_preferences(self) -> dict[str, float]:
        """Only the settled ones. Unresolved propositions are absent, not zero."""
        return {
            o.proposition: _round(o.group_position)
            for o in self.outcomes
            if o.adopted is not None
        }

    def constraints_for(self, member_id: str) -> tuple[HardConstraint, ...]:
        return tuple(c for c in self.hard_constraints if c.member_id == member_id)

    def dissent_for(self, member_id: str) -> tuple[DissentRecord, ...]:
        return tuple(d for d in self.dissent if d.member_id == member_id)

    def aggregate_intent(self) -> dict[str, Any]:
        """``gc_group_intents.aggregate_intent``."""
        return {
            "hard_constraints": [c.to_json() for c in self.hard_constraints],
            "irreducible_conflicts": [c.to_json() for c in self.irreducible_conflicts],
            "shared_goals": list(self.shared_goals),
            "ranked_goals": [{"goal_id": g, "score": s} for g, s in self.ranked_goals],
            "ordinary_preferences": self.adopted_preferences(),
            "outcomes": [o.to_json() for o in self.outcomes],
            "confidence": self.confidence,
            "unresolved": self.is_unresolved,
            "consensus": self.consensus.to_json(),
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "group_id": self.group_id,
            "revision": self.revision,
            "supersedes_revision": self.supersedes_revision,
            "aggregate_intent": self.aggregate_intent(),
            "member_intents": {m.member_id: m.to_json() for m in self.members},
            "decision_policy": self.decision_policy.to_json(),
            "dissent": [d.to_json() for d in self.dissent],
            "role_weights": {k: _round(v) for k, v in sorted(self.role_weights.items())},
            "backend": self.backend,
            "created_at": self.created_at,
        }


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------


def resolve_role_weights(
    members: Sequence[GroupMember], role_weights: Mapping[str, float] | None
) -> dict[str, float]:
    """``{member_id: weight}`` from a ``{role: weight}`` table.

    Returned keyed by member so that every downstream reader sees a weight that
    is already attached to a person — a role table passed further down could be
    re-applied a second time, and a doubly-weighted role is a silent way for a
    role to gain influence it was never granted.
    """
    table = dict(role_weights or {})
    for role, w in table.items():
        if not math.isfinite(float(w)) or float(w) < 0:
            raise ValueError(
                f"role weight for {role!r} must be finite and >= 0, got {w!r}"
            )
    return {m.member_id: float(table.get(m.role, 1.0)) for m in members}


def _member_positions(
    members: Sequence[GroupMember],
    propositions: Sequence[str],
    weights: Mapping[str, float],
) -> list[MemberPosition]:
    return [
        MemberPosition(
            member_id=m.member_id,
            positions={p: m.preferences.get(p, 0.0) for p in propositions},
            weight=weights[m.member_id],
            participation=m.participation,
        )
        for m in members
    ]


def _empty_consensus(propositions: Sequence[str]) -> ConsensusResult:
    """The honest result when there is nothing ordinary to decide.

    Not an average of nothing and not a neutral "agreement": every derived field
    is empty and ``overall_confidence`` is 0, so a caller cannot mistake "no
    ordinary preferences were on the table" for "the group agreed".
    """
    return ConsensusResult(
        propositions=tuple(propositions),
        magnetization=(),
        decision=(),
        confidence=(),
        dissent=(),
        mean_energy=0.0,
        backend="none/no-ordinary-preferences",
        gate=None,
        diagnostics={"n_propositions": 0, "reason": "no ordinary preferences declared"},
    )


def aggregate_group_intent(
    members: Sequence[GroupMember],
    *,
    tenant_id: str,
    group_id: str,
    propositions: Sequence[str] | None = None,
    role_weights: Mapping[str, float] | None = None,
    decision_policy: DecisionPolicy | None = None,
    previous: GroupIntentRevision | None = None,
    card: AgentModelCard | None = None,
    now: datetime | None = None,
    check_energy_gate: bool = False,
    **consensus_kwargs: Any,
) -> GroupIntentRevision:
    """Aggregate members into one append-only group-intent revision.

    The four components are computed in the card's own precedence order, and the
    order is load-bearing: the hard-constraint union and its conflicts are
    established *before* the EBM runs, so nothing the sampler produces can be
    read as having settled a boundary question.

    ``check_energy_gate`` defaults off. The gate only chooses where the sampler
    runs, and group-intent aggregation is on ``runtime.critical_path`` — a
    revision must not wait on GPU telemetry to be written. Callers that want the
    governor consulted pass ``True``.
    """
    if not members:
        raise ValueError(
            "cannot form a group intent from zero members; a group with no "
            "participants has no intent to aggregate"
        )
    ids = [m.member_id for m in members]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"members appear more than once: {duplicates}")
    card = card or cached_model_card()
    policy = decision_policy or DecisionPolicy.from_card(card)
    now = now or datetime.now(timezone.utc)

    if previous is not None:
        if previous.group_id != group_id or previous.tenant_id != tenant_id:
            raise ValueError(
                f"previous revision belongs to {previous.tenant_id}/"
                f"{previous.group_id}, not {tenant_id}/{group_id}"
            )
    revision = (previous.revision + 1) if previous is not None else 1

    # 1. hard constraints — computed from constraints alone. ``weights`` is not
    #    in scope of this call and cannot be made to be.
    all_constraints = [c for m in members for c in m.hard_constraints]
    for c in all_constraints:
        c.assert_category_declared(card)
    constraint_union, conflicts = hard_constraint_union(all_constraints)

    # 2. role weights — from here on, and nowhere above.
    weights = resolve_role_weights(members, role_weights)
    if math.fsum(weights.values()) <= 0:
        raise ValueError(
            "every member's role weight is zero; there is no group left to "
            "aggregate. Role weighting may re-balance influence, not remove "
            "every participant"
        )

    # 3. shared goals.
    shared = shared_goal_intersection(members)
    ranked = ranked_goal_union(members, weights)

    # 4. ordinary preferences — the only sampled component.
    props = (
        tuple(propositions)
        if propositions is not None
        else tuple(sorted({p for m in members for p in m.preferences}))
    )
    if props:
        consensus = aggregate_consensus(
            _member_positions(members, props, weights),
            props,
            check_gate=check_energy_gate,
            **consensus_kwargs,
        )
    else:
        consensus = _empty_consensus(props)

    dissent_by_prop: dict[str, list[str]] = {}
    for d in consensus.dissent:
        dissent_by_prop.setdefault(d.proposition, []).append(d.member_id)
    outcomes = tuple(
        _decide(p, consensus.magnetization[i], sorted(dissent_by_prop.get(p, ())), policy)
        for i, p in enumerate(consensus.propositions)
    )

    # 5. confidence — evidence-weighted, per ``components.confidence``. The
    #    consensus' own decisiveness is scaled by how much of the group actually
    #    supplied evidence: a decisive answer from two of ten members is not as
    #    strong a group signal as the same answer from all ten.
    coverage = sum(1 for m in members if m.states_a_preference) / len(members)
    confidence = _clamp01(consensus.overall_confidence * coverage)

    return GroupIntentRevision(
        tenant_id=tenant_id,
        group_id=group_id,
        revision=revision,
        hard_constraints=constraint_union,
        irreducible_conflicts=conflicts,
        shared_goals=shared,
        ranked_goals=ranked,
        outcomes=outcomes,
        consensus=consensus,
        dissent=tuple(consensus.dissent),
        decision_policy=policy,
        confidence=confidence,
        members=tuple(members),
        role_weights=weights,
        backend=consensus.backend,
        created_at=now,
        supersedes_revision=previous.revision if previous is not None else None,
    )


def constraint_subjects(constraints: Iterable[HardConstraint]) -> tuple[str, ...]:
    """Distinct subjects, sorted — a convenience for the sovereignty guard."""
    return tuple(sorted({c.subject for c in constraints}))
