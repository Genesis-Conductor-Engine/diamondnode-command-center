"""Tension snapshots: six dimensions per member, one class for the group.

This is the domain layer over ``torx.kernels.tension_classification``. The
kernel already computes ``T_i = W_i * (I_i - G)`` and the class pdit on the real
torx circuits; what it does not know is anything about *time, evidence or
groups*. Those three are the card clauses this module exists to enforce.

**Evidence** (``tension_gradient.evidence_requirements``). Two separate rules,
enforced two different ways because they are not the same rule:

* ``single_message_high_impact_inference_forbidden`` is a prohibition, so an
  observation marked ``high_impact`` that rests on fewer than
  ``minimum_distinct_events`` distinct events is **refused** —
  :class:`EvidenceInsufficient`, no snapshot. Discounting it would still let a
  single message move a consequential estimate, only more slowly.
* ``minimum_distinct_events`` for an ordinary observation is a *quality* bar. The
  observation is admitted, its confidence is scaled by
  :data:`PROVISIONAL_EVIDENCE_FACTOR`, and it is barred from advancing the
  hysteresis counter. Dropping it instead would discard a real gradient, and
  ``preserve_inference_uncertainty`` wants the weak estimate kept *and* marked
  weak, not deleted.
* ``provenance_required`` means an observation with no event ids cannot be built
  at all.

**Hysteresis** (``temporal_model.minimum_stable_observations: 3``). A class
change commits only after the new class has been observed three times running.
The committed class is what ``gc_tension_snapshots.tension_class`` stores, and
the class actually observed this cycle is stored beside it — suppressing a flip
must not hide that the flip was seen, or the audit record would disagree with
what the estimator did.

**Stale-state decay** (``temporal_model.stale_state_decay: true``). A member who
was not observed this cycle keeps their last gradient with its covariance
inflated by ``1 / 0.5**(age / half_life)``. Confidence therefore falls out of
:attr:`~src.torx_layer.state.PMode.confidence` rather than being written down
separately, and the mean is never moved: forgetting must make the estimate less
certain, not invent a different opinion. The half-life is session-scoped
(minutes), not the profile layer's 90 days — tension is a property of the
conversation happening now.

**Group tension** is the card's formula verbatim::

    T_group = robust_norm(T_i) + constraint_collision_penalty +
              participation_imbalance_penalty

where ``robust_norm`` is imported from the kernel rather than re-derived, so the
group scalar stays comparable with the per-member scalars it aggregates.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from src.model_card import AgentModelCard, cached_model_card
from src.personalization.evidence import content_hash
from src.torx_layer.kernels import TensionEstimate, tension_classification
# The group-level aggregate must use the *same* robust norm the kernel applies
# per member. Re-deriving the formula here would let the two definitions drift,
# and ``group_tension`` would stop being comparable with the member scalars it
# is built from — which is the one comparison the rollback trigger
# ("observed tension increases beyond hysteresis margin") depends on.
from src.torx_layer.kernels import _robust_norm as robust_norm
from src.torx_layer.state import (
    STORAGE_PRECISION,
    TENSION_CLASSES,
    TENSION_DIMENSIONS,
    PDit,
    PMode,
)

# --------------------------------------------------------------------------
# stable reason codes
# --------------------------------------------------------------------------

SINGLE_EVENT_HIGH_IMPACT_FORBIDDEN = "SINGLE_EVENT_HIGH_IMPACT_FORBIDDEN"
INSUFFICIENT_DISTINCT_EVENTS = "INSUFFICIENT_DISTINCT_EVENTS"
PROVENANCE_REQUIRED = "PROVENANCE_REQUIRED"

#: Confidence multiplier for an observation that is real but under-evidenced.
#: A half, not a tenth: the estimate is still the best available reading of that
#: member, and driving it to near-zero would let one well-evidenced member speak
#: for a group of provisionally-observed ones.
PROVISIONAL_EVIDENCE_FACTOR = 0.5

#: Half-life for an unobserved member's gradient. Session-scoped on purpose:
#: ``personalization.inference_update.decay`` measures preference drift in
#: months, but a tension gradient describes the disagreement in the room right
#: now, and a thirty-minute-old reading of that is already half a guess.
DEFAULT_STALE_HALF_LIFE_S = 1800.0

#: Weight on the constraint-collision term of the group tension formula. One
#: collision among four members lifts group tension by 0.125 — visible, and
#: nowhere near saturating on its own, because a collision is a reason to look
#: rather than a verdict.
CONSTRAINT_COLLISION_WEIGHT = 0.5

#: Weight on the participation-imbalance term. Lower than the collision weight:
#: an uneven conversation is evidence of tension, not tension itself.
PARTICIPATION_IMBALANCE_WEIGHT = 0.25

#: Bound on the retained class history. ``runtime.state_bounds`` caps per-session
#: state; an unbounded history would grow with the session.
MAX_CLASS_HISTORY = 16

#: Namespace for deterministic snapshot ids. Deriving the id from the snapshot's
#: own canonical content means the same inputs produce the same
#: ``gc_tension_snapshots.snapshot_id`` on a retry, which is what makes an
#: at-least-once consumer idempotent.
SNAPSHOT_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "torx-tension-snapshots")

_ZERO_REFERENCE = PMode.from_diagonal(
    TENSION_DIMENSIONS, tuple(0.0 for _ in TENSION_DIMENSIONS), 0.0, label="origin"
)


def _round(x: float) -> float:
    return round(float(x), STORAGE_PRECISION)


def _clamp01(x: float) -> float:
    return min(max(float(x), 0.0), 1.0)


class EvidenceInsufficient(ValueError):
    """A refusal to infer, carrying the stable code an audit record quotes."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


# --------------------------------------------------------------------------
# observations
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MemberObservation:
    """One member's intent as observed this cycle, with its provenance.

    ``intent`` is a pmode over :data:`TENSION_DIMENSIONS` — the projection of the
    member's intent vector into tension space, which is the only space in which
    it is comparable with the group's. ``high_impact`` marks an observation the
    caller intends to draw a consequential conclusion from; it is the flag that
    arms ``single_message_high_impact_inference_forbidden``.
    """

    member_id: str
    intent: PMode
    event_ids: tuple[str, ...]
    observed_at: datetime
    participation: float = 1.0
    authorization_gap: bool = False
    high_impact: bool = False

    def __post_init__(self) -> None:
        if not str(self.member_id).strip():
            raise ValueError("observation requires a member_id")
        if tuple(self.intent.dimensions) != TENSION_DIMENSIONS:
            raise ValueError(
                f"member {self.member_id!r}: intent must be a pmode over "
                f"{list(TENSION_DIMENSIONS)}, got {list(self.intent.dimensions)}"
            )
        if not isinstance(self.observed_at, datetime) or self.observed_at.tzinfo is None:
            raise ValueError(
                f"member {self.member_id!r}: observed_at must be timezone-aware; "
                "a naive timestamp cannot be aged against a decay half-life"
            )
        if not math.isfinite(self.participation) or not 0.0 <= self.participation <= 1.0:
            raise ValueError(
                f"member {self.member_id!r}: participation must be in [0, 1], got "
                f"{self.participation!r}"
            )
        events = tuple(dict.fromkeys(str(e) for e in self.event_ids if str(e).strip()))
        if not events:
            raise EvidenceInsufficient(
                PROVENANCE_REQUIRED,
                f"member {self.member_id!r}: an observation with no event ids "
                "cannot be admitted (evidence_requirements.provenance_required)",
            )
        object.__setattr__(self, "event_ids", events)

    @property
    def distinct_events(self) -> int:
        return len(self.event_ids)

    def qualify(self, minimum_distinct_events: int) -> bool:
        """Does this observation meet the card's evidence bar?

        Raises when the forbidden combination — a high-impact inference from
        fewer than ``minimum_distinct_events`` events — is requested. Returns
        ``False``, rather than raising, for an ordinary under-evidenced
        observation: that one is admitted and marked provisional.
        """
        if self.distinct_events >= minimum_distinct_events:
            return True
        if self.high_impact:
            raise EvidenceInsufficient(
                SINGLE_EVENT_HIGH_IMPACT_FORBIDDEN,
                f"member {self.member_id!r}: a high-impact inference rests on "
                f"{self.distinct_events} distinct event(s) "
                f"({list(self.event_ids)}); the card requires "
                f"{minimum_distinct_events} and forbids a high-impact inference "
                "from a single message",
            )
        return False


# --------------------------------------------------------------------------
# per-member gradient
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MemberGradient:
    """One member's row of ``gc_tension_snapshots.member_gradients``.

    Two renderings are kept because they answer different questions. The card
    declares every dimension on ``[0, 1]``, so the stored contract row carries
    magnitudes; but ``T_i = W_i * (I_i - G)`` is signed, and the sign is the
    *direction* of the disagreement — which side of the group this member is on.
    Storing only the magnitude would discard that, so both are written and the
    contract row is the magnitudes.
    """

    member_id: str
    gradient: PMode
    tension_class: PDit
    scalar: float
    confidence: float
    participation: float
    observed_at: datetime
    event_ids: tuple[str, ...]
    qualified: bool
    stale: bool = False
    age_seconds: float = 0.0
    backend: str = "unknown"

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0 or not math.isfinite(self.confidence):
            raise ValueError(
                f"member {self.member_id!r}: confidence must be in [0, 1], got "
                f"{self.confidence!r}"
            )
        object.__setattr__(self, "confidence", _round(self.confidence))
        object.__setattr__(self, "scalar", _round(self.scalar))
        object.__setattr__(self, "participation", _round(self.participation))
        object.__setattr__(self, "age_seconds", _round(max(self.age_seconds, 0.0)))

    def magnitudes(self) -> dict[str, float]:
        return {d: _round(abs(self.gradient.value(d))) for d in TENSION_DIMENSIONS}

    def signed(self) -> dict[str, float]:
        return {d: _round(self.gradient.value(d)) for d in TENSION_DIMENSIONS}

    def to_contract_json(self) -> dict[str, Any]:
        """The exact shape ``jsonb_contracts.tension_snapshot`` shows per member."""
        row = self.magnitudes()
        row["confidence"] = self.confidence
        return row

    def to_json(self) -> dict[str, Any]:
        return {
            **self.to_contract_json(),
            "signed": self.signed(),
            "variance": list(self.gradient.variance),
            "tension_class": self.tension_class.to_json(),
            "scalar": self.scalar,
            "participation": self.participation,
            "observed_at": self.observed_at,
            "event_ids": list(self.event_ids),
            "qualified": self.qualified,
            "stale": self.stale,
            "age_seconds": self.age_seconds,
            "backend": self.backend,
        }

    def decayed(self, now: datetime, half_life_s: float) -> MemberGradient:
        """Carry this gradient forward with its uncertainty widened.

        The mean is untouched. Only the covariance grows, by ``1 / retain`` where
        ``retain = 0.5 ** (age / half_life)``, so
        :attr:`~src.torx_layer.state.PMode.confidence` — which is
        ``1 / (1 + mean variance)`` — falls monotonically with age. Participation
        decays alongside it, which is what makes a member who has gone quiet
        register in the participation-imbalance penalty.
        """
        if half_life_s <= 0:
            raise ValueError(f"stale half-life must be > 0 seconds, got {half_life_s}")
        age = max((now - self.observed_at).total_seconds(), 0.0)
        retain = 0.5 ** (age / half_life_s)
        # Guard the reciprocal: an ancient observation must widen to a large but
        # finite variance rather than overflow to inf, because inf would make the
        # stored covariance non-serialisable.
        inflate = 1.0 / max(retain, 1e-12)
        cov = tuple(
            tuple(v * inflate for v in row) for row in self.gradient.covariance
        )
        widened = PMode(
            self.gradient.dimensions, self.gradient.mean, cov, self.gradient.label
        )
        return replace(
            self,
            gradient=widened,
            confidence=widened.confidence,
            participation=_clamp01(self.participation * retain),
            stale=True,
            age_seconds=age,
        )


# --------------------------------------------------------------------------
# hysteresis
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ClassHysteresis:
    """The rolling state behind ``minimum_stable_observations``.

    Immutable and returned on every snapshot rather than held in a module-level
    cache: the card's runtime is ``horizontally-partitioned-stateful-workers``
    with at-least-once delivery, so hysteresis state that lived in one worker's
    memory would be lost on rebalance and would silently re-arm the flip it was
    meant to suppress. Carried on the snapshot, it survives wherever the snapshot
    does.
    """

    committed: str | None = None
    candidate: str | None = None
    streak: int = 0
    minimum_stable_observations: int = 3
    history: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.minimum_stable_observations < 1:
            raise ValueError(
                "minimum_stable_observations must be >= 1, got "
                f"{self.minimum_stable_observations}"
            )
        for name, value in (("committed", self.committed), ("candidate", self.candidate)):
            if value is not None and value not in TENSION_CLASSES:
                raise ValueError(
                    f"{name} class {value!r} is not one of {list(TENSION_CLASSES)}"
                )
        object.__setattr__(self, "history", tuple(self.history)[-MAX_CLASS_HISTORY:])

    @classmethod
    def from_card(cls, card: AgentModelCard | None = None) -> ClassHysteresis:
        temporal = (card or cached_model_card()).tension_gradient.temporal_model
        return cls(
            minimum_stable_observations=(
                temporal.minimum_stable_observations if temporal.hysteresis_enabled else 1
            )
        )

    def observe(self, observed: str) -> ClassHysteresis:
        """Fold one observation in and return the next state.

        Nothing is discarded: the observation always lands in ``history`` even
        when it does not move ``committed``, so an auditor can see the flip that
        was suppressed and how close it came to committing.
        """
        if observed not in TENSION_CLASSES:
            raise ValueError(
                f"observed class {observed!r} is not one of {list(TENSION_CLASSES)}"
            )
        history = (*self.history, observed)
        if self.committed is None:
            # Nothing to hold on to. Hysteresis resists *change*; the first
            # reading is not a change, and refusing to commit it would leave the
            # snapshot with no class at all.
            return replace(self, committed=observed, candidate=None, streak=0,
                           history=history)
        if observed == self.committed:
            return replace(self, candidate=None, streak=0, history=history)
        streak = self.streak + 1 if observed == self.candidate else 1
        if streak >= self.minimum_stable_observations:
            return replace(self, committed=observed, candidate=None, streak=0,
                           history=history)
        return replace(self, candidate=observed, streak=streak, history=history)

    @property
    def pending_change(self) -> bool:
        return self.candidate is not None

    def to_json(self) -> dict[str, Any]:
        return {
            "committed": self.committed,
            "candidate": self.candidate,
            "streak": self.streak,
            "minimum_stable_observations": self.minimum_stable_observations,
            "history": list(self.history),
        }


# --------------------------------------------------------------------------
# the two penalty terms
# --------------------------------------------------------------------------


def constraint_collision_penalty(
    collisions: int,
    member_count: int,
    *,
    weight: float = CONSTRAINT_COLLISION_WEIGHT,
) -> float:
    """``constraint_collision_penalty`` from the card's group-tension formula.

    Scaled by group size: two irreducible conflicts among three people is a
    fractured group, the same two among two hundred is a local dispute. Capped at
    ``weight`` so the penalty can never on its own claim maximum tension — the
    robust norm of the actual gradients has to agree.
    """
    if member_count <= 0:
        return 0.0
    if collisions < 0:
        raise ValueError(f"collision count must be >= 0, got {collisions}")
    return _round(min(weight * collisions / member_count, weight))


def participation_imbalance_penalty(
    participations: Sequence[float],
    *,
    weight: float = PARTICIPATION_IMBALANCE_WEIGHT,
) -> float:
    """``participation_imbalance_penalty`` — a Gini coefficient over turn-share.

    Gini rather than variance because it is scale-free: a group where everyone
    speaks half as much is not more imbalanced, and a variance-based term would
    say it was. Zero when participation is equal, approaching 1 when one member
    holds the floor alone.
    """
    vals = [max(float(p), 0.0) for p in participations]
    n = len(vals)
    if n < 2:
        return 0.0
    mean = math.fsum(vals) / n
    if mean <= 0:
        # Nobody participated. That is not imbalance; it is an empty cycle, and
        # inventing a penalty for it would inflate tension out of silence.
        return 0.0
    spread = math.fsum(abs(a - b) for a in vals for b in vals)
    return _round(min(weight * spread / (2 * n * n * mean), weight))


# --------------------------------------------------------------------------
# the snapshot
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TensionSnapshot:
    """``jsonb_contracts.tension_snapshot``, plus everything it would discard.

    :meth:`contract_document` returns exactly the six required keys — that is the
    document the MCP ``group.tension.evaluate`` tool returns and the one the
    snapshot id is derived from. :meth:`to_json` returns it together with the
    class distribution, the signed gradients, the hysteresis state and the
    staleness flags, none of which the contract has a slot for and all of which
    ``preserve_inference_uncertainty`` requires be kept.
    """

    tenant_id: str
    group_id: str
    group_intent_revision: int
    member_gradients: Mapping[str, MemberGradient]
    group_tension: float
    tension_class: str
    observed_tension_class: str
    tension_class_distribution: PDit
    confidence: float
    hysteresis: ClassHysteresis
    penalties: Mapping[str, float]
    observed_at: datetime
    backend: str = "unknown"
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.group_intent_revision < 1:
            raise ValueError(
                f"group_intent_revision must be >= 1, got {self.group_intent_revision}"
            )
        for name in ("tension_class", "observed_tension_class"):
            value = getattr(self, name)
            if value not in TENSION_CLASSES:
                raise ValueError(
                    f"{name} {value!r} is not one of {list(TENSION_CLASSES)}"
                )
        if not 0.0 <= self.group_tension <= 1.0 or not math.isfinite(self.group_tension):
            raise ValueError(
                f"group_tension must be in [0, 1], got {self.group_tension!r}"
            )
        if not 0.0 <= self.confidence <= 1.0 or not math.isfinite(self.confidence):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence!r}")
        object.__setattr__(self, "group_tension", _round(self.group_tension))
        object.__setattr__(self, "confidence", _round(self.confidence))
        object.__setattr__(self, "member_gradients", dict(self.member_gradients))
        object.__setattr__(self, "penalties", dict(self.penalties))

    @property
    def class_changed(self) -> bool:
        """True when the committed class differs from what was observed.

        The one number an operator wants when asking "why does the snapshot say
        aligned when the room is clearly not": hysteresis is holding.
        """
        return self.tension_class != self.observed_tension_class

    @property
    def stale_members(self) -> tuple[str, ...]:
        return tuple(sorted(k for k, g in self.member_gradients.items() if g.stale))

    @property
    def provisional_members(self) -> tuple[str, ...]:
        return tuple(sorted(k for k, g in self.member_gradients.items() if not g.qualified))

    def contract_document(self) -> dict[str, Any]:
        """Exactly ``jsonb_contracts.tension_snapshot``'s required keys."""
        return {
            "group_id": self.group_id,
            "group_intent_revision": self.group_intent_revision,
            "member_gradients": {
                mid: g.to_contract_json()
                for mid, g in sorted(self.member_gradients.items())
            },
            "group_tension": self.group_tension,
            "tension_class": self.tension_class,
            "confidence": self.confidence,
        }

    @property
    def snapshot_id(self) -> str:
        """Deterministic ``gc_tension_snapshots.snapshot_id``.

        uuid5 over the canonical contract document plus the tenant and the
        observation time, so a redelivered event recomputes the same id instead
        of writing a second row for the same observation.
        """
        seed = content_hash(
            {
                "tenant_id": self.tenant_id,
                "observed_at": self.observed_at,
                "contract": self.contract_document(),
            }
        )
        return str(uuid.uuid5(SNAPSHOT_NAMESPACE, seed))

    def validate_contract(self, card: AgentModelCard | None = None) -> None:
        """Fail when the emitted document misses a key the card requires."""
        card = card or cached_model_card()
        required = card.jsonb_contracts["tension_snapshot"].required
        doc = self.contract_document()
        missing = [k for k in required if k not in doc]
        if missing:
            raise ValueError(
                f"tension snapshot is missing required contract keys: {missing}"
            )
        declared = set(card.tension_gradient.dimensions)
        for mid, row in doc["member_gradients"].items():
            absent = declared - set(row)
            if absent:
                raise ValueError(
                    f"member {mid!r} gradient is missing declared dimensions: "
                    f"{sorted(absent)}"
                )

    def to_json(self) -> dict[str, Any]:
        return {
            **self.contract_document(),
            "tenant_id": self.tenant_id,
            "snapshot_id": self.snapshot_id,
            "observed_at": self.observed_at,
            "observed_tension_class": self.observed_tension_class,
            "tension_class_distribution": self.tension_class_distribution.to_json(),
            "class_changed": self.class_changed,
            "member_gradients_detail": {
                mid: g.to_json() for mid, g in sorted(self.member_gradients.items())
            },
            "stale_members": list(self.stale_members),
            "provisional_members": list(self.provisional_members),
            "hysteresis": self.hysteresis.to_json(),
            "penalties": {k: _round(v) for k, v in sorted(self.penalties.items())},
            "backend": self.backend,
            "diagnostics": dict(self.diagnostics),
        }


# --------------------------------------------------------------------------
# estimation
# --------------------------------------------------------------------------


def _aggregate_gradient(gradients: Sequence[MemberGradient]) -> PMode:
    """Fold per-member gradients into one group gradient pmode.

    Per dimension the mean is the robust norm of the members' magnitudes — the
    same blend of max and mean the kernel uses across dimensions, so one member
    in severe tension always moves the group figure and no single member can
    saturate it. The variance is the *mean* of the members' variances rather
    than a pooled (shrinking) estimate: a confident member must not be able to
    make the group look certain about members who were barely observed.
    """
    n_dims = len(TENSION_DIMENSIONS)
    mean = []
    var = []
    for i, dim in enumerate(TENSION_DIMENSIONS):
        mean.append(_clamp01(robust_norm([abs(g.gradient.value(dim)) for g in gradients])))
        var.append(
            math.fsum(g.gradient.covariance[i][i] for g in gradients) / len(gradients)
        )
    del n_dims
    return PMode.from_diagonal(
        TENSION_DIMENSIONS, mean, var, label="group_tension_gradient"
    )


def estimate_tension(
    observations: Sequence[MemberObservation],
    group_intent: PMode,
    *,
    tenant_id: str,
    group_id: str,
    group_intent_revision: int | None = None,
    group_revision: Any = None,
    constraint_collisions: int | None = None,
    previous: TensionSnapshot | None = None,
    hysteresis: ClassHysteresis | None = None,
    dimension_weights: Mapping[str, float] | None = None,
    now: datetime | None = None,
    stale_half_life_s: float = DEFAULT_STALE_HALF_LIFE_S,
    card: AgentModelCard | None = None,
) -> TensionSnapshot:
    """Estimate the group's tension state and return one append-only snapshot.

    ``group_revision`` may be a
    :class:`~src.intent.group_aggregate.GroupIntentRevision`; when given, its
    ``revision`` and ``constraint_collision_count`` supply the two fields that
    otherwise have to be passed by hand. It is typed loosely on purpose so this
    module does not import the aggregator — tension estimation must remain usable
    against a group intent that came from persistence rather than from a fresh
    aggregation.

    Members present in ``previous`` but absent from ``observations`` are carried
    forward decayed rather than dropped: a member going quiet is a change in
    confidence, not the disappearance of their position.
    """
    card = card or cached_model_card()
    now = now or datetime.now(timezone.utc)
    if tuple(group_intent.dimensions) != TENSION_DIMENSIONS:
        raise ValueError(
            "group intent must be a pmode over "
            f"{list(TENSION_DIMENSIONS)}, got {list(group_intent.dimensions)}"
        )

    if group_revision is not None:
        if group_intent_revision is None:
            group_intent_revision = int(getattr(group_revision, "revision"))
        if constraint_collisions is None:
            constraint_collisions = int(
                getattr(group_revision, "constraint_collision_count")
            )
    if group_intent_revision is None:
        raise ValueError(
            "group_intent_revision is required: a tension snapshot is bound to "
            "the group intent it was measured against (gc_tension_snapshots."
            "group_intent_revision)"
        )
    collisions = 0 if constraint_collisions is None else int(constraint_collisions)

    seen = [o.member_id for o in observations]
    duplicates = sorted({m for m in seen if seen.count(m) > 1})
    if duplicates:
        raise ValueError(
            f"members observed more than once in one cycle: {duplicates}; fold "
            "repeated events into a single observation so their event ids count "
            "toward one evidence bar"
        )

    minimum_events = card.tension_gradient.evidence_requirements.minimum_distinct_events
    hyst = hysteresis or (previous.hysteresis if previous else ClassHysteresis.from_card(card))

    gradients: dict[str, MemberGradient] = {}
    fresh_qualified = 0
    for obs in observations:
        qualified = obs.qualify(minimum_events)  # raises on the forbidden case
        estimate: TensionEstimate = tension_classification(
            obs.intent,
            group_intent,
            weights=dimension_weights,
            has_authorization_gap=obs.authorization_gap,
        )
        confidence = estimate.confidence * (
            1.0 if qualified else PROVISIONAL_EVIDENCE_FACTOR
        )
        gradients[obs.member_id] = MemberGradient(
            member_id=obs.member_id,
            gradient=estimate.gradient,
            tension_class=estimate.tension_class,
            scalar=estimate.scalar,
            confidence=_clamp01(confidence),
            participation=obs.participation,
            observed_at=obs.observed_at,
            event_ids=obs.event_ids,
            qualified=qualified,
            backend=estimate.backend,
        )
        if qualified:
            fresh_qualified += 1

    # Stale carry-forward. Done after the fresh pass so a member observed this
    # cycle always wins over their own previous row.
    if previous is not None:
        for mid, old in previous.member_gradients.items():
            if mid not in gradients:
                gradients[mid] = old.decayed(now, stale_half_life_s)

    if not gradients:
        raise ValueError(
            "cannot estimate tension with no observations and no previous "
            "snapshot to carry forward"
        )

    ordered = [gradients[k] for k in sorted(gradients)]
    aggregate = _aggregate_gradient(ordered)
    group_estimate: TensionEstimate = tension_classification(
        aggregate,
        _ZERO_REFERENCE,
        has_authorization_gap=any(o.authorization_gap for o in observations),
    )

    norm_term = robust_norm([g.scalar for g in ordered])
    collision_term = constraint_collision_penalty(collisions, len(ordered))
    imbalance_term = participation_imbalance_penalty([g.participation for g in ordered])
    group_tension = _clamp01(norm_term + collision_term + imbalance_term)

    observed_class = group_estimate.tension_class.argmax
    # Hysteresis only advances on evidence that met the card's bar. A cycle made
    # entirely of stale carry-forward or provisional readings must not be able to
    # commit a class change, or the "3 stable observations" rule would be
    # satisfiable by saying nothing three times.
    next_hyst = hyst.observe(observed_class) if fresh_qualified else hyst
    committed = next_hyst.committed or observed_class

    # Evidence-weighted: the estimator's own confidence, scaled by the share of
    # the group that supplied qualifying evidence this cycle.
    coverage = fresh_qualified / len(ordered)
    confidence = _clamp01(group_estimate.confidence * coverage)

    return TensionSnapshot(
        tenant_id=tenant_id,
        group_id=group_id,
        group_intent_revision=int(group_intent_revision),
        member_gradients=gradients,
        group_tension=group_tension,
        tension_class=committed,
        observed_tension_class=observed_class,
        tension_class_distribution=group_estimate.tension_class,
        confidence=confidence,
        hysteresis=next_hyst,
        penalties={
            "robust_norm": _round(norm_term),
            "constraint_collision_penalty": collision_term,
            "participation_imbalance_penalty": imbalance_term,
        },
        observed_at=now,
        backend=group_estimate.backend,
        diagnostics={
            "minimum_distinct_events": minimum_events,
            "observed_members": len(observations),
            "carried_forward_members": len(gradients) - len(observations),
            "qualified_members": fresh_qualified,
            "constraint_collisions": collisions,
            "hysteresis_held": committed != observed_class,
            "stale_half_life_s": _round(stale_half_life_s),
        },
    )
