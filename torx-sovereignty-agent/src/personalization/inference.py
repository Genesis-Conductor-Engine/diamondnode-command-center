"""Automatic personalization: one structured event, one profile delta.

``personalization.mode: fully-automatic`` with
``automatic_mutation.human_approval_required: false`` means nothing stands
between an interaction event and stored state except the preconditions the card
lists. This module is those preconditions, executed in order:

1. *source evidence is attributable* — enforced upstream in
   :mod:`.evidence` (a source id is mandatory) and re-checked here against the
   profile's own tenant and user.
2. *confidence threshold is satisfied* — the three thresholds are **read from
   the card** (``inference_update.confidence``), never written down here. A
   deployment that tightens ``persistent_minimum`` tightens this module by
   editing one line of YAML; a hardcoded constant would let the card and the
   behaviour disagree while both look correct.
3. *mutation does not widen permissions* — an event carrying an implied grant is
   **rejected, not clamped**. Clamping would quietly accept a poisoned event and
   keep its non-permission half; rejecting makes the attempt visible in
   ``gc_profile_mutation_rejected_total`` and leaves nothing of it behind.
4. *mutation does not infer a prohibited sensitive trait* — enforced in
   :mod:`.evidence` at construction, so an event naming a prohibited target
   cannot even be built.
5. *mutation preserves rollback state* — :func:`apply_delta` never mutates the
   profile it is given. It returns a new one at ``revision + 1``, which is what
   makes ``append_only_revision_history`` and ``reversible_personalization``
   structural rather than procedural.

**Decay is the card's, and explicit state has none.** ``ordinary_preference``
halves in 90 days, ``workflow_preference`` in 180, and ``explicit_boundary`` has
a ``null`` half-life — the loader already refuses a card that puts a finite
number there, because a boundary that expires is consent with a timer on it.
Decay enters the estimate through the TORX kernel's own ``decay`` argument,
which inflates prior variance rather than pulling the mean toward a default:
forgetting must make the agent *less sure*, never make it believe something the
user never said.

**Contradictions are retained.** ``explicit_overrides_inferred`` and
``recent_high_confidence_overrides_old_low_confidence`` resolve the cases they
name. Everything else is stored as an unresolved :class:`Contradiction` and the
feature is marked ``contested`` — the profile keeps both readings and the
compiled manifest says so. That is the personalization-layer form of
``no_false_consensus``: a conflict the evidence does not settle must not be
settled by the estimator.

**The kernel.** The continuous update is ``torx.kernels.intent_update`` — the
same bounded-stochastic kernel the intent and tension layers run on. A
preference is an intent about how to be worked with, so it gets the same
confidence-weighted Gaussian posterior and the same honest variance, rather than
a second, private update rule that could disagree with the rest of the system.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

from src.model_card import AgentModelCard, cached_model_card
from src.torx_layer.circuits import backend_status
from src.torx_layer.kernels import intent_update
from src.torx_layer.state import STORAGE_PRECISION, PMode

from .evidence import (
    EXPLICIT_DOMAIN,
    HARD_BOUNDARY_DOMAIN,
    EvidenceEvent,
    EvidenceRejection,
    content_hash,
)

# --------------------------------------------------------------------------
# stable reason codes (extend the set in ``evidence``)
# --------------------------------------------------------------------------

#: Card stable code; reused verbatim so an MCP error and an audit record agree.
INSUFFICIENT_CONFIDENCE = "INSUFFICIENT_CONFIDENCE"
#: Maps to the card's ``AUTHORIZATION_DENIED`` at the MCP surface, but is kept
#: distinct internally: the request was not merely unauthorized, it was an
#: attempt to *widen* the envelope, which is a different metric and a different
#: alert.
PERMISSION_EXPANSION_FORBIDDEN = "PERMISSION_EXPANSION_FORBIDDEN"
INSUFFICIENT_DISTINCT_EVENTS = "INSUFFICIENT_DISTINCT_EVENTS"
TENANT_MISMATCH = "TENANT_MISMATCH"
DUPLICATE_EVENT = "DUPLICATE_EVENT"

# --------------------------------------------------------------------------
# profile layers and tiers
# --------------------------------------------------------------------------

#: ``personalization.profile_layers`` keys that a stored feature can carry.
LAYER_EXPLICIT = "explicit"
LAYER_PERSISTENT = "inferred_persistent"
LAYER_EPHEMERAL = "inferred_ephemeral"

TIER_PROVISIONAL = "provisional"
TIER_PERSISTENT = "persistent"
TIER_CONSEQUENTIAL = "consequential"

#: Domains whose features govern what the agent does *without asking*. The card
#: puts ``execution-threshold-within-granted-permissions`` and
#: ``proposal-threshold`` here, so a change to any of them is a "consequential
#: adjustment" and must clear the 0.90 threshold rather than the 0.75 one.
CONSEQUENTIAL_DOMAINS: frozenset[str] = frozenset({"autonomy"})

#: Difference in a ``[0, 1]`` feature that counts as a contradiction rather than
#: drift. A quarter of the declared range: smaller gaps are exactly what the
#: Kalman posterior exists to absorb, larger ones are two observations that
#: cannot both be describing the same preference.
CONTRADICTION_TOLERANCE = 0.25

#: How much more confident the newcomer must be before
#: ``recent_high_confidence_overrides_old_low_confidence`` fires. Without a
#: margin, two evenly matched readings would ping-pong the stored value on every
#: event and each flip would look like a resolution.
CONFIDENCE_DOMINANCE_MARGIN = 0.15

#: Prior for a feature with no history: the midpoint of the declared range with
#: the kernel's diffuse variance. Not zero — zero is a *position* on a ``[0, 1]``
#: preference scale, and starting there would bias every first observation
#: downward.
NEUTRAL_PRIOR_VALUE = 0.5
NEUTRAL_PRIOR_VARIANCE = 1.0

_SECONDS_PER_DAY = 86400.0


def _round(x: float) -> float:
    return round(float(x), STORAGE_PRECISION)


def _clamp01(x: float) -> float:
    return min(max(float(x), 0.0), 1.0)


# --------------------------------------------------------------------------
# card-derived configuration
# --------------------------------------------------------------------------


def confidence_thresholds(card: AgentModelCard | None = None) -> dict[str, float]:
    """``{provisional, persistent, consequential}`` straight from the card."""
    conf = (card or cached_model_card()).personalization.inference_update.confidence
    return {
        TIER_PROVISIONAL: conf.provisional_minimum,
        TIER_PERSISTENT: conf.persistent_minimum,
        TIER_CONSEQUENTIAL: conf.consequential_adjustment_minimum,
    }


def half_life_days(
    domain: str, layer: str, card: AgentModelCard | None = None
) -> float | None:
    """Half-life for a feature, or ``None`` when it must never decay.

    ``None`` is returned for everything the user owns outright — explicit
    statements and hard boundaries. The card only names a finite half-life for
    the two *inferred* families (``ordinary_preference``, ``workflow_preference``)
    and explicitly sets ``explicit_boundary: null``; extending non-decay to every
    explicit statement is the same direction of caution, since forgetting a
    declared preference is a change the user never asked for.
    """
    decay = (card or cached_model_card()).personalization.inference_update.decay
    if not decay.enabled:
        return None
    if layer == LAYER_EXPLICIT or domain == HARD_BOUNDARY_DOMAIN:
        return None
    if domain == "workflow":
        return decay.half_life_days.workflow_preference
    return decay.half_life_days.ordinary_preference


def minimum_distinct_events(card: AgentModelCard | None = None) -> int:
    """``tension_gradient.evidence_requirements.minimum_distinct_events``.

    Declared for the tension engine, applied here for the same reason it exists
    there: ``single_message_high_impact_inference_forbidden``. A consequential
    adjustment inferred from one message is exactly that forbidden thing.
    """
    return (card or cached_model_card()).tension_gradient.evidence_requirements.minimum_distinct_events


# --------------------------------------------------------------------------
# stored state
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FeatureState:
    """One feature as the profile holds it.

    Both ``variance`` and ``confidence`` are stored because they answer different
    questions. ``variance`` is the kernel's posterior spread — how tightly the
    estimate is pinned. ``confidence`` is the evidence weight behind it at
    ``observed_at``, and it is the number that decays and that the card's
    thresholds are expressed in.
    """

    domain: str
    name: str
    value: float
    variance: float
    confidence: float
    layer: str
    observed_at: datetime
    event_ids: tuple[str, ...] = ()
    value_type: str = "scalar"
    half_life: float | None = None
    contested: bool = False

    def __post_init__(self) -> None:
        if self.layer not in (LAYER_EXPLICIT, LAYER_PERSISTENT, LAYER_EPHEMERAL):
            raise ValueError(
                f"{self.domain}.{self.name}: unknown profile layer {self.layer!r}"
            )
        if not 0.0 <= self.value <= 1.0 or not math.isfinite(self.value):
            raise ValueError(
                f"{self.domain}.{self.name}: value {self.value!r} is outside "
                "the declared [0, 1] range"
            )
        if not 0.0 <= self.confidence <= 1.0 or not math.isfinite(self.confidence):
            raise ValueError(
                f"{self.domain}.{self.name}: confidence {self.confidence!r} is "
                "not a probability"
            )
        if self.variance < 0 or not math.isfinite(self.variance):
            raise ValueError(
                f"{self.domain}.{self.name}: variance must be finite and >= 0, "
                f"got {self.variance!r}"
            )
        if self.observed_at.tzinfo is None:
            raise ValueError(
                f"{self.domain}.{self.name}: observed_at must be timezone-aware"
            )
        if self.half_life is not None and self.half_life <= 0:
            raise ValueError(
                f"{self.domain}.{self.name}: half_life must be > 0 or None, got "
                f"{self.half_life!r}"
            )
        object.__setattr__(self, "value", _round(self.value))
        object.__setattr__(self, "variance", _round(self.variance))
        object.__setattr__(self, "confidence", _round(self.confidence))
        object.__setattr__(self, "event_ids", tuple(dict.fromkeys(self.event_ids)))

    @property
    def key(self) -> str:
        return f"{self.domain}.{self.name}"

    def confidence_at(self, now: datetime) -> float:
        """Confidence after exponential decay to ``now``.

        ``half_life is None`` means the state is user-owned: it is returned
        untouched however old it is.
        """
        if self.half_life is None:
            return self.confidence
        age_days = (now - self.observed_at).total_seconds() / _SECONDS_PER_DAY
        if age_days <= 0:
            return self.confidence
        return _round(self.confidence * 0.5 ** (age_days / self.half_life))

    def rendered_value(self) -> Any:
        return bool(round(self.value)) if self.value_type == "boolean" else self.value

    def to_json(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "name": self.name,
            "value": self.value,
            "variance": self.variance,
            "confidence": self.confidence,
            "layer": self.layer,
            "observed_at": self.observed_at,
            "event_ids": list(self.event_ids),
            "value_type": self.value_type,
            "half_life": self.half_life,
            "contested": self.contested,
        }


@dataclass(frozen=True, slots=True)
class Contradiction:
    """Two readings of one feature that disagree, kept verbatim.

    Stored whether or not the policy resolved it — a resolved contradiction is
    still the evidence that the resolution was needed, and
    ``poisoning.contradiction_retention`` is what lets an operator notice a
    source that keeps losing these.
    """

    domain: str
    name: str
    existing_value: float
    existing_confidence: float
    existing_layer: str
    incoming_value: float
    incoming_confidence: float
    incoming_layer: str
    incoming_event_id: str
    resolution: str
    observed_at: datetime
    retained: str  # "existing" | "incoming" | "both"

    @property
    def key(self) -> str:
        return f"{self.domain}.{self.name}"

    @property
    def is_unresolved(self) -> bool:
        return self.resolution == "unresolved"

    def to_json(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "name": self.name,
            "existing_value": _round(self.existing_value),
            "existing_confidence": _round(self.existing_confidence),
            "existing_layer": self.existing_layer,
            "incoming_value": _round(self.incoming_value),
            "incoming_confidence": _round(self.incoming_confidence),
            "incoming_layer": self.incoming_layer,
            "incoming_event_id": self.incoming_event_id,
            "resolution": self.resolution,
            "observed_at": self.observed_at,
            "retained": self.retained,
        }


@dataclass(frozen=True, slots=True)
class UserProfile:
    """A profile revision — ``jsonb_contracts.user_profile`` in memory.

    ``revoked_capabilities`` is a *subtraction list*, not a grant list. There is
    deliberately no field on this type that can add a capability, so
    ``personalization_may_not_widen_permissions`` holds no matter what any caller
    does: the widest a profile can ever be is the base contract itself.
    """

    tenant_id: str
    user_id: str
    revision: int = 1
    features: Mapping[str, FeatureState] = field(default_factory=dict)
    contradictions: tuple[Contradiction, ...] = ()
    applied_event_keys: frozenset[str] = frozenset()
    revoked_capabilities: frozenset[str] = frozenset()
    updated_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.revision < 1:
            raise ValueError(f"revision must be >= 1, got {self.revision}")
        for key, state in self.features.items():
            if key != state.key:
                raise ValueError(
                    f"feature stored under {key!r} but names itself {state.key!r}"
                )
        object.__setattr__(self, "features", dict(self.features))
        object.__setattr__(self, "applied_event_keys", frozenset(self.applied_event_keys))
        object.__setattr__(
            self, "revoked_capabilities", frozenset(self.revoked_capabilities)
        )

    def get(self, domain: str, name: str) -> FeatureState | None:
        return self.features.get(f"{domain}.{name}")

    @property
    def contested_features(self) -> tuple[str, ...]:
        return tuple(sorted(k for k, s in self.features.items() if s.contested))

    @property
    def unresolved_contradictions(self) -> tuple[Contradiction, ...]:
        return tuple(c for c in self.contradictions if c.is_unresolved)

    def hard_boundaries(self) -> dict[str, Any]:
        return {
            s.name: s.rendered_value()
            for s in self.features.values()
            if s.domain == HARD_BOUNDARY_DOMAIN
        }

    def to_json(self) -> dict[str, Any]:
        """The card's ``user_profile`` contract, plus retained-conflict state."""
        explicit: dict[str, Any] = {}
        inferred: dict[str, dict[str, Any]] = {}
        confidence_sums: dict[str, list[float]] = {}
        event_ids: list[str] = []
        for state in sorted(self.features.values(), key=lambda s: s.key):
            event_ids.extend(state.event_ids)
            if state.domain == HARD_BOUNDARY_DOMAIN:
                continue
            if state.domain == EXPLICIT_DOMAIN:
                explicit[state.name] = state.rendered_value()
                continue
            inferred.setdefault(state.domain, {})[state.name] = state.rendered_value()
            confidence_sums.setdefault(state.domain, []).append(state.confidence)
        return {
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
            "revision": self.revision,
            "explicit": explicit,
            "inferred": inferred,
            "hard_boundaries": self.hard_boundaries(),
            "confidence": {
                domain: _round(math.fsum(vals) / len(vals))
                for domain, vals in sorted(confidence_sums.items())
            },
            "provenance": {"event_ids": sorted(dict.fromkeys(event_ids))},
            "contested": list(self.contested_features),
            "contradictions": [c.to_json() for c in self.contradictions],
            "revoked_capabilities": sorted(self.revoked_capabilities),
            "updated_at": self.updated_at,
        }

    @property
    def profile_hash(self) -> str:
        """``gc_user_profiles.profile_hash`` — canonical-JSON SHA-256."""
        return content_hash(self.to_json())


# --------------------------------------------------------------------------
# delta
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProfileDelta:
    """What one event does to a profile — the ``gc_profile_revisions.delta`` row.

    A rejected delta is still a first-class result rather than an exception:
    ``observability.audit_events`` requires
    ``personalization.observation.rejected`` to be recorded, and a caller cannot
    record what it never receives.
    """

    tenant_id: str
    user_id: str
    base_revision: int
    accepted: bool
    tier: str | None = None
    updates: tuple[FeatureState, ...] = ()
    contradictions: tuple[Contradiction, ...] = ()
    capability_revocations: frozenset[str] = frozenset()
    rejection: EvidenceRejection | None = None
    source_event_ids: tuple[str, ...] = ()
    dedup_keys: tuple[str, ...] = ()
    persist: bool = False
    backend: str = "unknown"
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.accepted and self.rejection is not None:
            raise ValueError("an accepted delta cannot carry a rejection")
        if not self.accepted and self.rejection is None:
            raise ValueError("a rejected delta must carry its reason code")
        object.__setattr__(
            self, "capability_revocations", frozenset(self.capability_revocations)
        )

    @property
    def is_noop(self) -> bool:
        return not self.updates and not self.contradictions and not self.capability_revocations

    def to_json(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "base_revision": self.base_revision,
            "accepted": self.accepted,
            "tier": self.tier,
            "updates": [u.to_json() for u in self.updates],
            "contradictions": [c.to_json() for c in self.contradictions],
            "capability_revocations": sorted(self.capability_revocations),
            "rejection": self.rejection.to_json() if self.rejection else None,
            "source_event_ids": list(self.source_event_ids),
            "dedup_keys": list(self.dedup_keys),
            "persist": self.persist,
            "backend": self.backend,
            "diagnostics": dict(self.diagnostics),
        }


def _reject(
    event: EvidenceEvent, profile: UserProfile, code: str, detail: str
) -> ProfileDelta:
    return ProfileDelta(
        tenant_id=profile.tenant_id,
        user_id=profile.user_id,
        base_revision=profile.revision,
        accepted=False,
        rejection=EvidenceRejection(code=code, detail=detail, dedup_key=event.dedup_key),
        source_event_ids=(event.event_id,),
        dedup_keys=(event.dedup_key,),
        backend=backend_status()["engine"],
    )


# --------------------------------------------------------------------------
# the kernel update
# --------------------------------------------------------------------------


def _decay_fraction(existing: FeatureState | None, now: datetime) -> float:
    """Prior-widening fraction for ``intent_update``.

    ``intent_update`` inflates the prior covariance by ``1 / (1 - decay)``.
    Choosing ``decay = 1 - 0.5 ** (age / half_life)`` makes that inflation
    ``2 ** (age / half_life)``: the prior's variance doubles every half-life,
    which is the same curve the stored confidence follows, expressed in the unit
    the kernel actually consumes.
    """
    if existing is None or existing.half_life is None:
        return 0.0
    age_days = (now - existing.observed_at).total_seconds() / _SECONDS_PER_DAY
    if age_days <= 0:
        return 0.0
    decay = 1.0 - 0.5 ** (age_days / existing.half_life)
    # ``intent_update`` requires decay < 1; an ancient prior is *very* wide, not
    # infinitely wide.
    return min(max(decay, 0.0), 1.0 - 1e-6)


def _posterior(
    existing: FeatureState | None,
    observed_value: float,
    *,
    confidence: float,
    now: datetime,
    label: str,
) -> tuple[float, float]:
    """Run ``intent_update`` for a single feature; return ``(value, variance)``."""
    if existing is None:
        prior = PMode.from_diagonal(
            (label,), (NEUTRAL_PRIOR_VALUE,), (NEUTRAL_PRIOR_VARIANCE,), label="intent"
        )
        decay = 0.0
    else:
        prior = PMode.from_diagonal(
            (label,), (existing.value,), (max(existing.variance, 1e-9),), label="intent"
        )
        decay = _decay_fraction(existing, now)
    posterior = intent_update(prior, (observed_value,), confidence=confidence, decay=decay)
    return _clamp01(posterior.mean[0]), max(posterior.variance[0], 0.0)


# --------------------------------------------------------------------------
# infer_profile_delta
# --------------------------------------------------------------------------


def _classify_contradiction(
    existing: FeatureState,
    event: EvidenceEvent,
    *,
    incoming_confidence: float,
    now: datetime,
) -> str | None:
    """Return the contradiction policy that applies, or ``None`` for agreement."""
    incoming_value = event.observation.value
    boolean = event.observation.value_type == "boolean" or existing.value_type == "boolean"
    gap = abs(existing.value - incoming_value)
    if boolean:
        if round(existing.value) == round(incoming_value):
            return None
    elif gap <= CONTRADICTION_TOLERANCE:
        return None

    incoming_explicit = event.is_explicit
    existing_explicit = existing.layer == LAYER_EXPLICIT
    if incoming_explicit != existing_explicit:
        return "explicit-overrides-inferred"
    if incoming_confidence - existing.confidence_at(now) >= CONFIDENCE_DOMINANCE_MARGIN:
        return "recent-high-confidence-overrides-old-low-confidence"
    return "unresolved"


def infer_profile_delta(
    event: EvidenceEvent,
    profile: UserProfile,
    *,
    now: datetime | None = None,
    card: AgentModelCard | None = None,
) -> ProfileDelta:
    """Turn one structured event into a delta against ``profile``.

    Never mutates ``profile``. The returned delta is either accepted — with the
    feature states it would write and every contradiction it observed — or
    rejected with a stable code, and in both cases it is a record the audit
    trail can store as-is.
    """
    card = card or cached_model_card()
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("`now` must be timezone-aware")
    thresholds = confidence_thresholds(card)
    obs = event.observation

    if (event.tenant_id, event.user_id) != (profile.tenant_id, profile.user_id):
        return _reject(
            event,
            profile,
            TENANT_MISMATCH,
            f"event is for {event.tenant_id}:{event.user_id} but the profile is "
            f"{profile.tenant_id}:{profile.user_id}; personalization scope is "
            "per-user-per-tenant",
        )

    # ``event_deduplication_key`` under at-least-once delivery.
    if event.dedup_key in profile.applied_event_keys:
        return _reject(
            event,
            profile,
            DUPLICATE_EVENT,
            f"{event.dedup_key} is already applied at revision {profile.revision}",
        )

    # Precondition: the mutation must not widen permissions. Rejected whole, not
    # stripped of its permission half.
    if event.implied_capability_grants:
        return _reject(
            event,
            profile,
            PERMISSION_EXPANSION_FORBIDDEN,
            "evidence implies granting "
            f"{sorted(event.implied_capability_grants)}; personalization may "
            "narrow the capability envelope but never widen it",
        )

    # A feature is consequential when *writing it* changes what the agent does
    # unattended. A capability revocation also clears the high bar, because
    # taking a tool away is disruptive even though it is the safe direction.
    consequential_feature = obs.domain in CONSEQUENTIAL_DOMAINS or obs.is_boundary
    consequential = consequential_feature or bool(event.implied_capability_revocations)
    confidence = event.effective_confidence
    required = thresholds[TIER_CONSEQUENTIAL if consequential else TIER_PROVISIONAL]
    if confidence < required:
        return _reject(
            event,
            profile,
            INSUFFICIENT_CONFIDENCE,
            f"effective confidence {confidence} is below the "
            f"{'consequential' if consequential else 'provisional'} minimum "
            f"{required} for {obs.key}",
        )

    existing = profile.features.get(obs.key)
    distinct_events = len(
        dict.fromkeys((*(existing.event_ids if existing else ()), event.event_id))
    )
    minimum_events = minimum_distinct_events(card)
    # ``single_message_high_impact_inference_forbidden``. The *impact* is
    # withheld, not the observation: a confident single reading is recorded at
    # the card's session-only ``inferred_ephemeral`` authority, where it is
    # advisory and unpersisted, and a second corroborating event promotes it.
    # Discarding it instead would deadlock — nothing would ever accumulate the
    # second event the rule asks for.
    withheld: str | None = None
    if consequential_feature and not event.is_explicit and distinct_events < minimum_events:
        withheld = INSUFFICIENT_DISTINCT_EVENTS

    if withheld is not None:
        tier = TIER_PROVISIONAL
    elif consequential:
        tier = TIER_CONSEQUENTIAL
    elif confidence >= thresholds[TIER_PERSISTENT]:
        tier = TIER_PERSISTENT
    else:
        tier = TIER_PROVISIONAL

    if event.is_explicit:
        layer = LAYER_EXPLICIT
    elif tier == TIER_PROVISIONAL:
        layer = LAYER_EPHEMERAL
    else:
        layer = LAYER_PERSISTENT

    contradiction: Contradiction | None = None
    resolution = None
    if existing is not None:
        resolution = _classify_contradiction(
            existing, event, incoming_confidence=confidence, now=now
        )

    if resolution is not None:
        incoming_wins = (
            resolution == "recent-high-confidence-overrides-old-low-confidence"
            or (resolution == "explicit-overrides-inferred" and event.is_explicit)
        )
        contradiction = Contradiction(
            domain=obs.domain,
            name=obs.name,
            existing_value=existing.value,
            existing_confidence=existing.confidence_at(now),
            existing_layer=existing.layer,
            incoming_value=obs.value,
            incoming_confidence=confidence,
            incoming_layer=layer,
            incoming_event_id=event.event_id,
            resolution=resolution,
            observed_at=event.observed_at,
            retained="incoming" if incoming_wins else (
                "both" if resolution == "unresolved" else "existing"
            ),
        )
        if not incoming_wins:
            # The incoming reading loses (explicit state stands) or nothing
            # settles it (unresolved). Either way the stored value does not move
            # and the conflict is what gets written down. ``contested`` is set
            # for the unresolved case so the compiled manifest can surface a
            # feature the evidence does not agree on instead of presenting it as
            # settled.
            updates: tuple[FeatureState, ...] = ()
            if resolution == "unresolved":
                updates = (
                    replace(
                        existing,
                        contested=True,
                        event_ids=(*existing.event_ids, event.event_id),
                    ),
                )
            return ProfileDelta(
                tenant_id=profile.tenant_id,
                user_id=profile.user_id,
                base_revision=profile.revision,
                accepted=True,
                tier=tier,
                updates=updates,
                contradictions=(contradiction,),
                capability_revocations=event.implied_capability_revocations,
                source_event_ids=(event.event_id,),
                dedup_keys=(event.dedup_key,),
                persist=True,
                backend=backend_status()["engine"],
                diagnostics={
                    "effective_confidence": confidence,
                    "resolution": resolution,
                    "value_unchanged": True,
                    "distinct_events": distinct_events,
                    "withheld": withheld,
                },
            )

    life = half_life_days(obs.domain, layer, card)
    if event.is_explicit or obs.value_type == "boolean":
        # A declaration is not an estimate: it is written through verbatim. Also
        # the only sane treatment of a boolean — a Kalman blend of two booleans
        # is a number that is neither.
        value, variance = obs.value, 0.0
    else:
        value, variance = _posterior(
            existing, obs.value, confidence=confidence, now=now, label=obs.key
        )

    new_state = FeatureState(
        domain=obs.domain,
        name=obs.name,
        value=value,
        variance=variance,
        confidence=confidence,
        layer=layer,
        observed_at=event.observed_at,
        event_ids=(*(existing.event_ids if existing else ()), event.event_id),
        value_type=obs.value_type,
        half_life=life,
        # A contradiction the policy *resolved* is no longer contested; an
        # unresolved one never reaches here.
        contested=False,
    )

    return ProfileDelta(
        tenant_id=profile.tenant_id,
        user_id=profile.user_id,
        base_revision=profile.revision,
        accepted=True,
        tier=tier,
        updates=(new_state,),
        contradictions=(contradiction,) if contradiction else (),
        capability_revocations=event.implied_capability_revocations,
        source_event_ids=(event.event_id,),
        dedup_keys=(event.dedup_key,),
        # ``inferred_ephemeral`` is session-only authority, so a provisional
        # delta is applied in memory but never written as a revision.
        persist=layer != LAYER_EPHEMERAL,
        backend=backend_status()["engine"],
        diagnostics={
            "effective_confidence": confidence,
            "resolution": resolution,
            "prior_value": existing.value if existing else None,
            "prior_confidence": existing.confidence_at(now) if existing else None,
            "decay_fraction": _round(_decay_fraction(existing, now)),
            "half_life_days": life,
            "distinct_events": distinct_events,
            "consequential": consequential,
            "withheld": withheld,
        },
    )


# --------------------------------------------------------------------------
# append-only application
# --------------------------------------------------------------------------


def apply_delta(profile: UserProfile, delta: ProfileDelta) -> UserProfile:
    """Return a **new** profile at ``revision + 1``.

    ``profile`` is never touched, so the caller still holds the exact state a
    rollback would restore. Contradictions accumulate rather than replace:
    ``unresolved_contradictions_are_retained`` is a storage property, and an
    implementation that overwrote the list on each revision would satisfy it only
    until the next event.
    """
    if not delta.accepted:
        raise ValueError(
            f"cannot apply a rejected delta ({delta.rejection.code if delta.rejection else '?'})"
        )
    if (delta.tenant_id, delta.user_id) != (profile.tenant_id, profile.user_id):
        raise ValueError("delta and profile disagree on tenant/user")
    if delta.base_revision != profile.revision:
        raise ValueError(
            f"delta targets revision {delta.base_revision} but the profile is at "
            f"{profile.revision}: profile_revision uses optimistic concurrency "
            "control"
        )
    features = dict(profile.features)
    for state in delta.updates:
        features[state.key] = state
    observed = [s.observed_at for s in delta.updates]
    return UserProfile(
        tenant_id=profile.tenant_id,
        user_id=profile.user_id,
        revision=profile.revision + 1,
        features=features,
        contradictions=(*profile.contradictions, *delta.contradictions),
        applied_event_keys=profile.applied_event_keys | set(delta.dedup_keys),
        # Revocations only ever add to the subtraction list.
        revoked_capabilities=profile.revoked_capabilities | delta.capability_revocations,
        updated_at=max(observed) if observed else profile.updated_at,
    )


def ingest(
    events: Iterable[EvidenceEvent],
    profile: UserProfile,
    *,
    now: datetime | None = None,
    card: AgentModelCard | None = None,
) -> tuple[UserProfile, tuple[ProfileDelta, ...]]:
    """Fold a batch of events into a profile, keeping every delta.

    Rejected deltas are returned alongside accepted ones so the caller emits one
    audit event per observation, accepted or not.
    """
    deltas: list[ProfileDelta] = []
    current = profile
    for event in events:
        delta = infer_profile_delta(event, current, now=now, card=card)
        deltas.append(delta)
        if delta.accepted:
            current = apply_delta(current, delta)
    return current, tuple(deltas)


def rollback(profile: UserProfile, target: UserProfile) -> UserProfile:
    """Restore ``target``'s content as a **new** forward revision.

    Rollback never rewinds ``revision``. ``append_only_revision_history`` and
    ``replay.monotonic_revisions`` both break if a revision number is ever
    reused, and a Rule 30 VDF chain over reused numbers cannot be verified at
    all. The applied-event set and the contradiction log are carried forward
    from the *current* profile, because undoing a value is not the same as
    pretending the evidence never arrived.
    """
    if (profile.tenant_id, profile.user_id) != (target.tenant_id, target.user_id):
        raise ValueError("cannot roll back onto a different user's profile")
    if target.revision > profile.revision:
        raise ValueError(
            f"rollback target revision {target.revision} is ahead of the current "
            f"revision {profile.revision}"
        )
    return UserProfile(
        tenant_id=profile.tenant_id,
        user_id=profile.user_id,
        revision=profile.revision + 1,
        features=dict(target.features),
        contradictions=profile.contradictions,
        applied_event_keys=profile.applied_event_keys,
        revoked_capabilities=target.revoked_capabilities,
        updated_at=profile.updated_at,
    )
