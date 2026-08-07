"""Typed TORX state: ``pbit``, ``pdit``, ``pmode``.

These are the wire and storage forms of the model card's TORX state types. They
are plain frozen dataclasses — JSONB-serialisable, comparable, and usable without
JAX — while :mod:`src.torx_layer.circuits` and :mod:`src.torx_layer.kernels`
compute them on the real Extropic ``torx`` kernels.

The split matters for two card requirements:

* ``preserve_inference_uncertainty`` — a decision is stored as a *distribution*,
  not a collapsed label. ``PDit`` keeps the full categorical and ``PMode`` keeps
  the covariance, so "semantic-gap at 0.51 against priority-gap at 0.49" is
  distinguishable from "semantic-gap at 0.99" downstream and in the audit record.
* ``degradation`` — the last rung of the ladder is a
  ``non-topological-torx-baseline``. Because these types carry no JAX objects,
  every consumer keeps working when the kernels are unavailable; only the
  *estimator* degrades, never the state contract.

Numbers are stored as Python floats and normalised on construction so that the
canonical JSON the Rule 30 VDF signs is stable across backends: a value computed
on GPU JAX and the same value computed by the pure-Python fallback must produce
identical bytes or every proof would be backend-specific.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

# Distributions are rounded to this many decimals before storage. float32 JAX
# output carries ~7 significant digits; rounding to 9 decimals keeps every real
# bit while discarding the backend-dependent noise below it.
STORAGE_PRECISION = 9

# Probabilities within this distance of 0 or 1 are treated as certain. Used only
# for reporting (``is_certain``); the guard logic never relies on it.
CERTAINTY_EPSILON = 1e-9


def _round(x: float) -> float:
    return round(float(x), STORAGE_PRECISION)


def _normalise(weights: Sequence[float]) -> tuple[float, ...]:
    """Project a non-negative weight vector onto the simplex.

    Negative weights are a caller bug rather than a representable state, so they
    raise instead of being clipped — clipping would silently turn an inverted
    sign into a plausible-looking distribution.
    """
    vals = [float(w) for w in weights]
    if not vals:
        raise ValueError("distribution must have at least one outcome")
    for i, v in enumerate(vals):
        if v < 0 or not math.isfinite(v):
            raise ValueError(f"weight {i} is {v}; weights must be finite and >= 0")
    total = math.fsum(vals)
    if total <= 0:
        raise ValueError("weights sum to zero; cannot normalise")
    out = [_round(v / total) for v in vals]
    # Rounding can move the sum off 1.0 by up to len*1e-9; absorb the residual
    # into the largest entry so the stored vector sums to exactly 1 after
    # rounding, keeping canonical JSON reproducible.
    residual = _round(1.0 - math.fsum(out))
    if residual:
        k = max(range(len(out)), key=lambda i: out[i])
        out[k] = _round(out[k] + residual)
    return tuple(out)


def shannon_entropy(probs: Iterable[float], *, base: float = 2.0) -> float:
    """Entropy in bits, used as the stored uncertainty scalar."""
    total = 0.0
    for p in probs:
        if p > 0:
            total -= p * math.log(p, base)
    return _round(max(total, 0.0))


# --------------------------------------------------------------------------
# pbit
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PBit:
    """A probabilistic bit: ``P(x = 1) = p``.

    Used for ``bridge_viability``, ``boundary_violation`` and
    ``action_authorized`` — the three nodes the sovereignty guard reads.

    The guard never thresholds a pbit on its own; see
    :meth:`is_certainly_false`. A boundary check that passed at ``p = 0.02``
    would be a 2%-chance violation waved through, which is not what
    ``cannot-be-overridden`` means. Hard vetoes are constructed as exact 0/1
    (:meth:`certain`) by the deterministic-veto path in ``circuits``.
    """

    p: float
    label: str = ""

    def __post_init__(self) -> None:
        if not math.isfinite(self.p) or not 0.0 <= self.p <= 1.0:
            raise ValueError(f"pbit probability must be in [0, 1], got {self.p!r}")
        object.__setattr__(self, "p", _round(self.p))

    @classmethod
    def certain(cls, value: bool, label: str = "") -> PBit:
        return cls(1.0 if value else 0.0, label)

    @property
    def is_certain(self) -> bool:
        return self.p <= CERTAINTY_EPSILON or self.p >= 1.0 - CERTAINTY_EPSILON

    def is_certainly_true(self) -> bool:
        """Exactly 1. Only a deterministic construction reaches this."""
        return self.p >= 1.0 - CERTAINTY_EPSILON

    def is_certainly_false(self) -> bool:
        """Exactly 0 — the only state that clears a hard-boundary check."""
        return self.p <= CERTAINTY_EPSILON

    @property
    def entropy(self) -> float:
        return shannon_entropy((self.p, 1.0 - self.p))

    def to_json(self) -> dict[str, Any]:
        return {"type": "pbit", "label": self.label, "p": self.p}

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> PBit:
        return cls(float(doc["p"]), str(doc.get("label", "")))


# --------------------------------------------------------------------------
# pdit
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PDit:
    """A probabilistic k-ary digit over *named* outcomes.

    Carries ``tension_class``, ``bridge_candidate``, ``topology_descriptor`` and
    ``resolution_status``. Outcomes are named rather than indexed because the
    labels are written into JSONB and read back by other services; an index would
    silently re-map if the class list were ever reordered.
    """

    outcomes: tuple[str, ...]
    probs: tuple[float, ...]
    label: str = ""

    def __post_init__(self) -> None:
        if len(self.outcomes) != len(self.probs):
            raise ValueError(
                f"pdit has {len(self.outcomes)} outcomes but "
                f"{len(self.probs)} probabilities"
            )
        if len(set(self.outcomes)) != len(self.outcomes):
            raise ValueError(f"pdit outcomes must be unique, got {self.outcomes}")
        object.__setattr__(self, "outcomes", tuple(str(o) for o in self.outcomes))
        object.__setattr__(self, "probs", _normalise(self.probs))

    @classmethod
    def uniform(cls, outcomes: Sequence[str], label: str = "") -> PDit:
        return cls(tuple(outcomes), tuple(1.0 for _ in outcomes), label)

    @classmethod
    def certain(cls, outcomes: Sequence[str], outcome: str, label: str = "") -> PDit:
        outcomes = tuple(outcomes)
        if outcome not in outcomes:
            raise ValueError(f"{outcome!r} not among outcomes {outcomes}")
        return cls(
            outcomes, tuple(1.0 if o == outcome else 0.0 for o in outcomes), label
        )

    @classmethod
    def from_scores(
        cls, scores: Mapping[str, float], *, temperature: float = 1.0, label: str = ""
    ) -> PDit:
        """Softmax over named scores, at ``temperature``.

        Lower temperature sharpens toward the argmax; the bridge selector uses
        this to trade "commit to the top strategy" against "keep the runner-up
        alive in the audit record".
        """
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if not scores:
            raise ValueError("from_scores requires at least one score")
        names = tuple(scores)
        raw = [scores[n] / temperature for n in names]
        peak = max(raw)
        return cls(names, tuple(math.exp(v - peak) for v in raw), label)

    @property
    def argmax(self) -> str:
        return self.outcomes[max(range(len(self.probs)), key=lambda i: self.probs[i])]

    @property
    def max_prob(self) -> float:
        return max(self.probs)

    @property
    def entropy(self) -> float:
        return shannon_entropy(self.probs)

    @property
    def margin(self) -> float:
        """Gap between the top two outcomes.

        This is the decision margin the persistence-stability result refers to:
        a descriptor is only claimed invariant while perturbation stays below
        ``margin / 2``.
        """
        if len(self.probs) < 2:
            return _round(self.probs[0])
        top, second = sorted(self.probs, reverse=True)[:2]
        return _round(top - second)

    def prob(self, outcome: str) -> float:
        try:
            return self.probs[self.outcomes.index(outcome)]
        except ValueError:
            raise KeyError(f"{outcome!r} not among outcomes {self.outcomes}") from None

    def as_mapping(self) -> dict[str, float]:
        return dict(zip(self.outcomes, self.probs))

    def to_json(self) -> dict[str, Any]:
        return {
            "type": "pdit",
            "label": self.label,
            "outcomes": list(self.outcomes),
            "probs": list(self.probs),
            "argmax": self.argmax,
            "entropy": self.entropy,
            "margin": self.margin,
        }

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> PDit:
        return cls(
            tuple(doc["outcomes"]), tuple(doc["probs"]), str(doc.get("label", ""))
        )


# --------------------------------------------------------------------------
# pmode
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PMode:
    """A Gaussian over named continuous dimensions: ``N(mean, cov)``.

    Carries ``individual_intent``, ``group_intent``, ``tension_gradient``,
    ``group_tension`` and ``confidence_state``.

    Only the diagonal is required on construction; the full covariance is kept
    when the kernel produces one (the affine-Gaussian simulator returns a dense
    posterior covariance, and its off-diagonal terms are what distinguish "these
    two participants disagree independently" from "they disagree together").
    """

    dimensions: tuple[str, ...]
    mean: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]
    label: str = ""

    def __post_init__(self) -> None:
        n = len(self.dimensions)
        if n == 0:
            raise ValueError("pmode must have at least one dimension")
        if len(set(self.dimensions)) != n:
            raise ValueError(f"pmode dimensions must be unique, got {self.dimensions}")
        if len(self.mean) != n:
            raise ValueError(
                f"pmode has {n} dimensions but {len(self.mean)} mean entries"
            )
        cov = tuple(tuple(_round(v) for v in row) for row in self.covariance)
        if len(cov) != n or any(len(row) != n for row in cov):
            raise ValueError(f"pmode covariance must be {n}x{n}")
        for i in range(n):
            if cov[i][i] < 0:
                raise ValueError(
                    f"pmode variance for {self.dimensions[i]!r} is negative: "
                    f"{cov[i][i]}"
                )
        object.__setattr__(self, "dimensions", tuple(str(d) for d in self.dimensions))
        object.__setattr__(self, "mean", tuple(_round(v) for v in self.mean))
        object.__setattr__(self, "covariance", cov)

    @classmethod
    def from_diagonal(
        cls,
        dimensions: Sequence[str],
        mean: Sequence[float],
        variance: Sequence[float] | float,
        label: str = "",
    ) -> PMode:
        n = len(dimensions)
        var = [float(variance)] * n if isinstance(variance, (int, float)) else list(
            variance
        )
        cov = tuple(
            tuple(var[i] if i == j else 0.0 for j in range(n)) for i in range(n)
        )
        return cls(tuple(dimensions), tuple(mean), cov, label)

    @property
    def variance(self) -> tuple[float, ...]:
        return tuple(self.covariance[i][i] for i in range(len(self.dimensions)))

    def value(self, dimension: str) -> float:
        try:
            return self.mean[self.dimensions.index(dimension)]
        except ValueError:
            raise KeyError(f"{dimension!r} not among {self.dimensions}") from None

    def uncertainty(self, dimension: str) -> float:
        try:
            i = self.dimensions.index(dimension)
        except ValueError:
            raise KeyError(f"{dimension!r} not among {self.dimensions}") from None
        return _round(math.sqrt(max(self.covariance[i][i], 0.0)))

    @property
    def confidence(self) -> float:
        """A single ``[0, 1]`` confidence derived from mean posterior variance.

        ``1 / (1 + var)`` — zero variance is confidence 1, and confidence falls
        off smoothly rather than saturating, so a stale high-variance estimate
        never presents as certain. This is the scalar the card stores as
        ``confidence`` on tension snapshots and profile revisions.
        """
        mean_var = math.fsum(self.variance) / len(self.variance)
        return _round(1.0 / (1.0 + max(mean_var, 0.0)))

    def as_mapping(self) -> dict[str, float]:
        return dict(zip(self.dimensions, self.mean))

    def clamped(self, lo: float = 0.0, hi: float = 1.0) -> PMode:
        """Clamp the mean into ``[lo, hi]``, leaving covariance untouched.

        Tension dimensions are declared on ``[0, 1]`` in the card. A Gaussian is
        unbounded, so the posterior mean can land just outside after an update;
        clamping the mean keeps the stored snapshot inside its declared range
        while the retained variance still reports how far outside it wanted to
        go.
        """
        return PMode(
            self.dimensions,
            tuple(min(max(v, lo), hi) for v in self.mean),
            self.covariance,
            self.label,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "type": "pmode",
            "label": self.label,
            "dimensions": list(self.dimensions),
            "mean": list(self.mean),
            "variance": list(self.variance),
            "covariance": [list(r) for r in self.covariance],
            "confidence": self.confidence,
        }

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> PMode:
        if "covariance" in doc:
            cov = tuple(tuple(float(v) for v in row) for row in doc["covariance"])
            return cls(
                tuple(doc["dimensions"]),
                tuple(float(v) for v in doc["mean"]),
                cov,
                str(doc.get("label", "")),
            )
        return cls.from_diagonal(
            tuple(doc["dimensions"]),
            tuple(float(v) for v in doc["mean"]),
            tuple(float(v) for v in doc["variance"]),
            str(doc.get("label", "")),
        )


# --------------------------------------------------------------------------
# the six tension dimensions
# --------------------------------------------------------------------------

#: Declared in ``tension_gradient.dimensions``. Order is fixed here because it is
#: the vector order used by every kernel and stored in JSONB; reordering it would
#: silently re-label stored gradients.
TENSION_DIMENSIONS: tuple[str, ...] = (
    "goal_divergence",
    "constraint_collision",
    "semantic_misalignment",
    "priority_mismatch",
    "temporal_pressure",
    "participation_imbalance",
)

#: ``tension_gradient.classes``, in card order.
TENSION_CLASSES: tuple[str, ...] = (
    "aligned",
    "semantic-gap",
    "priority-gap",
    "constraint-collision",
    "value-conflict",
    "authorization-conflict",
    "unresolved",
)

#: ``bridge_engine.ordered_strategies``, in try-order.
BRIDGE_STRATEGIES: tuple[str, ...] = (
    "semantic-translation",
    "constraint-reconciliation",
    "priority-sequencing",
    "perspective-adaptation",
    "pareto-option",
    "parallel-fork",
    "explicit-escalation",
)

#: Terminal states of the resolution_status pdit.
RESOLUTION_STATES: tuple[str, ...] = (
    "resolved",
    "partially-resolved",
    "unresolved",
    "blocked",
)


@dataclass(frozen=True, slots=True)
class TorxDecisionState:
    """One full pass through the card's factor graph.

    Bundling the nodes keeps the guard's inputs and the audit record's contents
    identical by construction — everything the decision was made on is what gets
    attested.
    """

    tension_gradient: PMode
    tension_class: PDit
    topology_descriptor: PDit
    bridge_candidate: PDit
    bridge_viability: PBit
    boundary_violation: PBit
    action_authorized: PBit
    resolution_status: PDit
    backend: str = "unknown"
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "tension_gradient": self.tension_gradient.to_json(),
            "tension_class": self.tension_class.to_json(),
            "topology_descriptor": self.topology_descriptor.to_json(),
            "bridge_candidate": self.bridge_candidate.to_json(),
            "bridge_viability": self.bridge_viability.to_json(),
            "boundary_violation": self.boundary_violation.to_json(),
            "action_authorized": self.action_authorized.to_json(),
            "resolution_status": self.resolution_status.to_json(),
            "diagnostics": dict(self.diagnostics),
        }
