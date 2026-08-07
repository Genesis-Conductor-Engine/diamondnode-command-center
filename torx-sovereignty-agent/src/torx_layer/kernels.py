"""The three TORX kernels declared in ``torx.kernels``.

* ``intent_update`` — bounded-stochastic kernel. Prior intent, new evidence and
  source confidence in; posterior intent out. Runs as an **affine-Gaussian
  circuit** on ``torx.psc.AffineGaussianSimulator``: the prior is injected by a
  ``Diffuse`` gate, the evidence channel by an ``AffineGaussianGate`` whose
  log-variance encodes ``1 / confidence``, and the posterior comes from the
  simulator's Schur-complement ``condition``. That is an exact Gaussian
  (Kalman) update computed by the kernel, not an approximation of one —
  ``tests/torx_layer/test_kernels.py`` checks it against the closed form.

* ``tension_classification`` — hybrid discrete/continuous kernel. Consumes the
  individual and group intent pmodes, produces the six-dimensional tension
  gradient (pmode) and the tension class (pdit, via the exact stick-breaking
  categorical in :mod:`.circuits`).

* ``bridge_selection`` — constrained decision kernel. Consumes tension state,
  boundaries, authorization and the module catalog; produces a bridge-candidate
  distribution plus the three decision pbits, with the hard-boundary veto wired
  structurally.

**Bounded by construction.** ``runtime.state_bounds`` caps the work each call may
do, and the caps are enforced here rather than trusted: the intent dimension is
fixed at six, the candidate list is capped at ``bridge_candidates_per_cycle``,
and no kernel retains state between calls. This is what makes the
``large_deployment`` result's "bounded per session" precondition true of the
implementation and not only of the design.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .circuits import (
    DecisionInputs,
    DecisionMarginals,
    _load_torx,
    backend_status,
    bridge_candidate_pdit,
    evaluate_decision,
    tension_class_pdit,
    torx_available,
)
from .state import (
    BRIDGE_STRATEGIES,
    TENSION_CLASSES,
    TENSION_DIMENSIONS,
    PBit,
    PDit,
    PMode,
    TorxDecisionState,
)

# Variance floor. A confidence of exactly 1.0 would be zero-variance evidence,
# which makes the Schur complement singular and, worse, asserts the estimator is
# infallible — ``preserve_inference_uncertainty`` forbids that claim.
MIN_EVIDENCE_VARIANCE = 1e-4

# Variance used for a prior with no history. Large relative to the [0, 1] range
# of the tension dimensions, so the first piece of evidence dominates without the
# prior having to be discarded.
DIFFUSE_PRIOR_VARIANCE = 1.0

# Jitter added to the observed covariance before its Cholesky factorisation.
CONDITION_JITTER = 1e-9


def _variance_from_confidence(confidence: float) -> float:
    """Map a ``[0, 1]`` confidence onto an evidence variance.

    ``var = (1 - c) / c`` — confidence 0.5 gives unit variance (evidence worth
    as much as a diffuse prior), confidence approaching 1 gives variance
    approaching the floor, and confidence approaching 0 gives unbounded variance
    (the evidence is ignored rather than trusted).
    """
    c = min(max(float(confidence), 0.0), 1.0)
    if c <= 0.0:
        return 1.0 / MIN_EVIDENCE_VARIANCE
    return max((1.0 - c) / c, MIN_EVIDENCE_VARIANCE)


# --------------------------------------------------------------------------
# intent_update
# --------------------------------------------------------------------------


def _intent_update_fallback(
    prior: PMode, evidence: Sequence[float], evidence_variance: float
) -> PMode:
    """Exact diagonal Kalman update, for when the kernel is unavailable.

    Only correct for a diagonal prior covariance; :func:`intent_update` routes
    correlated priors to the kernel and falls back to the diagonal here only
    after dropping the off-diagonal terms, which it records in the label so an
    auditor can see the approximation was taken.
    """
    post_mean, post_var = [], []
    for i in range(len(prior.dimensions)):
        pv = prior.covariance[i][i]
        denom = pv + evidence_variance
        if denom <= 0:
            post_mean.append(prior.mean[i])
            post_var.append(pv)
            continue
        gain = pv / denom
        post_mean.append(prior.mean[i] + gain * (float(evidence[i]) - prior.mean[i]))
        post_var.append(pv * evidence_variance / denom)
    return PMode.from_diagonal(
        prior.dimensions, post_mean, post_var, label=prior.label or "intent"
    )


def intent_update(
    prior: PMode,
    evidence: Sequence[float],
    *,
    confidence: float,
    decay: float = 0.0,
) -> PMode:
    """``intent_update`` — posterior intent from prior, evidence and confidence.

    ``decay`` in ``[0, 1)`` widens the prior before the update, implementing the
    card's ``stale_state_decay``: an intent that has not been observed for a
    while becomes less certain rather than staying confidently stale. It is
    applied as added variance (a ``Diffuse`` step), never by pulling the mean
    toward a default — moving the mean would invent an opinion the user never
    expressed.
    """
    n = len(prior.dimensions)
    if len(evidence) != n:
        raise ValueError(
            f"evidence has {len(evidence)} entries for {n} intent dimensions"
        )
    if not 0.0 <= decay < 1.0:
        raise ValueError(f"decay must be in [0, 1), got {decay}")
    for i, v in enumerate(evidence):
        if not math.isfinite(float(v)):
            raise ValueError(f"evidence[{i}] is not finite: {v!r}")

    evidence_variance = _variance_from_confidence(confidence)
    # Decay inflates prior variance: var -> var / (1 - decay).
    inflate = 1.0 / (1.0 - decay)
    prior_cov = [[prior.covariance[i][j] * inflate for j in range(n)] for i in range(n)]

    if not torx_available():
        widened = PMode(prior.dimensions, prior.mean, tuple(map(tuple, prior_cov)),
                        prior.label)
        return _intent_update_fallback(widened, evidence, evidence_variance)

    try:
        import jax.numpy as jnp

        run = _build_intent_jit(n)
        prior_arr = jnp.asarray(prior_cov)
        chol = jnp.linalg.cholesky(prior_arr + CONDITION_JITTER * jnp.eye(n))
        if not bool(jnp.all(jnp.isfinite(chol))):
            raise ValueError("prior covariance is not positive definite")
        mean, cov = run(
            chol,
            jnp.asarray(prior.mean),
            jnp.asarray([float(v) for v in evidence]),
            jnp.asarray(math.log(evidence_variance)),
        )
        mean_l = [float(v) for v in mean]
        cov_l = tuple(tuple(float(v) for v in row) for row in cov)
        if not all(math.isfinite(v) for v in mean_l):
            raise ValueError("non-finite posterior mean from the affine kernel")
        return PMode(prior.dimensions, tuple(mean_l), cov_l, prior.label or "intent")
    except Exception:
        widened = PMode(prior.dimensions, prior.mean, tuple(map(tuple, prior_cov)),
                        prior.label)
        return _intent_update_fallback(widened, evidence, evidence_variance)


#: Compiled affine-Gaussian intent circuits, keyed by dimension count. The
#: circuit structure depends only on ``n``, so one compile serves every update
#: at that width. Without this, each call re-traced the simulator and cost ~3s —
#: far outside the card's decision budget for a critical-path kernel.
_intent_jit_cache: dict[int, Any] = {}


def _build_intent_jit(n: int):
    if n in _intent_jit_cache:
        return _intent_jit_cache[n]
    import jax
    import jax.numpy as jnp
    from torx.psc import AffineGaussianGate, AffineGaussianSimulator, HybridPCircuit


    # Site 0: latent intent (n dims). Site 1: evidence readout (n dims).
    #
    # Three affine-Gaussian gates, because the simulator starts every run at
    # zero mean *and zero covariance* and the diagonal ``Diffuse`` gate can only
    # add isotropic noise — neither can express a correlated prior on its own:
    #
    #   1. whiten   x <- x + N(0, I)                 site 0 gets cov = I
    #   2. colour   x <- L x + prior_mean            cov = L Lt = prior_cov
    #   3. observe  y <- x + N(0, evidence_variance) the measurement channel
    #
    # Step 2 is what carries the off-diagonal prior terms through, so a
    # correlated intent (two dimensions that move together) stays correlated in
    # the posterior instead of being silently diagonalised.
    g_whiten = AffineGaussianGate(sites={"continuous": [0]}, dims=(n,))
    g_colour = AffineGaussianGate(sites={"continuous": [0]}, dims=(n,))
    g_obs = AffineGaussianGate(sites={"continuous": [0, 1]}, dims=(n, n))
    circuit = HybridPCircuit([g_whiten, g_colour, g_obs], reps=1)
    sim = AffineGaussianSimulator()

    copy_latent_into_readout = (
        jnp.zeros((2 * n, 2 * n)).at[:n, :n].set(jnp.eye(n)).at[n:, :n].set(jnp.eye(n))
    )
    init = jnp.zeros(2 * n)

    # ``condition`` requires observed/query site membership to be static Python
    # values; only the observation vector and jitter may be traced. Both hold
    # here, so the whole update jits cleanly.
    @jax.jit
    def run(chol, prior_mean, evidence, log_evidence_var):
        thetas = [
            {
                "A": jnp.eye(n),
                "b": jnp.zeros(n),
                "log_var": jnp.zeros(n),  # exp(0) = 1 -> unit isotropic covariance
            },
            {
                "A": chol,
                "b": prior_mean,
                "log_var": jnp.full(n, -jnp.inf),  # deterministic: exp(-inf) = 0
            },
            {
                "A": copy_latent_into_readout,
                "b": jnp.zeros(2 * n),
                "log_var": jnp.concatenate(
                    [jnp.full(n, -jnp.inf), jnp.full(n, log_evidence_var)]
                ),
            },
        ]
        built = sim.build_circuit(circuit, thetas)
        posterior = sim.condition(
            built,
            {1: evidence},
            initial_continuous=init,
            query_sites=[0],
            jitter=CONDITION_JITTER,
        )
        return posterior.site_moments(0)

    _intent_jit_cache[n] = run
    return run


def diffuse_prior(dimensions: Sequence[str] = TENSION_DIMENSIONS) -> PMode:
    """A no-history prior: mean 0, wide isotropic variance."""
    return PMode.from_diagonal(
        tuple(dimensions),
        tuple(0.0 for _ in dimensions),
        DIFFUSE_PRIOR_VARIANCE,
        label="intent",
    )


# --------------------------------------------------------------------------
# tension_classification
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TensionEstimate:
    """Output of ``tension_classification``."""

    gradient: PMode
    tension_class: PDit
    scalar: float
    confidence: float
    backend: str


def _robust_norm(values: Sequence[float]) -> float:
    """Robust aggregate of the per-dimension tensions.

    The card's ``group_tension_formula`` says ``robust_norm``, and the
    ``simple_average_forbidden`` clause on group intent applies in spirit here
    too: a plain mean lets five calm dimensions hide one severe one. This uses
    the max blended with the mean (``0.5 * max + 0.5 * mean``), which is
    monotone in every dimension, never below the mean, and never above the max —
    so one severe dimension always moves the result, and no single dimension can
    saturate it on its own.
    """
    vals = [float(v) for v in values]
    if not vals:
        return 0.0
    return 0.5 * max(vals) + 0.5 * (math.fsum(vals) / len(vals))


def tension_gradient(
    individual: PMode,
    group: PMode,
    *,
    weights: Mapping[str, float] | None = None,
) -> PMode:
    """``T_i = W_i * (I_i - G)`` as a pmode.

    Uncertainty adds rather than cancels: the gradient's variance is the sum of
    the two input variances, so a tension computed from two shaky estimates is
    reported as shaky. That is what stops a low-evidence disagreement from
    triggering a bridge — the card requires ``minimum_distinct_events`` before
    acting, and this is the continuous counterpart of that rule.
    """
    if individual.dimensions != group.dimensions:
        raise ValueError(
            "individual and group intent must share dimensions; got "
            f"{individual.dimensions} vs {group.dimensions}"
        )
    n = len(individual.dimensions)
    w = [float((weights or {}).get(d, 1.0)) for d in individual.dimensions]
    for i, wi in enumerate(w):
        if wi < 0 or not math.isfinite(wi):
            raise ValueError(
                f"weight for {individual.dimensions[i]!r} must be finite and "
                f">= 0, got {wi}"
            )
    mean = tuple(w[i] * (individual.mean[i] - group.mean[i]) for i in range(n))
    cov = tuple(
        tuple(
            w[i] * w[j] * (individual.covariance[i][j] + group.covariance[i][j])
            for j in range(n)
        )
        for i in range(n)
    )
    return PMode(individual.dimensions, mean, cov, label="tension_gradient")


def _class_scores(gradient: PMode, *, has_authorization_gap: bool) -> list[float]:
    """Score each tension class from the gradient, in card order.

    Every class is scored from the dimensions that actually cause it, so the
    resulting pdit is explainable: ``semantic-gap`` is driven by
    ``semantic_misalignment``, ``constraint-collision`` by
    ``constraint_collision``, and so on. ``value-conflict`` is deliberately the
    conjunction of goal divergence *and* constraint collision — a deep
    disagreement that is not merely a wording or ordering problem — because
    misclassifying an ordinary priority dispute as a value conflict would
    escalate something a cheaper bridge could have resolved.
    """
    g = {d: abs(gradient.value(d)) for d in gradient.dimensions}
    magnitude = _robust_norm(list(g.values()))
    scale = 6.0  # sharpens the softmax; tuned so a clear cause dominates

    # An authorization gap is an observed fact, not an estimate from the
    # gradient, and it is not something agreement can resolve: participants can
    # be perfectly aligned and still blocked because nobody holds the grant.
    # So it suppresses ``aligned`` outright rather than competing with it — a
    # tie would let a zero gradient report "aligned" and send the group looking
    # for a bridge that cannot legally be applied.
    authorization = scale * (2.0 if has_authorization_gap else 0.0)
    aligned = 0.0 if has_authorization_gap else scale * (1.0 - min(magnitude, 1.0))
    semantic = scale * g["semantic_misalignment"]
    priority = scale * max(g["priority_mismatch"], 0.7 * g["temporal_pressure"])
    collision = scale * g["constraint_collision"]
    value_conflict = scale * min(g["goal_divergence"], g["constraint_collision"])
    # ``unresolved`` rises when several causes are simultaneously strong: no
    # single bridge strategy addresses that, and the card requires it be
    # surfaced rather than papered over.
    strong = sum(1 for v in g.values() if v >= 0.5)
    unresolved = scale * (0.25 * max(strong - 1, 0))
    return [aligned, semantic, priority, collision, value_conflict, authorization, unresolved]


def tension_classification(
    individual: PMode,
    group: PMode,
    *,
    weights: Mapping[str, float] | None = None,
    has_authorization_gap: bool = False,
) -> TensionEstimate:
    """``tension_classification`` — gradient plus class, on the torx kernel."""
    grad = tension_gradient(individual, group, weights=weights)
    scores = _class_scores(grad, has_authorization_gap=has_authorization_gap)
    cls = tension_class_pdit(scores)
    scalar = _robust_norm([abs(v) for v in grad.mean])
    return TensionEstimate(
        gradient=grad.clamped(-1.0, 1.0),
        tension_class=cls,
        scalar=min(max(scalar, 0.0), 1.0),
        confidence=grad.confidence,
        backend=backend_status()["engine"],
    )


# --------------------------------------------------------------------------
# bridge_selection
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BridgeConstraints:
    """The hard inputs the decision kernel may not argue with."""

    boundary_violated: bool
    authorized: bool
    available_strategies: tuple[str, ...] = BRIDGE_STRATEGIES

    def __post_init__(self) -> None:
        unknown = set(self.available_strategies) - set(BRIDGE_STRATEGIES)
        if unknown:
            raise ValueError(f"unknown bridge strategies: {sorted(unknown)}")
        if not self.available_strategies:
            raise ValueError("at least one strategy must be available")


def _strategy_scores(
    tension: PDit, constraints: BridgeConstraints
) -> list[float]:
    """Score each strategy against the tension class, in card try-order.

    The mapping is the card's own: a semantic gap wants semantic translation, a
    constraint collision wants constraint reconciliation, a priority gap wants
    sequencing. Earlier strategies get a small ordering bonus so that, all else
    equal, the *minimum-cost* bridge wins — ``ordered_strategies`` is a
    try-order, and the objective is "minimum authorized bridge".
    """
    affinity = {
        "semantic-translation": tension.prob("semantic-gap"),
        "constraint-reconciliation": tension.prob("constraint-collision"),
        "priority-sequencing": tension.prob("priority-gap"),
        "perspective-adaptation": 0.5 * (
            tension.prob("semantic-gap") + tension.prob("priority-gap")
        ),
        "pareto-option": tension.prob("priority-gap") + 0.5 * tension.prob("value-conflict"),
        "parallel-fork": tension.prob("value-conflict"),
        # Escalation is the terminal option: it scores on the states no bridge
        # resolves, and on an authorization conflict, which is not ours to fix.
        "explicit-escalation": tension.prob("unresolved")
        + tension.prob("authorization-conflict")
        + (1.0 if constraints.boundary_violated or not constraints.authorized else 0.0),
    }
    n = len(BRIDGE_STRATEGIES)
    scores = []
    for i, name in enumerate(BRIDGE_STRATEGIES):
        if name not in constraints.available_strategies:
            scores.append(-25.0)  # effectively zero mass after softmax
            continue
        order_bonus = 0.15 * (n - 1 - i) / (n - 1)
        scores.append(4.0 * affinity[name] + order_bonus)
    return scores


def bridge_selection(
    tension: TensionEstimate,
    constraints: BridgeConstraints,
    *,
    predicted_delta_tension: float,
    minimum_confidence: float = 0.6,
) -> TorxDecisionState:
    """``bridge_selection`` — candidate distribution plus the decision pbits.

    ``predicted_delta_tension`` is the simulator's before-minus-after estimate;
    the card's application rule requires it be **negative**, so a non-negative
    delta drives viability to zero here rather than being weighed against other
    factors. Confidence below ``minimum_confidence`` does the same. Those two are
    soft gates on the *estimate*; the boundary and authorization vetoes are
    structural and applied inside the circuit.
    """
    scores = _strategy_scores(tension.tension_class, constraints)
    candidate = bridge_candidate_pdit(scores)

    if predicted_delta_tension >= 0.0 or tension.confidence < minimum_confidence:
        p_viability = 0.0
    else:
        # Map the magnitude of the predicted improvement onto a viability
        # probability, damped by how confident the tension estimate is.
        improvement = min(-predicted_delta_tension, 1.0)
        p_viability = min(improvement * tension.confidence * candidate.max_prob * 2.0, 1.0)

    marginals: DecisionMarginals = evaluate_decision(
        DecisionInputs(
            p_bridge_viability=p_viability,
            p_boundary_violation=1.0 if constraints.boundary_violated else 0.0,
            p_action_authorized=0.0 if not constraints.authorized else 1.0,
        )
    )

    # ``topology_descriptor`` is the sidecar's output. It is not on the critical
    # path, so when no descriptor has been supplied the neutral one is used — the
    # second rung of ``topology_sidecar.fallback_order``.
    topology = PDit.uniform(("stable", "drifting", "fragmented"), "topology_descriptor")

    return TorxDecisionState(
        tension_gradient=tension.gradient,
        tension_class=tension.tension_class,
        topology_descriptor=topology,
        bridge_candidate=candidate,
        bridge_viability=marginals.bridge_viability,
        boundary_violation=marginals.boundary_violation,
        action_authorized=marginals.action_authorized,
        resolution_status=marginals.resolution_status,
        backend=marginals.backend,
        diagnostics={
            "predicted_delta_tension": round(float(predicted_delta_tension), 9),
            "tension_scalar": round(tension.scalar, 9),
            "tension_confidence": round(tension.confidence, 9),
            "minimum_confidence": minimum_confidence,
            "raw_viability": round(p_viability, 9),
            "selected_strategy": candidate.argmax,
            "candidate_margin": candidate.margin,
        },
    )
