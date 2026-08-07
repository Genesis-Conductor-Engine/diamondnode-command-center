"""The three TORX kernels behave as the model card declares.

``intent_update`` is checked against the closed-form Gaussian (Kalman) update
rather than against a golden vector, because the point of running it on
``AffineGaussianSimulator`` is that it *is* the exact posterior — a golden
vector would only prove the code still does whatever it did last time.
"""

from __future__ import annotations

import math

import pytest

from src.torx_layer.kernels import (
    MIN_EVIDENCE_VARIANCE,
    BridgeConstraints,
    _robust_norm,
    _variance_from_confidence,
    bridge_selection,
    diffuse_prior,
    intent_update,
    tension_classification,
    tension_gradient,
)
from src.torx_layer.state import TENSION_DIMENSIONS, PMode

TOL = 1e-6
DIMS = TENSION_DIMENSIONS


def kalman(prior_mean, prior_var, evidence, evidence_var):
    """Closed-form scalar Gaussian posterior — the oracle."""
    gain = prior_var / (prior_var + evidence_var)
    return (
        prior_mean + gain * (evidence - prior_mean),
        prior_var * evidence_var / (prior_var + evidence_var),
    )


# --------------------------------------------------------------------------
# intent_update
# --------------------------------------------------------------------------


@pytest.mark.parametrize("confidence", [0.1, 0.5, 0.75, 0.9, 0.99])
@pytest.mark.parametrize("prior_var", [0.05, 0.25, 1.0])
def test_intent_update_matches_the_closed_form(confidence, prior_var):
    prior_mean = [0.2, 0.0, 0.5, 0.3, 0.6, 0.1]
    evidence = [0.8, 0.0, 0.5, 0.3, 0.2, 0.1]
    prior = PMode.from_diagonal(DIMS, prior_mean, prior_var)

    posterior = intent_update(prior, evidence, confidence=confidence)
    evidence_var = _variance_from_confidence(confidence)

    for i, dim in enumerate(DIMS):
        exp_mean, exp_var = kalman(
            prior_mean[i], prior_var, evidence[i], evidence_var
        )
        assert posterior.value(dim) == pytest.approx(exp_mean, abs=TOL)
        assert posterior.variance[i] == pytest.approx(exp_var, abs=TOL)


def test_high_confidence_evidence_dominates_the_prior():
    prior = PMode.from_diagonal(DIMS, [0.0] * 6, 1.0)
    evidence = [0.9] * 6
    posterior = intent_update(prior, evidence, confidence=0.999)
    for dim in DIMS:
        assert posterior.value(dim) == pytest.approx(0.9, abs=0.01)


def test_zero_confidence_evidence_barely_moves_the_prior():
    """Confidence 0 must be ignored, not trusted.

    ``_variance_from_confidence`` maps it to a huge variance, so the Kalman gain
    goes to zero — the opposite of what a naive ``weight = confidence`` blend
    would do at the boundary.
    """
    prior = PMode.from_diagonal(DIMS, [0.25] * 6, 0.25)
    posterior = intent_update(prior, [0.95] * 6, confidence=0.0)
    for dim in DIMS:
        assert posterior.value(dim) == pytest.approx(0.25, abs=1e-3)


def test_a_correlated_prior_stays_correlated():
    """Off-diagonal covariance must survive the update.

    Two intent dimensions that move together carry information about each other;
    diagonalising the prior would silently discard it and report a more
    confident, less coupled posterior than the evidence supports.
    """
    n = len(DIMS)
    cov = tuple(
        tuple(0.25 if i == j else 0.12 for j in range(n)) for i in range(n)
    )
    prior = PMode(DIMS, tuple([0.2] * n), cov)
    posterior = intent_update(prior, [0.8] * n, confidence=0.8)
    assert abs(posterior.covariance[0][1]) > 1e-6, "prior correlation was lost"


def test_decay_strictly_widens_the_posterior():
    prior = PMode.from_diagonal(DIMS, [0.3] * 6, 0.2)
    evidence = [0.7] * 6
    fresh = intent_update(prior, evidence, confidence=0.8, decay=0.0)
    stale = intent_update(prior, evidence, confidence=0.8, decay=0.6)
    assert stale.variance[0] > fresh.variance[0]
    assert stale.confidence < fresh.confidence


def test_decay_never_pulls_the_mean_toward_a_default():
    """Staleness increases uncertainty; it must not invent an opinion."""
    prior = PMode.from_diagonal(DIMS, [0.9] * 6, 0.2)
    decayed = intent_update(prior, [0.9] * 6, confidence=0.5, decay=0.8)
    for dim in DIMS:
        assert decayed.value(dim) == pytest.approx(0.9, abs=1e-6)


def test_intent_update_rejects_bad_input():
    prior = diffuse_prior()
    with pytest.raises(ValueError, match="intent dimensions"):
        intent_update(prior, [0.1, 0.2], confidence=0.8)
    with pytest.raises(ValueError, match="decay"):
        intent_update(prior, [0.1] * 6, confidence=0.8, decay=1.0)
    with pytest.raises(ValueError, match="finite"):
        intent_update(prior, [float("nan")] * 6, confidence=0.8)


def test_confidence_one_is_floored_not_infinite():
    """A zero-variance measurement would claim an infallible estimator."""
    assert _variance_from_confidence(1.0) >= MIN_EVIDENCE_VARIANCE


# --------------------------------------------------------------------------
# tension_gradient / tension_classification
# --------------------------------------------------------------------------


def test_gradient_is_the_weighted_difference():
    ind = PMode.from_diagonal(DIMS, [0.9, 0.1, 0.8, 0.2, 0.5, 0.1], 0.01)
    grp = PMode.from_diagonal(DIMS, [0.4, 0.1, 0.3, 0.2, 0.5, 0.1], 0.01)
    grad = tension_gradient(ind, grp)
    assert grad.value("goal_divergence") == pytest.approx(0.5, abs=TOL)
    assert grad.value("constraint_collision") == pytest.approx(0.0, abs=TOL)


def test_gradient_uncertainty_adds_and_never_cancels():
    """Two shaky estimates make a shaky tension, not a confident one."""
    ind = PMode.from_diagonal(DIMS, [0.5] * 6, 0.30)
    grp = PMode.from_diagonal(DIMS, [0.5] * 6, 0.20)
    grad = tension_gradient(ind, grp)
    for v in grad.variance:
        assert v == pytest.approx(0.50, abs=TOL)
    assert grad.confidence < ind.confidence
    assert grad.confidence < grp.confidence


def test_gradient_requires_matching_dimensions():
    ind = PMode.from_diagonal(("a", "b"), [0.1, 0.2], 0.1)
    grp = PMode.from_diagonal(("a", "c"), [0.1, 0.2], 0.1)
    with pytest.raises(ValueError, match="share dimensions"):
        tension_gradient(ind, grp)


def test_gradient_rejects_negative_weights():
    ind = PMode.from_diagonal(DIMS, [0.5] * 6, 0.1)
    with pytest.raises(ValueError, match=">= 0"):
        tension_gradient(ind, ind, weights={"goal_divergence": -1.0})


def test_aligned_intents_classify_as_aligned():
    same = PMode.from_diagonal(DIMS, [0.3, 0.0, 0.2, 0.1, 0.2, 0.0], 0.01)
    est = tension_classification(same, same)
    assert est.tension_class.argmax == "aligned"
    assert est.scalar == pytest.approx(0.0, abs=TOL)


def test_a_semantic_gap_classifies_as_semantic_gap():
    ind = PMode.from_diagonal(DIMS, [0.05, 0.0, 0.85, 0.05, 0.0, 0.0], 0.01)
    grp = PMode.from_diagonal(DIMS, [0.05, 0.0, 0.05, 0.05, 0.0, 0.0], 0.01)
    est = tension_classification(ind, grp)
    assert est.tension_class.argmax == "semantic-gap"


def test_an_authorization_gap_is_classified_as_such():
    ind = PMode.from_diagonal(DIMS, [0.1] * 6, 0.01)
    grp = PMode.from_diagonal(DIMS, [0.1] * 6, 0.01)
    est = tension_classification(ind, grp, has_authorization_gap=True)
    assert est.tension_class.argmax == "authorization-conflict"


def test_classification_keeps_the_full_distribution():
    """``preserve_inference_uncertainty``: a near-tie must remain a near-tie."""
    ind = PMode.from_diagonal(DIMS, [0.0, 0.0, 0.5, 0.5, 0.0, 0.0], 0.01)
    grp = PMode.from_diagonal(DIMS, [0.0] * 6, 0.01)
    est = tension_classification(ind, grp)
    assert est.tension_class.margin < 0.35, "a near-tie collapsed to a hard label"
    assert est.tension_class.entropy > 1.0


def test_robust_norm_is_bounded_by_mean_and_max():
    for values in ([0.1, 0.2, 0.9], [0.5] * 6, [0.0, 0.0, 1.0]):
        mean = math.fsum(values) / len(values)
        assert mean <= _robust_norm(values) <= max(values)


def test_robust_norm_is_moved_by_one_severe_dimension():
    """A plain mean would let five calm dimensions hide one severe one."""
    calm = _robust_norm([0.1] * 6)
    one_severe = _robust_norm([0.1] * 5 + [1.0])
    assert one_severe > calm + 0.3


# --------------------------------------------------------------------------
# bridge_selection
# --------------------------------------------------------------------------


def semantic_gap_estimate():
    ind = PMode.from_diagonal(DIMS, [0.05, 0.0, 0.85, 0.05, 0.0, 0.0], 0.01)
    grp = PMode.from_diagonal(DIMS, [0.05, 0.0, 0.05, 0.05, 0.0, 0.0], 0.01)
    return tension_classification(ind, grp)


def test_a_beneficial_bridge_is_viable():
    est = semantic_gap_estimate()
    decision = bridge_selection(
        est, BridgeConstraints(False, True), predicted_delta_tension=-0.4
    )
    assert decision.bridge_viability.p > 0.0
    assert decision.bridge_candidate.argmax == "semantic-translation"


@pytest.mark.parametrize("delta", [0.0, 0.05, 0.5])
def test_a_non_negative_delta_drives_viability_to_zero(delta):
    """The card's application rule requires predicted delta tension < 0."""
    est = semantic_gap_estimate()
    decision = bridge_selection(
        est, BridgeConstraints(False, True), predicted_delta_tension=delta
    )
    assert decision.bridge_viability.p == 0.0


def test_low_confidence_blocks_the_bridge():
    ind = PMode.from_diagonal(DIMS, [0.05, 0.0, 0.85, 0.05, 0.0, 0.0], 5.0)
    grp = PMode.from_diagonal(DIMS, [0.05, 0.0, 0.05, 0.05, 0.0, 0.0], 5.0)
    est = tension_classification(ind, grp)
    decision = bridge_selection(
        est,
        BridgeConstraints(False, True),
        predicted_delta_tension=-0.5,
        minimum_confidence=0.9,
    )
    assert decision.bridge_viability.p == 0.0


def test_a_boundary_violation_selects_escalation_and_blocks():
    est = semantic_gap_estimate()
    decision = bridge_selection(
        est, BridgeConstraints(True, True), predicted_delta_tension=-0.4
    )
    assert decision.bridge_viability.is_certainly_false()
    assert decision.bridge_candidate.argmax == "explicit-escalation"
    assert decision.resolution_status.argmax == "blocked"


def test_missing_authorization_selects_escalation_and_blocks():
    est = semantic_gap_estimate()
    decision = bridge_selection(
        est, BridgeConstraints(False, False), predicted_delta_tension=-0.4
    )
    assert decision.bridge_viability.is_certainly_false()
    assert decision.bridge_candidate.argmax == "explicit-escalation"


def test_unavailable_strategies_get_no_mass():
    est = semantic_gap_estimate()
    decision = bridge_selection(
        est,
        BridgeConstraints(
            False, True, available_strategies=("parallel-fork", "explicit-escalation")
        ),
        predicted_delta_tension=-0.4,
    )
    assert decision.bridge_candidate.prob("semantic-translation") < 1e-6
    assert decision.bridge_candidate.argmax in {"parallel-fork", "explicit-escalation"}


def test_constraints_reject_unknown_strategies():
    with pytest.raises(ValueError, match="unknown bridge strategies"):
        BridgeConstraints(False, True, available_strategies=("teleportation",))
    with pytest.raises(ValueError, match="at least one strategy"):
        BridgeConstraints(False, True, available_strategies=())


def test_decision_state_serialises_every_factor_graph_node():
    est = semantic_gap_estimate()
    decision = bridge_selection(
        est, BridgeConstraints(False, True), predicted_delta_tension=-0.3
    )
    doc = decision.to_json()
    for node in (
        "tension_gradient",
        "tension_class",
        "topology_descriptor",
        "bridge_candidate",
        "bridge_viability",
        "boundary_violation",
        "action_authorized",
        "resolution_status",
    ):
        assert node in doc, f"{node} missing from the attested decision record"
    assert doc["diagnostics"]["selected_strategy"] == decision.bridge_candidate.argmax
