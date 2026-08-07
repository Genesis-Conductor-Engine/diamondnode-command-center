"""The TORX decision circuit does what its docstring claims.

The central assertion here is :func:`test_veto_is_structural_for_any_estimate`.
Everything else in the system — the sovereignty guard, the MCP surface, the
bridge engine — trusts that a violated hard boundary yields a *certainly*
non-viable bridge. That guarantee comes from circuit wiring (a deterministic
``PCSWAP`` against a site pinned at 0), not from a parameter, so the test
generates arbitrary viability estimates and demands exactly 0 every time.
"""

from __future__ import annotations

import math
import re

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.torx_layer import circuits
from src.torx_layer.circuits import (
    DecisionInputs,
    _categorical_fallback,
    _decision_fallback,
    backend_status,
    bridge_candidate_pdit,
    evaluate_categorical,
    evaluate_decision,
    logit,
    sigmoid,
    tension_class_pdit,
    torx_available,
    veto_is_structural,
)
from src.torx_layer.state import BRIDGE_STRATEGIES, TENSION_CLASSES

KERNEL_TOL = 1e-6


@pytest.fixture
def forced_fallback(monkeypatch):
    """Force the pure-Python path.

    ``circuits`` caches its backend probe in a module-level dict, but the probe
    result is only consulted through ``torx_available()``, which re-reads the
    env var each call — so setting it is enough and no cache reset is needed.
    """
    monkeypatch.setenv("TORX_FORCE_FALLBACK", "1")
    assert not torx_available()
    yield


# --------------------------------------------------------------------------
# the structural veto
# --------------------------------------------------------------------------


@settings(max_examples=60, deadline=None)
@given(
    p=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    violated=st.booleans(),
    authorized=st.booleans(),
)
def test_veto_is_structural_for_any_estimate(p, violated, authorized):
    """No viability estimate can survive a boundary or authorization veto."""
    out = evaluate_decision(
        DecisionInputs(
            p_bridge_viability=p,
            p_boundary_violation=1.0 if violated else 0.0,
            p_action_authorized=1.0 if authorized else 0.0,
        )
    )
    if violated or not authorized:
        assert out.bridge_viability.p == 0.0, (
            f"veto leaked: viability={out.bridge_viability.p} with "
            f"violated={violated} authorized={authorized} estimate={p}"
        )
        assert out.bridge_viability.is_certainly_false()
    else:
        # With no veto the estimate passes through unchanged.
        assert out.bridge_viability.p == pytest.approx(p, abs=KERNEL_TOL)


def test_veto_helper_agrees_with_the_circuit():
    assert veto_is_structural(1.0) is True
    assert veto_is_structural(1.0, violated=False, authorized=False) is True
    # With no veto and a certain estimate the bridge stays viable, so the helper
    # reports False — it answers "did the veto fire?", not "is this safe?".
    assert veto_is_structural(1.0, violated=False, authorized=True) is False


def test_a_vetoed_bridge_reports_blocked_not_unresolved():
    """Blocked and unresolved are different outcomes and must stay distinct.

    Blocked is a sovereign refusal; unresolved is "nothing was forbidden but
    nothing helped". Collapsing them would let a refusal read as a mere failure
    to find an option.
    """
    blocked = evaluate_decision(DecisionInputs(0.9, 1.0, 1.0))
    assert blocked.resolution_status.argmax == "blocked"

    unresolved = evaluate_decision(DecisionInputs(0.0, 0.0, 1.0))
    assert unresolved.resolution_status.argmax == "unresolved"
    assert unresolved.resolution_status.prob("blocked") == 0.0


# --------------------------------------------------------------------------
# kernel and fallback agree
# --------------------------------------------------------------------------


GRID = [0.0, 0.15, 0.5, 0.85, 1.0]


@pytest.mark.parametrize("p_v", GRID)
@pytest.mark.parametrize("p_b", GRID)
def test_kernel_matches_the_exact_fallback(p_v, p_b):
    for p_a in (0.0, 0.4, 1.0):
        inputs = DecisionInputs(p_v, p_b, p_a)
        kernel = evaluate_decision(inputs)
        exact = _decision_fallback(inputs)
        assert kernel.bridge_viability.p == pytest.approx(
            exact.bridge_viability.p, abs=KERNEL_TOL
        )
        assert kernel.boundary_violation.p == pytest.approx(
            exact.boundary_violation.p, abs=KERNEL_TOL
        )
        assert kernel.action_authorized.p == pytest.approx(
            exact.action_authorized.p, abs=KERNEL_TOL
        )


def test_forced_fallback_produces_the_same_decision(forced_fallback):
    out = evaluate_decision(DecisionInputs(0.6, 0.3, 0.8))
    assert out.backend == "torx-fallback/python-exact"
    # 0.6 * (1 - 0.3) * 0.8
    assert out.bridge_viability.p == pytest.approx(0.336, abs=KERNEL_TOL)


def test_fallback_still_enforces_the_veto(forced_fallback):
    out = evaluate_decision(DecisionInputs(1.0, 1.0, 1.0))
    assert out.bridge_viability.p == 0.0


def test_decision_inputs_reject_non_probabilities():
    for bad in (-0.1, 1.5, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="probability"):
            DecisionInputs(bad, 0.0, 1.0)


# --------------------------------------------------------------------------
# categorical (pdit) construction
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "scores",
    [
        [0.1, 3.0, 0.5, 0.0, 0.0, 0.0, 0.2],
        [0.0] * 7,
        [5.0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 9.0],
        [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0],
    ],
)
def test_categorical_reproduces_the_softmax_target(scores):
    kernel = evaluate_categorical(scores, TENSION_CLASSES, label="t")
    exact = _categorical_fallback(scores, TENSION_CLASSES, "t")
    for outcome in TENSION_CLASSES:
        assert kernel.prob(outcome) == pytest.approx(
            exact.prob(outcome), abs=KERNEL_TOL
        )
    assert math.fsum(kernel.probs) == pytest.approx(1.0, abs=1e-9)


def test_degenerate_scores_give_a_uniform_distribution():
    pdit = tension_class_pdit([0.0] * 7)
    for p in pdit.probs:
        assert p == pytest.approx(1.0 / 7, abs=1e-6)
    assert pdit.margin == pytest.approx(0.0, abs=1e-6)


def test_bridge_candidate_uses_the_card_strategy_order():
    pdit = bridge_candidate_pdit([2.0, 1.0, 0.5, 0.2, 0.1, 0.0, -1.0])
    assert pdit.outcomes == BRIDGE_STRATEGIES
    assert pdit.argmax == "semantic-translation"


def test_categorical_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="scores for"):
        evaluate_categorical([1.0, 2.0], TENSION_CLASSES)


# --------------------------------------------------------------------------
# backend reporting
# --------------------------------------------------------------------------


def test_backend_status_reports_a_resolvable_engine_label():
    """The node's HUD validator rejects a non-numeric engine version.

    ``webgpu-self-observer-state.py`` allows the ``thrml`` and ``torx`` engine
    families and fails closed on an unresolvable version, so the label must
    carry a real one.
    """
    status = backend_status()
    label = status["engine"]
    if status["torx_available"]:
        m = re.fullmatch(r"torx-(\d+(?:\.\d+)*)/jax-(\w+)", label)
        assert m, f"engine label not in torx-<version>/jax-<backend> form: {label}"
        assert all(part.isdigit() for part in m.group(1).split("."))
    else:
        assert label == "torx-fallback/python-exact"


def test_forced_fallback_is_reported_honestly(forced_fallback):
    status = backend_status()
    assert status["forced_fallback"] is True
    assert status["torx_available"] is False
    assert status["engine"] == "torx-fallback/python-exact"


# --------------------------------------------------------------------------
# numeric helpers
# --------------------------------------------------------------------------


@pytest.mark.parametrize("p", [0.0, 1e-9, 0.25, 0.5, 0.75, 1 - 1e-9, 1.0])
def test_logit_sigmoid_round_trip(p):
    """Saturating at a finite cap keeps NaN out of the gate matrices."""
    theta = logit(p)
    assert math.isfinite(theta)
    if 0.0 < p < 1.0:
        assert sigmoid(theta) == pytest.approx(p, abs=1e-6)
    assert sigmoid(logit(0.0)) == pytest.approx(0.0, abs=1e-15)
    assert sigmoid(logit(1.0)) == pytest.approx(1.0, abs=1e-15)


def test_logit_clamps_out_of_range_input():
    assert logit(-5.0) == logit(0.0)
    assert logit(7.0) == logit(1.0)
