"""Task 5 part 1 — the bridge simulator contract.

``bridge_engine.simulation`` fixes the *shape* of a simulation before any
ranking happens: every strategy in card order, before/after estimates, a
per-participant estimate, and a preserved counterfactual. These tests pin
that shape and the strategy semantics the card names.
"""

from __future__ import annotations

import uuid

import pytest

from src.bridge.simulator import (
    BRIDGE_STRATEGIES,
    BridgeSimulation,
    BridgeSimulator,
)
from src.model_card import load_model_card

#: Stable per-test identities so a failed assertion prints something readable.
TENANT = uuid.UUID("00000000-0000-0000-0000-0000000000aa")
GROUP = uuid.UUID("00000000-0000-0000-0000-0000000000bb")
SNAP = uuid.UUID("00000000-0000-0000-0000-0000000000cc")


@pytest.fixture(scope="module")
def card():
    return load_model_card()


@pytest.fixture
def simulator(card):
    return BridgeSimulator(card)


def make_snapshot(**overrides):
    """A TensionSnapshot with a divergence structure that responds to bridges."""
    from src.persistence.repositories import TensionSnapshot

    kwargs = dict(
        tenant_id=TENANT,
        group_id=GROUP,
        snapshot_id=SNAP,
        group_intent_revision=1,
        member_gradients={
            "00000000-0000-0000-0000-000000000001": {
                "goal_divergence": 0.2,
                "semantic_misalignment": 0.5,
            },
            "00000000-0000-0000-0000-000000000002": {
                "goal_divergence": 0.8,
                "semantic_misalignment": 0.3,
            },
        },
        group_tension=0.6,
        tension_class="semantic-gap",
        confidence=0.7,
        topology_descriptor={"betti": [1, 0]},
        vdf_proof_id=uuid.UUID("00000000-0000-0000-0000-0000000000dd"),
        observed_at=__import__("datetime").datetime(
            2026, 8, 6, 12, 0, 0, tzinfo=__import__("datetime").timezone.utc
        ),
    )
    kwargs.update(overrides)
    return TensionSnapshot(**kwargs)


def test_simulate_returns_every_strategy_in_card_order(simulator):
    snapshot = make_snapshot()
    results = simulator.simulate(snapshot, None)

    assert [s.strategy_id for s in results] == list(BRIDGE_STRATEGIES)
    assert results[-1].strategy_id == "explicit-escalation"


def test_every_strategy_carries_the_full_simulation_shape(simulator):
    results = simulator.simulate(make_snapshot(), None)
    for result in results:
        assert result.strategy_id in BRIDGE_STRATEGIES
        assert 0.0 <= result.predicted_before <= 1.0
        assert 0.0 <= result.predicted_after <= 1.0
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.proposal, dict)
        # Counterfactual preserved per the card's simulation contract.
        assert "predicted_without_bridge" in result.counterfactual
        assert "observed_trajectory" in result.counterfactual


def test_delta_is_before_minus_after(simulator):
    result = simulator.simulate_strategy("semantic-translation", make_snapshot(), None)
    assert result.delta == pytest.approx(
        result.predicted_after - result.predicted_before
    )


def test_escalation_never_reduces_tension(simulator):
    result = simulator.simulate_strategy("explicit-escalation", make_snapshot(), None)
    # Escalation surfaces irreducible conflict; it must not manufacture a drop.
    assert result.predicted_after == pytest.approx(result.predicted_before)
    assert result.delta == pytest.approx(0.0)
    assert "manufactured" in result.note


def test_simulator_is_deterministic_for_fixed_inputs(simulator):
    snapshot = make_snapshot()
    first = simulator.simulate(snapshot, None)
    second = simulator.simulate(snapshot, None)
    assert [s.predicted_after for s in first] == [s.predicted_after for s in second]
    assert [s.proposal for s in first] == [s.proposal for s in second]


def test_unknown_strategy_is_rejected(simulator):
    with pytest.raises(ValueError, match="unknown bridge strategy"):
        simulator.simulate_strategy("just-agree", make_snapshot(), None)


def test_per_participant_estimates_preserve_member_ids(simulator):
    result = simulator.simulate_strategy("semantic-translation", make_snapshot(), None)
    participant_ids = {p.participant for p in result.participants}
    assert participant_ids == set(make_snapshot().member_gradients)


def test_bridge_simulation_documents_round_trip(simulator):
    result = simulator.simulate_strategy("pareto-option", make_snapshot(), None)
    document = result.as_simulation_document()
    assert document["strategy_id"] == "pareto-option"
    assert "predicted_before" in document and "predicted_after" in document
    # The persisted document is the JSON-facing shape minus the advisory note.
    assert document["predicted_delta"] == pytest.approx(result.delta)
    assert document["counterfactual"] == result.counterfactual
    assert document["participants"] == [p.to_json() for p in result.participants]
