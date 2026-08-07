"""Task 5 part 3 — the bridge engine lifecycle.

The engine owns the ``application_rule`` and the rollback triggers. These
tests pin the refusal paths (with their card stable codes), the persisted
lifecycle (propose -> applied -> rolled-back), and the degradation gate.
"""

from __future__ import annotations

import datetime as _dt
import uuid

import pytest

from src.bridge.engine import (
    CODE_AUTHORIZATION_DENIED,
    CODE_BOUNDARY_VIOLATION,
    CODE_BRIDGE_INCREASES_TENSION,
    CODE_INSUFFICIENT_CONFIDENCE,
    CODE_STALE_INTENT_REVISION,
    DEFAULT_MINIMUM_CONFIDENCE,
    BridgeEngine,
    BridgeRejected,
)
from src.model_card import load_model_card
from src.observability.metrics import MetricRegistry
from src.persistence.repositories import (
    BridgeRepository,
    GroupIntentRepository,
    InMemoryBackend,
    TensionRepository,
)
from src.runtime.degradation import CAP_BRIDGE_ENGINE, DegradationController

T0 = _dt.datetime(2026, 8, 6, 12, 0, 0, tzinfo=_dt.timezone.utc)
MEMBER = "00000000-0000-0000-0000-000000000001"

GRANTS = {"bridge": {"granted": True}}


def at(seconds: int) -> _dt.datetime:
    return T0 + _dt.timedelta(seconds=seconds)


@pytest.fixture(scope="module")
def card():
    return load_model_card()


@pytest.fixture
def registry(card) -> MetricRegistry:
    return MetricRegistry(card, use_otel=False)


@pytest.fixture
def repos():
    backend = InMemoryBackend()
    return {
        "bridges": BridgeRepository(backend),
        "intents": GroupIntentRepository(backend),
        "tensions": TensionRepository(backend),
    }


@pytest.fixture
def tenant() -> uuid.UUID:
    return uuid.uuid4()


@pytest.fixture
def group() -> uuid.UUID:
    return uuid.uuid4()


def seed_intent(repos, tenant, group, *, now=at(0)):
    return repos["intents"].append_intent(
        tenant,
        group,
        expected_revision=0,
        aggregate_intent={"goals": {"ship": 0.9}},
        member_intents={str(uuid.uuid4()): {"ship": 0.8}},
        decision_policy={"rule": "weighted-consent"},
        dissent=[],
        vdf_proof_id=uuid.uuid4(),
        now=now,
    )


def record_snapshot(repos, tenant, group, intent, snapshot_id=None, **overrides):
    kwargs = dict(
        group_intent_revision=intent.revision,
        member_gradients={
            MEMBER: {"goal_divergence": 0.2, "semantic_misalignment": 0.5}
        },
        group_tension=0.6,
        tension_class="semantic-gap",
        confidence=0.8,
        topology_descriptor={"betti": [1, 0]},
        vdf_proof_id=uuid.uuid4(),
        observed_at=at(1),
    )
    kwargs.update(overrides)
    return repos["tensions"].record_snapshot(
        tenant, group, snapshot_id or uuid.uuid4(), **kwargs
    )


def make_engine(card, repos, **kwargs):
    return BridgeEngine(
        card=card,
        bridges=repos["bridges"],
        intents=repos["intents"],
        tensions=repos["tensions"],
        **kwargs,
    )


def propose_and_pick(repos, tenant, group, engine, snapshot):
    engine.propose(tenant, group, snapshot.snapshot_id, grants=GRANTS)
    return repos["bridges"].list_for_group(tenant, group)[0]


# --------------------------------------------------------------------------
# propose
# --------------------------------------------------------------------------


def test_propose_returns_ranked_candidates_and_persists_applicable(card, repos, registry, tenant, group):
    intent = seed_intent(repos, tenant, group)
    snapshot = record_snapshot(repos, tenant, group, intent)
    engine = make_engine(card, repos, registry=registry)

    candidates = engine.propose(tenant, group, snapshot.snapshot_id, grants=GRANTS)

    assert len(candidates) == len(engine.ordered_strategies)
    # Applicable (delta<0, authorized, confident) candidates are persisted.
    persisted = repos["bridges"].list_for_group(tenant, group)
    assert len(persisted) > 0
    assert all(a.status == "proposed" for a in persisted)
    # Ranking puts applicable candidates first, then best delta first.
    applicable = [c for c in candidates if c.applicable]
    assert applicable  # with grants the non-escalation strategies apply
    assert candidates[0].applicable
    for strategy in applicable:
        assert strategy.applicable
    best = min(c.simulation.delta for c in applicable)
    assert candidates[0].simulation.delta == best


def test_propose_default_denies_without_grants(card, repos, registry, tenant, group):
    intent = seed_intent(repos, tenant, group)
    snapshot = record_snapshot(repos, tenant, group, intent)
    engine = make_engine(card, repos, registry=registry)

    candidates = engine.propose(tenant, group, snapshot.snapshot_id)

    assert len(candidates) == len(engine.ordered_strategies)
    assert all(not c.applicable for c in candidates)
    assert all(c.refusal_code == CODE_AUTHORIZATION_DENIED for c in candidates)
    assert repos["bridges"].list_for_group(tenant, group) == []


def test_propose_respects_maximum_candidates(card, repos, registry, tenant, group):
    intent = seed_intent(repos, tenant, group)
    snapshot = record_snapshot(repos, tenant, group, intent)
    engine = make_engine(card, repos, registry=registry)

    engine.propose(tenant, group, snapshot.snapshot_id, grants=GRANTS, maximum_candidates=2)
    assert len(repos["bridges"].list_for_group(tenant, group)) == 2


def test_propose_refused_when_snapshot_is_missing(card, repos, registry, tenant, group):
    seed_intent(repos, tenant, group)
    engine = make_engine(card, repos, registry=registry)
    with pytest.raises(BridgeRejected) as excinfo:
        engine.propose(tenant, group, uuid.uuid4())
    assert excinfo.value.code == CODE_INSUFFICIENT_CONFIDENCE


def test_propose_refused_on_stale_intent_revision(card, repos, registry, tenant, group):
    intent = seed_intent(repos, tenant, group)
    snapshot = record_snapshot(repos, tenant, group, intent)
    engine = make_engine(card, repos, registry=registry)
    with pytest.raises(BridgeRejected) as excinfo:
        engine.propose(
            tenant,
            group,
            snapshot.snapshot_id,
            expected_group_intent_revision=intent.revision + 1,
        )
    assert excinfo.value.code == CODE_STALE_INTENT_REVISION


def test_propose_refused_under_bridge_engine_degradation(card, repos, registry, tenant, group):
    intent = seed_intent(repos, tenant, group)
    snapshot = record_snapshot(repos, tenant, group, intent)
    controller = DegradationController(card, stability_dwell=1, min_dwell_seconds=0.0)
    # Walk the ladder to the last rung, which sheds the bridge engine.
    for _ in range(controller.max_level):
        controller.observe(True, source="load-shed")
    assert not controller.is_active(CAP_BRIDGE_ENGINE)

    engine = make_engine(card, repos, registry=registry, degradation=controller)
    with pytest.raises(BridgeRejected) as excinfo:
        engine.propose(tenant, group, snapshot.snapshot_id)
    assert excinfo.value.code == CODE_AUTHORIZATION_DENIED
    assert "shed" in excinfo.value.detail


def test_propose_escalation_ranks_last_and_preserves_tension(card, repos, registry, tenant, group):
    intent = seed_intent(repos, tenant, group)
    snapshot = record_snapshot(repos, tenant, group, intent)
    engine = make_engine(card, repos, registry=registry)

    candidates = engine.propose(tenant, group, snapshot.snapshot_id, grants=GRANTS)
    escalation = next(c for c in candidates if c.simulation.strategy_id == "explicit-escalation")
    # Escalation is exempt from the delta test (surface-and-preserve: it may be
    # *seen* as the terminal fallback), so it applies but never reduces tension
    # and is ranked last.
    assert escalation.applicable
    assert escalation.simulation.delta == 0.0
    assert candidates[-1].simulation.strategy_id == "explicit-escalation"


# --------------------------------------------------------------------------
# apply
# --------------------------------------------------------------------------


def test_apply_re_verifies_and_marks_applied(card, repos, registry, tenant, group):
    intent = seed_intent(repos, tenant, group)
    snapshot = record_snapshot(repos, tenant, group, intent)
    engine = make_engine(card, repos, registry=registry)
    best = propose_and_pick(repos, tenant, group, engine, snapshot)

    applied = engine.apply(tenant, best.bridge_id, grants=GRANTS)
    assert applied.status == "applied"
    assert repos["bridges"].get(tenant, best.bridge_id).status == "applied"


def test_apply_refused_when_not_proposed(card, repos, registry, tenant, group):
    intent = seed_intent(repos, tenant, group)
    snapshot = record_snapshot(repos, tenant, group, intent)
    engine = make_engine(card, repos, registry=registry)
    best = propose_and_pick(repos, tenant, group, engine, snapshot)
    repos["bridges"].mark_applied(tenant, best.bridge_id, applied_at=at(2))

    with pytest.raises(BridgeRejected) as excinfo:
        engine.apply(tenant, best.bridge_id)
    assert excinfo.value.code == CODE_STALE_INTENT_REVISION


def test_apply_refused_when_snapshot_moved_on(card, repos, registry, tenant, group):
    intent = seed_intent(repos, tenant, group)
    snapshot = record_snapshot(repos, tenant, group, intent)
    engine = make_engine(card, repos, registry=registry)
    best = propose_and_pick(repos, tenant, group, engine, snapshot)

    # A newer snapshot lands after the bridge was proposed.
    record_snapshot(
        repos, tenant, group, intent, snapshot_id=uuid.uuid4(), observed_at=at(2)
    )
    with pytest.raises(BridgeRejected) as excinfo:
        engine.apply(tenant, best.bridge_id, expected_snapshot_id=snapshot.snapshot_id)
    assert excinfo.value.code == CODE_STALE_INTENT_REVISION


def test_apply_refused_when_confidence_drops_below_minimum(card, repos, registry, tenant, group):
    intent = seed_intent(repos, tenant, group)
    snapshot = record_snapshot(repos, tenant, group, intent, confidence=0.8)
    # Propose under a permissive engine so the bridge is persisted...
    engine = make_engine(card, repos, registry=registry)
    best = propose_and_pick(repos, tenant, group, engine, snapshot)

    # ...then apply under a stricter minimum: the rule is re-verified at apply
    # time from the persisted simulation document.
    strict = make_engine(card, repos, registry=registry, minimum_confidence=0.99)
    with pytest.raises(BridgeRejected) as excinfo:
        strict.apply(tenant, best.bridge_id, grants=GRANTS)
    assert excinfo.value.code == CODE_INSUFFICIENT_CONFIDENCE


def test_apply_refused_without_grants(card, repos, registry, tenant, group):
    intent = seed_intent(repos, tenant, group)
    snapshot = record_snapshot(repos, tenant, group, intent)
    engine = make_engine(card, repos, registry=registry)
    best = propose_and_pick(repos, tenant, group, engine, snapshot)

    # Re-verified at apply time: default-deny without a grant still blocks.
    with pytest.raises(BridgeRejected) as excinfo:
        engine.apply(tenant, best.bridge_id)
    assert excinfo.value.code == CODE_AUTHORIZATION_DENIED
    assert "default-deny" in excinfo.value.detail


# --------------------------------------------------------------------------
# rollback
# --------------------------------------------------------------------------


def test_rollback_requires_a_trigger(card, repos, registry, tenant, group):
    intent = seed_intent(repos, tenant, group)
    snapshot = record_snapshot(repos, tenant, group, intent)
    engine = make_engine(card, repos, registry=registry)
    best = propose_and_pick(repos, tenant, group, engine, snapshot)
    engine.apply(tenant, best.bridge_id, grants=GRANTS)

    with pytest.raises(BridgeRejected) as excinfo:
        engine.rollback(tenant, best.bridge_id, reason="testing")
    assert excinfo.value.code == CODE_STALE_INTENT_REVISION
    assert "no rollback trigger" in excinfo.value.detail


def test_rollback_fires_on_observed_tension_spike(card, repos, registry, tenant, group):
    intent = seed_intent(repos, tenant, group)
    snapshot = record_snapshot(repos, tenant, group, intent)
    engine = make_engine(card, repos, registry=registry)
    best = propose_and_pick(repos, tenant, group, engine, snapshot)
    engine.apply(tenant, best.bridge_id, grants=GRANTS)

    # A current snapshot well above the predicted_after trips the hysteresis.
    current = record_snapshot(
        repos, tenant, group, intent, snapshot_id=uuid.uuid4(),
        group_tension=0.99, observed_at=at(2),
    )
    undo = engine.rollback(
        tenant, best.bridge_id, reason="tension spiked", current_snapshot=current
    )
    assert undo.status == "applied"
    assert undo.rollback_of == best.bridge_id
    assert repos["bridges"].get(tenant, best.bridge_id).status == "rolled-back"


def test_rollback_fires_on_new_hard_boundary(card, repos, registry, tenant, group):
    intent = seed_intent(repos, tenant, group)
    snapshot = record_snapshot(repos, tenant, group, intent)
    engine = make_engine(card, repos, registry=registry)
    best = propose_and_pick(repos, tenant, group, engine, snapshot)
    engine.apply(tenant, best.bridge_id, grants=GRANTS)

    current = record_snapshot(
        repos, tenant, group, intent, snapshot_id=uuid.uuid4(),
        member_gradients={MEMBER: {"boundary_violation": True}}, observed_at=at(2),
    )
    triggers = engine.check_rollback_triggers(
        tenant, best.bridge_id, current_snapshot=current
    )
    assert any("hard boundary" in t["condition"] for t in triggers)

    undo = engine.rollback(
        tenant, best.bridge_id, reason="boundary", current_snapshot=current
    )
    assert undo.rollback_of == best.bridge_id


def test_rollback_fires_on_revoked_authorization(card, repos, registry, tenant, group):
    # The engine's simulated proposals never carry authorization flags, so the
    # revoked-authorization trigger is tested by persisting a proposal that
    # does (as a downstream writer would when a grant is revoked).
    intent = seed_intent(repos, tenant, group)
    snapshot = record_snapshot(repos, tenant, group, intent)
    bridge = repos["bridges"].propose(
        tenant,
        uuid.uuid4(),
        group_id=group,
        bridge_type="semantic-translation",
        proposal={
            "strategy_id": "semantic-translation",
            "authorization_revoked": True,
        },
        simulation={
            "predicted_before": 0.6,
            "predicted_after": 0.4,
            "predicted_delta": -0.2,
            "confidence": 0.8,
            "counterfactual": {},
        },
        authorization={"authorized": True, "boundary_violation": False},
        vdf_proof_id=uuid.uuid4(),
        now=at(1),
    )
    repos["bridges"].mark_applied(tenant, bridge.bridge_id, applied_at=at(2))

    engine = make_engine(card, repos, registry=registry)
    triggers = engine.check_rollback_triggers(tenant, bridge.bridge_id)
    assert any(t["condition"] == "authorization is revoked" for t in triggers)

    undo = engine.rollback(tenant, bridge.bridge_id, reason="authorization revoked")
    assert undo.rollback_of == bridge.bridge_id
    assert repos["bridges"].get(tenant, bridge.bridge_id).status == "rolled-back"


# --------------------------------------------------------------------------
# error-code / stable-code surface
# --------------------------------------------------------------------------


def test_rule_error_codes_are_stable_and_reusable(card):
    codes = {
        CODE_BOUNDARY_VIOLATION,
        CODE_INSUFFICIENT_CONFIDENCE,
        CODE_AUTHORIZATION_DENIED,
        CODE_STALE_INTENT_REVISION,
        CODE_BRIDGE_INCREASES_TENSION,
    }
    assert len(codes) == 5, "five distinct stable codes"
    assert DEFAULT_MINIMUM_CONFIDENCE == 0.5
