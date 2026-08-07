import pytest
from datetime import datetime, timezone, timedelta
from src.intent.tension import MemberObservation, MemberGradient, ClassHysteresis, TensionSnapshot, EvidenceInsufficient, PROVISIONAL_EVIDENCE_FACTOR, SINGLE_EVENT_HIGH_IMPACT_FORBIDDEN, INSUFFICIENT_DISTINCT_EVENTS
from src.torx_layer.state import PMode, PDit, TENSION_DIMENSIONS, TENSION_CLASSES

def make_pmode(vals):
    # simple diagonal pmode with given values and unit variance
    return PMode.from_diagonal(TENSION_DIMENSIONS, vals, 0.1)

def test_member_observation_validation():
    obs = MemberObservation(
        member_id="m1",
        intent=make_pmode([0.0]*6),
        event_ids=("e1",),
        observed_at=datetime.now(timezone.utc),
    )
    assert obs.distinct_events == 1
    # high impact with insufficient events should raise
    obs_hi = MemberObservation(
        member_id="m2",
        intent=make_pmode([0.0]*6),
        event_ids=("e1",),
        observed_at=datetime.now(timezone.utc),
        high_impact=True,
    )
    with pytest.raises(EvidenceInsufficient):
        # qualification will be called inside estimate, but we test directly
        obs_hi.qualify(minimum_distinct_events=2)
    # non-high-impact under-evidenced returns False
    assert not obs.qualify(minimum_distinct_events=2)

def test_class_hysteresis_transitions():
    hyst = ClassHysteresis.from_card(None)  # defaults minimum 3
    # first observation becomes committed
    hyst1 = hyst.observe("aligned")
    assert hyst1.committed == "aligned"
    # second different observation sets candidate
    hyst2 = hyst1.observe("semantic-gap")
    assert hyst2.candidate == "semantic-gap"
    # repeat to reach streak
    hyst3 = hyst2.observe("semantic-gap")
    hyst4 = hyst3.observe("semantic-gap")
    assert hyst4.committed == "semantic-gap"
    assert hyst4.candidate is None

def test_tension_snapshot_contract_and_id():
    # create simple member gradient
    grad = MemberGradient(
        member_id="m1",
        gradient=make_pmode([0.2]*6),
        tension_class=PDit.certain(TENSION_CLASSES, "aligned"),
        scalar=0.5,
        confidence=0.8,
        participation=1.0,
        observed_at=datetime.now(timezone.utc),
        event_ids=("e1",),
        qualified=True,
    )
    snap = TensionSnapshot(
        tenant_id="t",
        group_id="g",
        group_intent_revision=1,
        member_gradients={"m1": grad},
        group_tension=0.3,
        tension_class="aligned",
        observed_tension_class="aligned",
        tension_class_distribution=PDit.certain(TENSION_CLASSES, "aligned"),
        confidence=0.9,
        hysteresis=ClassHysteresis.from_card(None),
        penalties={"a": 0.1},
        observed_at=datetime.now(timezone.utc),
    )
    doc = snap.contract_document()
    assert doc["group_id"] == "g"
    assert "m1" in doc["member_gradients"]
    # deterministic id should be reproducible
    id1 = snap.snapshot_id
    id2 = snap.snapshot_id
    assert id1 == id2
