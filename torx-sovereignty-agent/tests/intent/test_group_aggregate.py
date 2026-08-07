import pytest
from src.intent.group_aggregate import HardConstraint, IrreducibleConflict, hard_constraint_union, GroupMember, GoalStatement, shared_goal_intersection, ranked_goal_union, DecisionPolicy, aggregate_group_intent
from src.torx_layer.state import STORAGE_PRECISION

def test_hard_constraint_validation():
    # valid constraint
    c = HardConstraint(member_id="m1", subject="s", mode="require", event_ids=("e1",))
    assert c.member_id == "m1"
    # invalid mode
    with pytest.raises(ValueError):
        HardConstraint(member_id="m1", subject="s", mode="invalid", event_ids=("e1",))
    # missing event ids
    with pytest.raises(ValueError):
        HardConstraint(member_id="m1", subject="s", event_ids=())

def test_hard_constraint_union_and_conflict():
    c1 = HardConstraint(member_id="a", subject="x", mode="require", event_ids=("e1",))
    c2 = HardConstraint(member_id="b", subject="x", mode="forbid", event_ids=("e2",))
    union, conflicts = hard_constraint_union([c1, c2])
    assert len(union) == 2
    assert isinstance(conflicts[0], IrreducibleConflict)
    assert conflicts[0].subject == "x"
    assert set(conflicts[0].requiring) == {"a"}
    assert set(conflicts[0].forbidding) == {"b"}

def test_shared_goal_intersection_and_ranked_union():
    g1 = GoalStatement(member_id="m1", goal_id="g1", rank=0)
    g2 = GoalStatement(member_id="m1", goal_id="g2", rank=1)
    m1 = GroupMember(member_id="m1", goals=(g1, g2))
    g3 = GoalStatement(member_id="m2", goal_id="g1", rank=0)
    m2 = GroupMember(member_id="m2", goals=(g3,))
    # shared intersection should be only g1
    assert shared_goal_intersection([m1, m2]) == ("g1",)
    # weighted ranking
    weights = {"m1": 2.0, "m2": 1.0}
    ranked = ranked_goal_union([m1, m2], weights)
    # g1 should have highest score
    assert ranked[0][0] == "g1"

def test_decision_policy_parse_and_json():
    dp = DecisionPolicy.parse("consent-for-boundaries-majority-for-ordinary-preferences")
    assert dp.boundary_rule == "consent"
    assert dp.ordinary_rule == "majority"
    json = dp.to_json()
    assert json["boundary_rule"] == "consent"
    assert json["ordinary_rule"] == "majority"

def test_aggregate_group_intent_basic():
    # simple members with no constraints or goals
    m1 = GroupMember(member_id="a", preferences={"p": 0.5})
    m2 = GroupMember(member_id="b", preferences={"p": -0.5})
    rev = aggregate_group_intent([m1, m2], tenant_id="t", group_id="g")
    assert rev.revision == 1
    assert rev.members == (m1, m2)
    # outcomes should contain proposition "p"
    assert any(o.proposition == "p" for o in rev.outcomes)
