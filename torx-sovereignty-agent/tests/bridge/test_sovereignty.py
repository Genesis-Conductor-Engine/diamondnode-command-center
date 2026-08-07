"""Task 5 part 2 — the contextual sovereignty guard.

``contextual_sovereignty`` fixes a 5-level precedence ladder and default-deny
authorization. These tests pin: hard-boundary blocks, the cannot-be-overridden
levels, the level-3 cap, and the default-deny rule.
"""

from __future__ import annotations

import pytest

from src.bridge.sovereignty import (
    CLASS_GROUP_RULE,
    CLASS_HARD_BOUNDARY,
    CLASS_INFERRED,
    CLASS_TENANT_POLICY,
    CLASS_TOOL_AUTHORIZATION,
    EFFECT_ADVISORY,
    EFFECT_CANNOT_OVERRIDE,
    EFFECT_CAPS_AUTONOMOUS,
    EFFECT_GOVERNS_ORDINARY,
    SovereigntyGuard,
    evaluate_proposal,
)
from src.model_card import load_model_card


@pytest.fixture(scope="module")
def card():
    return load_model_card()


@pytest.fixture
def guard(card):
    return SovereigntyGuard(card)


def test_hard_boundary_category_blocks_regardless_of_grant(guard):
    proposal = {
        "strategy_id": "semantic-translation",
        "effect_class": "consent",
    }
    verdict = guard.evaluate(proposal, grants={"bridge": {"granted": True}})
    assert verdict.boundary_violation is True
    assert verdict.authorized is False
    assert verdict.precedence_class == CLASS_HARD_BOUNDARY
    assert verdict.precedence_level == 1


def test_any_hard_boundary_key_phrasing_flags(guard):
    for key, value in [
        ("boundary_category", "safety"),
        ("targets", ["privacy"]),
        ("category", "data-residency"),
    ]:
        verdict = guard.evaluate({key: value}, grants={"bridge": True})
        assert verdict.boundary_violation is True, key
        assert verdict.precedence_level == 1


def test_cannot_be_overridden_levels_are_never_authorized(guard):
    # A proposal reaching a level-2 class (legal-safety-tenant-policy) can be
    # refused even though the class is not itself a hard boundary category.
    proposal = {"strategy_id": "constraint-reconciliation", "effect_class": CLASS_TENANT_POLICY}
    verdict = guard.evaluate(proposal, grants={"bridge": {"granted": True, "level": 3}})
    assert verdict.boundary_violation is False
    assert verdict.authorized is False
    assert verdict.precedence_level == 2
    assert "cannot-be-overridden" in verdict.reason


def test_default_deny_without_an_explicit_grant(guard):
    proposal = {"strategy_id": "priority-sequencing", "effect_class": CLASS_INFERRED}
    verdict = guard.evaluate(proposal)
    assert verdict.authorized is False
    assert verdict.precedence_level == 5
    assert "default-deny" in verdict.reason


def test_explicit_grant_authorizes_an_inferred_proposal(guard):
    proposal = {"strategy_id": "semantic-translation", "effect_class": CLASS_INFERRED}
    verdict = guard.evaluate(proposal, grants={"bridge": {"granted": True}})
    assert verdict.boundary_violation is False
    assert verdict.authorized is True
    assert verdict.precedence_class == CLASS_INFERRED
    assert verdict.precedence_level == 5


def test_revoked_grant_is_denied(guard):
    proposal = {"strategy_id": "semantic-translation", "effect_class": CLASS_INFERRED}
    verdict = guard.evaluate(
        proposal, grants={"bridge": {"granted": True, "revoked": True}}
    )
    assert verdict.authorized is False


def test_scope_from_grant_scopes_is_required(guard):
    proposal = {
        "strategy_id": "priority-sequencing",
        "effect_class": CLASS_GROUP_RULE,
        "grant_scopes": ["reorder-goals"],
    }
    verdict = guard.evaluate(proposal, grants={"bridge": True})
    assert verdict.authorized is False, "the grant covers a different scope"

    verdict = guard.evaluate(proposal, grants={"reorder-goals": True})
    assert verdict.authorized is True


def test_level_order_matches_the_card(guard):
    precedence = guard.sovereignty.precedence
    levels = [p.level for p in precedence]
    assert levels == [1, 2, 3, 4, 5], "the ladder is dense 1..5"
    assert precedence[0].cls == CLASS_HARD_BOUNDARY
    assert precedence[0].effect == EFFECT_CANNOT_OVERRIDE
    assert precedence[2].cls == CLASS_TOOL_AUTHORIZATION
    assert precedence[2].effect == EFFECT_CAPS_AUTONOMOUS
    assert precedence[3].cls == CLASS_GROUP_RULE
    assert precedence[3].effect == EFFECT_GOVERNS_ORDINARY
    assert precedence[4].cls == CLASS_INFERRED
    assert precedence[4].effect == EFFECT_ADVISORY


def test_module_level_evaluate_proposal_wrapper(card):
    verdict = evaluate_proposal(
        {"strategy_id": "parallel-fork", "effect_class": CLASS_INFERRED},
        grants={"bridge": True},
        card=card,
    )
    assert verdict.authorized is True
