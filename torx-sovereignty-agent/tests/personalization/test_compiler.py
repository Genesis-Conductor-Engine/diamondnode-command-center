"""Task 3b — the effective-agent compiler.

The centrepiece is :func:`test_no_layer_can_widen_the_capability_grant`, a
hypothesis property over arbitrary profiles, overlays and session contexts. The
card's ``permission-non-expansion`` acceptance is a universal claim ("*no*
inferred preference or group intent can widen the MCP capability grant"), and a
handful of examples cannot support a universal claim — only a property test
over generated inputs can, and only if the generator is allowed to produce
exactly the inputs an attacker would.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.model_card import cached_model_card, parse_model_card
from src.model_card.loader import DEFAULT_CARD_PATH
from src.personalization.compiler import (
    HARD_BOUNDARY_KEY,
    MANDATORY_MODULES,
    MODULE_SCORES_KEY,
    REVOKED_CAPABILITIES_KEY,
    STAGE_AUTHORIZATION_FILTER,
    AgentLayer,
    BaseContract,
    CompileContext,
    CompileRejected,
    EffectiveAgentManifest,
    HardBoundaryConflict,
    compile_effective_agent,
    deep_merge,
    layers_from_profile,
    select_modules,
)
from src.personalization.evidence import (
    EvidenceEvent,
    EvidenceSource,
    FeatureObservation,
)
from src.personalization.inference import UserProfile, apply_delta, infer_profile_delta

CARD = cached_model_card()
PERMISSIONS = tuple(sorted(CARD.mcp.permissions))
BASE_CAPABILITIES = frozenset(
    {"profile:read", "profile:write", "bridge:propose", "agent:compile"}
)
TENANT = "00000000-0000-0000-0000-0000000000aa"
USER = "00000000-0000-0000-0000-000000000001"
T0 = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


def base_contract(**kwargs) -> BaseContract:
    params = {
        "contract_id": CARD.base_agent.contract_id,
        "capabilities": BASE_CAPABILITIES,
        "settings": {
            "style": {"verbosity": 0.5, "technicality": 0.5},
            "tools": ["read", "write"],
            MODULE_SCORES_KEY: {
                "communication-adapter": 0.9,
                "workflow-sequencer": 0.3,
                "audit-explainer": 0.7,
            },
        },
    }
    params.update(kwargs)
    return BaseContract(**params)


@pytest.fixture(scope="module")
def card_text() -> str:
    return DEFAULT_CARD_PATH.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# permission non-expansion — the property
# --------------------------------------------------------------------------

# Capability names deliberately include ones outside the base grant *and* ones
# outside the card's permission universe: a generator that could only produce
# already-granted capabilities would never exercise the property it claims.
_capability = st.sampled_from(
    list(PERMISSIONS) + ["fabricated:capability", "profile:superuser", "*"]
)
_capability_set = st.frozensets(_capability, max_size=6)

# Keys avoid the compiler's reserved names (``hard_boundaries``,
# ``module_scores``, ...) so the generated documents exercise the merge rather
# than the boundary and selection rules, which have their own tests.
_key = st.text(alphabet="abcdefg_", min_size=1, max_size=6)
_scalar = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-1000, max_value=1000),
    st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
    st.text(alphabet="abcxyz ", max_size=8),
)
_document = st.dictionaries(
    _key,
    st.recursive(
        _scalar,
        lambda children: st.one_of(
            st.lists(children, max_size=3), st.dictionaries(_key, children, max_size=3)
        ),
        max_leaves=6,
    ),
    max_size=4,
)


def _layer(name: str):
    return st.builds(
        AgentLayer,
        name=st.just(name),
        data=_document,
        capabilities=st.one_of(st.none(), _capability_set),
        modules=st.none(),
    )


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    tenant_policy=st.one_of(st.none(), _layer("tenant_policy")),
    profile_layer=st.one_of(st.none(), _layer("automatic_user_profile")),
    overlay=st.one_of(st.none(), _layer("group_relative_overlay")),
    session=st.one_of(st.none(), _layer("session_context")),
    requested=st.one_of(st.none(), _capability_set),
    revoked=st.frozensets(_capability, max_size=3),
)
def test_no_layer_can_widen_the_capability_grant(
    tenant_policy, profile_layer, overlay, session, requested, revoked
):
    """``permission-non-expansion``, for arbitrary layers.

    Whatever an inferred profile, a group overlay, a session context or an
    explicit request asks for, the compiled grant is a subset of the base
    contract's — and of the card's permission universe.
    """
    boundaries = (
        AgentLayer(
            "hard_user_boundaries",
            {HARD_BOUNDARY_KEY: {REVOKED_CAPABILITIES_KEY: sorted(revoked)}},
        )
        if revoked
        else None
    )
    manifest = compile_effective_agent(
        CompileContext(
            base=base_contract(),
            tenant_policy=tenant_policy,
            hard_user_boundaries=boundaries,
            automatic_user_profile=profile_layer,
            group_relative_overlay=overlay,
            session_context=session,
            requested_capabilities=requested,
        )
    )
    granted = manifest.granted_capabilities
    assert granted <= BASE_CAPABILITIES
    assert granted <= CARD.mcp.permissions
    assert granted.isdisjoint(revoked)
    if requested is not None:
        assert granted <= requested


# Fewer examples than the pure-merge property above: every generated feature
# runs a real TORX ``intent_update``, so this one buys its coverage in kernel
# calls rather than in cheap set operations.
@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    values=st.dictionaries(
        st.sampled_from(["verbosity", "technicality", "challenge_level"]),
        st.floats(min_value=0.0, max_value=1.0),
        min_size=1,
        max_size=2,
    ),
    revoked=st.frozensets(_capability, max_size=3),
)
def test_an_inferred_profile_can_only_narrow(values, revoked):
    """The same property, driven through a real inferred profile."""
    profile = UserProfile(
        tenant_id=TENANT, user_id=USER, revoked_capabilities=revoked
    )
    for index, (name, value) in enumerate(values.items()):
        event = EvidenceEvent(
            tenant_id=TENANT,
            user_id=USER,
            event_id=f"p{index}",
            observed_at=T0,
            kind="observed-behavior",
            observation=FeatureObservation("communication", name, value),
            source=EvidenceSource("relay", "tool-result", 1.0),
            source_confidence=0.9,
        )
        profile = apply_delta(profile, infer_profile_delta(event, profile, now=T0))
    boundaries, preferences = layers_from_profile(profile)
    manifest = compile_effective_agent(
        CompileContext(
            base=base_contract(),
            hard_user_boundaries=boundaries,
            automatic_user_profile=preferences,
        )
    )
    assert manifest.granted_capabilities <= BASE_CAPABILITIES
    assert manifest.granted_capabilities.isdisjoint(revoked)


# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------


def _reference_context() -> CompileContext:
    return CompileContext(
        base=base_contract(),
        tenant_policy=AgentLayer(
            "tenant_policy",
            {"style": {"verbosity": 0.2}, "tools": ["audit", "read"]},
            capabilities=frozenset({"profile:read", "bridge:propose", "agent:compile"}),
        ),
        automatic_user_profile=AgentLayer(
            "automatic_user_profile", {"preferences": {"communication": {"verbosity": 0.8}}}
        ),
        session_context=AgentLayer(
            "session_context", {MODULE_SCORES_KEY: {"workflow-sequencer": 0.8}}
        ),
        requested_capabilities=frozenset({"profile:read", "bridge:apply"}),
    )


def test_same_inputs_produce_an_identical_manifest_and_hash():
    first = compile_effective_agent(_reference_context())
    second = compile_effective_agent(_reference_context())
    assert first.manifest_hash == second.manifest_hash
    assert first == second
    assert first.canonical_json() == second.canonical_json()


def test_key_order_in_the_inputs_does_not_change_the_hash():
    """Canonical JSON is lexicographic, so insertion order cannot leak in."""
    a = compile_effective_agent(
        CompileContext(
            base=base_contract(),
            session_context=AgentLayer("session_context", {"a": 1, "b": 2}),
        )
    )
    b = compile_effective_agent(
        CompileContext(
            base=base_contract(),
            session_context=AgentLayer("session_context", {"b": 2, "a": 1}),
        )
    )
    assert a.manifest_hash == b.manifest_hash


def test_manifest_is_hashable_and_keyed_by_content():
    first = compile_effective_agent(_reference_context())
    same = compile_effective_agent(_reference_context())
    other = compile_effective_agent(
        CompileContext(base=base_contract(), requested_capabilities=frozenset({"profile:read"}))
    )
    index = {first: "a"}
    assert index[same] == "a"
    assert other != first
    assert len({first, same, other}) == 2


def test_a_manifest_cannot_disagree_with_its_own_hash():
    manifest = compile_effective_agent(_reference_context())
    with pytest.raises(ValueError, match="does not match its content"):
        EffectiveAgentManifest(content=manifest.content, manifest_hash="0" * 64)
    tampered = {**manifest.content, "granted_capabilities": sorted(CARD.mcp.permissions)}
    with pytest.raises(ValueError, match="does not match its content"):
        EffectiveAgentManifest(content=tampered, manifest_hash=manifest.manifest_hash)


def test_manifest_hash_is_outside_the_hashed_content():
    """Same exclusion rule the VDF applies to ``vdf_proof``."""
    manifest = compile_effective_agent(_reference_context())
    assert "manifest_hash" not in manifest.content
    assert manifest.to_json()["manifest_hash"] == manifest.manifest_hash


# --------------------------------------------------------------------------
# deterministic merge rules
# --------------------------------------------------------------------------


def test_scalar_conflict_is_won_by_the_higher_precedence_layer():
    manifest = compile_effective_agent(_reference_context())
    assert manifest.setting("style", "verbosity") == 0.2  # tenant over base
    assert manifest.setting("style", "technicality") == 0.5  # untouched by tenant


def test_list_conflict_is_a_stable_deduplicating_union():
    manifest = compile_effective_agent(_reference_context())
    # base ["read", "write"] + tenant ["audit", "read"] -> first occurrences kept
    assert manifest.setting("tools") == ["read", "write", "audit"]


def test_deep_merge_deduplicates_unhashable_list_entries():
    owners: dict = {}
    merged = deep_merge(
        {"rules": [{"a": 1, "b": 2}]},
        {"rules": [{"b": 2, "a": 1}, {"c": 3}]},
        stage="session_context",
        owners=owners,
    )
    assert merged["rules"] == [{"a": 1, "b": 2}, {"c": 3}]


def test_higher_precedence_layer_replaces_a_scalar_with_a_mapping():
    owners: dict = {}
    merged = deep_merge(
        {"style": 0.5}, {"style": {"verbosity": 0.9}}, stage="session_context", owners=owners
    )
    assert merged == {"style": {"verbosity": 0.9}}


# --------------------------------------------------------------------------
# hard boundaries
# --------------------------------------------------------------------------


def test_hard_boundary_conflict_rejects_the_compile(card_text):
    assert (
        parse_model_card(card_text).base_agent.deterministic_merge.hard_boundary_conflict
        == "reject-compile"
    )
    context = CompileContext(
        base=base_contract(
            settings={HARD_BOUNDARY_KEY: {"permission_expansion": False}}
        ),
        tenant_policy=AgentLayer(
            "tenant_policy", {HARD_BOUNDARY_KEY: {"permission_expansion": True}}
        ),
    )
    with pytest.raises(HardBoundaryConflict) as exc:
        compile_effective_agent(context)
    assert exc.value.code == "BOUNDARY_VIOLATION"
    assert "immutable_base" in str(exc.value)


def test_an_agreeing_restatement_of_a_boundary_is_not_a_conflict():
    context = CompileContext(
        base=base_contract(settings={HARD_BOUNDARY_KEY: {"preserve_dissent": True}}),
        hard_user_boundaries=AgentLayer(
            "hard_user_boundaries", {HARD_BOUNDARY_KEY: {"preserve_dissent": True}}
        ),
    )
    manifest = compile_effective_agent(context)
    assert manifest.hard_boundaries["preserve_dissent"] is True


@pytest.mark.parametrize(
    "stage", ["automatic_user_profile", "group_relative_overlay", "session_context"]
)
def test_layers_below_the_boundary_stages_cannot_declare_one(stage):
    """A group or a session cannot mint a hard boundary for an individual."""
    context = CompileContext(
        base=base_contract(),
        **{stage: AgentLayer(stage, {HARD_BOUNDARY_KEY: {"preserve_dissent": False}})},
    )
    with pytest.raises(HardBoundaryConflict, match="precedence levels 1 and 2"):
        compile_effective_agent(context)


def test_revocations_from_two_layers_union_rather_than_conflict():
    context = CompileContext(
        base=base_contract(
            settings={HARD_BOUNDARY_KEY: {REVOKED_CAPABILITIES_KEY: ["bridge:propose"]}}
        ),
        hard_user_boundaries=AgentLayer(
            "hard_user_boundaries",
            {HARD_BOUNDARY_KEY: {REVOKED_CAPABILITIES_KEY: ["profile:write"]}},
        ),
    )
    manifest = compile_effective_agent(context)
    assert manifest.granted_capabilities == frozenset({"profile:read", "agent:compile"})
    assert "hard_user_boundaries.revoked_capabilities" in manifest.content["narrowed_by"]


# --------------------------------------------------------------------------
# compile order
# --------------------------------------------------------------------------


def test_the_compile_order_is_the_cards(card_text):
    card = parse_model_card(card_text)
    assert card.base_agent.compile_order == (
        "immutable_base",
        "tenant_policy",
        "hard_user_boundaries",
        "automatic_user_profile",
        "group_relative_overlay",
        "session_context",
        "module_selection",
        "authorization_filter",
    )
    manifest = compile_effective_agent(_reference_context())
    assert manifest.content["compile_order"] == list(card.base_agent.compile_order)
    applied = manifest.content["layers_applied"]
    assert applied[-1] == STAGE_AUTHORIZATION_FILTER
    assert applied.index("module_selection") < applied.index(STAGE_AUTHORIZATION_FILTER)
    # Stages with no layer supplied are simply absent, in card order otherwise.
    assert applied == [s for s in card.base_agent.compile_order if s in applied]


def test_an_authorization_filter_that_is_not_last_is_rejected(card_text):
    mutated = parse_model_card(
        card_text.replace(
            '    - "authorization_filter"',
            '    - "authorization_filter"\n    - "module_selection"',
            1,
        )
    )
    context = CompileContext(base=base_contract(), card=mutated)
    with pytest.raises(CompileRejected, match="mandatory final stage"):
        compile_effective_agent(context)


def test_an_unknown_compile_stage_is_refused_not_skipped(card_text):
    mutated = parse_model_card(
        card_text.replace(
            '    - "session_context"\n', '    - "session_context"\n    - "telemetry_layer"\n', 1
        )
    )
    context = CompileContext(base=base_contract(), card=mutated)
    with pytest.raises(CompileRejected, match="telemetry_layer"):
        compile_effective_agent(context)


def test_a_base_grant_outside_the_cards_permission_universe_is_refused():
    context = CompileContext(
        base=base_contract(capabilities=frozenset({"profile:read", "profile:superuser"}))
    )
    with pytest.raises(CompileRejected, match="profile:superuser"):
        compile_effective_agent(context)


# --------------------------------------------------------------------------
# module selection
# --------------------------------------------------------------------------


def test_module_selection_uses_the_cards_threshold_and_cap():
    spec = CARD.modular_differentiation.module_selection
    assert (spec.activation_threshold, spec.max_active_modules) == (0.60, 12)
    manifest = compile_effective_agent(_reference_context())
    active = manifest.active_modules
    assert "communication-adapter" in active  # 0.9
    assert "audit-explainer" in active  # 0.7
    assert "workflow-sequencer" in active  # 0.3 in base, raised to 0.8 by session
    assert "intent-estimator" not in active  # unscored -> 0.0
    assert len(active) <= spec.max_active_modules


def test_a_score_below_the_threshold_does_not_activate():
    manifest = compile_effective_agent(
        CompileContext(
            base=base_contract(settings={MODULE_SCORES_KEY: {"communication-adapter": 0.59}})
        )
    )
    assert "communication-adapter" not in manifest.active_modules


def test_mandatory_modules_activate_at_any_score():
    manifest = compile_effective_agent(CompileContext(base=base_contract(settings={})))
    assert set(MANDATORY_MODULES) <= set(manifest.active_modules)
    activations = {m["name"]: m for m in manifest.content["active_modules"]}
    assert activations["contextual-sovereignty-guard"]["activation"] == "rule"
    assert activations["contextual-sovereignty-guard"]["score"] == 0.0


def test_the_cap_drops_the_lowest_scoring_modules():
    catalog = [f"module-{i:02d}" for i in range(20)]
    scores = {name: 0.6 + i / 100 for i, name in enumerate(catalog)}
    selected = select_modules(catalog, scores, threshold=0.60, maximum=12)
    assert len(selected) == 12
    assert selected[0].name == "module-19"  # highest score first
    assert [m.name for m in selected] == [f"module-{i:02d}" for i in range(19, 7, -1)]


def test_the_cap_cannot_evict_a_mandatory_module():
    """A never-disable control must not fall off the end of the list."""
    catalog = [*MANDATORY_MODULES, "module-00"]
    scores = {name: 0.9 for name in catalog}
    with pytest.raises(CompileRejected, match="mandatory"):
        select_modules(catalog, scores, threshold=0.6, maximum=1, mandatory=MANDATORY_MODULES)
    selected = select_modules(
        catalog, scores, threshold=0.6, maximum=2, mandatory=MANDATORY_MODULES
    )
    assert [m.name for m in selected] == sorted(MANDATORY_MODULES)


def test_an_out_of_range_module_score_is_refused():
    with pytest.raises(CompileRejected, match="score_range"):
        select_modules(["a"], {"a": 1.7}, threshold=0.6, maximum=12)


def test_an_unavailable_topology_sidecar_is_ruled_out_whatever_it_scores():
    manifest = compile_effective_agent(
        CompileContext(
            base=base_contract(settings={MODULE_SCORES_KEY: {"topology-stability-sidecar": 1.0}}),
            session_context=AgentLayer("session_context", {"topology_sidecar_available": False}),
        )
    )
    assert "topology-stability-sidecar" not in manifest.active_modules


def test_tenant_policy_may_narrow_the_module_catalog():
    manifest = compile_effective_agent(
        CompileContext(
            base=base_contract(),
            tenant_policy=AgentLayer(
                "tenant_policy",
                {},
                modules=frozenset({*MANDATORY_MODULES, "communication-adapter"}),
            ),
        )
    )
    assert "audit-explainer" not in manifest.active_modules
    assert "communication-adapter" in manifest.active_modules


# --------------------------------------------------------------------------
# authorization filter
# --------------------------------------------------------------------------


def test_the_request_can_only_narrow_and_denials_are_recorded():
    manifest = compile_effective_agent(_reference_context())
    assert manifest.granted_capabilities == frozenset({"profile:read"})
    assert manifest.content["denied_capabilities"] == ["bridge:apply"]
    assert "tenant_policy" in manifest.content["narrowed_by"]


def test_a_module_needing_an_ungranted_capability_is_deauthorized():
    manifest = compile_effective_agent(
        CompileContext(
            base=base_contract(
                module_requirements={"communication-adapter": frozenset({"bridge:apply"})}
            )
        )
    )
    assert "communication-adapter" not in manifest.active_modules
    assert manifest.content["deauthorized_modules"] == ["communication-adapter"]


def test_a_mandatory_module_that_cannot_be_authorized_rejects_the_compile():
    with pytest.raises(CompileRejected, match="mandatory"):
        compile_effective_agent(
            CompileContext(
                base=base_contract(
                    module_requirements={
                        "contextual-sovereignty-guard": frozenset({"bridge:apply"})
                    }
                )
            )
        )


# --------------------------------------------------------------------------
# end to end
# --------------------------------------------------------------------------


def test_a_compiled_manifest_surfaces_contested_features():
    """A contested preference must not be presented as settled."""
    profile = UserProfile(tenant_id=TENANT, user_id=USER)

    def event(event_id: str, value: float, confidence: float) -> EvidenceEvent:
        return EvidenceEvent(
            tenant_id=TENANT,
            user_id=USER,
            event_id=event_id,
            observed_at=T0,
            kind="observed-behavior",
            observation=FeatureObservation("communication", "verbosity", value),
            source=EvidenceSource("relay", "tool-result", 1.0),
            source_confidence=confidence,
        )

    profile = apply_delta(profile, infer_profile_delta(event("c1", 0.9, 0.80), profile, now=T0))
    profile = apply_delta(profile, infer_profile_delta(event("c2", 0.1, 0.85), profile, now=T0))
    assert profile.contested_features == ("communication.verbosity",)

    boundaries, preferences = layers_from_profile(profile)
    manifest = compile_effective_agent(
        CompileContext(
            base=base_contract(),
            hard_user_boundaries=boundaries,
            automatic_user_profile=preferences,
        )
    )
    assert manifest.content["contested_features"] == ["communication.verbosity"]
    assert manifest.content["unresolved_contradictions"] == 1
    assert manifest.setting("preferences", "communication", "verbosity") is not None


def test_a_user_boundary_survives_into_the_manifest():
    profile = UserProfile(tenant_id=TENANT, user_id=USER)
    boundary_event = EvidenceEvent(
        tenant_id=TENANT,
        user_id=USER,
        event_id="b1",
        observed_at=T0,
        kind="explicit-boundary",
        observation=FeatureObservation.boolean(
            HARD_BOUNDARY_KEY, "permission_expansion", False
        ),
        source=EvidenceSource("console", "user-declaration", 1.0),
        source_confidence=1.0,
        implied_capability_revocations=frozenset({"profile:write"}),
    )
    profile = apply_delta(profile, infer_profile_delta(boundary_event, profile, now=T0))
    boundaries, preferences = layers_from_profile(profile)
    manifest = compile_effective_agent(
        CompileContext(
            base=base_contract(),
            hard_user_boundaries=boundaries,
            automatic_user_profile=preferences,
        )
    )
    assert manifest.hard_boundaries["permission_expansion"] is False
    assert "profile:write" not in manifest.granted_capabilities
    # And a tenant policy cannot put it back.
    widened = compile_effective_agent(
        CompileContext(
            base=base_contract(),
            tenant_policy=AgentLayer("tenant_policy", {}, capabilities=CARD.mcp.permissions),
            hard_user_boundaries=boundaries,
            automatic_user_profile=preferences,
        )
    )
    assert widened.granted_capabilities <= BASE_CAPABILITIES
    assert "profile:write" not in widened.granted_capabilities
