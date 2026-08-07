"""Task 1 — the card loads, and the controls it declares cannot be edited away."""

from __future__ import annotations

import textwrap

import pytest

from src.model_card import ModelCardError, load_model_card, parse_model_card
from src.model_card.loader import DEFAULT_CARD_PATH


@pytest.fixture(scope="module")
def card():
    return load_model_card()


@pytest.fixture(scope="module")
def card_text() -> str:
    return DEFAULT_CARD_PATH.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# invariant fields
# --------------------------------------------------------------------------


def test_card_loads_from_default_path(card):
    assert card.kind == "GenesisConductorAgentModelCard"
    assert card.metadata.id == "torx-contextual-sovereignty-agent"
    assert card.schema_version == "1.0.0"


def test_invariants_are_all_asserted(card):
    inv = card.invariants
    assert inv.immutable_base_contract
    assert inv.contextual_sovereignty
    assert inv.no_false_consensus
    assert inv.preserve_minority_positions
    assert inv.preserve_inference_uncertainty
    assert inv.append_only_revision_history
    assert inv.reversible_personalization
    assert inv.evidence_provenance_required
    assert inv.mcp_permission_envelope_cannot_self_expand
    assert inv.personalization_may_narrow_permissions
    assert not inv.personalization_may_not_widen_permissions is False


def test_base_contract_is_immutable_at_runtime(card):
    assert card.base_agent.mutation == "forbidden-at-runtime"
    assert card.base_agent.compile_order[0] == "immutable_base"
    assert card.base_agent.compile_order[-1] == "authorization_filter"


def test_precedence_puts_hard_boundaries_first(card):
    cs = card.contextual_sovereignty
    assert cs.level_of("hard-individual-boundary") == 1
    assert cs.level_of("automatically-inferred-preference") == 5
    assert cs.irreducible_conflict.manufactured_consensus == "forbidden"
    assert cs.irreducible_conflict.minority_position_storage == "required"


def test_no_module_may_modify_tool_permissions(card):
    modules = card.modular_differentiation.modules
    assert modules, "module catalog must not be empty"
    assert all(not m.can_modify_tool_permissions for m in modules.values())


def test_torx_declares_the_three_state_types_and_decision_pbits(card):
    assert set(card.torx.state_types) == {"pbit", "pdit", "pmode"}
    graph = card.torx.factor_graph
    assert graph.nodes["bridge_viability"].type == "pbit"
    assert graph.nodes["tension_class"].type == "pdit"
    assert graph.nodes["individual_intent"].type == "pmode"
    # resolution_status is the sink: it reads all three decision pbits
    parents = set(graph.parents_of("resolution_status"))
    assert {"bridge_viability", "boundary_violation", "action_authorized"} <= parents


def test_topology_sidecar_is_off_the_critical_path(card):
    assert card.torx.topology_sidecar.critical_path is False
    assert "topology-stability-sidecar" in card.runtime.critical_path.excluded
    assert card.torx.topology_sidecar.fallback_order == (
        "last-valid-descriptor",
        "neutral-descriptor",
        "non-topological-torx-baseline",
    )


def test_degradation_never_disables_the_protected_controls(card):
    deg = card.runtime.degradation
    assert deg.order[-1] == "retain-base-agent-plus-hard-boundaries"
    assert set(deg.never_disable) == {
        "authorization-filter",
        "hard-boundary-enforcement",
        "dissent-preservation",
    }


def test_confidence_thresholds_are_monotonic(card):
    c = card.personalization.inference_update.confidence
    assert c.provisional_minimum < c.persistent_minimum
    assert c.persistent_minimum < c.consequential_adjustment_minimum


def test_explicit_boundaries_do_not_decay(card):
    assert card.personalization.inference_update.decay.half_life_days.explicit_boundary is None


def test_mcp_defaults_to_deny_and_forbids_self_expansion(card):
    srv = card.mcp.server
    assert srv.authorization.default == "deny"
    assert srv.authorization.self_expansion == "forbidden"
    assert srv.idempotency.required_for_mutations
    assert card.mcp.tool("profile.correct").mutating
    assert not card.mcp.tool("profile.inspect").mutating
    assert "profile:export" in card.mcp.permissions


def test_bridge_strategies_are_ordered_and_end_in_escalation(card):
    ids = [s.id for s in card.bridge_engine.ordered_strategies]
    assert ids == [
        "semantic-translation",
        "constraint-reconciliation",
        "priority-sequencing",
        "perspective-adaptation",
        "pareto-option",
        "parallel-fork",
        "explicit-escalation",
    ]


def test_every_persisted_table_is_tenant_scoped(card):
    for name, table in card.jsonb_persistence.tables.items():
        assert "tenant_id" in table.columns, name
        assert table.primary_key[0] == "tenant_id", name


def test_jsonb_contract_examples_satisfy_their_own_required_keys(card):
    for name, contract in card.jsonb_contracts.items():
        for key in contract.required:
            assert key in contract.example, f"{name}.{key}"


def test_vdf_excludes_its_own_proof_from_the_canonical_input(card):
    assert "vdf_proof" in card.rule30_vdf.canonical_input.excluded_fields
    assert card.rule30_vdf.canonical_input.key_order == "lexicographic"
    assert "not-proof-of-correctness-for-the-underlying-inference" in (
        card.rule30_vdf.non_claims
    )


# --------------------------------------------------------------------------
# rejection behaviour
# --------------------------------------------------------------------------


def test_duplicate_yaml_keys_are_rejected():
    """PyYAML keeps the last value; here that is an error.

    The duplicate silently discards the first ``prohibited_persistence`` list —
    exactly the failure mode that would leave the guard open.
    """
    doc = textwrap.dedent(
        """
        schema_version: "1.0.0"
        kind: GenesisConductorAgentModelCard
        prohibited_persistence:
          - "credentials"
        prohibited_persistence: []
        """
    )
    with pytest.raises(ModelCardError, match="duplicate key"):
        parse_model_card(doc)


def test_unknown_top_level_field_is_rejected(card_text):
    mutated = card_text + '\nunexpected_section:\n  enabled: true\n'
    with pytest.raises(ModelCardError, match="validation"):
        parse_model_card(mutated)


def test_unknown_nested_field_is_rejected(card_text):
    mutated = card_text.replace(
        "invariants:\n  immutable_base_contract: true",
        "invariants:\n  immutable_base_contract: true\n  typo_control: true",
    )
    with pytest.raises(ModelCardError, match="validation"):
        parse_model_card(mutated)


def test_module_claiming_permission_modification_is_rejected(card_text):
    mutated = card_text.replace(
        '    responsibility: "adapt style, density, terminology, and explanation form"\n'
        '      state_scope: "per-user"\n'
        "      can_modify_tool_permissions: false",
        '    responsibility: "adapt style, density, terminology, and explanation form"\n'
        '      state_scope: "per-user"\n'
        "      can_modify_tool_permissions: true",
    )
    if mutated == card_text:  # whitespace drift — patch the first occurrence
        mutated = card_text.replace(
            "can_modify_tool_permissions: false", "can_modify_tool_permissions: true", 1
        )
    with pytest.raises(ModelCardError, match="can_modify_tool_permissions"):
        parse_model_card(mutated)


def test_empty_prohibited_persistence_is_rejected(card_text):
    lines = card_text.splitlines(keepends=True)
    out, skipping = [], False
    for line in lines:
        if line.startswith("  prohibited_persistence:"):
            out.append("  prohibited_persistence: []\n")
            skipping = True
            continue
        if skipping:
            if line.startswith("    - "):
                continue
            skipping = False
        out.append(line)
    with pytest.raises(ModelCardError, match="prohibited_persistence"):
        parse_model_card("".join(out))


def test_topology_sidecar_on_the_critical_path_is_rejected(card_text):
    mutated = card_text.replace(
        "    critical_path: false", "    critical_path: true", 1
    )
    with pytest.raises(ModelCardError, match="critical_path"):
        parse_model_card(mutated)


def test_tenant_disabling_hard_boundaries_is_rejected(card_text):
    mutated = card_text.replace(
        "may_not_disable_hard_boundary_enforcement: true",
        "may_not_disable_hard_boundary_enforcement: false",
    )
    with pytest.raises(ModelCardError, match="hard_boundary_enforcement"):
        parse_model_card(mutated)


def test_non_mapping_document_is_rejected():
    with pytest.raises(ModelCardError, match="mapping at the top level"):
        parse_model_card("- just\n- a\n- list\n")


def test_missing_file_reports_the_path(tmp_path):
    with pytest.raises(ModelCardError, match="cannot read model card"):
        load_model_card(tmp_path / "absent.yaml")
