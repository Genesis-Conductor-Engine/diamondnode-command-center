"""Strict Pydantic types for the TORX Contextual Sovereignty agent model card.

The model card is the *contract*, not documentation: thresholds, precedence
levels, degradation order, and the MCP permission map are all read from it at
runtime. A card that parses loosely would let a typo silently disable a control
(``prohibited_persistance: [...]`` would leave the prohibited list empty and the
guard permanently open), so every model here forbids unknown fields and the
loader rejects duplicate YAML keys.

Sections the plan calls out as strict — metadata, invariants, personalization,
TORX, JSONB persistence, MCP, deployment — get fully typed models. Sections that
exist to be read by humans and echoed into audit records (``result_cards``,
``objective``, prose ``validation`` assertions) are typed at their outer shape
and carry their inner mapping verbatim, so an editorial change does not require
a code change to keep the card loadable.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Probability = Annotated[float, Field(ge=0.0, le=1.0)]


class Strict(BaseModel):
    """Base for every card model: unknown keys are an error, not a warning."""

    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------
# metadata
# --------------------------------------------------------------------------


class Owner(Strict):
    name: str
    organization: str
    orcid: str | None = None


class Metadata(Strict):
    id: str
    name: str
    version: str
    status: str
    owner: Owner
    labels: tuple[str, ...] = ()


class Objective(Strict):
    statement: str
    optimization_order: tuple[str, ...]


class ResultCard(Strict):
    title: str
    status: str
    claim: str
    deployment_effect: str | None = None
    invariant_condition: str | None = None
    readiness: str | None = None


# --------------------------------------------------------------------------
# invariants
# --------------------------------------------------------------------------


class SignalBoundary(Strict):
    """Which observation classes may cross into the estimation layer.

    ``rejected`` is the enforceable half: it names the raw-signal classes the
    agent must never ingest, and `boundary_integrity` in the result cards is only
    true while that holds.
    """

    accepted: tuple[str, ...]
    rejected: tuple[str, ...]


class Invariants(Strict):
    immutable_base_contract: bool
    contextual_sovereignty: bool
    no_false_consensus: bool
    preserve_minority_positions: bool
    preserve_inference_uncertainty: bool
    append_only_revision_history: bool
    reversible_personalization: bool
    evidence_provenance_required: bool
    mcp_permission_envelope_cannot_self_expand: bool
    personalization_may_narrow_permissions: bool
    personalization_may_not_widen_permissions: bool
    signal_boundary: SignalBoundary

    @model_validator(mode="after")
    def _permission_direction_is_coherent(self) -> Invariants:
        """A card that permits widening contradicts its own MCP envelope clause.

        Both flags are declared independently in the card, so a hand-edit can set
        ``may_not_widen: false`` while leaving the envelope clause true. That
        combination has no consistent enforcement, so reject it at load.
        """
        if not self.personalization_may_not_widen_permissions and (
            self.mcp_permission_envelope_cannot_self_expand
        ):
            raise ValueError(
                "invariants: personalization_may_not_widen_permissions=false "
                "contradicts mcp_permission_envelope_cannot_self_expand=true"
            )
        return self


# --------------------------------------------------------------------------
# base agent
# --------------------------------------------------------------------------


class DeterministicMerge(Strict):
    strategy: Literal["ordered-deep-merge"]
    scalar_conflict: Literal["higher-precedence-layer-wins"]
    list_conflict: Literal["stable-deduplicating-union"]
    hard_boundary_conflict: Literal["reject-compile"]


class BaseAgent(Strict):
    contract_id: str
    mutation: Literal["forbidden-at-runtime"]
    effective_agent_formula: str
    compile_order: tuple[str, ...]
    deterministic_merge: DeterministicMerge


# --------------------------------------------------------------------------
# personalization
# --------------------------------------------------------------------------


class ProfileLayer(Strict):
    authority: str
    examples: tuple[str, ...] = ()


class ProfileLayers(Strict):
    explicit: ProfileLayer
    inferred_persistent: ProfileLayer
    inferred_ephemeral: ProfileLayer
    group_relative: ProfileLayer


class ConfidenceThresholds(Strict):
    provisional_minimum: Probability
    persistent_minimum: Probability
    consequential_adjustment_minimum: Probability

    @model_validator(mode="after")
    def _monotonic(self) -> ConfidenceThresholds:
        if not (
            self.provisional_minimum
            <= self.persistent_minimum
            <= self.consequential_adjustment_minimum
        ):
            raise ValueError(
                "confidence thresholds must be non-decreasing: "
                "provisional <= persistent <= consequential"
            )
        return self


class HalfLifeDays(Strict):
    ordinary_preference: float | None
    workflow_preference: float | None
    explicit_boundary: float | None = None

    @model_validator(mode="after")
    def _explicit_boundaries_never_decay(self) -> HalfLifeDays:
        """A finite half-life on an explicit boundary would expire consent."""
        if self.explicit_boundary is not None:
            raise ValueError(
                "explicit_boundary half-life must be null: explicit boundaries "
                "are user-owned and may not decay"
            )
        return self


class Decay(Strict):
    enabled: bool
    half_life_days: HalfLifeDays


class ContradictionPolicy(Strict):
    explicit_overrides_inferred: bool
    recent_high_confidence_overrides_old_low_confidence: bool
    unresolved_contradictions_are_retained: bool


class InferenceUpdate(Strict):
    trigger: str
    event_deduplication_key: str
    confidence: ConfidenceThresholds
    decay: Decay
    contradiction_policy: ContradictionPolicy


class AutomaticMutation(Strict):
    enabled: bool
    human_approval_required: bool
    preconditions: tuple[str, ...]
    postconditions: tuple[str, ...]


class UserControls(Strict):
    inspect: bool
    explain: bool
    correct: bool
    pin: bool
    rollback: bool
    export: bool
    delete: bool


class Personalization(Strict):
    mode: Literal["fully-automatic"]
    scope: str
    persistence: str
    profile_layers: ProfileLayers
    feature_domains: dict[str, tuple[str, ...]]
    inference_update: InferenceUpdate
    automatic_mutation: AutomaticMutation
    prohibited_persistence: tuple[str, ...]
    user_controls: UserControls

    @model_validator(mode="after")
    def _prohibited_list_is_non_empty(self) -> Personalization:
        if not self.prohibited_persistence:
            raise ValueError(
                "personalization.prohibited_persistence must not be empty: an "
                "empty list disables the prohibited-inference guard entirely"
            )
        return self


# --------------------------------------------------------------------------
# individual and group intent
# --------------------------------------------------------------------------


class IndividualIntentVector(Strict):
    type: Literal["pmode"]
    dimensions: dict[str, str]


class GroupIntentComponents(Strict):
    shared_goals: str
    hard_constraints: str
    ordinary_preferences: str
    dissent: str
    confidence: str


class RoleWeighting(Strict):
    enabled: bool
    restriction: str


class GroupIntent(Strict):
    aggregation: str
    simple_average_forbidden: bool
    components: GroupIntentComponents
    default_decision_rule: str
    role_weighting: RoleWeighting


class IndividualAndGroupIntent(Strict):
    individual_intent_vector: IndividualIntentVector
    group_intent: GroupIntent


# --------------------------------------------------------------------------
# tension gradient
# --------------------------------------------------------------------------


class DimensionRange(Strict):
    range: tuple[float, float]

    @model_validator(mode="after")
    def _ordered(self) -> DimensionRange:
        lo, hi = self.range
        if lo >= hi:
            raise ValueError(f"dimension range must be increasing, got {self.range}")
        return self


class TemporalModel(Strict):
    window: str
    hysteresis_enabled: bool
    minimum_stable_observations: Annotated[int, Field(ge=1)]
    stale_state_decay: bool


class EvidenceRequirements(Strict):
    minimum_distinct_events: Annotated[int, Field(ge=1)]
    single_message_high_impact_inference_forbidden: bool
    provenance_required: bool


class TensionGradientSpec(Strict):
    engine_id: str
    purpose: str
    individual_gradient_formula: str
    group_tension_formula: str
    dimensions: dict[str, DimensionRange]
    classes: tuple[str, ...]
    temporal_model: TemporalModel
    evidence_requirements: EvidenceRequirements

    @model_validator(mode="after")
    def _classes_include_terminal_states(self) -> TensionGradientSpec:
        missing = {"aligned", "unresolved"} - set(self.classes)
        if missing:
            raise ValueError(
                f"tension_gradient.classes must include {sorted(missing)}: "
                "'aligned' and 'unresolved' are the terminal states the "
                "no-false-consensus invariant depends on"
            )
        return self


# --------------------------------------------------------------------------
# bridge engine
# --------------------------------------------------------------------------


class BridgeStrategySpec(Strict):
    id: str
    action: str


class BridgeSimulationSpec(Strict):
    required_before_apply: bool
    estimate_before_and_after: bool
    evaluate_each_participant: bool
    preserve_counterfactual: bool


class BridgeRollbackSpec(Strict):
    enabled: bool
    trigger_conditions: tuple[str, ...]


class BridgeEngine(Strict):
    engine_id: str
    objective: str
    ordered_strategies: tuple[BridgeStrategySpec, ...]
    simulation: BridgeSimulationSpec
    application_rule: str
    rollback: BridgeRollbackSpec

    @model_validator(mode="after")
    def _escalation_is_last(self) -> BridgeEngine:
        """Escalation must be the terminal strategy.

        The ordered list *is* the try-order. If ``explicit-escalation`` were not
        last, the engine would surface an irreducible conflict while cheaper
        bridges remained untried — and worse, an escalation ranked before
        ``parallel-fork`` would report conflict where none needed to exist.
        """
        if not self.ordered_strategies:
            raise ValueError("bridge_engine.ordered_strategies must not be empty")
        if self.ordered_strategies[-1].id != "explicit-escalation":
            raise ValueError(
                "bridge_engine.ordered_strategies must end with "
                f"'explicit-escalation', got {self.ordered_strategies[-1].id!r}"
            )
        return self


# --------------------------------------------------------------------------
# contextual sovereignty
# --------------------------------------------------------------------------


class PrecedenceLevel(Strict):
    level: Annotated[int, Field(ge=1)]
    cls: str = Field(alias="class")
    effect: str

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class IrreducibleConflict(Strict):
    action: str
    manufactured_consensus: Literal["forbidden"]
    minority_position_storage: Literal["required"]


class ContextualSovereignty(Strict):
    precedence: tuple[PrecedenceLevel, ...]
    hard_boundary_categories: tuple[str, ...]
    irreducible_conflict: IrreducibleConflict

    @model_validator(mode="after")
    def _precedence_is_dense_and_ordered(self) -> ContextualSovereignty:
        """Levels must be 1..N with no gaps or ties.

        Precedence is compared numerically at runtime; a duplicated level makes
        two different classes equally authoritative, which has no defined
        resolution, and a gap usually means a level was deleted by mistake.
        """
        levels = [p.level for p in self.precedence]
        if levels != list(range(1, len(levels) + 1)):
            raise ValueError(
                f"contextual_sovereignty.precedence levels must be 1..N in "
                f"order, got {levels}"
            )
        if self.precedence[0].cls != "hard-individual-boundary":
            raise ValueError(
                "precedence level 1 must be 'hard-individual-boundary'"
            )
        return self

    def level_of(self, cls: str) -> int:
        for p in self.precedence:
            if p.cls == cls:
                return p.level
        raise KeyError(f"unknown precedence class: {cls!r}")


# --------------------------------------------------------------------------
# modular differentiation
# --------------------------------------------------------------------------


class ModuleSelection(Strict):
    method: str
    score_range: tuple[float, float]
    activation_threshold: Probability
    max_active_modules: Annotated[int, Field(ge=1)]
    conflict_resolution: str
    permission_filter: Literal["mandatory-final-stage"]


class ModuleSpec(Strict):
    responsibility: str
    state_scope: str
    can_modify_tool_permissions: bool
    execution: str | None = None

    @model_validator(mode="after")
    def _no_module_may_modify_permissions(self) -> ModuleSpec:
        """No module in this card is allowed to widen the permission envelope.

        This is the structural half of ``permission_non_expansion``: rather than
        trusting each module's implementation, the card asserts the flag is false
        for every module and the loader refuses any card that says otherwise.
        """
        if self.can_modify_tool_permissions:
            raise ValueError(
                "can_modify_tool_permissions must be false for every module: "
                "the permission envelope may only be narrowed by the "
                "authorization filter, never widened by a module"
            )
        return self


class HotReload(Strict):
    enabled: bool
    atomic_manifest_swap: bool
    in_flight_action_policy: str


class ModularDifferentiation(Strict):
    compiler_id: str
    output: str
    compilation_mode: str
    deterministic: bool
    inputs: tuple[str, ...]
    module_selection: ModuleSelection
    modules: dict[str, ModuleSpec]
    hot_reload: HotReload


# --------------------------------------------------------------------------
# TORX
# --------------------------------------------------------------------------


class TorxStateType(Strict):
    uses: tuple[str, ...]
    domain: tuple[int, ...] | None = None


class FactorGraphNode(Strict):
    type: Literal["pbit", "pdit", "pmode"]


class FactorGraph(Strict):
    nodes: dict[str, FactorGraphNode]
    directed_edges: tuple[tuple[str, str], ...]

    @model_validator(mode="after")
    def _edges_reference_declared_nodes(self) -> FactorGraph:
        for src, dst in self.directed_edges:
            for endpoint in (src, dst):
                if endpoint not in self.nodes:
                    raise ValueError(
                        f"factor_graph edge references undeclared node "
                        f"{endpoint!r}"
                    )
        return self

    def parents_of(self, node: str) -> tuple[str, ...]:
        return tuple(s for s, d in self.directed_edges if d == node)


class TorxKernelSpec(Strict):
    cls: str = Field(alias="class")
    inputs: tuple[str, ...]
    output: str

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class TopologyDefaults(Strict):
    rolling_window_seconds: Annotated[float, Field(gt=0)]
    maximum_points: Annotated[int, Field(ge=1)]
    maximum_homology_dimension: Annotated[int, Field(ge=0)]
    update_interval_ms: Annotated[int, Field(ge=1)]


class StabilityContract(Strict):
    bound: str
    descriptor_invariance: str


class TopologySidecar(Strict):
    enabled: bool
    critical_path: bool
    fallback_order: tuple[str, ...]
    defaults: TopologyDefaults
    stability_contract: StabilityContract

    @model_validator(mode="after")
    def _sidecar_is_off_the_critical_path(self) -> TopologySidecar:
        """``large_deployment`` is claimed only while this stays false."""
        if self.critical_path:
            raise ValueError(
                "torx.topology_sidecar.critical_path must be false: the "
                "horizontal-scalability result is conditioned on topology work "
                "remaining an asynchronous degradable sidecar"
            )
        return self


class TorxSpec(Strict):
    integration_id: str
    execution_role: str
    state_types: dict[str, TorxStateType]
    factor_graph: FactorGraph
    kernels: dict[str, TorxKernelSpec]
    topology_sidecar: TopologySidecar

    @model_validator(mode="after")
    def _state_types_are_the_three_torx_types(self) -> TorxSpec:
        if set(self.state_types) != {"pbit", "pdit", "pmode"}:
            raise ValueError(
                "torx.state_types must be exactly {pbit, pdit, pmode}, got "
                f"{sorted(self.state_types)}"
            )
        return self


# --------------------------------------------------------------------------
# Rule 30 VDF
# --------------------------------------------------------------------------


class CanonicalInput(Strict):
    encoding: str
    key_order: Literal["lexicographic"]
    floating_point: str
    excluded_fields: tuple[str, ...]


class ProofEnvelopeSpec(Strict):
    required_fields: tuple[str, ...]


class VdfVerification(Strict):
    deterministic: bool
    offline_verification: bool
    mcp_tool: str


class Rule30Vdf(Strict):
    integration_id: str
    purpose: str
    algorithm: Literal["rule30-vdf"]
    canonical_input: CanonicalInput
    attested_events: tuple[str, ...]
    proof_envelope: ProofEnvelopeSpec
    verification: VdfVerification
    non_claims: tuple[str, ...]

    @model_validator(mode="after")
    def _proof_field_is_excluded_from_its_own_input(self) -> Rule30Vdf:
        """The proof cannot be part of the bytes it attests.

        Leaving ``vdf_proof`` in the canonical input makes the hash
        self-referential — nothing would ever verify.
        """
        if "vdf_proof" not in self.canonical_input.excluded_fields:
            raise ValueError(
                "rule30_vdf.canonical_input.excluded_fields must contain "
                "'vdf_proof'"
            )
        return self


# --------------------------------------------------------------------------
# JSONB persistence
# --------------------------------------------------------------------------


class TableSpec(Strict):
    primary_key: tuple[str, ...]
    columns: dict[str, str]
    indexes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _primary_key_columns_exist(self) -> TableSpec:
        missing = [c for c in self.primary_key if c not in self.columns]
        if missing:
            raise ValueError(f"primary key references undeclared columns: {missing}")
        return self


class FieldLevelEncryption(Strict):
    enabled_for: tuple[str, ...]


class Encryption(Strict):
    in_transit: str
    at_rest: str
    field_level: FieldLevelEncryption


class JsonbPersistence(Strict):
    engine: Literal["PostgreSQL"]
    minimum_version: str
    tenant_isolation: Literal["row-level-security"]
    encryption: Encryption
    tables: dict[str, TableSpec]
    retention: dict[str, str]

    @model_validator(mode="after")
    def _every_table_is_tenant_scoped(self) -> JsonbPersistence:
        """Row-level security is only meaningful with a tenant discriminator.

        A table missing ``tenant_id`` cannot carry an RLS policy, so its rows
        would be readable across tenants.
        """
        for name, table in self.tables.items():
            if "tenant_id" not in table.columns:
                raise ValueError(
                    f"table {name!r} has no tenant_id column but "
                    "tenant_isolation is row-level-security"
                )
        return self


class JsonbContract(Strict):
    required: tuple[str, ...]
    example: dict[str, Any]

    @model_validator(mode="after")
    def _example_satisfies_its_own_contract(self) -> JsonbContract:
        """The example doubles as the contract's first test vector."""
        missing = [k for k in self.required if k not in self.example]
        if missing:
            raise ValueError(
                f"jsonb contract example is missing required keys: {missing}"
            )
        return self


# --------------------------------------------------------------------------
# MCP
# --------------------------------------------------------------------------


class McpDiscovery(Strict):
    agent_card_path: str
    mcp_path: str


class McpAuthentication(Strict):
    required: bool
    mechanisms: tuple[str, ...]


class McpAuthorization(Strict):
    model: str
    default: Literal["deny"]
    self_expansion: Literal["forbidden"]


class McpIdempotency(Strict):
    required_for_mutations: bool
    header: str


class McpServer(Strict):
    id: str
    protocol_version: str
    transports: tuple[str, ...]
    discovery: McpDiscovery
    content_types: tuple[str, ...]
    authentication: McpAuthentication
    authorization: McpAuthorization
    idempotency: McpIdempotency


class McpTool(Strict):
    name: str
    mutating: bool
    permission: str
    input_schema: dict[str, Any]
    output: str

    @model_validator(mode="after")
    def _schema_is_a_json_object_schema(self) -> McpTool:
        if self.input_schema.get("type") != "object":
            raise ValueError(
                f"tool {self.name!r}: input_schema.type must be 'object'"
            )
        declared = set(self.input_schema.get("properties", {}))
        required = set(self.input_schema.get("required", []))
        if not required <= declared:
            raise ValueError(
                f"tool {self.name!r}: required names not in properties: "
                f"{sorted(required - declared)}"
            )
        return self


class McpResource(Strict):
    uri_template: str
    permission: str


class McpPrompt(Strict):
    name: str
    purpose: str


class McpErrors(Strict):
    format: str
    required_fields: tuple[str, ...]
    stable_codes: tuple[str, ...]


class Mcp(Strict):
    server: McpServer
    tools: tuple[McpTool, ...]
    resources: tuple[McpResource, ...]
    prompts: tuple[McpPrompt, ...]
    errors: McpErrors

    @model_validator(mode="after")
    def _tool_names_are_unique(self) -> Mcp:
        names = [t.name for t in self.tools]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"duplicate MCP tool names: {dupes}")
        return self

    def tool(self, name: str) -> McpTool:
        for t in self.tools:
            if t.name == name:
                return t
        raise KeyError(f"unknown MCP tool: {name!r}")

    @property
    def permissions(self) -> frozenset[str]:
        """Every permission the card can ever grant.

        The authorization filter uses this as the closed world: a grant naming a
        permission outside this set is a configuration error, not a new
        capability.
        """
        return frozenset(
            [t.permission for t in self.tools] + [r.permission for r in self.resources]
        )


# --------------------------------------------------------------------------
# runtime
# --------------------------------------------------------------------------


class EventModel(Strict):
    ordering_key: str
    delivery: str
    consumers_must_be_idempotent: bool


class StateBounds(Strict):
    profile_max_document_bytes: Annotated[int, Field(ge=1)]
    group_member_count_soft_limit: Annotated[int, Field(ge=1)]
    active_tension_window_events_per_member: Annotated[int, Field(ge=1)]
    bridge_candidates_per_cycle: Annotated[int, Field(ge=1)]
    active_modules_per_effective_agent: Annotated[int, Field(ge=1)]


class CriticalPath(Strict):
    components: tuple[str, ...]
    excluded: tuple[str, ...]


class Degradation(Strict):
    order: tuple[str, ...]
    never_disable: tuple[str, ...]

    @model_validator(mode="after")
    def _degradation_never_touches_the_protected_set(self) -> Degradation:
        """No step in the ladder may name a never-disable control.

        The controller walks ``order`` under load. If a step named e.g.
        ``disable-authorization-filter`` the ladder would eventually switch off a
        control the same section declares undisablable.
        """
        protected_words = {w for c in self.never_disable for w in c.split("-")}
        for step in self.order:
            if step.startswith("disable-"):
                target = step.removeprefix("disable-")
                if target in self.never_disable:
                    raise ValueError(
                        f"degradation step {step!r} disables a never_disable "
                        f"control {target!r}"
                    )
                if set(target.split("-")) & protected_words & {
                    "authorization", "boundary", "dissent"
                }:
                    raise ValueError(
                        f"degradation step {step!r} appears to disable a "
                        f"protected control ({sorted(self.never_disable)})"
                    )
        return self


class Runtime(Strict):
    event_model: EventModel
    state_bounds: StateBounds
    critical_path: CriticalPath
    degradation: Degradation

    @model_validator(mode="after")
    def _sidecar_excluded_from_critical_path(self) -> Runtime:
        if "topology-stability-sidecar" not in self.critical_path.excluded:
            raise ValueError(
                "runtime.critical_path.excluded must contain "
                "'topology-stability-sidecar'"
            )
        return self


# --------------------------------------------------------------------------
# observability
# --------------------------------------------------------------------------


class Tracing(Strict):
    required_span_attributes: tuple[str, ...]


class Observability(Strict):
    audit_events: tuple[str, ...]
    metrics: tuple[str, ...]
    tracing: Tracing


# --------------------------------------------------------------------------
# security and privacy
# --------------------------------------------------------------------------


class TenantPolicySpec(Strict):
    may_disable_automatic_learning: bool
    may_reduce_retention: bool
    may_narrow_module_catalog: bool
    may_not_disable_hard_boundary_enforcement: bool

    @model_validator(mode="after")
    def _hard_boundaries_are_not_tenant_disablable(self) -> TenantPolicySpec:
        if not self.may_not_disable_hard_boundary_enforcement:
            raise ValueError(
                "tenant_policy.may_not_disable_hard_boundary_enforcement must "
                "be true: tenant policy sits below hard individual boundaries "
                "in the precedence order"
            )
        return self


class DataMinimization(Strict):
    store_features_not_raw_content: bool
    purpose_binding: bool
    bounded_retention: bool


class SecurityAndPrivacy(Strict):
    threat_controls: dict[str, dict[str, Any]]
    data_minimization: DataMinimization
    tenant_policy: TenantPolicySpec


# --------------------------------------------------------------------------
# validation and deployment
# --------------------------------------------------------------------------


class SchemaValidation(Strict):
    yaml_parse_required: bool
    jsonb_contract_validation_required: bool
    mcp_input_schema_validation_required: bool


class BehavioralAcceptance(Strict):
    id: str
    assertion: str


class LoadQualification(Strict):
    status: str
    gates: tuple[str, ...]


class Validation(Strict):
    schema_: SchemaValidation = Field(alias="schema")
    behavioral_acceptance: tuple[BehavioralAcceptance, ...]
    load_qualification: LoadQualification

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    @property
    def acceptance_ids(self) -> frozenset[str]:
        return frozenset(a.id for a in self.behavioral_acceptance)


class Consistency(Strict):
    profile_revision: str
    group_intent_revision: str
    bridge_apply: str


class VdfAttestationPolicy(Strict):
    synchronous_for: tuple[str, ...]
    asynchronous_for: tuple[str, ...]

    @model_validator(mode="after")
    def _no_event_is_both(self) -> VdfAttestationPolicy:
        both = set(self.synchronous_for) & set(self.asynchronous_for)
        if both:
            raise ValueError(
                f"vdf_attestation events listed both sync and async: {sorted(both)}"
            )
        return self


class Availability(Strict):
    base_agent_fallback: bool
    hard_boundaries_cached: bool
    topology_sidecar_optional: bool
    vdf_attestation: VdfAttestationPolicy


class Deployment(Strict):
    topology: str
    partition_key: str
    recommended_components: tuple[str, ...]
    consistency: Consistency
    availability: Availability
    readiness_claim: str


# --------------------------------------------------------------------------
# top level
# --------------------------------------------------------------------------


class AgentModelCard(Strict):
    """The whole card. Unknown top-level keys are rejected."""

    schema_version: str
    kind: Literal["GenesisConductorAgentModelCard"]
    metadata: Metadata
    objective: Objective
    result_cards: dict[str, ResultCard]
    invariants: Invariants
    base_agent: BaseAgent
    personalization: Personalization
    individual_and_group_intent: IndividualAndGroupIntent
    tension_gradient: TensionGradientSpec
    bridge_engine: BridgeEngine
    contextual_sovereignty: ContextualSovereignty
    modular_differentiation: ModularDifferentiation
    torx: TorxSpec
    rule30_vdf: Rule30Vdf
    jsonb_persistence: JsonbPersistence
    jsonb_contracts: dict[str, JsonbContract]
    mcp: Mcp
    runtime: Runtime
    observability: Observability
    security_and_privacy: SecurityAndPrivacy
    validation: Validation
    deployment: Deployment

    @model_validator(mode="after")
    def _cross_section_coherence(self) -> AgentModelCard:
        """Checks that only make sense once every section has parsed.

        Each of these is a place where two sections restate the same fact; if
        they drift, the runtime silently follows one of them and the other
        becomes a false claim in the audit record.
        """
        # Every bridge strategy the engine may emit must be a known strategy id
        # when the bridge tables store ``bridge_type``.
        strategy_ids = {s.id for s in self.bridge_engine.ordered_strategies}
        if len(strategy_ids) != len(self.bridge_engine.ordered_strategies):
            raise ValueError("bridge_engine.ordered_strategies has duplicate ids")

        # The TORX factor graph must expose the three decision pbits the
        # sovereignty guard reads before any action.
        required_pbits = {"bridge_viability", "boundary_violation", "action_authorized"}
        declared_pbits = {
            n for n, spec in self.torx.factor_graph.nodes.items() if spec.type == "pbit"
        }
        missing = required_pbits - declared_pbits
        if missing:
            raise ValueError(
                f"torx.factor_graph is missing required pbit nodes: {sorted(missing)}"
            )

        # Attested events must be a subset of what observability can audit.
        audit_stems = {e.replace(".", "-").replace("_", "-") for e in
                       self.observability.audit_events}
        for event in self.rule30_vdf.attested_events:
            stem = event.replace("_", "-")
            if not any(stem.rstrip("d").rstrip("e") in a or a in stem
                       for a in audit_stems):
                raise ValueError(
                    f"attested event {event!r} has no corresponding entry in "
                    "observability.audit_events"
                )

        # Each behavioral acceptance id the tests key off must be present.
        required_acceptance = {
            "personalization-auto-fit",
            "permission-non-expansion",
            "contextual-sovereignty",
            "no-false-consensus",
            "bridge-benefit",
            "vdf-binding",
            "degraded-operation",
        }
        missing_acc = required_acceptance - self.validation.acceptance_ids
        if missing_acc:
            raise ValueError(
                f"validation.behavioral_acceptance is missing: {sorted(missing_acc)}"
            )
        return self
