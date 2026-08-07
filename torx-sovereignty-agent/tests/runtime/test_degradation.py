"""Task 8 part 1 — the ladder, the topology chain, and the observability contract.

The card's ``degraded-operation`` acceptance says a topology sidecar failure must
preserve TORX decisions, authorization, hard boundaries, and the last valid or
neutral descriptor. These tests are that assertion made executable, plus the two
contract checks that make a card edit fail the build: the metric registry must
name exactly ``observability.metrics``, and a span must emit exactly
``observability.tracing.required_span_attributes``.

Metrics and tracing are exercised here rather than under ``tests/observability``
because this task owns one test module; the sections below are kept separate.
"""

from __future__ import annotations

import threading
import uuid

import pytest

from src.model_card import load_model_card, parse_model_card
from src.model_card.loader import DEFAULT_CARD_PATH
from src.observability import metrics as metrics_mod
from src.observability import tracing as tracing_mod
from src.observability.metrics import (
    COUNTER,
    GAUGE,
    HISTOGRAM,
    METRIC_KINDS,
    MetricContractError,
    MetricRegistry,
    verify_metric_names,
)
from src.observability.tracing import (
    SPAN_ATTRIBUTE_KEYS,
    SpanAttributes,
    SpanContractError,
    SpanRecorder,
    hash_identifier,
    span,
    verify_span_attributes,
)
from src.runtime.degradation import (
    CAP_BRIDGE_ENGINE,
    CAP_MODULAR_DIFFERENTIATION,
    CAP_TOPOLOGY_DESCRIPTOR_INPUT,
    CAP_TOPOLOGY_RECOMPUTE,
    CAP_TOPOLOGY_SIDECAR,
    SHEDDABLE_CAPABILITIES,
    STAGE_BASELINE,
    STAGE_FRESH,
    STAGE_LAST_VALID,
    STAGE_NEUTRAL,
    STEP_EFFECTS,
    TOPOLOGY_DESCRIPTOR_STATES,
    DegradationController,
    DegradationRung,
    StepEffect,
    TopologyFallback,
    guarded_decision,
    neutral_descriptor,
)
from src.torx_layer.circuits import DecisionInputs
from src.torx_layer.state import PDit


@pytest.fixture(scope="module")
def card():
    return load_model_card()


@pytest.fixture(scope="module")
def card_text() -> str:
    return DEFAULT_CARD_PATH.read_text(encoding="utf-8")


@pytest.fixture
def registry(card) -> MetricRegistry:
    """An isolated in-process registry, so one test cannot read another's data."""
    return MetricRegistry(card, use_otel=False)


class FakeClock:
    """Monotonic clock the test drives, so dwell is tested without sleeping."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def controller(card, clock) -> DegradationController:
    return DegradationController(
        card, stability_dwell=3, min_dwell_seconds=30.0, clock=clock
    )


def a_descriptor(peak: str = "connected") -> PDit:
    return PDit.certain(TOPOLOGY_DESCRIPTOR_STATES, peak, "topology_descriptor")


# --------------------------------------------------------------------------
# the ladder is the card's ladder
# --------------------------------------------------------------------------


def test_ladder_order_is_read_from_the_card(controller, card):
    assert controller.order == tuple(card.runtime.degradation.order)
    assert controller.order == (
        "reduce-topology-refresh-frequency",
        "use-last-valid-topology-descriptor",
        "disable-topology-sidecar",
        "freeze-profile-learning",
        "retain-base-agent-plus-hard-boundaries",
    )
    assert controller.max_level == 5


def test_every_card_step_has_an_implementation(card):
    assert set(card.runtime.degradation.order) <= set(STEP_EFFECTS)


def test_a_card_step_with_no_implementation_is_rejected(card_text):
    mutated = card_text.replace(
        '- "reduce-topology-refresh-frequency"', '- "invent-a-new-rung"'
    )
    assert mutated != card_text
    card = parse_model_card(mutated, source="<mutated>")
    with pytest.raises(ValueError, match="no implementation"):
        DegradationController(card)


# --------------------------------------------------------------------------
# monotone escalation
# --------------------------------------------------------------------------


def test_sustained_pressure_walks_every_rung_without_skipping(controller):
    seen = [controller.level]
    for _ in range(controller.max_level):
        rung = controller.observe(True, source="load-shed")
        seen.append(rung.index)
    assert seen == [0, 1, 2, 3, 4, 5]
    # each rung names the card step at its position
    assert controller.rung.step == controller.order[-1]


def test_escalation_saturates_at_the_last_rung(controller):
    for _ in range(20):
        controller.observe(True, source="load-shed")
    assert controller.level == controller.max_level
    assert controller.rung.step == "retain-base-agent-plus-hard-boundaries"


def test_each_rung_sheds_a_superset_of_the_previous_rung(controller):
    previous: set[str] = set()
    for index in range(controller.max_level + 1):
        disabled = set(controller.rung_at(index).disabled)
        assert previous <= disabled, f"rung {index} un-sheds {previous - disabled}"
        previous = disabled
    assert previous  # the last rung sheds something


def test_refresh_slows_then_stops_as_the_ladder_descends(controller, card):
    base = float(card.torx.topology_sidecar.defaults.update_interval_ms)
    assert controller.rung_at(0).refresh_interval_ms == base
    assert controller.rung_at(1).refresh_interval_ms == base * 4.0
    # from "use-last-valid-topology-descriptor" onward there is no recompute at all
    for index in range(2, controller.max_level + 1):
        assert controller.rung_at(index).refresh_interval_ms is None


# --------------------------------------------------------------------------
# never_disable — asserted at every rung
# --------------------------------------------------------------------------


def test_protected_controls_are_active_at_every_rung(controller, card):
    protected = tuple(card.runtime.degradation.never_disable)
    assert protected == (
        "authorization-filter",
        "hard-boundary-enforcement",
        "dissent-preservation",
    )
    for index in range(controller.max_level + 1):
        rung = controller.rung_at(index)
        for control in protected:
            assert rung.is_active(control), (
                f"{control} inactive at rung {index} ({rung.step})"
            )
        assert not set(rung.disabled) & set(protected)


def test_protected_controls_survive_sustained_pressure(controller, card):
    protected = tuple(card.runtime.degradation.never_disable)
    for _ in range(controller.max_level * 3):
        rung = controller.observe(True, source="load-shed")
        for control in protected:
            assert rung.is_active(control)
    assert controller.level == controller.max_level
    for control in protected:
        assert controller.is_active(control)


def test_a_rung_that_disables_a_protected_control_cannot_be_constructed():
    with pytest.raises(ValueError, match="never_disable"):
        DegradationRung(
            index=1,
            step="disable-authorization-filter",
            summary="",
            disabled=("authorization-filter",),
            protected=("authorization-filter", "hard-boundary-enforcement"),
            refresh_interval_ms=None,
        )


def test_a_step_effect_targeting_a_protected_control_is_rejected(
    card, monkeypatch
):
    """The whole effect table is checked once, at controller construction."""
    monkeypatch.setitem(
        STEP_EFFECTS,
        "freeze-profile-learning",
        StepEffect.__new__(StepEffect),
    )
    # Build the malicious effect without tripping StepEffect's own validation,
    # which only knows about sheddable capabilities.
    object.__setattr__(
        STEP_EFFECTS["freeze-profile-learning"], "disables", ("dissent-preservation",)
    )
    object.__setattr__(
        STEP_EFFECTS["freeze-profile-learning"], "refresh_multiplier", 1.0
    )
    object.__setattr__(STEP_EFFECTS["freeze-profile-learning"], "summary", "bad")
    with pytest.raises(ValueError, match="never_disable forbids"):
        DegradationController(card)


def test_step_effects_may_only_name_sheddable_capabilities():
    with pytest.raises(ValueError, match="SHEDDABLE_CAPABILITIES"):
        StepEffect(
            disables=("hard-boundary-enforcement",),
            refresh_multiplier=1.0,
            summary="",
        )


def test_is_active_rejects_an_unknown_control(controller):
    with pytest.raises(ValueError, match="unknown control"):
        controller.is_active("topology-sidcar")
    for cap in SHEDDABLE_CAPABILITIES:
        assert controller.is_active(cap)  # nominal: nothing shed


# --------------------------------------------------------------------------
# recovery: dwell, no flapping
# --------------------------------------------------------------------------


def test_recovery_needs_the_consecutive_healthy_count(controller, clock):
    controller.observe(True)
    controller.observe(True)
    assert controller.level == 2

    clock.advance(60.0)
    controller.observe(False)
    controller.observe(False)
    assert controller.level == 2, "recovered before the dwell count was met"
    assert controller.observe(False).index == 1


def test_recovery_needs_the_minimum_time_at_the_rung(controller, clock):
    controller.observe(True)
    assert controller.level == 1
    for _ in range(10):
        controller.observe(False)  # count satisfied many times over
    assert controller.level == 1, "recovered before min_dwell_seconds elapsed"

    clock.advance(30.0)
    controller.observe(False)
    controller.observe(False)
    assert controller.observe(False).index == 0


def test_one_pressure_sample_resets_the_healthy_streak(controller, clock):
    controller.observe(True)
    controller.observe(True)
    assert controller.level == 2
    clock.advance(60.0)

    controller.observe(False)
    controller.observe(False)
    controller.observe(True)  # flap
    assert controller.level == 3

    clock.advance(60.0)
    controller.observe(False)
    controller.observe(False)
    assert controller.level == 3, "streak was not reset by the pressure sample"


def test_recovery_is_one_rung_at_a_time(controller, clock):
    for _ in range(5):
        controller.observe(True)
    assert controller.level == 5
    levels = []
    for _ in range(5):
        clock.advance(60.0)
        for _ in range(3):
            rung = controller.observe(False)
        levels.append(rung.index)
    assert levels == [4, 3, 2, 1, 0]


def test_nominal_stays_nominal_under_health(controller, clock):
    for _ in range(10):
        clock.advance(60.0)
        assert controller.observe(False).index == 0
    assert controller.history == ()


def test_history_records_only_actual_transitions(controller, clock):
    controller.observe(True, source="sidecar")
    clock.advance(60.0)
    controller.observe(False)
    controller.observe(False)
    controller.observe(False)
    directions = [t.direction for t in controller.history]
    assert directions == ["escalate", "recover"]
    assert controller.history[0].to_json()["from_index"] == 0
    assert controller.history[-1].to_json()["to_index"] == 0


def test_reset_returns_to_nominal_and_is_recorded(controller):
    for _ in range(3):
        controller.observe(True)
    assert controller.reset(source="operator").index == 0
    assert controller.history[-1].direction == "reset"
    assert controller.to_json()["level"] == 0


# --------------------------------------------------------------------------
# topology fallback chain
# --------------------------------------------------------------------------


def test_fallback_order_is_read_from_the_card(card, registry):
    fb = TopologyFallback(card, registry=registry)
    assert fb.fallback_order == tuple(card.torx.topology_sidecar.fallback_order)
    assert fb.fallback_order == (
        STAGE_LAST_VALID,
        STAGE_NEUTRAL,
        STAGE_BASELINE,
    )


def test_a_fresh_probe_is_used_and_cached(card, registry):
    fb = TopologyFallback(card, registry=registry)
    result = fb.resolve(lambda: a_descriptor("cyclic"))
    assert result.stage == STAGE_FRESH
    assert not result.degraded
    assert result.sidecar_ok
    assert result.descriptor.argmax == "cyclic"
    assert fb.last_valid is not None


def test_a_raising_sidecar_falls_back_to_the_last_valid_descriptor(card, registry):
    fb = TopologyFallback(card, registry=registry)
    fb.resolve(lambda: a_descriptor("fragmented"))

    def boom() -> PDit:
        raise RuntimeError("sidecar worker died")

    result = fb.resolve(boom)
    assert result.stage == STAGE_LAST_VALID
    assert result.descriptor.argmax == "fragmented"
    assert result.informative and result.use_topology_edge
    assert not result.sidecar_ok
    assert "sidecar worker died" in (result.error or "")


def test_with_no_cache_a_failure_falls_through_to_neutral(card, registry):
    fb = TopologyFallback(card, registry=registry)

    def boom() -> PDit:
        raise RuntimeError("cold start, no descriptor yet")

    result = fb.resolve(boom)
    assert result.stage == STAGE_NEUTRAL
    assert not result.informative
    assert result.use_topology_edge
    assert result.descriptor.margin == 0.0


def test_a_hanging_sidecar_is_abandoned_within_its_budget(card, registry):
    fb = TopologyFallback(card, registry=registry, timeout_s=0.05)
    fb.remember(a_descriptor("connected"))
    release = threading.Event()

    def hang() -> PDit:
        release.wait(30.0)
        return a_descriptor("cyclic")

    try:
        result = fb.resolve(hang)
    finally:
        release.set()

    assert result.stage == STAGE_LAST_VALID
    assert result.descriptor.argmax == "connected"
    assert "SidecarTimeout" in (result.error or "")
    assert result.elapsed_s < 5.0, "the timeout did not bound the caller"


def test_a_relabelled_descriptor_is_refused_rather_than_stored(card, registry):
    fb = TopologyFallback(card, registry=registry)
    result = fb.resolve(lambda: PDit.uniform(("a", "b"), "topology_descriptor"))
    assert result.stage == STAGE_NEUTRAL
    assert "do not match" in (result.error or "")
    assert fb.last_valid is None


def test_the_floor_pins_the_stage_even_when_a_cache_exists(card, registry):
    fb = TopologyFallback(card, registry=registry)
    fb.remember(a_descriptor("connected"))
    result = fb.resolve(lambda: a_descriptor("cyclic"), floor=STAGE_NEUTRAL)
    assert result.stage == STAGE_NEUTRAL
    assert not result.sidecar_attempted, "probed while the sidecar was disabled"


def test_the_baseline_stage_drops_the_edge_but_still_returns_a_descriptor(
    card, registry
):
    fb = TopologyFallback(card, registry=registry)
    fb.remember(a_descriptor("connected"))
    result = fb.resolve(floor=STAGE_BASELINE)
    assert result.stage == STAGE_BASELINE
    assert not result.use_topology_edge
    assert result.descriptor.outcomes == TOPOLOGY_DESCRIPTOR_STATES


def test_the_neutral_descriptor_asserts_nothing():
    d = neutral_descriptor()
    assert d.margin == 0.0
    assert len(set(d.probs)) == 1
    assert d.entropy == pytest.approx(2.0)  # log2(4)


def test_the_ladder_drives_the_topology_floor(controller):
    floors = [controller.topology_stage_floor]
    for _ in range(controller.max_level):
        controller.observe(True)
        floors.append(controller.topology_stage_floor)
    assert floors == [
        "",  # nominal: probe freely
        "",  # reduce-topology-refresh-frequency: still probing, slower
        STAGE_LAST_VALID,  # use-last-valid-topology-descriptor
        STAGE_NEUTRAL,  # disable-topology-sidecar
        STAGE_NEUTRAL,  # freeze-profile-learning
        STAGE_BASELINE,  # retain-base-agent-plus-hard-boundaries
    ]


def test_topology_sidecar_latency_is_recorded(card, registry):
    fb = TopologyFallback(card, registry=registry)
    fb.resolve(lambda: a_descriptor())
    assert registry.count("gc_topology_sidecar_seconds", stage=STAGE_FRESH) == 1


# --------------------------------------------------------------------------
# a sidecar failure never reaches the decision
# --------------------------------------------------------------------------


def test_a_broken_sidecar_still_yields_a_decision(card, controller, registry):
    fb = TopologyFallback(card, registry=registry)

    def boom() -> PDit:
        raise RuntimeError("sidecar unavailable")

    outcome = guarded_decision(
        DecisionInputs(0.8, 0.0, 1.0),
        fallback=fb,
        controller=controller,
        fetch=boom,
        registry=registry,
    )
    assert outcome.topology.stage == STAGE_NEUTRAL
    assert outcome.decision.resolution_status.outcomes
    assert outcome.decision.action_authorized.is_certainly_true()
    assert controller.level == 1, "a probe failure should cost exactly one rung"
    assert registry.count("gc_torx_update_seconds", stage=STAGE_NEUTRAL) == 1


def test_hard_boundaries_still_block_at_the_last_rung(card, controller, registry):
    for _ in range(controller.max_level):
        controller.observe(True)
    assert controller.level == controller.max_level

    fb = TopologyFallback(card, registry=registry)
    outcome = guarded_decision(
        DecisionInputs(0.99, 1.0, 1.0),
        fallback=fb,
        controller=controller,
        fetch=lambda: a_descriptor(),
        registry=registry,
    )
    assert outcome.rung.index == controller.max_level
    assert outcome.decision.boundary_violation.is_certainly_true()
    assert outcome.decision.bridge_viability.is_certainly_false()
    assert outcome.rung.is_active("hard-boundary-enforcement")
    assert outcome.rung.is_active("authorization-filter")
    assert outcome.rung.is_active("dissent-preservation")
    assert not outcome.rung.is_active(CAP_BRIDGE_ENGINE)
    assert not outcome.rung.is_active(CAP_MODULAR_DIFFERENTIATION)


def test_no_probe_means_no_pressure_so_the_ladder_does_not_ratchet(
    card, controller, registry
):
    for _ in range(3):
        controller.observe(True)
    assert controller.level == 3  # sidecar disabled: resolve will not probe
    fb = TopologyFallback(card, registry=registry)
    for _ in range(5):
        outcome = guarded_decision(
            DecisionInputs(0.5, 0.0, 1.0),
            fallback=fb,
            controller=controller,
            fetch=lambda: a_descriptor(),
            registry=registry,
        )
        assert not outcome.topology.sidecar_attempted
    assert controller.level == 3


def test_guarded_decision_works_without_a_controller(card, registry):
    fb = TopologyFallback(card, registry=registry)
    outcome = guarded_decision(
        DecisionInputs(0.6, 0.0, 1.0), fallback=fb, registry=registry
    )
    assert outcome.rung.index == 0
    assert outcome.rung.is_active("dissent-preservation")
    assert outcome.to_json()["topology"]["stage"] == STAGE_NEUTRAL


def test_capability_shedding_matches_the_named_rungs(controller):
    controller.observe(True)
    controller.observe(True)
    assert not controller.is_active(CAP_TOPOLOGY_RECOMPUTE)
    assert controller.is_active(CAP_TOPOLOGY_SIDECAR)
    controller.observe(True)
    assert not controller.is_active(CAP_TOPOLOGY_SIDECAR)
    assert controller.is_active(CAP_TOPOLOGY_DESCRIPTOR_INPUT)
    controller.observe(True)
    controller.observe(True)
    assert not controller.is_active(CAP_TOPOLOGY_DESCRIPTOR_INPUT)


# --------------------------------------------------------------------------
# metrics: the registry is exactly the card's list
# --------------------------------------------------------------------------


def test_registry_names_are_exactly_the_card_metric_list(card, registry):
    assert registry.names == tuple(card.observability.metrics)
    assert set(METRIC_KINDS) == set(card.observability.metrics)
    assert len(registry.names) == 12


def test_a_card_metric_with_no_implementation_fails(card_text):
    mutated = card_text.replace(
        '    - "gc_mcp_tool_call_total"',
        '    - "gc_mcp_tool_call_total"\n    - "gc_unimplemented_total"',
    )
    assert mutated != card_text
    mutated_card = parse_model_card(mutated, source="<mutated>")
    with pytest.raises(MetricContractError, match="gc_unimplemented_total"):
        verify_metric_names(mutated_card)
    with pytest.raises(MetricContractError):
        MetricRegistry(mutated_card, use_otel=False)


def test_a_metric_dropped_from_the_card_also_fails(card_text):
    mutated = card_text.replace('    - "gc_torx_update_seconds"\n', "")
    assert mutated != card_text
    with pytest.raises(MetricContractError, match="gc_torx_update_seconds"):
        verify_metric_names(parse_model_card(mutated, source="<mutated>"))


def test_metric_kinds_are_declared_for_every_name(card):
    for name in card.observability.metrics:
        assert METRIC_KINDS[name] in (COUNTER, GAUGE, HISTOGRAM)
    assert METRIC_KINDS["gc_group_tension_current"] == GAUGE
    assert METRIC_KINDS["gc_profile_revision_total"] == COUNTER
    assert METRIC_KINDS["gc_torx_update_seconds"] == HISTOGRAM


def test_counters_accumulate_and_gauges_replace(registry):
    registry.counter("gc_boundary_violation_block_total").add(1, tenant="acme")
    registry.counter("gc_boundary_violation_block_total").add(2, tenant="acme")
    assert registry.value("gc_boundary_violation_block_total", tenant="acme") == 3.0

    registry.gauge("gc_group_tension_current").set(0.4, group="g")
    registry.gauge("gc_group_tension_current").set(0.7, group="g")
    assert registry.value("gc_group_tension_current", group="g") == 0.7


def test_attributes_separate_series(registry):
    registry.counter("gc_mcp_tool_call_total").add(1, tool="a", outcome="ok")
    registry.counter("gc_mcp_tool_call_total").add(1, tool="b", outcome="ok")
    assert registry.value("gc_mcp_tool_call_total", tool="a", outcome="ok") == 1.0
    assert len(registry.points("gc_mcp_tool_call_total")) == 2


def test_the_wrong_record_method_raises_rather_than_coercing(registry):
    with pytest.raises(ValueError, match="Use add"):
        registry.instrument("gc_profile_revision_total").set(1.0)
    with pytest.raises(ValueError, match="Use record"):
        registry.instrument("gc_torx_update_seconds").add(1.0)
    with pytest.raises(ValueError, match="cannot decrease"):
        registry.counter("gc_profile_revision_total").add(-1)


def test_unknown_metric_names_are_rejected(registry):
    with pytest.raises(KeyError):
        registry.instrument("gc_made_up_total")


def test_recorded_values_are_rounded_for_stable_storage(registry):
    registry.histogram("gc_bridge_predicted_delta").record(1 / 3)
    point = registry.points("gc_bridge_predicted_delta")[0]
    assert point.value == round(1 / 3, 9)
    assert point.to_json()["count"] == 1


def test_timed_records_even_when_the_block_raises(registry):
    with pytest.raises(RuntimeError):
        with registry.timed("gc_effective_agent_compile_seconds", tenant="acme"):
            raise RuntimeError("compile failed")
    assert registry.count("gc_effective_agent_compile_seconds", tenant="acme") == 1


def test_the_backend_is_reported(registry):
    assert registry.backend == "in-process"
    assert metrics_mod.backend_status()["backend"] in (
        "opentelemetry",
        "in-process",
    )
    snapshot = registry.snapshot()
    assert snapshot["backend"] == "in-process"
    assert snapshot["names"] == list(registry.names)


def test_the_otel_path_records_the_same_values_when_available(card):
    if not metrics_mod.otel_available():  # pragma: no cover - environment dependent
        pytest.skip("opentelemetry-api not installed")
    reg = MetricRegistry(card, use_otel=True)
    assert reg.backend == "opentelemetry"
    reg.counter("gc_false_consensus_guard_total").add(2, group="g")
    # The OTel API without a configured SDK is a no-op, so the in-process
    # accumulator remains the readable source of truth.
    assert reg.value("gc_false_consensus_guard_total", group="g") == 2.0


# --------------------------------------------------------------------------
# tracing: hashes, never raw ids
# --------------------------------------------------------------------------


def test_span_attribute_keys_are_exactly_the_card_list(card):
    required = tuple(card.observability.tracing.required_span_attributes)
    assert SPAN_ATTRIBUTE_KEYS == required
    attrs = SpanAttributes.for_subject(
        tenant_id="acme", user_id="u-1", group_id="g-1"
    )
    assert tuple(attrs.attributes()) == required
    assert verify_span_attributes(attrs.attributes(), card) == required


def test_a_card_span_attribute_with_no_implementation_fails(card_text):
    mutated = card_text.replace(
        '      - "vdf.proof_id"', '      - "vdf.proof_id"\n      - "session.id"'
    )
    assert mutated != card_text
    mutated_card = parse_model_card(mutated, source="<mutated>")
    attrs = SpanAttributes(tenant_id="acme")
    with pytest.raises(SpanContractError, match="session.id"):
        verify_span_attributes(attrs.attributes(), mutated_card)


def test_a_raw_uuid_never_appears_in_emitted_attributes(card):
    user_id = str(uuid.uuid4())
    group_id = str(uuid.uuid4())
    attrs = SpanAttributes.for_subject(
        tenant_id="acme",
        user_id=user_id,
        group_id=group_id,
        profile_revision=7,
        tension_class="semantic-gap",
        bridge_type="semantic-translation",
    )
    recorder = SpanRecorder()
    with span("torx.update", attrs, card=card, recorder=recorder):
        pass

    (recorded,) = recorder.spans()
    blob = repr(recorded.to_json())
    assert user_id not in blob
    assert group_id not in blob
    assert recorded.attributes["user.id_hash"] == hash_identifier(
        user_id, scope="acme"
    )
    assert len(recorded.attributes["user.id_hash"]) == 32
    # The card lists tenant.id unhashed: a tenant is an organisation, and
    # tenant-scoped querying is why traces exist.
    assert recorded.attributes["tenant.id"] == "acme"


def test_the_attributes_object_cannot_hold_a_raw_identifier():
    raw = str(uuid.uuid4())
    with pytest.raises(ValueError, match="raw identifier"):
        SpanAttributes(tenant_id="acme", user_id_hash=raw)
    with pytest.raises(ValueError, match="raw identifier"):
        SpanAttributes(tenant_id="acme", group_id_hash=raw)
    with pytest.raises(ValueError, match="hex digest"):
        SpanAttributes(tenant_id="acme", user_id_hash="not-a-hash")

    attrs = SpanAttributes.for_subject(tenant_id="acme", user_id=raw)
    assert not hasattr(attrs, "user_id")
    assert raw not in repr(attrs)


def test_hashes_are_tenant_scoped_and_peppered(monkeypatch):
    raw = "user-42"
    a = hash_identifier(raw, scope="tenant-a")
    b = hash_identifier(raw, scope="tenant-b")
    assert a != b, "the same user must not link across tenants"

    monkeypatch.setenv(tracing_mod.ID_HASH_SALT_ENV, "a-different-pepper")
    assert hash_identifier(raw, scope="tenant-a") != a


def test_unknown_facet_values_are_rejected():
    with pytest.raises(ValueError, match="tension_class"):
        SpanAttributes(tenant_id="acme", tension_class="mildly-annoyed")
    with pytest.raises(ValueError, match="bridge_type"):
        SpanAttributes(tenant_id="acme", bridge_type="just-agree")
    with pytest.raises(ValueError, match="profile_revision"):
        SpanAttributes(tenant_id="acme", profile_revision=-1)
    with pytest.raises(ValueError, match="tenant_id is required"):
        SpanAttributes(tenant_id="")


def test_late_known_facets_are_revalidated_inside_the_span(card):
    attrs = SpanAttributes.for_subject(tenant_id="acme", user_id="u-1")
    recorder = SpanRecorder()
    with span("bridge.apply", attrs, card=card, recorder=recorder) as handle:
        assert "bridge.type" in handle.attrs.unset
        handle.update(bridge_type="pareto-option", vdf_proof_id="proof-9")
        with pytest.raises(ValueError, match="raw identifier"):
            handle.update(user_id_hash=str(uuid.uuid4()))

    (recorded,) = recorder.spans()
    assert recorded.attributes["bridge.type"] == "pareto-option"
    assert recorded.attributes["vdf.proof_id"] == "proof-9"


def test_a_span_records_its_duration_into_a_named_metric(card, registry):
    attrs = SpanAttributes(tenant_id="acme")
    recorder = SpanRecorder()
    with span(
        "vdf.attest",
        attrs,
        metric="gc_vdf_attestation_seconds",
        registry=registry,
        card=card,
        recorder=recorder,
    ):
        pass
    assert registry.count("gc_vdf_attestation_seconds") == 1
    assert recorder.spans()[0].error is None


def test_a_failing_span_is_still_recorded(card):
    recorder = SpanRecorder()
    with pytest.raises(RuntimeError):
        with span("torx.update", SpanAttributes(tenant_id="acme"), card=card,
                  recorder=recorder):
            raise RuntimeError("kernel unavailable")
    (recorded,) = recorder.spans()
    assert "kernel unavailable" in (recorded.error or "")
    assert tuple(recorded.attributes) == SPAN_ATTRIBUTE_KEYS


def test_the_span_buffer_is_bounded():
    recorder = SpanRecorder(maxlen=2)
    for _ in range(5):
        with span("torx.update", SpanAttributes(tenant_id="acme"), recorder=recorder):
            pass
    assert len(recorder.spans()) == 2
    with pytest.raises(ValueError, match="span buffer size"):
        SpanRecorder(maxlen=0)
