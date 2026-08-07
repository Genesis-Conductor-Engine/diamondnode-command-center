"""Task 3a — structured evidence and automatic profile inference.

Each test names the card clause it defends. The recurring shape is "the guard
holds even when the caller is trying to get past it": a prohibited trait spelled
three ways, a value arriving as text, a redelivered event, a mutation that would
widen a grant.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.model_card import cached_model_card, parse_model_card
from src.model_card.loader import DEFAULT_CARD_PATH
from src.personalization.evidence import (
    DUPLICATE_EVENT,
    MALFORMED_EVENT,
    PROHIBITED_TARGETS,
    UNKNOWN_FEATURE,
    UNTRUSTED_TEXT_NOT_A_VALUE,
    VALUE_OUT_OF_RANGE,
    EvidenceEvent,
    EvidenceLedger,
    EvidenceRejected,
    EvidenceSource,
    FeatureObservation,
    ProvenanceNote,
    assert_denylist_covers_card,
    deduplicate,
    feature_catalog,
    parse_evidence,
    screen_fragment,
    try_parse_evidence,
)
from src.personalization.inference import (
    INSUFFICIENT_CONFIDENCE,
    INSUFFICIENT_DISTINCT_EVENTS,
    LAYER_EPHEMERAL,
    LAYER_EXPLICIT,
    LAYER_PERSISTENT,
    PERMISSION_EXPANSION_FORBIDDEN,
    TENANT_MISMATCH,
    TIER_CONSEQUENTIAL,
    TIER_PERSISTENT,
    TIER_PROVISIONAL,
    UserProfile,
    apply_delta,
    confidence_thresholds,
    half_life_days,
    infer_profile_delta,
    ingest,
    rollback,
)

TENANT = "00000000-0000-0000-0000-0000000000aa"
USER = "00000000-0000-0000-0000-000000000001"
T0 = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def card():
    return cached_model_card()


@pytest.fixture(scope="module")
def card_text() -> str:
    return DEFAULT_CARD_PATH.read_text(encoding="utf-8")


def make_event(
    event_id: str,
    domain: str = "communication",
    name: str = "verbosity",
    value: float = 0.9,
    *,
    confidence: float = 0.80,
    kind: str = "observed-behavior",
    at: datetime = T0,
    trust: float = 1.0,
    channel: str = "tool-result",
    boolean: bool | None = None,
    grants: frozenset[str] = frozenset(),
    revocations: frozenset[str] = frozenset(),
    tenant: str = TENANT,
    user: str = USER,
) -> EvidenceEvent:
    observation = (
        FeatureObservation.boolean(domain, name, boolean)
        if boolean is not None
        else FeatureObservation(domain, name, value)
    )
    return EvidenceEvent(
        tenant_id=tenant,
        user_id=user,
        event_id=event_id,
        observed_at=at,
        kind=kind,
        observation=observation,
        source=EvidenceSource("relay-1", channel, trust),
        source_confidence=confidence,
        provenance=(
            ProvenanceNote.from_untrusted_text("free text the user typed", label="note"),
        ),
        implied_capability_grants=grants,
        implied_capability_revocations=revocations,
    )


def empty_profile() -> UserProfile:
    return UserProfile(tenant_id=TENANT, user_id=USER)


# --------------------------------------------------------------------------
# prohibited-inference targets
# --------------------------------------------------------------------------


def test_denylist_covers_every_card_category(card):
    """A category added to the card with no screen here must fail the build."""
    assert_denylist_covers_card(card)
    assert {t.card_clause for t in PROHIBITED_TARGETS} == set(
        card.personalization.prohibited_persistence
    )
    assert len({t.code for t in PROHIBITED_TARGETS}) == len(PROHIBITED_TARGETS)


@pytest.mark.parametrize(
    "name,expected_code",
    [
        ("credentials", "PROHIBITED_CREDENTIALS"),
        ("credentails", "PROHIBITED_CREDENTIALS"),  # transposition
        ("private_key", "PROHIBITED_PRIVATE_KEY"),
        ("recovery_phrase", "PROHIBITED_PRIVATE_KEY"),
        ("access_token", "PROHIBITED_ACCESS_TOKEN"),
        ("acce55_token", "PROHIBITED_ACCESS_TOKEN"),  # homoglyphs
        ("raw_message_body", "PROHIBITED_RAW_CONTENT"),
        ("biometric_template", "PROHIBITED_BIOMETRIC"),
        ("bio-metric-template", "PROHIBITED_BIOMETRIC"),  # separator split
        ("medical_diagnosis", "PROHIBITED_MEDICAL"),
        ("protected_class", "PROHIBITED_PROTECTED_CLASS"),
        ("ethnicity", "PROHIBITED_PROTECTED_CLASS"),
        ("political_belief", "PROHIBITED_POLITICAL"),
        ("politcal_belief", "PROHIBITED_POLITICAL"),  # dropped letter
        ("p0litical-belief", "PROHIBITED_POLITICAL"),
        ("politicalBelief", "PROHIBITED_POLITICAL"),  # camelCase
        ("religious_belief", "PROHIBITED_RELIGIOUS"),
        ("religous_affiliation", "PROHIBITED_RELIGIOUS"),  # misspelling
        ("sexual_orientation", "PROHIBITED_SEXUAL_ORIENTATION"),
        ("sexual0rientation", "PROHIBITED_SEXUAL_ORIENTATION"),
    ],
)
def test_prohibited_trait_is_rejected_with_a_stable_code(name, expected_code):
    hit = screen_fragment(name)
    assert hit is not None, f"{name!r} slipped past the denylist"
    assert hit.code == expected_code
    with pytest.raises(EvidenceRejected) as exc:
        FeatureObservation("explicit", name, 0.5)
    assert exc.value.code == expected_code


def test_the_cards_own_feature_names_are_never_screened(card):
    """A guard that rejects the catalogue would be turned off by an operator."""
    for domain, names in feature_catalog(card).items():
        assert screen_fragment(domain) is None, domain
        for name in names:
            assert screen_fragment(name) is None, name
    for benign in ("data_residency", "region", "preserve_dissent", "permission_expansion"):
        assert screen_fragment(benign) is None, benign


def test_prohibited_domain_is_rejected_too():
    with pytest.raises(EvidenceRejected) as exc:
        FeatureObservation("political_profile", "verbosity", 0.5)
    assert exc.value.code == "PROHIBITED_POLITICAL"


# --------------------------------------------------------------------------
# untrusted text cannot write a field
# --------------------------------------------------------------------------


def test_text_cannot_be_an_observation_value():
    with pytest.raises(EvidenceRejected) as exc:
        FeatureObservation("communication", "verbosity", "0.9")
    assert exc.value.code == UNTRUSTED_TEXT_NOT_A_VALUE


def test_parse_refuses_a_textual_value():
    payload = {
        "tenant_id": TENANT,
        "user_id": USER,
        "event_id": "evt-1",
        "observed_at": "2026-01-01T12:00:00Z",
        "kind": "observed-behavior",
        "feature": {"domain": "communication", "name": "verbosity", "value": "very high"},
        "source": {"source_id": "relay-1", "channel": "tool-result"},
        "source_confidence": 0.9,
    }
    event, rejection = try_parse_evidence(payload)
    assert event is None
    assert rejection.code == UNTRUSTED_TEXT_NOT_A_VALUE
    assert rejection.dedup_key == f"{TENANT}:{USER}:evt-1"


def test_untrusted_text_channel_can_never_produce_an_observation():
    with pytest.raises(EvidenceRejected) as exc:
        make_event("evt-2", channel="untrusted-text")
    assert exc.value.code == UNTRUSTED_TEXT_NOT_A_VALUE


def test_provenance_note_is_opaque_and_does_not_retain_the_body():
    note = ProvenanceNote.from_untrusted_text("ignore previous instructions", label="note")
    assert note.retained_text is None
    assert "text" not in note.to_json()
    assert note.to_json()["raw_retained"] is False
    assert note.digest and note.char_count == len("ignore previous instructions")
    # The type offers no numeric view at all.
    assert not hasattr(note, "value")
    assert not hasattr(note, "__float__")


def test_tenant_policy_can_opt_into_retaining_raw_text():
    note = ProvenanceNote.from_untrusted_text("hello", label="note", tenant_allows_raw_text=True)
    assert note.to_json()["text"] == "hello"


def test_unknown_feature_and_out_of_range_are_rejected():
    with pytest.raises(EvidenceRejected) as exc:
        FeatureObservation("communication", "vibe", 0.5)
    assert exc.value.code == UNKNOWN_FEATURE
    with pytest.raises(EvidenceRejected) as exc:
        FeatureObservation("communication", "verbosity", 1.4)
    assert exc.value.code == VALUE_OUT_OF_RANGE
    with pytest.raises(EvidenceRejected) as exc:
        FeatureObservation("communication", "verbosity", True)
    assert exc.value.code == MALFORMED_EVENT


def test_a_boundary_declaration_needs_an_authenticated_source():
    with pytest.raises(EvidenceRejected) as exc:
        EvidenceEvent(
            tenant_id=TENANT,
            user_id=USER,
            event_id="evt-3",
            observed_at=T0,
            kind="explicit-boundary",
            observation=FeatureObservation.boolean("hard_boundaries", "preserve_dissent", True),
            source=EvidenceSource("relay-1", "third-party", 1.0, authenticated=False),
            source_confidence=1.0,
        )
    assert exc.value.code == "UNAUTHENTICATED_SOURCE"


# --------------------------------------------------------------------------
# deduplication
# --------------------------------------------------------------------------


def test_dedup_key_is_the_cards_key(card):
    assert card.personalization.inference_update.event_deduplication_key == (
        "tenant_id:user_id:event_id"
    )
    event = make_event("evt-9")
    assert event.dedup_key == f"{TENANT}:{USER}:evt-9"


def test_redelivery_is_deduplicated_by_the_ledger_and_the_profile():
    first = make_event("evt-10")
    again = make_event("evt-10")
    ledger = EvidenceLedger()
    assert ledger.admit(first) is True
    assert ledger.admit(again) is False
    accepted, rejected = deduplicate([first, again, make_event("evt-11")])
    assert [e.event_id for e in accepted] == ["evt-10", "evt-11"]
    assert [r.code for r in rejected] == [DUPLICATE_EVENT]

    profile = apply_delta(empty_profile(), infer_profile_delta(first, empty_profile(), now=T0))
    replay = infer_profile_delta(again, profile, now=T0)
    assert replay.accepted is False
    assert replay.rejection.code == DUPLICATE_EVENT


def test_event_for_another_user_is_refused():
    delta = infer_profile_delta(make_event("evt-12", user="someone-else"), empty_profile(), now=T0)
    assert delta.rejection.code == TENANT_MISMATCH


# --------------------------------------------------------------------------
# confidence thresholds — read from the card
# --------------------------------------------------------------------------


def test_thresholds_come_from_the_card(card):
    assert confidence_thresholds(card) == {
        TIER_PROVISIONAL: 0.55,
        TIER_PERSISTENT: 0.75,
        TIER_CONSEQUENTIAL: 0.90,
    }


def test_below_provisional_is_rejected_and_writes_nothing():
    delta = infer_profile_delta(make_event("evt-20", confidence=0.50), empty_profile(), now=T0)
    assert delta.accepted is False
    assert delta.rejection.code == INSUFFICIENT_CONFIDENCE
    assert delta.updates == ()
    with pytest.raises(ValueError, match="rejected delta"):
        apply_delta(empty_profile(), delta)


def test_provisional_evidence_is_session_only():
    delta = infer_profile_delta(make_event("evt-21", confidence=0.60), empty_profile(), now=T0)
    assert delta.accepted and delta.tier == TIER_PROVISIONAL
    assert delta.updates[0].layer == LAYER_EPHEMERAL
    # ``inferred_ephemeral`` authority is session-only, so it is never written as
    # a persistent revision.
    assert delta.persist is False


def test_persistent_threshold_promotes_the_layer():
    delta = infer_profile_delta(make_event("evt-22", confidence=0.80), empty_profile(), now=T0)
    assert delta.tier == TIER_PERSISTENT
    assert delta.updates[0].layer == LAYER_PERSISTENT
    assert delta.persist is True


def test_source_trust_multiplies_into_the_threshold_decision():
    """``evidence_source_weighting``: a confident but untrusted source loses."""
    delta = infer_profile_delta(
        make_event("evt-23", confidence=0.95, trust=0.5), empty_profile(), now=T0
    )
    assert delta.accepted is False
    assert delta.rejection.code == INSUFFICIENT_CONFIDENCE


def test_thresholds_are_read_from_the_card_not_hardcoded(card_text):
    """Tighten the card and the same event stops being accepted."""
    tightened = parse_model_card(
        card_text.replace(
            "      provisional_minimum: 0.55\n"
            "      persistent_minimum: 0.75\n"
            "      consequential_adjustment_minimum: 0.90",
            "      provisional_minimum: 0.85\n"
            "      persistent_minimum: 0.95\n"
            "      consequential_adjustment_minimum: 0.99",
        )
    )
    assert confidence_thresholds(tightened)[TIER_PROVISIONAL] == 0.85
    event = make_event("evt-24", confidence=0.80)
    assert infer_profile_delta(event, empty_profile(), now=T0).accepted is True
    tightened_delta = infer_profile_delta(event, empty_profile(), now=T0, card=tightened)
    assert tightened_delta.accepted is False
    assert tightened_delta.rejection.code == INSUFFICIENT_CONFIDENCE


def test_consequential_adjustment_needs_the_high_threshold(card):
    """An autonomy feature governs unattended action, so 0.75 is not enough."""
    delta = infer_profile_delta(
        make_event("evt-25", "autonomy", "proposal_threshold", 0.8, confidence=0.80),
        empty_profile(),
        now=T0,
    )
    assert delta.rejection.code == INSUFFICIENT_CONFIDENCE
    assert str(card.personalization.inference_update.confidence.consequential_adjustment_minimum) in str(
        delta.rejection.detail
    )


def test_a_single_message_cannot_carry_a_consequential_inference(card):
    """``single_message_high_impact_inference_forbidden``.

    The *impact* is withheld, not the observation: one confident reading lands in
    the session-only advisory layer, and only a second corroborating event makes
    it a consequential, persisted adjustment.
    """
    assert card.tension_gradient.evidence_requirements.minimum_distinct_events == 2
    profile = empty_profile()
    first = infer_profile_delta(
        make_event("evt-26", "autonomy", "proposal_threshold", 0.8, confidence=0.95),
        profile,
        now=T0,
    )
    assert first.tier == TIER_PROVISIONAL
    assert first.updates[0].layer == LAYER_EPHEMERAL
    assert first.persist is False
    assert first.diagnostics["withheld"] == INSUFFICIENT_DISTINCT_EVENTS

    profile = apply_delta(profile, first)
    second = infer_profile_delta(
        make_event("evt-27a", "autonomy", "proposal_threshold", 0.8, confidence=0.95),
        profile,
        now=T0,
    )
    assert second.tier == TIER_CONSEQUENTIAL
    assert second.updates[0].layer == LAYER_PERSISTENT
    assert second.persist is True
    assert second.diagnostics["withheld"] is None


def test_an_explicit_declaration_is_not_an_inference():
    """A user stating a boundary needs no corroborating second event."""
    delta = infer_profile_delta(
        make_event(
            "evt-27",
            "hard_boundaries",
            "preserve_dissent",
            boolean=True,
            confidence=1.0,
            kind="explicit-boundary",
        ),
        empty_profile(),
        now=T0,
    )
    assert delta.accepted is True
    assert delta.tier == TIER_CONSEQUENTIAL
    assert delta.updates[0].layer == LAYER_EXPLICIT
    assert delta.updates[0].rendered_value() is True


# --------------------------------------------------------------------------
# permissions
# --------------------------------------------------------------------------


def test_permission_widening_is_rejected_whole_not_clamped():
    event = make_event("evt-30", confidence=0.99, grants=frozenset({"bridge:apply"}))
    delta = infer_profile_delta(event, empty_profile(), now=T0)
    assert delta.accepted is False
    assert delta.rejection.code == PERMISSION_EXPANSION_FORBIDDEN
    # Nothing survives: not the capability, and not the preference it rode in on.
    assert delta.updates == ()
    assert delta.capability_revocations == frozenset()


def test_narrowing_a_capability_is_allowed():
    event = make_event(
        "evt-31", confidence=0.95, revocations=frozenset({"bridge:apply"})
    )
    delta = infer_profile_delta(event, empty_profile(), now=T0)
    assert delta.accepted is True
    assert delta.capability_revocations == frozenset({"bridge:apply"})
    profile = apply_delta(empty_profile(), delta)
    assert profile.revoked_capabilities == frozenset({"bridge:apply"})
    # The profile has no field that could ever add one back.
    assert not hasattr(profile, "granted_capabilities")


# --------------------------------------------------------------------------
# decay
# --------------------------------------------------------------------------


def test_half_lives_come_from_the_card(card):
    hl = card.personalization.inference_update.decay.half_life_days
    assert (hl.ordinary_preference, hl.workflow_preference, hl.explicit_boundary) == (
        90,
        180,
        None,
    )
    assert half_life_days("communication", LAYER_PERSISTENT, card) == 90
    assert half_life_days("workflow", LAYER_PERSISTENT, card) == 180
    assert half_life_days("hard_boundaries", LAYER_EXPLICIT, card) is None
    assert half_life_days("communication", LAYER_EXPLICIT, card) is None


def test_ordinary_and_workflow_preferences_decay_at_the_card_rate():
    ordinary = infer_profile_delta(
        make_event("evt-40", "communication", "verbosity", 0.9, confidence=0.80),
        empty_profile(),
        now=T0,
    ).updates[0]
    workflow = infer_profile_delta(
        make_event("evt-41", "workflow", "planning_depth", 0.9, confidence=0.80),
        empty_profile(),
        now=T0,
    ).updates[0]
    assert ordinary.confidence_at(T0) == pytest.approx(0.80)
    assert ordinary.confidence_at(T0 + timedelta(days=90)) == pytest.approx(0.40)
    assert workflow.confidence_at(T0 + timedelta(days=90)) == pytest.approx(
        0.80 * 0.5**0.5, abs=1e-6
    )
    assert workflow.confidence_at(T0 + timedelta(days=180)) == pytest.approx(0.40)


def test_explicit_boundaries_never_decay():
    boundary = infer_profile_delta(
        make_event(
            "evt-42",
            "hard_boundaries",
            "preserve_dissent",
            boolean=True,
            confidence=1.0,
            kind="explicit-boundary",
        ),
        empty_profile(),
        now=T0,
    ).updates[0]
    assert boundary.half_life is None
    assert boundary.confidence_at(T0 + timedelta(days=10_000)) == 1.0


def test_decay_widens_the_prior_so_stale_state_yields_faster():
    """Forgetting must reduce certainty, never invent a new opinion.

    Both runs see the same fresh observation; the only difference is how old the
    prior is. The stale prior must move further, because ``intent_update`` was
    handed a larger decay and therefore a wider prior variance.
    """
    profile = apply_delta(
        empty_profile(),
        infer_profile_delta(
            make_event("evt-43", value=0.9, confidence=0.80), empty_profile(), now=T0
        ),
    )
    # Stay inside the contradiction tolerance so this exercises the kernel
    # update rather than the conflict path.
    fresh = infer_profile_delta(
        make_event("evt-44", value=0.6, confidence=0.80, at=T0), profile, now=T0
    )
    stale = infer_profile_delta(
        make_event("evt-45", value=0.6, confidence=0.80, at=T0 + timedelta(days=360)),
        profile,
        now=T0 + timedelta(days=360),
    )
    assert fresh.contradictions == () and stale.contradictions == ()
    assert stale.diagnostics["decay_fraction"] > fresh.diagnostics["decay_fraction"]
    # The stale prior yields further toward the new observation ...
    assert stale.updates[0].value < fresh.updates[0].value
    # ... and says it is less sure, which is what decay is allowed to do.
    assert stale.updates[0].variance > fresh.updates[0].variance


# --------------------------------------------------------------------------
# contradictions
# --------------------------------------------------------------------------


def test_explicit_overrides_inferred_and_the_conflict_is_kept():
    profile = apply_delta(
        empty_profile(),
        infer_profile_delta(
            make_event("evt-50", value=0.9, confidence=0.80), empty_profile(), now=T0
        ),
    )
    delta = infer_profile_delta(
        make_event("evt-51", value=0.1, confidence=0.95, kind="explicit-declaration"),
        profile,
        now=T0,
    )
    assert delta.contradictions[0].resolution == "explicit-overrides-inferred"
    assert delta.contradictions[0].retained == "incoming"
    # A declaration is written through verbatim, not blended with the estimate.
    assert delta.updates[0].value == pytest.approx(0.1)
    assert delta.updates[0].layer == LAYER_EXPLICIT


def test_inferred_evidence_never_overwrites_an_explicit_value():
    profile = apply_delta(
        empty_profile(),
        infer_profile_delta(
            make_event("evt-52", value=0.1, confidence=0.99, kind="explicit-declaration"),
            empty_profile(),
            now=T0,
        ),
    )
    delta = infer_profile_delta(
        make_event("evt-53", value=0.95, confidence=0.99), profile, now=T0
    )
    assert delta.accepted is True
    assert delta.updates == ()  # the explicit value stands
    conflict = delta.contradictions[0]
    assert conflict.resolution == "explicit-overrides-inferred"
    assert conflict.retained == "existing"
    assert conflict.incoming_value == pytest.approx(0.95)
    after = apply_delta(profile, delta)
    assert after.get("communication", "verbosity").value == pytest.approx(0.1)
    assert len(after.contradictions) == 1


def test_recent_high_confidence_overrides_old_low_confidence():
    profile = apply_delta(
        empty_profile(),
        infer_profile_delta(
            make_event("evt-54", value=0.9, confidence=0.76), empty_profile(), now=T0
        ),
    )
    delta = infer_profile_delta(
        make_event("evt-55", value=0.1, confidence=0.95), profile, now=T0
    )
    conflict = delta.contradictions[0]
    assert conflict.resolution == "recent-high-confidence-overrides-old-low-confidence"
    assert conflict.retained == "incoming"
    assert delta.updates[0].value < profile.get("communication", "verbosity").value
    assert delta.updates[0].contested is False


def test_an_unresolved_contradiction_is_retained_and_marks_the_feature_contested():
    """Nothing settles it, so nothing is settled — both readings survive."""
    profile = apply_delta(
        empty_profile(),
        infer_profile_delta(
            make_event("evt-56", value=0.9, confidence=0.80), empty_profile(), now=T0
        ),
    )
    stored = profile.get("communication", "verbosity").value
    delta = infer_profile_delta(
        make_event("evt-57", value=0.1, confidence=0.85), profile, now=T0
    )
    conflict = delta.contradictions[0]
    assert conflict.resolution == "unresolved"
    assert conflict.retained == "both"
    assert conflict.existing_value == pytest.approx(stored)
    assert conflict.incoming_value == pytest.approx(0.1)
    assert delta.updates[0].value == pytest.approx(stored)  # not averaged away
    assert delta.updates[0].contested is True

    after = apply_delta(profile, delta)
    assert after.contested_features == ("communication.verbosity",)
    assert len(after.unresolved_contradictions) == 1
    assert after.to_json()["contradictions"][0]["resolution"] == "unresolved"


def test_contradictions_accumulate_across_revisions():
    profile = empty_profile()
    profile = apply_delta(
        profile,
        infer_profile_delta(make_event("evt-58", value=0.9, confidence=0.80), profile, now=T0),
    )
    for i, value in enumerate((0.1, 0.15), start=59):
        delta = infer_profile_delta(
            make_event(f"evt-{i}", value=value, confidence=0.85), profile, now=T0
        )
        profile = apply_delta(profile, delta)
    assert len(profile.contradictions) == 2
    assert all(c.is_unresolved for c in profile.contradictions)


# --------------------------------------------------------------------------
# auto-fit and append-only history
# --------------------------------------------------------------------------


def test_repeated_compatible_evidence_moves_the_profile(card):
    """``personalization-auto-fit``, without manual approval anywhere."""
    assert card.personalization.automatic_mutation.human_approval_required is False
    profile = empty_profile()
    events = [
        make_event(f"fit-{i}", value=0.92, confidence=0.85, at=T0 + timedelta(minutes=i))
        for i in range(5)
    ]
    profile, deltas = ingest(events, profile, now=T0 + timedelta(minutes=5))
    assert all(d.accepted for d in deltas)
    state = profile.get("communication", "verbosity")
    assert 0.85 < state.value <= 0.92
    assert len(state.event_ids) == 5
    assert profile.revision == 6
    # The kernel is the shared TORX one, and the delta says which path ran.
    assert deltas[0].backend.startswith("torx-")


def test_apply_delta_never_mutates_and_always_moves_forward():
    profile = empty_profile()
    delta = infer_profile_delta(make_event("evt-60"), profile, now=T0)
    updated = apply_delta(profile, delta)
    assert profile.revision == 1 and profile.features == {}
    assert updated.revision == 2
    assert updated.get("communication", "verbosity") is not None
    # Optimistic concurrency: a delta built against an older revision is refused.
    with pytest.raises(ValueError, match="optimistic concurrency"):
        apply_delta(updated, delta)


def test_rollback_creates_a_new_revision_rather_than_rewinding():
    profile = empty_profile()
    profile = apply_delta(
        profile, infer_profile_delta(make_event("evt-61", value=0.9), profile, now=T0)
    )
    checkpoint = profile
    profile = apply_delta(
        profile,
        infer_profile_delta(make_event("evt-62", value=0.2, confidence=0.95), profile, now=T0),
    )
    restored = rollback(profile, checkpoint)
    assert restored.revision == profile.revision + 1
    assert restored.get("communication", "verbosity").value == pytest.approx(
        checkpoint.get("communication", "verbosity").value
    )
    # The evidence is not un-seen: a rolled-back event must not be re-appliable.
    assert restored.applied_event_keys == profile.applied_event_keys
    assert restored.contradictions == profile.contradictions


def test_profile_json_satisfies_the_card_contract(card):
    profile = empty_profile()
    for event in (
        make_event("evt-70", "communication", "verbosity", 0.9, confidence=0.8),
        make_event("evt-71", "workflow", "planning_depth", 0.8, confidence=0.8),
        make_event(
            "evt-72",
            "hard_boundaries",
            "permission_expansion",
            boolean=False,
            confidence=1.0,
            kind="explicit-boundary",
        ),
    ):
        profile = apply_delta(profile, infer_profile_delta(event, profile, now=T0))
    doc = profile.to_json()
    for required in card.jsonb_contracts["user_profile"].required:
        assert required in doc, required
    assert doc["hard_boundaries"] == {"permission_expansion": False}
    assert set(doc["inferred"]) == {"communication", "workflow"}
    assert set(doc["confidence"]) == {"communication", "workflow"}
    assert doc["provenance"]["event_ids"] == ["evt-70", "evt-71", "evt-72"]
    assert profile.profile_hash == UserProfile(**{
        "tenant_id": profile.tenant_id,
        "user_id": profile.user_id,
        "revision": profile.revision,
        "features": profile.features,
        "contradictions": profile.contradictions,
        "applied_event_keys": profile.applied_event_keys,
        "revoked_capabilities": profile.revoked_capabilities,
        "updated_at": profile.updated_at,
    }).profile_hash


def test_parse_evidence_round_trips_a_structured_payload():
    event = parse_evidence(
        {
            "tenant_id": TENANT,
            "user_id": USER,
            "event_id": "evt-80",
            "observed_at": "2026-01-01T12:00:00Z",
            "kind": "tool-outcome",
            "feature": {"domain": "workflow", "name": "verification_frequency", "value": 0.87},
            "source": {"source_id": "runner-3", "channel": "system-telemetry", "trust": 0.9},
            "source_confidence": 0.9,
            "note": "the model said the user seems detail-oriented",
        }
    )
    assert event.observation.key == "workflow.verification_frequency"
    assert event.effective_confidence == pytest.approx(0.81)
    assert event.provenance[0].label == "note"
    assert event.provenance[0].retained_text is None
    assert event.observed_at == T0
