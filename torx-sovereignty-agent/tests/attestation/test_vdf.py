"""Task 6 — Rule 30 VDF prove/verify (``rule30_vdf`` card contract).

These tests pin the two claims the card's ``rule30_vdf.verification`` makes —
``deterministic: true`` and ``offline_verification: true`` — plus the
observability contract: proving and verifying emit the ``vdf.*`` audit spans
carrying ``vdf.proof_id``, and the chain work lands in the
``gc_vdf_attestation_seconds`` histogram rather than being skipped.
"""

from __future__ import annotations

import copy
import datetime as _dt
import uuid

import pytest

from src.attestation.vdf import (
    AttestationError,
    VERIFIER_VERSION,
    clear_verification_failures,
    input_hash,
    prove,
    verification_failures,
    verify,
)
from src.model_card import load_model_card
from src.observability.metrics import MetricRegistry
from src.observability.tracing import (
    ATTR_VDF_PROOF_ID,
    SPAN_ATTRIBUTE_KEYS,
    reset_recorded_spans,
    recorded_spans,
)
from src.personalization.evidence import content_hash

#: Small chain parameters keep the test run fast while still exercising the
#: checkpoint path (multiple checkpoints) and the delay loop.
DIFFICULTY = 128
WIDTH = 64
CHECKPOINT_EVERY = 16


@pytest.fixture(scope="module")
def card():
    return load_model_card()


@pytest.fixture
def registry(card) -> MetricRegistry:
    return MetricRegistry(card, use_otel=False)


@pytest.fixture(autouse=True)
def _fresh_telemetry():
    reset_recorded_spans()
    clear_verification_failures()
    yield
    reset_recorded_spans()
    clear_verification_failures()


@pytest.fixture
def document() -> dict:
    return {
        "event": "bridge-applied",
        "simulated_delta": 0.25,
        "observed_delta": 0.19,
        "confidence": 0.93,
        "nonce": "doc-1",
    }


def make_envelope(document, registry, card, **prove_kwargs) -> dict:
    params = {
        "difficulty": DIFFICULTY,
        "width": WIDTH,
        "checkpoint_every": CHECKPOINT_EVERY,
        "proof_id": "p-1",
        "event_id": "e-1",
        "tenant_id": "acme",
        "registry": registry,
        "card": card,
    }
    params.update(prove_kwargs)
    return prove(document, **params)


# --------------------------------------------------------------------------
# the card contract
# --------------------------------------------------------------------------


def test_the_envelope_carries_exactly_the_required_fields(card, document):
    env = prove(document, difficulty=DIFFICULTY, width=WIDTH)
    expected = {
        "proof_id",
        "event_id",
        "input_hash",
        "seed",
        "difficulty",
        "output",
        "proof",
        "created_at",
        "verifier_version",
    }
    assert set(env) == expected
    assert set(env["proof"]) == {"width", "checkpoint_every", "checkpoints"}
    assert env["verifier_version"] == VERIFIER_VERSION


def test_deterministic_same_document_proves_identically(document, registry, card):
    a = make_envelope(document, registry, card)
    b = make_envelope(document, registry, card)
    assert a["output"] == b["output"]
    assert a["seed"] == b["seed"]
    assert a["proof"]["checkpoints"] == b["proof"]["checkpoints"]
    assert a["input_hash"] == b["input_hash"]


def test_prove_verify_round_trip(document, registry, card):
    env = make_envelope(document, registry, card)
    assert verify(env, document=document, tenant_id="acme", registry=registry, card=card)


def test_offline_verification_needs_only_the_envelope(document, registry, card):
    env = make_envelope(document, registry, card)
    assert verify(env, tenant_id="acme", registry=registry, card=card)


def test_offline_verification_needs_no_registry_or_card(document):
    env = make_envelope(document, registry=None, card=None)
    assert verify(env, tenant_id="acme")


def test_verification_is_deterministic_across_runs(document, registry, card):
    env = make_envelope(document, registry, card)
    assert verify(env, tenant_id="acme", registry=registry, card=card)
    assert verify(env, tenant_id="acme", registry=registry, card=card)


# --------------------------------------------------------------------------
# tamper rejection
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mutation", "with_document"),
    [
        (lambda env: {"output": "0" * 64}, False),
        (lambda env: {"seed": "tampered-seed"}, False),
        (lambda env: {"difficulty": env["difficulty"] + 1}, False),
        (lambda env: {"input_hash": "f" * 64}, True),
    ],
    ids=["output", "seed", "difficulty", "input_hash"],
)
def test_tampered_envelope_fails(mutation, with_document, document, registry, card):
    env = make_envelope(document, registry, card)
    mutated = dict(env)
    mutated.update(mutation(env))
    assert (
        verify(
            mutated,
            document=document if with_document else None,
            tenant_id="acme",
            registry=registry,
            card=card,
        )
        is False
    )


def test_tampered_checkpoint_fails(document, registry, card):
    env = make_envelope(document, registry, card)
    mutated = copy.deepcopy(env)
    mutated["proof"]["checkpoints"][0] = "0" * 64
    assert verify(mutated, tenant_id="acme", registry=registry, card=card) is False


def test_tampered_proof_width_fails(document, registry, card):
    env = make_envelope(document, registry, card)
    mutated = copy.deepcopy(env)
    mutated["proof"]["width"] = WIDTH + 8
    assert verify(mutated, tenant_id="acme", registry=registry, card=card) is False


def test_document_that_hashes_elsewhere_fails(document, registry, card):
    env = make_envelope(document, registry, card)
    other = dict(document, nonce="doc-2")
    assert verify(env, document=other, tenant_id="acme", registry=registry, card=card) is False


def test_missing_required_field_is_an_attestation_error(document, registry, card):
    env = make_envelope(document, registry, card)
    del env["output"]
    with pytest.raises(AttestationError, match="required proof_envelope fields"):
        verify(env, tenant_id="acme", registry=registry, card=card)


def test_malformed_proof_is_an_attestation_error(document, registry, card):
    env = make_envelope(document, registry, card)
    env["proof"] = "not-an-object"
    with pytest.raises(AttestationError, match="must be an object"):
        verify(env, tenant_id="acme", registry=registry, card=card)


# --------------------------------------------------------------------------
# canonical input
# --------------------------------------------------------------------------


def test_excluded_fields_never_enter_the_input_hash(document):
    with_proof = dict(document, vdf_proof={"checkpoints": ["x"] * 8})
    with_handle = dict(document, **{"transient-runtime-handles": {"fd": 7}})
    base = input_hash(document)
    assert input_hash(with_proof) == base
    assert input_hash(with_handle) == base


def test_input_hash_matches_the_shared_content_hash(document):
    assert input_hash(document) == content_hash(document)


def test_proving_does_not_mutate_the_document(document, registry, card):
    original = copy.deepcopy(document)
    make_envelope(document, registry, card)
    assert document == original


# --------------------------------------------------------------------------
# observability contract
# --------------------------------------------------------------------------


def test_success_emits_attested_span_with_proof_id(document, registry, card):
    env = make_envelope(document, registry, card)
    verify(env, document=document, tenant_id="acme", registry=registry, card=card)
    spans = recorded_spans()
    assert any(s.name == "gc.vdf.attested" for s in spans)
    attested = next(s for s in spans if s.name == "gc.vdf.attested")
    assert attested.attributes[ATTR_VDF_PROOF_ID] == "p-1"
    assert attested.error is None


def test_failure_emits_verification_failed_span_with_proof_id(document, registry, card):
    env = make_envelope(document, registry, card)
    mutated = dict(env, output="0" * 64)
    assert verify(mutated, tenant_id="acme", registry=registry, card=card) is False
    spans = recorded_spans()
    assert any(s.name == "gc.vdf.verification_failed" for s in spans)
    failed = next(s for s in spans if s.name == "gc.vdf.verification_failed")
    assert failed.attributes[ATTR_VDF_PROOF_ID] == "p-1"


def test_emitted_spans_carry_the_full_card_attribute_set(document, registry, card):
    env = make_envelope(document, registry, card)
    verify(env, tenant_id="acme", registry=registry, card=card)
    spans = recorded_spans()
    assert spans
    for s in spans:
        assert tuple(s.attributes) == SPAN_ATTRIBUTE_KEYS
        assert s.attributes["tenant.id"] == "acme"


def test_no_span_is_emitted_without_a_tenant(document):
    env = prove(
        document,
        difficulty=DIFFICULTY,
        width=WIDTH,
        checkpoint_every=CHECKPOINT_EVERY,
        proof_id="p-9",
        event_id="e-9",
    )
    verify(env, document=document)
    assert recorded_spans() == ()


def test_attestation_seconds_histogram_moves_on_prove_and_verify(document, registry, card):
    make_envelope(document, registry, card)
    env = prove(
        document,
        difficulty=DIFFICULTY,
        width=WIDTH,
        checkpoint_every=CHECKPOINT_EVERY,
        proof_id="p-2",
        event_id="e-2",
        tenant_id="acme",
        registry=registry,
        card=card,
    )
    verify(env, tenant_id="acme", registry=registry, card=card)
    assert registry.count("gc_vdf_attestation_seconds") >= 2
    points = registry.points("gc_vdf_attestation_seconds")
    assert points
    assert sum(p.total for p in points) > 0


# --------------------------------------------------------------------------
# failure buffer
# --------------------------------------------------------------------------


def test_failed_verifications_are_recorded_in_the_buffer(document, registry, card):
    env = make_envelope(document, registry, card)
    mutated = dict(env, seed="evil")
    assert verify(mutated, tenant_id="acme", registry=registry, card=card) is False
    assert ("p-1", "recomputed Rule 30 chain does not match") in verification_failures()


def test_document_mismatch_records_the_right_reason(document, registry, card):
    env = make_envelope(document, registry, card)
    other = dict(document, nonce="doc-2")
    verify(env, document=other, tenant_id="acme", registry=registry, card=card)
    assert any(
        proof_id == "p-1" and "input_hash" in reason
        for proof_id, reason in verification_failures()
    )


def test_successful_verifications_leave_no_failure_record(document, registry, card):
    env = make_envelope(document, registry, card)
    verify(env, tenant_id="acme", registry=registry, card=card)
    assert verification_failures() == ()


# --------------------------------------------------------------------------
# edge cases
# --------------------------------------------------------------------------


def test_seed_defaults_to_the_input_hash(document, registry, card):
    env = make_envelope(document, registry, card)
    assert env["seed"] == env["input_hash"]


def test_explicit_seed_diverges_the_chain(document, registry, card):
    a = make_envelope(document, registry, card)
    b = make_envelope(document, registry, card, seed="explicit-seed")
    assert a["output"] != b["output"]
    assert verify(b, document=document, tenant_id="acme", registry=registry, card=card)


def test_invalid_parameters_are_rejected():
    with pytest.raises(ValueError, match="difficulty"):
        prove({"x": 1}, difficulty=0)
    with pytest.raises(ValueError, match="width"):
        prove({"x": 1}, width=2)
    with pytest.raises(ValueError, match="checkpoint_every"):
        prove({"x": 1}, checkpoint_every=0)


def test_created_at_is_canonical_utc_z(document):
    env = prove(
        document,
        difficulty=DIFFICULTY,
        width=WIDTH,
        checkpoint_every=CHECKPOINT_EVERY,
        created_at=_dt.datetime(2026, 8, 6, 12, 0, 0, tzinfo=_dt.timezone.utc),
    )
    assert env["created_at"].endswith("Z")
    assert env["created_at"] == "2026-08-06T12:00:00.000000Z"


def test_uuids_are_stringified(document, registry, card):
    env = make_envelope(
        document,
        registry,
        card,
        proof_id=uuid.UUID("12345678-1234-5678-1234-567812345678"),
        event_id=uuid.UUID("87654321-4321-8765-4321-876543218765"),
    )
    assert env["proof_id"] == "12345678-1234-5678-1234-567812345678"
    assert env["event_id"] == "87654321-4321-8765-4321-876543218765"
