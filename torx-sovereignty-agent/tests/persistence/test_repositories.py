"""Task 2 — the persistence invariants, executed rather than described.

Every invariant test runs twice: once against :class:`InMemoryBackend` and once
against PostgreSQL. The PostgreSQL parameter carries ``@pytest.mark.postgres``
and skips with a specific reason when ``TORX_TEST_DATABASE_URL`` is unset or
unreachable, so on a machine with no server the invariants are still genuinely
exercised — by one shared test body, not by a second suite that can drift.

Three classes of guarantee are NOT reachable from the in-memory backend and are
therefore tested only under the ``postgres`` mark, and only under it honestly:
the BEFORE UPDATE/DELETE triggers, the no-gap revision triggers, and row-level
security as an actual privilege boundary. Everything else — monotonic
revisions, optimistic concurrency, append-only, rollback-appends, tenant
scoping, foreign keys, JSONB canonicalisation — runs everywhere.
"""

from __future__ import annotations

import datetime as _dt
import os
import uuid
from pathlib import Path

import pytest

from src.persistence import models
from src.persistence.repositories import (
    AppendOnlyViolation,
    AttestationRepository,
    BridgeRepository,
    ContractViolation,
    ForeignKeyViolation,
    GroupIntentRepository,
    IdempotencyConflict,
    InMemoryBackend,
    ProfileRepository,
    RevisionConflict,
    SqlAlchemyBackend,
    StatusConflict,
    TensionRepository,
    canonical_jsonb,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

T0 = _dt.datetime(2026, 8, 6, 12, 0, 0, tzinfo=_dt.timezone.utc)


def at(seconds: int) -> _dt.datetime:
    return T0 + _dt.timedelta(seconds=seconds)


# --------------------------------------------------------------------------
# Payload builders — every document satisfies its jsonb_contracts entry
# --------------------------------------------------------------------------


def snapshot(user_id: uuid.UUID, revision: int, **overrides) -> dict:
    doc = {
        "user_id": str(user_id),
        "revision": revision,
        "explicit": {"language": "en-US"},
        "inferred": {"communication": {"technicality": 0.94}},
        "hard_boundaries": {"permission_expansion": False, "preserve_dissent": True},
        "confidence": {"communication": 0.93},
        "provenance": {"event_ids": [f"evt-{revision}"]},
    }
    doc.update(overrides)
    return doc


def envelope(proof_id: uuid.UUID, event_id: uuid.UUID) -> dict:
    return {
        "proof_id": str(proof_id),
        "event_id": str(event_id),
        "input_hash": "a" * 64,
        "seed": "b" * 32,
        "difficulty": 1024,
        "output": "c" * 64,
        "proof": "d" * 64,
        "created_at": T0.isoformat(),
        "verifier_version": "rule30-vdf/1",
    }


def simulation(before: float = 0.37, after: float = 0.19) -> dict:
    return {"predicted_before": before, "predicted_after": after}


def authorization(*, authorized: bool = True, violation: bool = False) -> dict:
    return {"boundary_violation": violation, "authorized": authorized}


def new_profile(repo: ProfileRepository, tenant: uuid.UUID, user: uuid.UUID):
    return repo.append_revision(
        tenant,
        user,
        expected_revision=0,
        delta={"explicit": {"language": "en-US"}},
        full_snapshot=snapshot(user, 1),
        source_event_ids=["evt-1"],
        confidence_summary={"communication": 0.93},
        vdf_proof_id=uuid.uuid4(),
        profile_hash="hash-1",
        effective_manifest_hash="manifest-1",
        now=at(0),
    )


def seed_group_intent(
    repo: GroupIntentRepository, tenant: uuid.UUID, group: uuid.UUID, *, dissent=()
):
    return repo.append_intent(
        tenant,
        group,
        expected_revision=0,
        aggregate_intent={"goals": {"ship": 0.9}},
        member_intents={str(uuid.uuid4()): {"ship": 0.8}},
        decision_policy={"rule": "weighted-consent"},
        dissent=list(dissent),
        vdf_proof_id=uuid.uuid4(),
        now=at(0),
    )


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------


def _skip_reason() -> str:
    return (
        "TORX_TEST_DATABASE_URL is not set: the PostgreSQL-backed parameter "
        "needs a live PostgreSQL 15+ instance. Export e.g. "
        "TORX_TEST_DATABASE_URL=postgresql+psycopg://user:pw@localhost/torx_test "
        "to run it."
    )


@pytest.fixture(scope="session")
def postgres_engine():
    url = os.environ.get("TORX_TEST_DATABASE_URL")
    if not url:
        pytest.skip(_skip_reason())
    sa = pytest.importorskip("sqlalchemy")
    from alembic import command
    from alembic.config import Config

    engine = sa.create_engine(url, future=True)
    try:
        with engine.connect() as conn:
            conn.execute(sa.text("SELECT 1"))
    except Exception as exc:  # pragma: no cover - environment dependent
        engine.dispose()
        pytest.skip(f"TORX_TEST_DATABASE_URL is unreachable: {exc}")

    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(REPO_ROOT / "alembic"))
    cfg.set_main_option("sqlalchemy.url", url)
    try:
        command.downgrade(cfg, "base")
    except Exception:  # pragma: no cover - nothing to undo on a clean database
        pass
    command.upgrade(cfg, "head")
    yield engine
    command.downgrade(cfg, "base")
    engine.dispose()


def _truncate(engine) -> None:
    import sqlalchemy as sa

    names = ", ".join(models.metadata.tables)
    with engine.begin() as conn:
        conn.execute(sa.text(f"TRUNCATE {names} CASCADE"))


@pytest.fixture(
    params=[
        pytest.param("in-memory", id="in-memory"),
        pytest.param("postgresql", id="postgresql", marks=pytest.mark.postgres),
    ]
)
def backend(request):
    """One shared body, two storage engines."""
    if request.param == "in-memory":
        return InMemoryBackend()
    engine = request.getfixturevalue("postgres_engine")
    _truncate(engine)
    return SqlAlchemyBackend(engine)


@pytest.fixture
def profiles(backend) -> ProfileRepository:
    return ProfileRepository(backend)


@pytest.fixture
def intents(backend) -> GroupIntentRepository:
    return GroupIntentRepository(backend)


@pytest.fixture
def tensions(backend) -> TensionRepository:
    return TensionRepository(backend)


@pytest.fixture
def bridges(backend) -> BridgeRepository:
    return BridgeRepository(backend)


@pytest.fixture
def attestations(backend) -> AttestationRepository:
    return AttestationRepository(backend)


@pytest.fixture
def tenant() -> uuid.UUID:
    return uuid.uuid4()


@pytest.fixture
def user() -> uuid.UUID:
    return uuid.uuid4()


@pytest.fixture
def group() -> uuid.UUID:
    return uuid.uuid4()


# --------------------------------------------------------------------------
# Monotonic, gapless revisions
# --------------------------------------------------------------------------


def test_first_revision_is_one_and_head_agrees(profiles, tenant, user):
    assert profiles.current_revision(tenant, user) == 0
    revision = new_profile(profiles, tenant, user)

    assert revision.revision == 1
    head = profiles.get_profile(tenant, user)
    assert head is not None
    assert head.current_revision == 1
    assert head.profile == snapshot(user, 1)


def test_revisions_are_dense_and_increasing(profiles, tenant, user):
    new_profile(profiles, tenant, user)
    for n in range(2, 6):
        profiles.append_revision(
            tenant,
            user,
            expected_revision=n - 1,
            delta={"inferred": {"verbosity": 0.1 * n}},
            full_snapshot=snapshot(user, n),
            source_event_ids=[f"evt-{n}"],
            confidence_summary={"communication": 0.9},
            vdf_proof_id=uuid.uuid4(),
            profile_hash=f"hash-{n}",
            effective_manifest_hash=f"manifest-{n}",
            now=at(n),
        )

    numbers = [r.revision for r in profiles.list_revisions(tenant, user)]
    assert numbers == [5, 4, 3, 2, 1], "list_revisions is newest-first and gapless"
    assert profiles.current_revision(tenant, user) == 5


def test_revision_numbers_cannot_be_chosen_by_the_caller(profiles, tenant, user):
    """There is no parameter for it: the next revision is always current + 1."""
    new_profile(profiles, tenant, user)
    second = profiles.append_revision(
        tenant,
        user,
        expected_revision=1,
        delta={"inferred": {}},
        full_snapshot=snapshot(user, 2),
        source_event_ids=["evt-2"],
        confidence_summary={},
        vdf_proof_id=uuid.uuid4(),
        profile_hash="hash-2",
        effective_manifest_hash="manifest-2",
        now=at(2),
    )
    assert second.revision == 2


# --------------------------------------------------------------------------
# Optimistic concurrency
# --------------------------------------------------------------------------


def test_stale_expected_revision_is_refused(profiles, tenant, user):
    new_profile(profiles, tenant, user)
    profiles.append_revision(
        tenant,
        user,
        expected_revision=1,
        delta={"a": 1},
        full_snapshot=snapshot(user, 2),
        source_event_ids=["evt-2"],
        confidence_summary={},
        vdf_proof_id=uuid.uuid4(),
        profile_hash="hash-2",
        effective_manifest_hash="manifest-2",
        now=at(2),
    )

    with pytest.raises(RevisionConflict) as excinfo:
        profiles.append_revision(
            tenant,
            user,
            expected_revision=1,  # read before the second write landed
            delta={"b": 2},
            full_snapshot=snapshot(user, 2),
            source_event_ids=["evt-3"],
            confidence_summary={},
            vdf_proof_id=uuid.uuid4(),
            profile_hash="hash-3",
            effective_manifest_hash="manifest-3",
            now=at(3),
        )

    error = excinfo.value
    assert error.expected == 1
    assert error.actual == 2
    message = str(error).lower()
    # Actionable: says what was expected, what is stored, and what to do next.
    assert "expected revision 1" in message
    assert "revision 2" in message
    assert "re-read" in message and "retry" in message
    # And the refused write left nothing behind.
    assert profiles.current_revision(tenant, user) == 2
    assert len(profiles.list_revisions(tenant, user)) == 2


def test_losing_writer_does_not_corrupt_history(profiles, tenant, user):
    """The refused write must not leave an orphan revision row behind."""
    new_profile(profiles, tenant, user)
    with pytest.raises(RevisionConflict):
        profiles.append_revision(
            tenant,
            user,
            expected_revision=7,
            delta={},
            full_snapshot=snapshot(user, 8),
            source_event_ids=["evt-x"],
            confidence_summary={},
            vdf_proof_id=uuid.uuid4(),
            profile_hash="hash-x",
            effective_manifest_hash="manifest-x",
            now=at(9),
        )
    assert [r.revision for r in profiles.list_revisions(tenant, user)] == [1]


def test_concurrent_creation_of_the_same_profile_is_refused(profiles, tenant, user):
    new_profile(profiles, tenant, user)
    with pytest.raises(RevisionConflict) as excinfo:
        new_profile(profiles, tenant, user)
    assert "expected revision 0" in str(excinfo.value).lower()


def test_group_intent_uses_compare_and_swap(intents, tenant, group):
    seed_group_intent(intents, tenant, group)
    intents.append_intent(
        tenant,
        group,
        expected_revision=1,
        aggregate_intent={"goals": {"ship": 0.95}},
        member_intents={},
        decision_policy={"rule": "weighted-consent"},
        dissent=[],
        vdf_proof_id=uuid.uuid4(),
        now=at(1),
    )
    assert intents.current_revision(tenant, group) == 2

    with pytest.raises(RevisionConflict) as excinfo:
        intents.append_intent(
            tenant,
            group,
            expected_revision=1,
            aggregate_intent={"goals": {"ship": 0.1}},
            member_intents={},
            decision_policy={"rule": "weighted-consent"},
            dissent=[],
            vdf_proof_id=uuid.uuid4(),
            now=at(2),
        )
    assert excinfo.value.actual == 2
    assert "re-aggregate" in str(excinfo.value)
    assert intents.latest(tenant, group).aggregate_intent == {"goals": {"ship": 0.95}}


# --------------------------------------------------------------------------
# Append-only
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "table",
    ["gc_profile_revisions", "gc_group_intents", "gc_tension_snapshots"],
)
def test_history_tables_reject_update_outright(backend, table):
    with pytest.raises(AppendOnlyViolation) as excinfo:
        backend.update(table, {"tenant_id": uuid.uuid4()}, {"delta": {}})
    assert "append-only" in str(excinfo.value).lower()
    assert "new row" in str(excinfo.value) or "supersedes" in str(excinfo.value)


def test_storage_backend_exposes_no_delete(backend):
    """Erasure is an audited out-of-band path, not a verb on the write path."""
    assert not hasattr(backend, "delete")


def test_immutable_columns_are_rejected_even_on_mutable_tables(backend):
    with pytest.raises(AppendOnlyViolation) as excinfo:
        backend.update(
            "gc_vdf_attestations",
            {"tenant_id": uuid.uuid4(), "proof_id": uuid.uuid4()},
            {"envelope": {"tampered": True}},
        )
    assert "envelope" in str(excinfo.value)

    with pytest.raises(AppendOnlyViolation) as excinfo:
        backend.update(
            "gc_bridge_actions",
            {"tenant_id": uuid.uuid4(), "bridge_id": uuid.uuid4()},
            {"simulation": {"predicted_after": 0.0}},
        )
    assert "simulation" in str(excinfo.value)


def test_appending_never_rewrites_an_existing_revision(profiles, tenant, user):
    first = new_profile(profiles, tenant, user)
    profiles.append_revision(
        tenant,
        user,
        expected_revision=1,
        delta={"inferred": {"verbosity": 0.9}},
        full_snapshot=snapshot(user, 2, explicit={"language": "de-DE"}),
        source_event_ids=["evt-2"],
        confidence_summary={"communication": 0.5},
        vdf_proof_id=uuid.uuid4(),
        profile_hash="hash-2",
        effective_manifest_hash="manifest-2",
        now=at(2),
    )
    assert profiles.get_revision(tenant, user, 1) == first


# --------------------------------------------------------------------------
# Rollback appends
# --------------------------------------------------------------------------


def test_rollback_creates_a_new_revision_pointing_at_the_undone_one(
    profiles, tenant, user
):
    new_profile(profiles, tenant, user)
    profiles.append_revision(
        tenant,
        user,
        expected_revision=1,
        delta={"inferred": {"technicality": 0.1}},
        full_snapshot=snapshot(user, 2, inferred={"communication": {"technicality": 0.1}}),
        source_event_ids=["evt-2"],
        confidence_summary={"communication": 0.4},
        vdf_proof_id=uuid.uuid4(),
        profile_hash="hash-2",
        effective_manifest_hash="manifest-2",
        now=at(2),
    )

    undo = profiles.rollback_to(
        tenant,
        user,
        1,
        expected_revision=2,
        source_event_ids=["evt-user-correction"],
        vdf_proof_id=uuid.uuid4(),
        profile_hash="hash-1",
        effective_manifest_hash="manifest-3",
        now=at(3),
    )

    assert undo.revision == 3, "rollback appends, it does not rewind the counter"
    assert undo.supersedes_revision == 2, "supersedes names the revision undone"
    assert undo.is_rollback
    assert undo.full_snapshot == profiles.get_revision(tenant, user, 1).full_snapshot
    assert undo.delta == {"rollback": {"to_revision": 1, "undoes_revision": 2}}

    # Nothing was deleted: all three revisions are still readable.
    assert [r.revision for r in profiles.list_revisions(tenant, user)] == [3, 2, 1]
    assert profiles.get_revision(tenant, user, 2).full_snapshot["inferred"] == {
        "communication": {"technicality": 0.1}
    }
    head = profiles.get_profile(tenant, user)
    assert head.current_revision == 3
    assert head.profile == profiles.get_revision(tenant, user, 1).full_snapshot


def test_a_rollback_can_itself_be_rolled_back(profiles, tenant, user):
    new_profile(profiles, tenant, user)
    profiles.append_revision(
        tenant,
        user,
        expected_revision=1,
        delta={"x": 1},
        full_snapshot=snapshot(user, 2, explicit={"language": "fr-FR"}),
        source_event_ids=["evt-2"],
        confidence_summary={},
        vdf_proof_id=uuid.uuid4(),
        profile_hash="hash-2",
        effective_manifest_hash="manifest-2",
        now=at(2),
    )
    profiles.rollback_to(
        tenant, user, 1,
        expected_revision=2,
        source_event_ids=["evt-undo"],
        vdf_proof_id=uuid.uuid4(),
        profile_hash="hash-1",
        effective_manifest_hash="manifest-3",
        now=at(3),
    )
    redo = profiles.rollback_to(
        tenant, user, 2,
        expected_revision=3,
        source_event_ids=["evt-redo"],
        vdf_proof_id=uuid.uuid4(),
        profile_hash="hash-2",
        effective_manifest_hash="manifest-4",
        now=at(4),
    )
    assert redo.revision == 4
    assert redo.supersedes_revision == 3
    assert redo.full_snapshot["explicit"] == {"language": "fr-FR"}


def test_rollback_with_a_stale_expectation_is_refused(profiles, tenant, user):
    new_profile(profiles, tenant, user)
    with pytest.raises(RevisionConflict):
        profiles.rollback_to(
            tenant, user, 1,
            expected_revision=5,
            source_event_ids=["evt-undo"],
            vdf_proof_id=uuid.uuid4(),
            profile_hash="hash-1",
            effective_manifest_hash="manifest-2",
            now=at(3),
        )
    assert profiles.current_revision(tenant, user) == 1


def test_rollback_to_a_future_revision_is_refused(profiles, tenant, user):
    new_profile(profiles, tenant, user)
    with pytest.raises(ValueError, match="ahead of the current revision"):
        profiles.rollback_to(
            tenant, user, 9,
            expected_revision=1,
            source_event_ids=["evt-undo"],
            vdf_proof_id=uuid.uuid4(),
            profile_hash="hash-1",
            effective_manifest_hash="manifest-2",
            now=at(3),
        )


# --------------------------------------------------------------------------
# Tenant scoping
# --------------------------------------------------------------------------


def test_a_tenant_cannot_read_another_tenants_profile(profiles, user):
    tenant_a, tenant_b = uuid.uuid4(), uuid.uuid4()
    new_profile(profiles, tenant_a, user)

    assert profiles.get_profile(tenant_b, user) is None
    assert profiles.current_revision(tenant_b, user) == 0
    assert profiles.list_revisions(tenant_b, user) == []
    assert profiles.get_revision(tenant_b, user, 1) is None


def test_tenants_keep_independent_revision_chains(profiles, user):
    tenant_a, tenant_b = uuid.uuid4(), uuid.uuid4()
    new_profile(profiles, tenant_a, user)
    profiles.append_revision(
        tenant_a, user,
        expected_revision=1,
        delta={},
        full_snapshot=snapshot(user, 2),
        source_event_ids=["evt-2"],
        confidence_summary={},
        vdf_proof_id=uuid.uuid4(),
        profile_hash="a2",
        effective_manifest_hash="m2",
        now=at(2),
    )
    # The same user id in another tenant starts from scratch.
    b_first = new_profile(profiles, tenant_b, user)

    assert b_first.revision == 1
    assert profiles.current_revision(tenant_a, user) == 2
    assert profiles.current_revision(tenant_b, user) == 1


def test_no_ambient_tenant_exists(profiles, user):
    """Every entry point demands a tenant; there is no default to fall back on."""
    with pytest.raises(TypeError):
        profiles.get_profile(user_id=user)  # type: ignore[call-arg]


# --------------------------------------------------------------------------
# Dissent and uncertainty are stored
# --------------------------------------------------------------------------


def test_dissent_is_stored_verbatim(intents, tenant, group):
    minority = uuid.uuid4()
    dissent = [
        {
            "member_id": str(minority),
            "position": -0.8,
            "magnitude": 0.9,
            "reason": "constraint-collision",
        }
    ]
    seed_group_intent(intents, tenant, group, dissent=dissent)

    stored = intents.latest(tenant, group)
    assert stored.dissent == dissent, "dissent survives the round trip unchanged"


def test_missing_dissent_is_rejected_but_empty_dissent_is_not(intents, tenant, group):
    with pytest.raises(ValueError, match="must not be None"):
        intents.append_intent(
            tenant, group,
            expected_revision=0,
            aggregate_intent={},
            member_intents={},
            decision_policy={},
            dissent=None,  # type: ignore[arg-type]
            vdf_proof_id=uuid.uuid4(),
            now=at(0),
        )
    # [] is a fact ("none observed") and is storable.
    intent = seed_group_intent(intents, tenant, group, dissent=[])
    assert intent.dissent == []


def test_confidence_summary_must_be_present(profiles, tenant, user):
    with pytest.raises(ValueError, match="confidence_summary must not be None"):
        profiles.append_revision(
            tenant, user,
            expected_revision=0,
            delta={},
            full_snapshot=snapshot(user, 1),
            source_event_ids=["evt-1"],
            confidence_summary=None,  # type: ignore[arg-type]
            vdf_proof_id=uuid.uuid4(),
            profile_hash="h",
            effective_manifest_hash="m",
            now=at(0),
        )


def test_a_revision_without_provenance_is_rejected(profiles, tenant, user):
    with pytest.raises(ValueError, match="evidence_provenance_required"):
        profiles.append_revision(
            tenant, user,
            expected_revision=0,
            delta={},
            full_snapshot=snapshot(user, 1),
            source_event_ids=[],
            confidence_summary={},
            vdf_proof_id=uuid.uuid4(),
            profile_hash="h",
            effective_manifest_hash="m",
            now=at(0),
        )
    assert profiles.get_profile(tenant, user) is None


def test_snapshot_must_satisfy_the_card_jsonb_contract(profiles, tenant, user):
    with pytest.raises(ContractViolation, match="jsonb_contracts.user_profile"):
        profiles.append_revision(
            tenant, user,
            expected_revision=0,
            delta={},
            full_snapshot={"user_id": str(user)},
            source_event_ids=["evt-1"],
            confidence_summary={},
            vdf_proof_id=uuid.uuid4(),
            profile_hash="h",
            effective_manifest_hash="m",
            now=at(0),
        )


# --------------------------------------------------------------------------
# Tension snapshots
# --------------------------------------------------------------------------


# Stable, so a replayed snapshot is byte-identical to the original.
MEMBER = "00000000-0000-0000-0000-000000000001"


def record_snapshot(tensions, tenant, group, snapshot_id, **overrides):
    kwargs = dict(
        group_intent_revision=1,
        member_gradients={
            MEMBER: {
                "goal_divergence": 0.18,
                "semantic_misalignment": 0.42,
                "confidence": 0.84,
            }
        },
        group_tension=0.37,
        tension_class="semantic-gap",
        confidence=0.84,
        topology_descriptor={"betti": [1, 0]},
        vdf_proof_id=uuid.uuid4(),
        observed_at=at(1),
    )
    kwargs.update(overrides)
    return tensions.record_snapshot(tenant, group, snapshot_id, **kwargs)


def test_tension_snapshot_round_trips(tensions, intents, tenant, group):
    seed_group_intent(intents, tenant, group)
    snapshot_id = uuid.uuid4()
    written = record_snapshot(tensions, tenant, group, snapshot_id)

    read = tensions.get(tenant, group, snapshot_id)
    assert read == written
    assert tensions.latest(tenant, group) == written
    assert tensions.list_by_class(tenant, "semantic-gap") == [written]
    assert tensions.list_by_class(tenant, "value-conflict") == []


def test_tension_scalars_are_rounded_to_storage_precision(
    tensions, intents, tenant, group
):
    seed_group_intent(intents, tenant, group)
    written = record_snapshot(
        tensions, tenant, group, uuid.uuid4(),
        group_tension=1 / 3, confidence=2 / 3,
    )
    assert written.group_tension == round(1 / 3, 9)
    assert tensions.get(tenant, group, written.snapshot_id).group_tension == round(
        1 / 3, 9
    ), "the value read back is byte-identical to the value written"


def test_unknown_tension_class_is_rejected(tensions, intents, tenant, group):
    seed_group_intent(intents, tenant, group)
    with pytest.raises(ValueError, match="tension_class"):
        record_snapshot(
            tensions, tenant, group, uuid.uuid4(), tension_class="mild-disagreement"
        )


def test_snapshot_must_reference_an_existing_group_intent(tensions, tenant, group):
    with pytest.raises(ForeignKeyViolation):
        record_snapshot(tensions, tenant, group, uuid.uuid4())


def test_replaying_an_identical_snapshot_is_idempotent(
    tensions, intents, tenant, group
):
    seed_group_intent(intents, tenant, group)
    snapshot_id = uuid.uuid4()
    proof = uuid.uuid4()
    first = record_snapshot(tensions, tenant, group, snapshot_id, vdf_proof_id=proof)
    again = record_snapshot(tensions, tenant, group, snapshot_id, vdf_proof_id=proof)
    assert first == again
    assert len(tensions.list_for_group(tenant, group)) == 1

    with pytest.raises(IdempotencyConflict):
        record_snapshot(
            tensions, tenant, group, snapshot_id, vdf_proof_id=proof, group_tension=0.9
        )


# --------------------------------------------------------------------------
# Bridge lifecycle
# --------------------------------------------------------------------------


def propose(bridges, tenant, group, bridge_id, **overrides):
    kwargs = dict(
        group_id=group,
        bridge_type="semantic-translation",
        proposal={"summary": "restate the constraint in the other frame"},
        simulation=simulation(),
        authorization=authorization(),
        vdf_proof_id=uuid.uuid4(),
        now=at(1),
    )
    kwargs.update(overrides)
    return bridges.propose(tenant, bridge_id, **kwargs)


def test_bridge_apply_then_rollback_appends_a_new_action(bridges, tenant, group):
    original_id, undo_id = uuid.uuid4(), uuid.uuid4()
    propose(bridges, tenant, group, original_id)
    applied = bridges.mark_applied(tenant, original_id, applied_at=at(2))
    assert applied.status == "applied"
    assert applied.applied_at == at(2)

    undo = bridges.rollback(
        tenant,
        original_id,
        new_bridge_id=undo_id,
        proposal={"summary": "revert the translation"},
        simulation=simulation(before=0.19, after=0.37),
        authorization=authorization(),
        vdf_proof_id=uuid.uuid4(),
        now=at(3),
    )

    assert undo.rollback_of == original_id
    assert undo.status == "applied"
    # The original is still there, marked rolled-back rather than removed.
    original = bridges.get(tenant, original_id)
    assert original.status == "rolled-back"
    assert original.simulation == simulation()
    assert {a.bridge_id for a in bridges.list_for_group(tenant, group)} == {
        original_id,
        undo_id,
    }


def test_a_bridge_cannot_be_applied_twice(bridges, tenant, group):
    bridge_id = uuid.uuid4()
    propose(bridges, tenant, group, bridge_id)
    bridges.mark_applied(tenant, bridge_id, applied_at=at(2))
    with pytest.raises(StatusConflict) as excinfo:
        bridges.mark_applied(tenant, bridge_id, applied_at=at(3))
    assert excinfo.value.actual == "applied"
    assert "re-read it with get()" in str(excinfo.value).lower()


def test_only_an_applied_bridge_can_be_rolled_back(bridges, tenant, group):
    bridge_id = uuid.uuid4()
    propose(bridges, tenant, group, bridge_id)
    with pytest.raises(StatusConflict, match="only an applied bridge"):
        bridges.rollback(
            tenant, bridge_id,
            new_bridge_id=uuid.uuid4(),
            proposal={},
            simulation=simulation(),
            authorization=authorization(),
            vdf_proof_id=uuid.uuid4(),
            now=at(3),
        )


def test_a_bridge_without_before_and_after_estimates_is_rejected(
    bridges, tenant, group
):
    with pytest.raises(ContractViolation, match="predicted_after"):
        propose(
            bridges, tenant, group, uuid.uuid4(),
            simulation={"predicted_before": 0.4},
        )


def test_a_bridge_without_a_boundary_verdict_is_rejected(bridges, tenant, group):
    with pytest.raises(ContractViolation, match="boundary_violation"):
        propose(
            bridges, tenant, group, uuid.uuid4(), authorization={"authorized": True}
        )


def test_unknown_bridge_strategy_is_rejected(bridges, tenant, group):
    with pytest.raises(ValueError, match="ordered_strategies"):
        propose(bridges, tenant, group, uuid.uuid4(), bridge_type="just-agree")


def test_bridge_status_index_read_is_tenant_scoped(bridges, group):
    tenant_a, tenant_b = uuid.uuid4(), uuid.uuid4()
    propose(bridges, tenant_a, group, uuid.uuid4())
    assert len(bridges.list_by_status(tenant_a, "proposed")) == 1
    assert bridges.list_by_status(tenant_b, "proposed") == []


# --------------------------------------------------------------------------
# Attestations
# --------------------------------------------------------------------------


def test_attestation_round_trips_and_is_findable_by_event(attestations, tenant):
    proof_id, event_id = uuid.uuid4(), uuid.uuid4()
    written = attestations.record(
        tenant, proof_id, event_id=event_id,
        envelope=envelope(proof_id, event_id), now=at(1),
    )
    assert attestations.get(tenant, proof_id) == written
    assert attestations.by_event(tenant, event_id) == [written]
    assert attestations.by_event(uuid.uuid4(), event_id) == []


def test_verification_status_is_the_only_thing_that_can_change(attestations, tenant):
    proof_id, event_id = uuid.uuid4(), uuid.uuid4()
    attestations.record(
        tenant, proof_id, event_id=event_id,
        envelope=envelope(proof_id, event_id), now=at(1),
    )
    verified = attestations.set_verification_status(
        tenant, proof_id, status="verified", expected_status="unverified"
    )
    assert verified.verification_status == "verified"
    assert verified.envelope == envelope(proof_id, event_id)

    with pytest.raises(StatusConflict):
        attestations.set_verification_status(
            tenant, proof_id, status="failed", expected_status="unverified"
        )


def test_envelope_must_carry_every_required_proof_field(attestations, tenant):
    proof_id, event_id = uuid.uuid4(), uuid.uuid4()
    incomplete = envelope(proof_id, event_id)
    del incomplete["input_hash"]
    with pytest.raises(ContractViolation, match="input_hash"):
        attestations.record(
            tenant, proof_id, event_id=event_id, envelope=incomplete, now=at(1)
        )


def test_reusing_a_proof_id_for_a_different_envelope_is_refused(attestations, tenant):
    proof_id, event_id = uuid.uuid4(), uuid.uuid4()
    attestations.record(
        tenant, proof_id, event_id=event_id,
        envelope=envelope(proof_id, event_id), now=at(1),
    )
    # Identical replay: fine (at-least-once delivery).
    attestations.record(
        tenant, proof_id, event_id=event_id,
        envelope=envelope(proof_id, event_id), now=at(1),
    )
    tampered = envelope(proof_id, event_id) | {"output": "0" * 64}
    with pytest.raises(IdempotencyConflict):
        attestations.record(
            tenant, proof_id, event_id=event_id, envelope=tampered, now=at(1)
        )


# --------------------------------------------------------------------------
# JSONB canonicalisation
# --------------------------------------------------------------------------


def test_stored_payloads_are_snapshots_not_references(profiles, tenant, user):
    """A caller mutating its own dict afterwards must not change what was stored."""
    payload = snapshot(user, 1)
    new_profile_kwargs = dict(
        expected_revision=0,
        delta={"explicit": {"language": "en-US"}},
        full_snapshot=payload,
        source_event_ids=["evt-1"],
        confidence_summary={"communication": 0.93},
        vdf_proof_id=uuid.uuid4(),
        profile_hash="h",
        effective_manifest_hash="m",
        now=at(0),
    )
    written = profiles.append_revision(tenant, user, **new_profile_kwargs)

    payload["explicit"]["language"] = "tampered"
    # ...and neither can the record that was handed back.
    written.full_snapshot["explicit"]["language"] = "tampered-via-record"
    assert profiles.get_revision(tenant, user, 1).full_snapshot["explicit"] == {
        "language": "en-US"
    }
    assert profiles.get_profile(tenant, user).profile["explicit"] == {
        "language": "en-US"
    }


def test_canonical_jsonb_rounds_floats_and_rejects_the_unstorable():
    assert canonical_jsonb({"x": 1 / 3}, "d") == {"x": round(1 / 3, 9)}
    assert canonical_jsonb({"xs": (1, 2)}, "d") == {"xs": [1, 2]}

    with pytest.raises(ValueError, match="not representable"):
        canonical_jsonb({"x": float("nan")}, "d")
    with pytest.raises(ValueError, match="keys must be strings"):
        canonical_jsonb({1: "a"}, "d")
    with pytest.raises(ValueError, match="not JSONB-serialisable"):
        canonical_jsonb({"x": object()}, "d")


def test_naive_timestamps_are_rejected(profiles, tenant, user):
    with pytest.raises(ValueError, match="naive"):
        profiles.append_revision(
            tenant, user,
            expected_revision=0,
            delta={},
            full_snapshot=snapshot(user, 1),
            source_event_ids=["evt-1"],
            confidence_summary={},
            vdf_proof_id=uuid.uuid4(),
            profile_hash="h",
            effective_manifest_hash="m",
            now=_dt.datetime(2026, 8, 6, 12, 0, 0),
        )


# --------------------------------------------------------------------------
# Schema agrees with the card
# --------------------------------------------------------------------------


def test_models_match_the_card():
    models.verify_schema_matches_card()


def test_every_declared_table_is_modelled():
    from src.model_card.loader import cached_model_card

    card = cached_model_card()
    assert set(models.TABLE_MODELS) == set(card.jsonb_persistence.tables)


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (
            lambda tables: tables["gc_user_profiles"]["columns"].update(
                {"shadow_profile": "jsonb"}
            ),
            "shadow_profile",
        ),
        (
            lambda tables: tables["gc_profile_revisions"]["columns"].update(
                {"revision": "text"}
            ),
            "revision",
        ),
        (
            lambda tables: tables["gc_tension_snapshots"]["indexes"].append(
                "gin(topology_descriptor jsonb_path_ops)"
            ),
            "topology_descriptor",
        ),
        (
            lambda tables: tables["gc_bridge_actions"].__setitem__(
                "primary_key", ["tenant_id", "group_id"]
            ),
            "primary key",
        ),
    ],
)
def test_card_drift_is_detected(mutate, expected):
    """Edit the card, and the models stop agreeing — loudly, and with a diff."""
    import yaml

    from src.model_card.loader import DEFAULT_CARD_PATH, parse_model_card

    raw = yaml.safe_load(DEFAULT_CARD_PATH.read_text(encoding="utf-8"))
    mutate(raw["jsonb_persistence"]["tables"])
    drifted = parse_model_card(yaml.safe_dump(raw), source="<drifted>")

    with pytest.raises(models.SchemaCardMismatch) as excinfo:
        models.verify_schema_matches_card(drifted)
    assert expected in str(excinfo.value)


# --------------------------------------------------------------------------
# Migration — verified offline, without a server
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def offline_ddl() -> str:
    """Render the whole migration as SQL. No connection is opened."""
    import contextlib
    import io

    from alembic import command
    from alembic.config import Config

    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(REPO_ROOT / "alembic"))
    cfg.set_main_option("sqlalchemy.url", "postgresql+psycopg://torx@localhost/torx")
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        command.upgrade(cfg, "head", sql=True)
    return buffer.getvalue()


def test_migration_creates_every_table_and_column(offline_ddl):
    for name, model in models.TABLE_MODELS.items():
        assert f"CREATE TABLE {name} (" in offline_ddl
        body = offline_ddl.split(f"CREATE TABLE {name} (", 1)[1].split(");", 1)[0]
        for column in model.__table__.c:
            # "authorization" is quoted in the DDL; compare on the bare name.
            assert str(column.name) in body, f"{name}.{column.name} missing"


def test_migration_uses_the_declared_postgres_types(offline_ddl):
    profiles = offline_ddl.split("CREATE TABLE gc_user_profiles (", 1)[1].split(");")[0]
    assert "tenant_id UUID NOT NULL" in profiles
    assert "profile JSONB NOT NULL" in profiles
    assert "current_revision BIGINT NOT NULL" in profiles
    assert "created_at TIMESTAMP WITH TIME ZONE" in profiles

    tension = offline_ddl.split("CREATE TABLE gc_tension_snapshots (", 1)[1].split(");")[0]
    assert "group_tension NUMERIC(12, 9) NOT NULL" in tension


def test_migration_creates_every_declared_index(offline_ddl):
    for model in models.TABLE_MODELS.values():
        for index in model.__table__.indexes:
            assert f"CREATE INDEX {index.name} " in offline_ddl, index.name
    assert offline_ddl.count("USING gin") == 6, "one GIN index per table"
    assert offline_ddl.count("jsonb_path_ops") == 6
    assert "revision DESC" in offline_ddl
    assert "observed_at DESC" in offline_ddl


def test_migration_enables_row_level_security_on_every_table(offline_ddl):
    for name in models.TABLE_MODELS:
        assert f"ALTER TABLE {name} ENABLE ROW LEVEL SECURITY;" in offline_ddl
        # Without FORCE the owner bypasses the policy, and the app is the owner.
        assert f"ALTER TABLE {name} FORCE ROW LEVEL SECURITY;" in offline_ddl
        assert f"CREATE POLICY {name}_tenant_isolation ON {name}" in offline_ddl
    assert offline_ddl.count("current_setting('app.tenant_id', true)::uuid") == 12
    assert offline_ddl.count("WITH CHECK") == 6, "writes are scoped, not only reads"


def test_migration_installs_the_append_only_triggers(offline_ddl):
    for name in ("gc_profile_revisions", "gc_group_intents", "gc_tension_snapshots"):
        assert f"CREATE TRIGGER {name}_append_only" in offline_ddl
        block = offline_ddl.split(f"CREATE TRIGGER {name}_append_only", 1)[1][:200]
        assert "BEFORE UPDATE OR DELETE" in block
        assert f"ON {name}" in block
    assert "gc_reject_history_mutation()" in offline_ddl
    assert "ERRCODE = 'GC001'" in offline_ddl

    # The two narrowly-mutable tables reject DELETE and every other UPDATE.
    for name in ("gc_bridge_actions", "gc_vdf_attestations"):
        assert f"CREATE TRIGGER {name}_append_only" in offline_ddl
    assert 'NEW."authorization" IS DISTINCT FROM OLD."authorization"' in offline_ddl
    assert "only verification_status may change" in offline_ddl


def test_migration_installs_the_no_gap_revision_triggers(offline_ddl):
    assert "CREATE TRIGGER gc_profile_revisions_dense" in offline_ddl
    assert "CREATE TRIGGER gc_group_intents_dense" in offline_ddl
    assert "COALESCE(MAX(revision), 0) + 1" in offline_ddl
    assert "must be gapless" in offline_ddl


def test_migration_downgrade_is_renderable():
    """A migration that cannot be undone is a migration nobody dares apply."""
    import contextlib
    import io

    from alembic import command
    from alembic.config import Config

    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(REPO_ROOT / "alembic"))
    cfg.set_main_option("sqlalchemy.url", "postgresql+psycopg://torx@localhost/torx")
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        command.downgrade(cfg, "001_agent_personalization:base", sql=True)
    sql = buffer.getvalue()
    for name in models.TABLE_MODELS:
        assert f"DROP TABLE {name}" in sql
        assert f"DROP POLICY IF EXISTS {name}_tenant_isolation" in sql
    assert sql.index("DROP TABLE gc_tension_snapshots") < sql.index(
        "DROP TABLE gc_group_intents"
    ), "children drop before parents"


# --------------------------------------------------------------------------
# The PostgreSQL backend's SQL, checked without a server
# --------------------------------------------------------------------------
#
# The invariant tests above skip their postgresql parameter when no database is
# reachable, which would leave SqlAlchemyBackend's statement construction —
# column names, the quoted "authorization" identifier, the Decimal conversion of
# NUMERIC columns — completely unexercised on a machine like CI-without-postgres.
# These tests drive the real backend against a connection that compiles each
# statement for the PostgreSQL dialect and throws it away.


class _StubResult:
    rowcount = 1

    def __iter__(self):
        return iter(())

    def scalar_one(self):  # pragma: no cover - not reached by these tests
        return None


class _CompileOnlyConnection:
    """Compiles every statement for PostgreSQL instead of sending it."""

    def __init__(self) -> None:
        from sqlalchemy.dialects import postgresql

        self._dialect = postgresql.dialect()
        self.sql: list[str] = []
        self.params: list[dict] = []

    def execute(self, statement, parameters=None):
        compiled = statement.compile(dialect=self._dialect)
        self.sql.append(str(compiled))
        # Statement-embedded binds for insert/update/select, overridden by any
        # explicitly-passed parameters (that is how set_config is invoked).
        self.params.append(
            {**(dict(getattr(compiled, "params", {}) or {})), **(parameters or {})}
        )
        return _StubResult()

    def begin(self):
        return self

    def commit(self) -> None:
        pass

    def rollback(self) -> None:  # pragma: no cover - no failures in these tests
        pass

    def close(self) -> None:
        pass


class _CompileOnlyEngine:
    def __init__(self, connection: _CompileOnlyConnection) -> None:
        self._connection = connection

    def connect(self):
        return self._connection


@pytest.fixture
def compile_only():
    connection = _CompileOnlyConnection()
    return connection, SqlAlchemyBackend(_CompileOnlyEngine(connection))


def test_postgres_backend_sets_the_tenant_for_row_level_security(
    compile_only, tenant, user
):
    connection, backend = compile_only
    ProfileRepository(backend).get_profile(tenant, user)
    assert "set_config('app.tenant_id'" in connection.sql[0]
    assert connection.params[0]["tid"] == str(tenant)
    assert "FROM gc_user_profiles" in connection.sql[1]


def test_postgres_backend_insert_compiles_for_every_table(compile_only, tenant, user):
    connection, backend = compile_only
    ProfileRepository(backend).append_revision(
        tenant, user,
        expected_revision=0,
        delta={"explicit": {}},
        full_snapshot=snapshot(user, 1),
        source_event_ids=["evt-1"],
        confidence_summary={"communication": 0.93},
        vdf_proof_id=uuid.uuid4(),
        profile_hash="h",
        effective_manifest_hash="m",
        now=at(0),
    )
    sql = " ".join(connection.sql)
    assert "INSERT INTO gc_user_profiles" in sql
    assert "INSERT INTO gc_profile_revisions" in sql


def test_postgres_backend_quotes_the_authorization_column(compile_only, tenant, group):
    connection, backend = compile_only
    propose(BridgeRepository(backend), tenant, group, uuid.uuid4())
    statement = next(s for s in connection.sql if "INSERT INTO gc_bridge_actions" in s)
    # AUTHORIZATION is a PostgreSQL keyword; unquoted it is a syntax error.
    assert '"authorization"' in statement


def test_postgres_backend_binds_numerics_as_decimals(compile_only, tenant, group):
    from decimal import Decimal

    connection, backend = compile_only
    record_snapshot(TensionRepository(backend), tenant, group, uuid.uuid4(),
                    group_tension=1 / 3)
    index = next(
        i for i, s in enumerate(connection.sql)
        if "INSERT INTO gc_tension_snapshots" in s
    )
    bound = connection.params[index]["group_tension"]
    assert isinstance(bound, Decimal)
    assert bound == Decimal(str(round(1 / 3, 9)))


def test_postgres_backend_update_is_a_compare_and_swap(compile_only, tenant):
    connection, backend = compile_only
    proof_id = uuid.uuid4()
    with backend.unit_of_work(tenant):
        backend.update(
            "gc_vdf_attestations",
            {"tenant_id": tenant, "proof_id": proof_id},
            {"verification_status": "verified"},
            expected={"verification_status": "unverified"},
        )
    statement = connection.sql[-1]
    assert statement.startswith("UPDATE gc_vdf_attestations SET verification_status")
    assert "WHERE" in statement and "verification_status =" in statement.split("WHERE")[1]


def test_postgres_backend_refuses_to_run_outside_a_unit_of_work(compile_only, tenant):
    from src.persistence.repositories import TenantScopeError

    _, backend = compile_only
    with pytest.raises(TenantScopeError, match="app.tenant_id"):
        backend.fetch_all("gc_user_profiles", {"tenant_id": tenant})


def test_a_unit_of_work_cannot_change_tenant_midway(backend, tenant):
    from src.persistence.repositories import TenantScopeError

    with backend.unit_of_work(tenant):
        with pytest.raises(TenantScopeError, match="already open for tenant"):
            with backend.unit_of_work(uuid.uuid4()):
                pass


# --------------------------------------------------------------------------
# Property: any legal write sequence keeps the chain dense
# --------------------------------------------------------------------------


def test_arbitrary_write_sequences_stay_dense():
    from hypothesis import HealthCheck, given, settings
    from hypothesis import strategies as st

    @given(count=st.integers(min_value=1, max_value=12))
    @settings(max_examples=25, deadline=None,
              suppress_health_check=[HealthCheck.function_scoped_fixture])
    def property_body(count: int) -> None:
        repo = ProfileRepository(InMemoryBackend())
        tenant, user = uuid.uuid4(), uuid.uuid4()
        for n in range(1, count + 1):
            repo.append_revision(
                tenant, user,
                expected_revision=n - 1,
                delta={"step": n},
                full_snapshot=snapshot(user, n),
                source_event_ids=[f"evt-{n}"],
                confidence_summary={},
                vdf_proof_id=uuid.uuid4(),
                profile_hash=f"h{n}",
                effective_manifest_hash=f"m{n}",
                now=at(n),
            )
        revisions = [r.revision for r in repo.list_revisions(tenant, user)]
        assert revisions == list(range(count, 0, -1))
        assert repo.current_revision(tenant, user) == count

    property_body()


# --------------------------------------------------------------------------
# PostgreSQL-only: the guards that only a server can enforce
# --------------------------------------------------------------------------


@pytest.mark.postgres
def test_db_trigger_blocks_a_raw_update_of_history(postgres_engine, tenant, user):
    """The append-only guard has to survive a caller that skips this module."""
    import sqlalchemy as sa

    _truncate(postgres_engine)
    repo = ProfileRepository(SqlAlchemyBackend(postgres_engine))
    new_profile(repo, tenant, user)

    with pytest.raises(sa.exc.DatabaseError) as excinfo:
        with postgres_engine.begin() as conn:
            conn.execute(
                sa.text("SELECT set_config('app.tenant_id', :t, true)"),
                {"t": str(tenant)},
            )
            conn.execute(
                sa.text(
                    "UPDATE gc_profile_revisions SET delta = '{\"tampered\": true}' "
                    "WHERE tenant_id = :t AND user_id = :u AND revision = 1"
                ),
                {"t": str(tenant), "u": str(user)},
            )
    assert excinfo.value.orig.sqlstate == "GC001"


@pytest.mark.postgres
def test_db_trigger_blocks_a_raw_delete_of_history(postgres_engine, tenant, user):
    import sqlalchemy as sa

    _truncate(postgres_engine)
    repo = ProfileRepository(SqlAlchemyBackend(postgres_engine))
    new_profile(repo, tenant, user)

    with pytest.raises(sa.exc.DatabaseError) as excinfo:
        with postgres_engine.begin() as conn:
            conn.execute(
                sa.text("SELECT set_config('app.tenant_id', :t, true)"),
                {"t": str(tenant)},
            )
            conn.execute(
                sa.text(
                    "DELETE FROM gc_profile_revisions WHERE tenant_id = :t "
                    "AND user_id = :u"
                ),
                {"t": str(tenant), "u": str(user)},
            )
    assert excinfo.value.orig.sqlstate == "GC001"


@pytest.mark.postgres
def test_db_trigger_blocks_a_gap_in_the_revision_chain(postgres_engine, tenant, user):
    import sqlalchemy as sa

    _truncate(postgres_engine)
    repo = ProfileRepository(SqlAlchemyBackend(postgres_engine))
    new_profile(repo, tenant, user)

    with pytest.raises(sa.exc.DatabaseError) as excinfo:
        with postgres_engine.begin() as conn:
            conn.execute(
                sa.text("SELECT set_config('app.tenant_id', :t, true)"),
                {"t": str(tenant)},
            )
            conn.execute(
                sa.text(
                    "INSERT INTO gc_profile_revisions (tenant_id, user_id, "
                    "revision, source_event_ids, delta, full_snapshot, "
                    "confidence_summary, vdf_proof_id, created_at) VALUES "
                    "(:t, :u, 9, '[\"e\"]', '{}', '{}', '{}', :p, now())"
                ),
                {"t": str(tenant), "u": str(user), "p": str(uuid.uuid4())},
            )
    assert excinfo.value.orig.sqlstate == "GC001"
    assert "gapless" in str(excinfo.value)


@pytest.mark.postgres
def test_row_level_security_hides_other_tenants(postgres_engine, user):
    """Not just the WHERE clause: the database itself refuses to show the row."""
    import sqlalchemy as sa

    _truncate(postgres_engine)
    tenant_a, tenant_b = uuid.uuid4(), uuid.uuid4()
    repo = ProfileRepository(SqlAlchemyBackend(postgres_engine))
    new_profile(repo, tenant_a, user)

    with postgres_engine.begin() as conn:
        conn.execute(
            sa.text("SELECT set_config('app.tenant_id', :t, true)"),
            {"t": str(tenant_b)},
        )
        # An unqualified select, deliberately: only RLS is standing between
        # tenant B and tenant A's profile here.
        visible = conn.execute(sa.text("SELECT count(*) FROM gc_user_profiles"))
        assert visible.scalar_one() == 0

    with postgres_engine.begin() as conn:
        conn.execute(
            sa.text("SELECT set_config('app.tenant_id', :t, true)"),
            {"t": str(tenant_a)},
        )
        visible = conn.execute(sa.text("SELECT count(*) FROM gc_user_profiles"))
        assert visible.scalar_one() == 1


@pytest.mark.postgres
def test_a_session_without_a_tenant_sees_nothing(postgres_engine, tenant, user):
    import sqlalchemy as sa

    _truncate(postgres_engine)
    repo = ProfileRepository(SqlAlchemyBackend(postgres_engine))
    new_profile(repo, tenant, user)

    with postgres_engine.begin() as conn:
        rows = conn.execute(sa.text("SELECT count(*) FROM gc_user_profiles"))
        assert rows.scalar_one() == 0, (
            "current_setting('app.tenant_id', true) is NULL, so the policy "
            "predicate is NULL and no row qualifies"
        )


@pytest.mark.postgres
def test_row_level_security_blocks_writing_into_another_tenant(
    postgres_engine, tenant, user
):
    import sqlalchemy as sa

    _truncate(postgres_engine)
    other = uuid.uuid4()
    with pytest.raises(sa.exc.DatabaseError):
        with postgres_engine.begin() as conn:
            conn.execute(
                sa.text("SELECT set_config('app.tenant_id', :t, true)"),
                {"t": str(tenant)},
            )
            conn.execute(
                sa.text(
                    "INSERT INTO gc_user_profiles (tenant_id, user_id, "
                    "current_revision, profile, profile_hash, "
                    "effective_manifest_hash) VALUES "
                    "(:other, :u, 1, '{}', 'h', 'm')"
                ),
                {"other": str(other), "u": str(user)},
            )
