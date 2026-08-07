"""SQLAlchemy schema for the card's ``jsonb_persistence`` tables.

The card is the schema's source of truth, not its documentation. Every column,
composite primary key, and index below is transcribed from
``jsonb_persistence.tables``, and :func:`verify_schema_matches_card` runs at
import time so a card edit and a schema edit cannot silently diverge — a
forgotten column would otherwise surface as a runtime ``UndefinedColumn`` in
production rather than an ImportError in CI.

Three design decisions are worth stating because they are not obvious from the
column list:

* **``vdf_proof_id`` is not a foreign key.** ``availability.vdf_attestation``
  makes attestation *asynchronous* for ordinary-preference learning and offline
  compaction. A referential constraint to ``gc_vdf_attestations`` would force
  every profile revision to wait for its Rule 30 proof before it could be
  written, converting an availability property into a hard dependency. The proof
  id is allocated by the writer up front; the envelope lands later.

* **``supersedes_revision`` and ``rollback_of`` are self-referential composite
  foreign keys.** Both are nullable, and PostgreSQL's default ``MATCH SIMPLE``
  skips the check when any column of the key is NULL, so "this revision undoes
  nothing" stays representable while "this revision undoes revision 9" is
  verified to actually point at revision 9 of the same user in the same tenant.

* **Some declared indexes duplicate the primary-key index**
  (``btree(tenant_id, user_id)`` on ``gc_user_profiles``). They are kept rather
  than optimised away because the schema check compares against the card, and a
  silent omission here would be indistinguishable from a transcription bug. The
  cost is write amplification on one table; the benefit is that the card and the
  database say the same thing.

Row-level security, the append-only triggers, and the no-gap revision trigger
live in the Alembic migration: they are DDL that SQLAlchemy's declarative layer
does not model, and they are the guards that survive a buggy caller.
"""

from __future__ import annotations

import datetime as _dt
import re
import uuid as _uuid
from typing import Any

from sqlalchemy import (
    TIMESTAMP,
    BigInteger,
    CheckConstraint,
    Column,
    ForeignKeyConstraint,
    Index,
    MetaData,
    Numeric,
    PrimaryKeyConstraint,
    Text,
    func,
)
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.sql import operators, quoted_name
from sqlalchemy.sql.elements import UnaryExpression

from src.model_card.loader import cached_model_card
from src.model_card.types import AgentModelCard
from src.torx_layer.state import STORAGE_PRECISION

# NUMERIC columns hold TORX scalars that are also fed to the Rule 30 canonical
# JSON. state.STORAGE_PRECISION is the rounding the whole repo uses, so pinning
# the DB scale to it makes "what the sampler produced" and "what the database
# returns" the same number rather than two numbers that usually agree.
NUMERIC_PRECISION = 12
NUMERIC_SCALE = STORAGE_PRECISION

# Named so Alembic autogenerate produces stable diffs instead of renaming
# constraints on every run.
NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_N_name)s",
    "pk": "pk_%(table_name)s",
}


class SchemaCardMismatch(ValueError):
    """Raised when the declarative models disagree with the model card."""


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


def _uuid_col(**kw: Any) -> Mapped[Any]:
    return mapped_column(postgresql.UUID(as_uuid=True), **kw)


def _jsonb_col(**kw: Any) -> Mapped[Any]:
    return mapped_column(postgresql.JSONB, **kw)


def _ts_col(**kw: Any) -> Mapped[Any]:
    return mapped_column(TIMESTAMP(timezone=True), **kw)


# --------------------------------------------------------------------------
# gc_user_profiles — the mutable head of each user's revision chain
# --------------------------------------------------------------------------


class UserProfile(Base):
    """Current effective profile for one user in one tenant.

    This is the only profile row that is ever UPDATEd, and the update is the
    concurrency gate: ``current_revision`` is the compare-and-swap target that
    ``deployment.consistency.profile_revision: optimistic-concurrency-control``
    requires. Everything historical lives in :class:`ProfileRevision`.
    """

    __tablename__ = "gc_user_profiles"

    tenant_id: Mapped[_uuid.UUID] = _uuid_col(nullable=False)
    user_id: Mapped[_uuid.UUID] = _uuid_col(nullable=False)
    current_revision: Mapped[int] = mapped_column(BigInteger, nullable=False)
    profile: Mapped[dict[str, Any]] = _jsonb_col(nullable=False)
    profile_hash: Mapped[str] = mapped_column(Text, nullable=False)
    effective_manifest_hash: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[_dt.datetime] = _ts_col(
        nullable=False, server_default=func.now()
    )
    updated_at: Mapped[_dt.datetime] = _ts_col(
        nullable=False, server_default=func.now()
    )

    __table_args__ = (
        PrimaryKeyConstraint("tenant_id", "user_id"),
        # Revision 0 means "no profile"; a head row that exists is at 1 or more.
        CheckConstraint("current_revision >= 1", name="current_revision_positive"),
    )


# --------------------------------------------------------------------------
# gc_profile_revisions — append-only history
# --------------------------------------------------------------------------


class ProfileRevision(Base):
    """One immutable step in a user's profile history.

    ``invariants.append_only_revision_history`` is enforced three times over: the
    repository refuses to update, the storage backend refuses to update, and a
    BEFORE UPDATE OR DELETE trigger in the migration raises. Rollback
    (``user_controls.rollback``) appends a new revision whose ``full_snapshot``
    is the target's and whose ``supersedes_revision`` names the revision being
    undone, so the undo is itself part of the audit trail.
    """

    __tablename__ = "gc_profile_revisions"

    tenant_id: Mapped[_uuid.UUID] = _uuid_col(nullable=False)
    user_id: Mapped[_uuid.UUID] = _uuid_col(nullable=False)
    revision: Mapped[int] = mapped_column(BigInteger, nullable=False)
    # invariants.evidence_provenance_required: a revision with no attributable
    # source event is not a revision, it is an unexplained mutation.
    source_event_ids: Mapped[list[str]] = _jsonb_col(nullable=False)
    delta: Mapped[dict[str, Any]] = _jsonb_col(nullable=False)
    full_snapshot: Mapped[dict[str, Any]] = _jsonb_col(nullable=False)
    # invariants.preserve_inference_uncertainty: the confidence that justified
    # the mutation is stored with it, not recomputed later from the result.
    confidence_summary: Mapped[dict[str, Any]] = _jsonb_col(nullable=False)
    supersedes_revision: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    vdf_proof_id: Mapped[_uuid.UUID] = _uuid_col(nullable=False)
    created_at: Mapped[_dt.datetime] = _ts_col(
        nullable=False, server_default=func.now()
    )

    __table_args__ = (
        PrimaryKeyConstraint("tenant_id", "user_id", "revision"),
        ForeignKeyConstraint(
            ["tenant_id", "user_id"],
            ["gc_user_profiles.tenant_id", "gc_user_profiles.user_id"],
            name="fk_gc_profile_revisions_profile",
        ),
        # A rollback points backwards, never forwards or at itself.
        ForeignKeyConstraint(
            ["tenant_id", "user_id", "supersedes_revision"],
            [
                "gc_profile_revisions.tenant_id",
                "gc_profile_revisions.user_id",
                "gc_profile_revisions.revision",
            ],
            name="fk_gc_profile_revisions_supersedes",
        ),
        CheckConstraint("revision >= 1", name="revision_positive"),
        CheckConstraint(
            "supersedes_revision IS NULL OR supersedes_revision < revision",
            name="supersedes_is_backwards",
        ),
    )


# --------------------------------------------------------------------------
# gc_group_intents — append-only group intent history
# --------------------------------------------------------------------------


class GroupIntent(Base):
    """One revision of a group's aggregated intent.

    ``dissent`` is NOT NULL by design. An absent dissent column and an empty
    dissent list are different claims — "nobody disagreed" versus "we did not
    record who disagreed" — and ``no_false_consensus`` only means something if
    the second one cannot be written.
    """

    __tablename__ = "gc_group_intents"

    tenant_id: Mapped[_uuid.UUID] = _uuid_col(nullable=False)
    group_id: Mapped[_uuid.UUID] = _uuid_col(nullable=False)
    revision: Mapped[int] = mapped_column(BigInteger, nullable=False)
    aggregate_intent: Mapped[dict[str, Any]] = _jsonb_col(nullable=False)
    member_intents: Mapped[dict[str, Any]] = _jsonb_col(nullable=False)
    decision_policy: Mapped[dict[str, Any]] = _jsonb_col(nullable=False)
    dissent: Mapped[Any] = _jsonb_col(nullable=False)
    vdf_proof_id: Mapped[_uuid.UUID] = _uuid_col(nullable=False)
    created_at: Mapped[_dt.datetime] = _ts_col(
        nullable=False, server_default=func.now()
    )

    __table_args__ = (
        PrimaryKeyConstraint("tenant_id", "group_id", "revision"),
        CheckConstraint("revision >= 1", name="revision_positive"),
    )


# --------------------------------------------------------------------------
# gc_tension_snapshots — immutable observations
# --------------------------------------------------------------------------


class TensionSnapshot(Base):
    """A measured tension gradient bound to the group intent it was measured against.

    ``group_intent_revision`` is a real foreign key: a snapshot that cannot name
    the intent it was computed from cannot support ``STALE_INTENT_REVISION``
    detection at bridge-apply time, which is what
    ``deployment.consistency.bridge_apply: snapshot-bound`` relies on.
    """

    __tablename__ = "gc_tension_snapshots"

    tenant_id: Mapped[_uuid.UUID] = _uuid_col(nullable=False)
    group_id: Mapped[_uuid.UUID] = _uuid_col(nullable=False)
    snapshot_id: Mapped[_uuid.UUID] = _uuid_col(nullable=False)
    group_intent_revision: Mapped[int] = mapped_column(BigInteger, nullable=False)
    member_gradients: Mapped[dict[str, Any]] = _jsonb_col(nullable=False)
    group_tension: Mapped[Any] = mapped_column(
        Numeric(NUMERIC_PRECISION, NUMERIC_SCALE), nullable=False
    )
    tension_class: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[Any] = mapped_column(
        Numeric(NUMERIC_PRECISION, NUMERIC_SCALE), nullable=False
    )
    topology_descriptor: Mapped[dict[str, Any]] = _jsonb_col(nullable=False)
    vdf_proof_id: Mapped[_uuid.UUID] = _uuid_col(nullable=False)
    observed_at: Mapped[_dt.datetime] = _ts_col(
        nullable=False, server_default=func.now()
    )

    __table_args__ = (
        PrimaryKeyConstraint("tenant_id", "group_id", "snapshot_id"),
        ForeignKeyConstraint(
            ["tenant_id", "group_id", "group_intent_revision"],
            [
                "gc_group_intents.tenant_id",
                "gc_group_intents.group_id",
                "gc_group_intents.revision",
            ],
            name="fk_gc_tension_snapshots_group_intent",
        ),
        CheckConstraint(
            "group_tension >= 0 AND group_tension <= 1", name="group_tension_ratio"
        ),
        CheckConstraint("confidence >= 0 AND confidence <= 1", name="confidence_ratio"),
    )


# --------------------------------------------------------------------------
# gc_bridge_actions
# --------------------------------------------------------------------------


class BridgeAction(Base):
    """A proposed, applied, rejected, or rolled-back bridge.

    ``status`` and ``applied_at`` are the only mutable columns; the trigger in
    the migration rejects any other UPDATE and every DELETE. A rollback is a new
    row whose ``rollback_of`` names the action it undoes, mirroring the profile
    rollback shape so ``bridge.rollback`` never erases the thing it reverses.
    """

    __tablename__ = "gc_bridge_actions"

    tenant_id: Mapped[_uuid.UUID] = _uuid_col(nullable=False)
    bridge_id: Mapped[_uuid.UUID] = _uuid_col(nullable=False)
    group_id: Mapped[_uuid.UUID] = _uuid_col(nullable=False)
    bridge_type: Mapped[str] = mapped_column(Text, nullable=False)
    proposal: Mapped[dict[str, Any]] = _jsonb_col(nullable=False)
    simulation: Mapped[dict[str, Any]] = _jsonb_col(nullable=False)
    # AUTHORIZATION is a type_func_name_keyword in PostgreSQL's grammar, so it
    # is not a legal unquoted ColId: CREATE TABLE ... (authorization jsonb) is a
    # syntax error. The card names the column, so the column is quoted rather
    # than renamed.
    authorization: Mapped[dict[str, Any]] = mapped_column(
        quoted_name("authorization", True), postgresql.JSONB, nullable=False
    )
    status: Mapped[str] = mapped_column(Text, nullable=False)
    rollback_of: Mapped[_uuid.UUID | None] = _uuid_col(nullable=True)
    vdf_proof_id: Mapped[_uuid.UUID] = _uuid_col(nullable=False)
    created_at: Mapped[_dt.datetime] = _ts_col(
        nullable=False, server_default=func.now()
    )
    applied_at: Mapped[_dt.datetime | None] = _ts_col(nullable=True)

    __table_args__ = (
        PrimaryKeyConstraint("tenant_id", "bridge_id"),
        ForeignKeyConstraint(
            ["tenant_id", "rollback_of"],
            ["gc_bridge_actions.tenant_id", "gc_bridge_actions.bridge_id"],
            name="fk_gc_bridge_actions_rollback_of",
        ),
        CheckConstraint("rollback_of IS NULL OR rollback_of <> bridge_id",
                        name="rollback_of_is_another_action"),
    )


# --------------------------------------------------------------------------
# gc_vdf_attestations
# --------------------------------------------------------------------------


class VdfAttestation(Base):
    """A Rule 30 proof envelope bound to the event it attests.

    ``verification_status`` is the single re-checkable field: verification is
    deterministic and offline (``rule30_vdf.verification``), so re-running it
    must be able to record its outcome without the envelope itself becoming
    editable. The migration's trigger enforces exactly that split.
    """

    __tablename__ = "gc_vdf_attestations"

    tenant_id: Mapped[_uuid.UUID] = _uuid_col(nullable=False)
    proof_id: Mapped[_uuid.UUID] = _uuid_col(nullable=False)
    event_id: Mapped[_uuid.UUID] = _uuid_col(nullable=False)
    envelope: Mapped[dict[str, Any]] = _jsonb_col(nullable=False)
    verification_status: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[_dt.datetime] = _ts_col(
        nullable=False, server_default=func.now()
    )

    __table_args__ = (PrimaryKeyConstraint("tenant_id", "proof_id"),)


# --------------------------------------------------------------------------
# Indexes
# --------------------------------------------------------------------------
#
# Declared after the classes so descending order and jsonb_path_ops can be
# expressed against real Column objects; ``_index_signature`` below renders each
# one back into the card's own notation for comparison.

Index(
    "ix_gc_user_profiles_tenant_user",
    UserProfile.__table__.c.tenant_id,
    UserProfile.__table__.c.user_id,
)
Index(
    "ix_gc_user_profiles_profile_gin",
    UserProfile.__table__.c.profile,
    postgresql_using="gin",
    postgresql_ops={"profile": "jsonb_path_ops"},
)

Index(
    "ix_gc_profile_revisions_tenant_user_revision",
    ProfileRevision.__table__.c.tenant_id,
    ProfileRevision.__table__.c.user_id,
    ProfileRevision.__table__.c.revision.desc(),
)
Index(
    "ix_gc_profile_revisions_delta_gin",
    ProfileRevision.__table__.c.delta,
    postgresql_using="gin",
    postgresql_ops={"delta": "jsonb_path_ops"},
)

Index(
    "ix_gc_group_intents_tenant_group_revision",
    GroupIntent.__table__.c.tenant_id,
    GroupIntent.__table__.c.group_id,
    GroupIntent.__table__.c.revision.desc(),
)
Index(
    "ix_gc_group_intents_aggregate_intent_gin",
    GroupIntent.__table__.c.aggregate_intent,
    postgresql_using="gin",
    postgresql_ops={"aggregate_intent": "jsonb_path_ops"},
)

Index(
    "ix_gc_tension_snapshots_tenant_group_observed",
    TensionSnapshot.__table__.c.tenant_id,
    TensionSnapshot.__table__.c.group_id,
    TensionSnapshot.__table__.c.observed_at.desc(),
)
Index(
    "ix_gc_tension_snapshots_tenant_class",
    TensionSnapshot.__table__.c.tenant_id,
    TensionSnapshot.__table__.c.tension_class,
)
Index(
    "ix_gc_tension_snapshots_member_gradients_gin",
    TensionSnapshot.__table__.c.member_gradients,
    postgresql_using="gin",
    postgresql_ops={"member_gradients": "jsonb_path_ops"},
)

Index(
    "ix_gc_bridge_actions_tenant_group_created",
    BridgeAction.__table__.c.tenant_id,
    BridgeAction.__table__.c.group_id,
    BridgeAction.__table__.c.created_at.desc(),
)
Index(
    "ix_gc_bridge_actions_tenant_status",
    BridgeAction.__table__.c.tenant_id,
    BridgeAction.__table__.c.status,
)
Index(
    "ix_gc_bridge_actions_simulation_gin",
    BridgeAction.__table__.c.simulation,
    postgresql_using="gin",
    postgresql_ops={"simulation": "jsonb_path_ops"},
)

Index(
    "ix_gc_vdf_attestations_tenant_event",
    VdfAttestation.__table__.c.tenant_id,
    VdfAttestation.__table__.c.event_id,
)
Index(
    "ix_gc_vdf_attestations_envelope_gin",
    VdfAttestation.__table__.c.envelope,
    postgresql_using="gin",
    postgresql_ops={"envelope": "jsonb_path_ops"},
)


#: Card table name -> declarative model. The single place the two vocabularies meet.
TABLE_MODELS: dict[str, type[Base]] = {
    "gc_user_profiles": UserProfile,
    "gc_profile_revisions": ProfileRevision,
    "gc_group_intents": GroupIntent,
    "gc_tension_snapshots": TensionSnapshot,
    "gc_bridge_actions": BridgeAction,
    "gc_vdf_attestations": VdfAttestation,
}

metadata = Base.metadata


# --------------------------------------------------------------------------
# Card agreement
# --------------------------------------------------------------------------

# Card type token -> (expected SQLAlchemy type, nullable). The nullability is
# carried by the token itself ("bigint-nullable"), so a column that the card
# declares as required cannot be made optional in the model without tripping
# this table.
_CARD_TYPE_TOKENS: dict[str, tuple[type, bool]] = {
    "uuid": (postgresql.UUID, False),
    "uuid-nullable": (postgresql.UUID, True),
    "bigint": (BigInteger, False),
    "bigint-nullable": (BigInteger, True),
    "jsonb": (postgresql.JSONB, False),
    "jsonb-nullable": (postgresql.JSONB, True),
    "text": (Text, False),
    "text-nullable": (Text, True),
    "numeric": (Numeric, False),
    "numeric-nullable": (Numeric, True),
    "timestamptz": (TIMESTAMP, False),
    "timestamptz-nullable": (TIMESTAMP, True),
}


def _expression_term(expr: Any) -> tuple[str, str]:
    """Render one index expression as ``(column_name, "asc"|"desc")``."""
    if isinstance(expr, UnaryExpression) and expr.modifier is operators.desc_op:
        inner = expr.element
        return getattr(inner, "name", str(inner)), "desc"
    if isinstance(expr, Column):
        return expr.name, "asc"
    return str(expr), "asc"


def _index_signature(index: Index) -> str:
    """Render an :class:`Index` in the card's own notation.

    The card writes ``gin(profile jsonb_path_ops)`` and
    ``btree(tenant_id, user_id, revision desc)``. Rendering the model back into
    that notation makes the comparison a string equality on the thing a reviewer
    actually reads, instead of a structural diff nobody checks.
    """
    method = (index.dialect_kwargs.get("postgresql_using") or "btree").lower()
    ops: dict[str, str] = dict(index.dialect_kwargs.get("postgresql_ops") or {})
    terms: list[str] = []
    for expr in index.expressions:
        name, direction = _expression_term(expr)
        term = name
        if direction == "desc":
            term += " desc"
        if name in ops:
            term += f" {ops[name]}"
        terms.append(term)
    return f"{method}({', '.join(terms)})"


def _normalise_index(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower()).replace(" ,", ",")


def _check_column(
    table_name: str, column_name: str, token: str, column: Column
) -> list[str]:
    problems: list[str] = []
    expected = _CARD_TYPE_TOKENS.get(token)
    if expected is None:
        return [
            f"{table_name}.{column_name}: card declares unknown type {token!r}; "
            f"add it to models._CARD_TYPE_TOKENS if it is intentional"
        ]
    sa_type, nullable = expected
    if not isinstance(column.type, sa_type):
        problems.append(
            f"{table_name}.{column_name}: card says {token!r} "
            f"(expects {sa_type.__name__}) but the model uses "
            f"{type(column.type).__name__}"
        )
    if sa_type is TIMESTAMP and not getattr(column.type, "timezone", False):
        problems.append(
            f"{table_name}.{column_name}: card says {token!r}; the model column "
            "must be TIMESTAMP(timezone=True) or timestamps lose their offset"
        )
    if column.nullable != nullable:
        want = "nullable" if nullable else "NOT NULL"
        problems.append(
            f"{table_name}.{column_name}: card says {token!r} so the column must "
            f"be {want}, but the model declares nullable={column.nullable}"
        )
    return problems


def verify_schema_matches_card(card: AgentModelCard | None = None) -> None:
    """Assert the declarative models are exactly what the card declares.

    Compares, per table: the column set, each column's type family and
    nullability, the composite primary key (order included — it decides the
    index's leading column and therefore which lookups are cheap), and the
    declared index set rendered back into card notation.

    Raises :class:`SchemaCardMismatch` listing every disagreement at once,
    because fixing them one ImportError at a time is how half a schema drift
    gets merged.
    """
    card = card or cached_model_card()
    declared = card.jsonb_persistence.tables
    problems: list[str] = []

    missing_models = sorted(set(declared) - set(TABLE_MODELS))
    extra_models = sorted(set(TABLE_MODELS) - set(declared))
    if missing_models:
        problems.append(
            f"card declares tables with no model: {missing_models}"
        )
    if extra_models:
        problems.append(
            f"models declare tables absent from the card: {extra_models}"
        )

    for name in sorted(set(declared) & set(TABLE_MODELS)):
        spec = declared[name]
        table = TABLE_MODELS[name].__table__

        card_columns = set(spec.columns)
        model_columns = set(table.c.keys())
        for missing in sorted(card_columns - model_columns):
            problems.append(
                f"{name}: card declares column {missing!r} that the model omits"
            )
        for extra in sorted(model_columns - card_columns):
            problems.append(
                f"{name}: model declares column {extra!r} that the card omits; "
                "add it to the card or drop it from the model"
            )
        for column_name in sorted(card_columns & model_columns):
            problems.extend(
                _check_column(
                    name, column_name, spec.columns[column_name], table.c[column_name]
                )
            )

        model_pk = tuple(c.name for c in table.primary_key.columns)
        if model_pk != tuple(spec.primary_key):
            problems.append(
                f"{name}: card primary key is {list(spec.primary_key)} but the "
                f"model's is {list(model_pk)} (order is significant)"
            )

        card_indexes = {_normalise_index(i) for i in spec.indexes}
        model_indexes = {_normalise_index(_index_signature(i)) for i in table.indexes}
        for missing_index in sorted(card_indexes - model_indexes):
            problems.append(
                f"{name}: card declares index {missing_index!r} that the model "
                "does not create"
            )
        for extra_index in sorted(model_indexes - card_indexes):
            problems.append(
                f"{name}: model creates index {extra_index!r} that the card does "
                "not declare"
            )

    if problems:
        raise SchemaCardMismatch(
            "SQLAlchemy models disagree with "
            "torx_contextual_sovereignty_agent.model-card.yaml "
            "(jsonb_persistence.tables):\n  - " + "\n  - ".join(problems)
        )


# Import-time, not test-time: a divergence found by a test that nobody ran is a
# divergence shipped.
verify_schema_matches_card()


__all__ = [
    "Base",
    "BridgeAction",
    "GroupIntent",
    "NAMING_CONVENTION",
    "NUMERIC_PRECISION",
    "NUMERIC_SCALE",
    "ProfileRevision",
    "SchemaCardMismatch",
    "TABLE_MODELS",
    "TensionSnapshot",
    "UserProfile",
    "VdfAttestation",
    "metadata",
    "verify_schema_matches_card",
]
