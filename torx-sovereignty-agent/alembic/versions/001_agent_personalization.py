"""Agent personalization schema: six JSONB tables, RLS, append-only triggers.

The tables are transcribed from ``jsonb_persistence.tables``; the *guards* are
what this migration adds on top of what SQLAlchemy can declare:

1. **Row-level security** on all six tables, keyed on
   ``current_setting('app.tenant_id', true)``. ``FORCE ROW LEVEL SECURITY`` is
   set as well, because without it the policy does not apply to the table owner
   — and the application usually connects as the owner, which would make the
   whole control decorative. ``current_setting(..., true)`` returns NULL when the
   setting is absent, and ``tenant_id = NULL`` is NULL, so a session that never
   declared a tenant sees nothing rather than everything.

2. **Append-only triggers.** ``invariants.append_only_revision_history`` has to
   survive a buggy caller, an ad-hoc psql session, and a future ORM that does not
   go through ``src.persistence.repositories``. ``gc_profile_revisions``,
   ``gc_group_intents`` and ``gc_tension_snapshots`` reject every UPDATE and
   DELETE. ``gc_bridge_actions`` and ``gc_vdf_attestations`` reject DELETE and
   any UPDATE outside their one narrow mutable field-set — a bridge's lifecycle
   status and an attestation's re-checkable verification result. The triggers
   raise SQLSTATE ``GC001``, a custom class so a history-rewrite attempt stays
   distinguishable from an ordinary check violation.

3. **A no-gap revision trigger.** ``revision`` must be ``max(revision) + 1`` per
   ``(tenant_id, user_id)`` / ``(tenant_id, group_id)``. The primary key already
   makes revisions unique; this makes them *dense*, which is what lets
   ``profile.rollback``'s ``target_revision`` and the Rule 30 chain assume that
   revision *n-1* exists.

Revision ID: 001_agent_personalization
Revises:
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "001_agent_personalization"
down_revision = None
branch_labels = None
depends_on = None

# Kept in step with src.persistence.repositories.APPEND_ONLY_SQLSTATE and
# MUTABLE_COLUMNS. The Python copy stops a bad write before the wire; this copy
# stops one that never came through Python.
APPEND_ONLY_SQLSTATE = "GC001"

TABLES = (
    "gc_user_profiles",
    "gc_profile_revisions",
    "gc_group_intents",
    "gc_tension_snapshots",
    "gc_bridge_actions",
    "gc_vdf_attestations",
)

#: History tables that reject every UPDATE and DELETE outright.
FROZEN_TABLES = (
    "gc_profile_revisions",
    "gc_group_intents",
    "gc_tension_snapshots",
)

NUMERIC = sa.Numeric(12, 9)


def _uuid() -> postgresql.UUID:
    return postgresql.UUID(as_uuid=True)


def _ts() -> sa.TIMESTAMP:
    return sa.TIMESTAMP(timezone=True)


def upgrade() -> None:
    _create_tables()
    _create_indexes()
    _enable_row_level_security()
    _create_append_only_guards()
    _create_revision_density_guards()


def downgrade() -> None:
    for table in FROZEN_TABLES:
        op.execute(f"DROP TRIGGER IF EXISTS {table}_append_only ON {table}")
    op.execute(
        "DROP TRIGGER IF EXISTS gc_bridge_actions_append_only ON gc_bridge_actions"
    )
    op.execute(
        "DROP TRIGGER IF EXISTS gc_vdf_attestations_append_only "
        "ON gc_vdf_attestations"
    )
    op.execute(
        "DROP TRIGGER IF EXISTS gc_profile_revisions_dense "
        "ON gc_profile_revisions"
    )
    op.execute("DROP TRIGGER IF EXISTS gc_group_intents_dense ON gc_group_intents")
    op.execute("DROP FUNCTION IF EXISTS gc_reject_history_mutation()")
    op.execute("DROP FUNCTION IF EXISTS gc_bridge_actions_guard()")
    op.execute("DROP FUNCTION IF EXISTS gc_vdf_attestations_guard()")
    op.execute("DROP FUNCTION IF EXISTS gc_profile_revision_is_dense()")
    op.execute("DROP FUNCTION IF EXISTS gc_group_intent_is_dense()")
    for table in TABLES:
        op.execute(f"DROP POLICY IF EXISTS {table}_tenant_isolation ON {table}")
    # Children first: gc_tension_snapshots references gc_group_intents and
    # gc_profile_revisions references gc_user_profiles.
    for table in reversed(TABLES):
        op.drop_table(table)


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------


def _create_tables() -> None:
    op.create_table(
        "gc_user_profiles",
        sa.Column("tenant_id", _uuid(), nullable=False),
        sa.Column("user_id", _uuid(), nullable=False),
        sa.Column("current_revision", sa.BigInteger(), nullable=False),
        sa.Column("profile", postgresql.JSONB(), nullable=False),
        sa.Column("profile_hash", sa.Text(), nullable=False),
        sa.Column("effective_manifest_hash", sa.Text(), nullable=False),
        sa.Column(
            "created_at", _ts(), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", _ts(), nullable=False, server_default=sa.func.now()
        ),
        sa.PrimaryKeyConstraint("tenant_id", "user_id", name="pk_gc_user_profiles"),
        sa.CheckConstraint(
            "current_revision >= 1", name="current_revision_positive"
        ),
    )

    op.create_table(
        "gc_profile_revisions",
        sa.Column("tenant_id", _uuid(), nullable=False),
        sa.Column("user_id", _uuid(), nullable=False),
        sa.Column("revision", sa.BigInteger(), nullable=False),
        sa.Column("source_event_ids", postgresql.JSONB(), nullable=False),
        sa.Column("delta", postgresql.JSONB(), nullable=False),
        sa.Column("full_snapshot", postgresql.JSONB(), nullable=False),
        sa.Column("confidence_summary", postgresql.JSONB(), nullable=False),
        sa.Column("supersedes_revision", sa.BigInteger(), nullable=True),
        sa.Column("vdf_proof_id", _uuid(), nullable=False),
        sa.Column(
            "created_at", _ts(), nullable=False, server_default=sa.func.now()
        ),
        sa.PrimaryKeyConstraint(
            "tenant_id", "user_id", "revision", name="pk_gc_profile_revisions"
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "user_id"],
            ["gc_user_profiles.tenant_id", "gc_user_profiles.user_id"],
            name="fk_gc_profile_revisions_profile",
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "user_id", "supersedes_revision"],
            [
                "gc_profile_revisions.tenant_id",
                "gc_profile_revisions.user_id",
                "gc_profile_revisions.revision",
            ],
            name="fk_gc_profile_revisions_supersedes",
        ),
        sa.CheckConstraint(
            "revision >= 1", name="revision_positive"
        ),
        sa.CheckConstraint(
            "supersedes_revision IS NULL OR supersedes_revision < revision",
            name="supersedes_is_backwards",
        ),
    )

    op.create_table(
        "gc_group_intents",
        sa.Column("tenant_id", _uuid(), nullable=False),
        sa.Column("group_id", _uuid(), nullable=False),
        sa.Column("revision", sa.BigInteger(), nullable=False),
        sa.Column("aggregate_intent", postgresql.JSONB(), nullable=False),
        sa.Column("member_intents", postgresql.JSONB(), nullable=False),
        sa.Column("decision_policy", postgresql.JSONB(), nullable=False),
        # NOT NULL: "nobody dissented" and "we did not record dissent" must not
        # be the same row. invariants.preserve_minority_positions.
        sa.Column("dissent", postgresql.JSONB(), nullable=False),
        sa.Column("vdf_proof_id", _uuid(), nullable=False),
        sa.Column(
            "created_at", _ts(), nullable=False, server_default=sa.func.now()
        ),
        sa.PrimaryKeyConstraint(
            "tenant_id", "group_id", "revision", name="pk_gc_group_intents"
        ),
        sa.CheckConstraint(
            "revision >= 1", name="revision_positive"
        ),
    )

    op.create_table(
        "gc_tension_snapshots",
        sa.Column("tenant_id", _uuid(), nullable=False),
        sa.Column("group_id", _uuid(), nullable=False),
        sa.Column("snapshot_id", _uuid(), nullable=False),
        sa.Column("group_intent_revision", sa.BigInteger(), nullable=False),
        sa.Column("member_gradients", postgresql.JSONB(), nullable=False),
        sa.Column("group_tension", NUMERIC, nullable=False),
        sa.Column("tension_class", sa.Text(), nullable=False),
        sa.Column("confidence", NUMERIC, nullable=False),
        sa.Column("topology_descriptor", postgresql.JSONB(), nullable=False),
        sa.Column("vdf_proof_id", _uuid(), nullable=False),
        sa.Column(
            "observed_at", _ts(), nullable=False, server_default=sa.func.now()
        ),
        sa.PrimaryKeyConstraint(
            "tenant_id", "group_id", "snapshot_id", name="pk_gc_tension_snapshots"
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "group_id", "group_intent_revision"],
            [
                "gc_group_intents.tenant_id",
                "gc_group_intents.group_id",
                "gc_group_intents.revision",
            ],
            name="fk_gc_tension_snapshots_group_intent",
        ),
        sa.CheckConstraint(
            "group_tension >= 0 AND group_tension <= 1",
            name="group_tension_ratio",
        ),
        sa.CheckConstraint(
            "confidence >= 0 AND confidence <= 1",
            name="confidence_ratio",
        ),
    )

    op.create_table(
        "gc_bridge_actions",
        sa.Column("tenant_id", _uuid(), nullable=False),
        sa.Column("bridge_id", _uuid(), nullable=False),
        sa.Column("group_id", _uuid(), nullable=False),
        sa.Column("bridge_type", sa.Text(), nullable=False),
        sa.Column("proposal", postgresql.JSONB(), nullable=False),
        sa.Column("simulation", postgresql.JSONB(), nullable=False),
        # Quoted: AUTHORIZATION is a type_func_name_keyword and is not a legal
        # unquoted column name in PostgreSQL.
        sa.Column(
            sa.sql.quoted_name("authorization", True),
            postgresql.JSONB(),
            nullable=False,
        ),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("rollback_of", _uuid(), nullable=True),
        sa.Column("vdf_proof_id", _uuid(), nullable=False),
        sa.Column(
            "created_at", _ts(), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("applied_at", _ts(), nullable=True),
        sa.PrimaryKeyConstraint(
            "tenant_id", "bridge_id", name="pk_gc_bridge_actions"
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "rollback_of"],
            ["gc_bridge_actions.tenant_id", "gc_bridge_actions.bridge_id"],
            name="fk_gc_bridge_actions_rollback_of",
        ),
        sa.CheckConstraint(
            "rollback_of IS NULL OR rollback_of <> bridge_id",
            name="rollback_of_is_another_action",
        ),
    )

    op.create_table(
        "gc_vdf_attestations",
        sa.Column("tenant_id", _uuid(), nullable=False),
        sa.Column("proof_id", _uuid(), nullable=False),
        sa.Column("event_id", _uuid(), nullable=False),
        sa.Column("envelope", postgresql.JSONB(), nullable=False),
        sa.Column("verification_status", sa.Text(), nullable=False),
        sa.Column(
            "created_at", _ts(), nullable=False, server_default=sa.func.now()
        ),
        sa.PrimaryKeyConstraint(
            "tenant_id", "proof_id", name="pk_gc_vdf_attestations"
        ),
    )


# --------------------------------------------------------------------------
# Indexes
# --------------------------------------------------------------------------


def _create_indexes() -> None:
    op.create_index(
        "ix_gc_user_profiles_tenant_user", "gc_user_profiles", ["tenant_id", "user_id"]
    )
    op.create_index(
        "ix_gc_user_profiles_profile_gin",
        "gc_user_profiles",
        ["profile"],
        postgresql_using="gin",
        postgresql_ops={"profile": "jsonb_path_ops"},
    )

    op.create_index(
        "ix_gc_profile_revisions_tenant_user_revision",
        "gc_profile_revisions",
        ["tenant_id", "user_id", sa.text("revision DESC")],
    )
    op.create_index(
        "ix_gc_profile_revisions_delta_gin",
        "gc_profile_revisions",
        ["delta"],
        postgresql_using="gin",
        postgresql_ops={"delta": "jsonb_path_ops"},
    )

    op.create_index(
        "ix_gc_group_intents_tenant_group_revision",
        "gc_group_intents",
        ["tenant_id", "group_id", sa.text("revision DESC")],
    )
    op.create_index(
        "ix_gc_group_intents_aggregate_intent_gin",
        "gc_group_intents",
        ["aggregate_intent"],
        postgresql_using="gin",
        postgresql_ops={"aggregate_intent": "jsonb_path_ops"},
    )

    op.create_index(
        "ix_gc_tension_snapshots_tenant_group_observed",
        "gc_tension_snapshots",
        ["tenant_id", "group_id", sa.text("observed_at DESC")],
    )
    op.create_index(
        "ix_gc_tension_snapshots_tenant_class",
        "gc_tension_snapshots",
        ["tenant_id", "tension_class"],
    )
    op.create_index(
        "ix_gc_tension_snapshots_member_gradients_gin",
        "gc_tension_snapshots",
        ["member_gradients"],
        postgresql_using="gin",
        postgresql_ops={"member_gradients": "jsonb_path_ops"},
    )

    op.create_index(
        "ix_gc_bridge_actions_tenant_group_created",
        "gc_bridge_actions",
        ["tenant_id", "group_id", sa.text("created_at DESC")],
    )
    op.create_index(
        "ix_gc_bridge_actions_tenant_status",
        "gc_bridge_actions",
        ["tenant_id", "status"],
    )
    op.create_index(
        "ix_gc_bridge_actions_simulation_gin",
        "gc_bridge_actions",
        ["simulation"],
        postgresql_using="gin",
        postgresql_ops={"simulation": "jsonb_path_ops"},
    )

    op.create_index(
        "ix_gc_vdf_attestations_tenant_event",
        "gc_vdf_attestations",
        ["tenant_id", "event_id"],
    )
    op.create_index(
        "ix_gc_vdf_attestations_envelope_gin",
        "gc_vdf_attestations",
        ["envelope"],
        postgresql_using="gin",
        postgresql_ops={"envelope": "jsonb_path_ops"},
    )


# --------------------------------------------------------------------------
# Row-level security
# --------------------------------------------------------------------------


def _enable_row_level_security() -> None:
    """One policy per table: rows belong to the tenant in ``app.tenant_id``.

    ``USING`` scopes reads and the pre-image of writes; ``WITH CHECK`` scopes the
    post-image, so a session cannot INSERT or UPDATE a row *into* another
    tenant either. ``tenant_isolation: row-level-security`` in the card is this
    pair of clauses, not just the first one.
    """
    predicate = "tenant_id = current_setting('app.tenant_id', true)::uuid"
    for table in TABLES:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        # Without FORCE the owner bypasses the policy, and the application
        # normally connects as the owner.
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY {table}_tenant_isolation ON {table} "
            f"USING ({predicate}) WITH CHECK ({predicate})"
        )


# --------------------------------------------------------------------------
# Append-only guards
# --------------------------------------------------------------------------


def _create_append_only_guards() -> None:
    op.execute(
        f"""
        CREATE OR REPLACE FUNCTION gc_reject_history_mutation()
        RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
            RAISE EXCEPTION
                'append-only violation: % on %.% is forbidden; append a new revision that supersedes the old one instead of rewriting it',
                TG_OP, TG_TABLE_SCHEMA, TG_TABLE_NAME
                USING ERRCODE = '{APPEND_ONLY_SQLSTATE}';
            RETURN NULL;
        END;
        $$
        """
    )
    for table in FROZEN_TABLES:
        op.execute(
            f"""
            CREATE TRIGGER {table}_append_only
            BEFORE UPDATE OR DELETE ON {table}
            FOR EACH ROW EXECUTE FUNCTION gc_reject_history_mutation()
            """
        )

    # A bridge action's lifecycle status is genuinely mutable; everything that
    # describes *what the bridge was* is not.
    op.execute(
        f"""
        CREATE OR REPLACE FUNCTION gc_bridge_actions_guard()
        RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
            IF TG_OP = 'DELETE' THEN
                RAISE EXCEPTION
                    'append-only violation: DELETE on gc_bridge_actions is forbidden; roll the action back with a new action instead'
                    USING ERRCODE = '{APPEND_ONLY_SQLSTATE}';
            END IF;
            IF NEW.tenant_id IS DISTINCT FROM OLD.tenant_id
               OR NEW.bridge_id IS DISTINCT FROM OLD.bridge_id
               OR NEW.group_id IS DISTINCT FROM OLD.group_id
               OR NEW.bridge_type IS DISTINCT FROM OLD.bridge_type
               OR NEW.proposal IS DISTINCT FROM OLD.proposal
               OR NEW.simulation IS DISTINCT FROM OLD.simulation
               OR NEW."authorization" IS DISTINCT FROM OLD."authorization"
               OR NEW.rollback_of IS DISTINCT FROM OLD.rollback_of
               OR NEW.vdf_proof_id IS DISTINCT FROM OLD.vdf_proof_id
               OR NEW.created_at IS DISTINCT FROM OLD.created_at THEN
                RAISE EXCEPTION
                    'append-only violation: only status and applied_at may change on gc_bridge_actions'
                    USING ERRCODE = '{APPEND_ONLY_SQLSTATE}';
            END IF;
            RETURN NEW;
        END;
        $$
        """
    )
    op.execute(
        """
        CREATE TRIGGER gc_bridge_actions_append_only
        BEFORE UPDATE OR DELETE ON gc_bridge_actions
        FOR EACH ROW EXECUTE FUNCTION gc_bridge_actions_guard()
        """
    )

    # Verification is deterministic and offline, so re-running it must be able
    # to record its outcome. The envelope it verifies must not move underneath.
    op.execute(
        f"""
        CREATE OR REPLACE FUNCTION gc_vdf_attestations_guard()
        RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
            IF TG_OP = 'DELETE' THEN
                RAISE EXCEPTION
                    'append-only violation: DELETE on gc_vdf_attestations is forbidden; a discarded proof cannot be re-verified'
                    USING ERRCODE = '{APPEND_ONLY_SQLSTATE}';
            END IF;
            IF NEW.tenant_id IS DISTINCT FROM OLD.tenant_id
               OR NEW.proof_id IS DISTINCT FROM OLD.proof_id
               OR NEW.event_id IS DISTINCT FROM OLD.event_id
               OR NEW.envelope IS DISTINCT FROM OLD.envelope
               OR NEW.created_at IS DISTINCT FROM OLD.created_at THEN
                RAISE EXCEPTION
                    'append-only violation: only verification_status may change on gc_vdf_attestations'
                    USING ERRCODE = '{APPEND_ONLY_SQLSTATE}';
            END IF;
            RETURN NEW;
        END;
        $$
        """
    )
    op.execute(
        """
        CREATE TRIGGER gc_vdf_attestations_append_only
        BEFORE UPDATE OR DELETE ON gc_vdf_attestations
        FOR EACH ROW EXECUTE FUNCTION gc_vdf_attestations_guard()
        """
    )


def _create_revision_density_guards() -> None:
    """Revisions must be dense, not merely unique.

    The primary key stops two writers from both claiming revision 5. It does not
    stop one writer from jumping to revision 900, which would leave 6..899 as
    holes that ``profile.rollback`` cannot land on and that break the Rule 30
    chain's "n follows n-1" assumption.
    """
    op.execute(
        f"""
        CREATE OR REPLACE FUNCTION gc_profile_revision_is_dense()
        RETURNS trigger LANGUAGE plpgsql AS $$
        DECLARE
            expected bigint;
        BEGIN
            SELECT COALESCE(MAX(revision), 0) + 1 INTO expected
            FROM gc_profile_revisions
            WHERE tenant_id = NEW.tenant_id AND user_id = NEW.user_id;
            IF NEW.revision <> expected THEN
                RAISE EXCEPTION
                    'revision % is not the next revision for this profile (expected %); revision history must be gapless',
                    NEW.revision, expected
                    USING ERRCODE = '{APPEND_ONLY_SQLSTATE}';
            END IF;
            RETURN NEW;
        END;
        $$
        """
    )
    op.execute(
        """
        CREATE TRIGGER gc_profile_revisions_dense
        BEFORE INSERT ON gc_profile_revisions
        FOR EACH ROW EXECUTE FUNCTION gc_profile_revision_is_dense()
        """
    )
    op.execute(
        f"""
        CREATE OR REPLACE FUNCTION gc_group_intent_is_dense()
        RETURNS trigger LANGUAGE plpgsql AS $$
        DECLARE
            expected bigint;
        BEGIN
            SELECT COALESCE(MAX(revision), 0) + 1 INTO expected
            FROM gc_group_intents
            WHERE tenant_id = NEW.tenant_id AND group_id = NEW.group_id;
            IF NEW.revision <> expected THEN
                RAISE EXCEPTION
                    'revision % is not the next revision for this group (expected %); group intent history must be gapless',
                    NEW.revision, expected
                    USING ERRCODE = '{APPEND_ONLY_SQLSTATE}';
            END IF;
            RETURN NEW;
        END;
        $$
        """
    )
    op.execute(
        """
        CREATE TRIGGER gc_group_intents_dense
        BEFORE INSERT ON gc_group_intents
        FOR EACH ROW EXECUTE FUNCTION gc_group_intent_is_dense()
        """
    )
