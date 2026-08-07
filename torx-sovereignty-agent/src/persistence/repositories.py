"""Repositories over the card's JSONB tables, with a storage-backend seam.

Every invariant this layer owns is enforced *here*, once, above a narrow
:class:`StorageBackend` interface, and the interface has two implementations:
:class:`SqlAlchemyBackend` (PostgreSQL, the real thing) and
:class:`InMemoryBackend` (a reference backend that derives its column set,
NOT NULL constraints, primary keys and foreign keys from the same
``models.metadata``). The seam exists so the invariants are executed by tests on
a machine with no database — not restated in a second codebase that drifts.

What this layer guarantees:

* **Monotonic revisions, no gaps.** ``next = current + 1``, always. Gaps would
  make ``profile.rollback``'s ``target_revision`` ambiguous and would break the
  Rule 30 chain, whose whole value is that revision *n* is bound to *n-1*.
* **Optimistic concurrency.** ``deployment.consistency`` asks for
  optimistic-concurrency-control on profile revisions and compare-and-swap on
  group intent. Both are the same mechanism here: the writer states the revision
  it read, and a stale statement raises :class:`RevisionConflict` rather than
  overwriting the work it never saw.
* **Append-only history.** ``invariants.append_only_revision_history``. There is
  no ``delete`` on this interface at all, and ``update`` is restricted by
  :data:`MUTABLE_COLUMNS` to the three genuinely mutable things: the profile
  head, a bridge's status, and an attestation's re-checkable verification
  result. Rollback appends; it never rewrites. The Alembic migration installs
  the same rules as triggers, so a caller that bypasses this module still cannot
  rewrite history.
* **Explicit tenancy.** Every method takes ``tenant_id`` as its first argument.
  There is no ambient tenant, no thread-local, no default. On PostgreSQL the
  same value is pushed into ``app.tenant_id`` for row-level security, so the
  predicate the code writes and the predicate the database enforces come from
  one place.
* **Dissent and uncertainty are stored.** ``dissent`` and ``confidence_summary``
  are NOT NULL and are rejected as ``None`` with an explanation: an absent field
  and an empty one are different claims, and only one of them is honest.

What this layer deliberately does *not* do: decide whether a permission grant
narrowed or widened. That comparison needs the base contract and the effective
manifest, which are the sovereignty guard's inputs, not the repository's. This
module stores what the guard decided; it does not re-decide it.

Floats headed for storage are rounded to ``state.STORAGE_PRECISION`` on the way
in, by both backends, so the canonical JSON the Rule 30 VDF signs is identical
whether the row came from PostgreSQL or from memory.
"""

from __future__ import annotations

import abc
import copy
import datetime as _dt
import math
import uuid as _uuid
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Iterator, Mapping, Sequence

from src.model_card.loader import cached_model_card
from src.torx_layer.state import BRIDGE_STRATEGIES, STORAGE_PRECISION, TENSION_CLASSES

from . import models

# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


class PersistenceError(RuntimeError):
    """Base class for every failure this layer raises deliberately."""


class RevisionConflict(PersistenceError):
    """A write carried a stale expectation and was refused.

    Carries the structured facts a retry needs (``expected`` and ``actual``) in
    addition to the message, so a caller can rebase programmatically instead of
    parsing prose.
    """

    def __init__(
        self,
        message: str,
        *,
        table: str,
        tenant_id: _uuid.UUID,
        entity: str,
        expected: Any,
        actual: Any,
    ) -> None:
        super().__init__(message)
        self.table = table
        self.tenant_id = tenant_id
        self.entity = entity
        self.expected = expected
        self.actual = actual


class StatusConflict(RevisionConflict):
    """A lifecycle transition was attempted from the wrong state.

    A subclass of :class:`RevisionConflict` because it is the same failure —
    the writer acted on a state it had already lost — and callers that retry one
    should retry the other.
    """


class AppendOnlyViolation(PersistenceError):
    """An UPDATE or DELETE was attempted against history."""


class DuplicateRow(PersistenceError):
    """A primary key already exists."""


class ForeignKeyViolation(PersistenceError):
    """A row referenced something that does not exist."""


class IdempotencyConflict(PersistenceError):
    """The same id was written twice with different content.

    ``runtime.event_model`` is at-least-once with idempotent consumers, so
    re-delivering an identical event must succeed silently. Re-using an id for
    *different* content is the opposite: a collision that would silently rewrite
    an attested record.
    """


class TenantScopeError(PersistenceError):
    """A tenant boundary was crossed, or none was established."""


class ContractViolation(ValueError):
    """A JSONB document is missing keys its ``jsonb_contracts`` entry requires."""


# --------------------------------------------------------------------------
# Value normalisation
# --------------------------------------------------------------------------


def _as_uuid(value: Any, field_name: str) -> _uuid.UUID:
    if isinstance(value, _uuid.UUID):
        return value
    if isinstance(value, str):
        try:
            return _uuid.UUID(value)
        except ValueError as exc:
            raise ValueError(
                f"{field_name} must be a UUID, got {value!r}: {exc}"
            ) from exc
    raise ValueError(
        f"{field_name} must be a uuid.UUID or a UUID string, got "
        f"{type(value).__name__}"
    )


def _as_utc(value: Any, field_name: str) -> _dt.datetime:
    if not isinstance(value, _dt.datetime):
        raise ValueError(
            f"{field_name} must be a datetime, got {type(value).__name__}"
        )
    if value.tzinfo is None:
        raise ValueError(
            f"{field_name} is naive; pass an aware datetime "
            "(datetime.now(timezone.utc)) so the stored instant is unambiguous"
        )
    return value.astimezone(_dt.timezone.utc)


def _as_ratio(value: Any, field_name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number, got {value!r}") from exc
    if not math.isfinite(out) or not 0.0 <= out <= 1.0:
        raise ValueError(f"{field_name} must be a finite value in [0, 1], got {out!r}")
    return round(out, STORAGE_PRECISION)


def _as_revision(value: Any, field_name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{field_name} must be an int, got {type(value).__name__}"
        )
    if value < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}, got {value}")
    return value


def canonical_jsonb(value: Any, field_name: str) -> Any:
    """Deep-copy a JSONB payload into its stored form.

    Three things happen, and each of them is a bug this layer refuses to ship:

    * floats are rounded to ``STORAGE_PRECISION`` so GPU and CPU producers write
      identical bytes and the VDF over those bytes verifies either way;
    * NaN and Infinity are rejected — ``json.dumps`` would happily emit them and
      PostgreSQL would reject the insert much later, from a different stack;
    * non-string mapping keys are rejected instead of being coerced, because
      ``{1: "a"}`` and ``{"1": "a"}` must not become the same stored row.

    The result is a fresh structure, so a caller that mutates its dict after the
    write does not retroactively change what was stored — the in-memory backend
    has to behave like a database here, not like a reference.
    """

    def convert(node: Any, path: str) -> Any:
        if isinstance(node, Mapping):
            out: dict[str, Any] = {}
            for key, item in node.items():
                if not isinstance(key, str):
                    raise ValueError(
                        f"{path}: JSONB object keys must be strings, got "
                        f"{type(key).__name__} ({key!r}); convert it explicitly "
                        "rather than relying on json coercion"
                    )
                out[key] = convert(item, f"{path}.{key}")
            return out
        if isinstance(node, (list, tuple)):
            return [convert(item, f"{path}[{i}]") for i, item in enumerate(node)]
        if isinstance(node, bool) or node is None or isinstance(node, str):
            return node
        if isinstance(node, int):
            return node
        if isinstance(node, float):
            if not math.isfinite(node):
                raise ValueError(
                    f"{path}: {node!r} is not representable in JSONB; a "
                    "non-finite tension or confidence is a computation bug, not "
                    "a storable value"
                )
            return round(node, STORAGE_PRECISION)
        if isinstance(node, Decimal):
            return round(float(node), STORAGE_PRECISION)
        if isinstance(node, _uuid.UUID):
            return str(node)
        if isinstance(node, _dt.datetime):
            return _as_utc(node, path).isoformat()
        raise ValueError(
            f"{path}: {type(node).__name__} is not JSONB-serialisable; convert "
            "it to a JSON type before storing"
        )

    return convert(value, field_name)


def _require_mapping(value: Any, field_name: str) -> dict[str, Any]:
    if value is None:
        raise ValueError(
            f"{field_name} must not be None; an absent field and an empty one "
            "are different claims, and only the empty one is storable"
        )
    if not isinstance(value, Mapping):
        raise ValueError(
            f"{field_name} must be a mapping, got {type(value).__name__}"
        )
    return canonical_jsonb(value, field_name)


def _require_sequence(value: Any, field_name: str, *, allow_empty: bool) -> list[Any]:
    if value is None:
        raise ValueError(
            f"{field_name} must not be None; pass [] to record 'none observed'"
        )
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(
            f"{field_name} must be a sequence, got {type(value).__name__}"
        )
    out = canonical_jsonb(list(value), field_name)
    if not allow_empty and not out:
        raise ValueError(
            f"{field_name} must not be empty; "
            "invariants.evidence_provenance_required means a stored revision "
            "must name the events that justified it"
        )
    return out


def _require_contract(document: Mapping[str, Any], contract: str, what: str) -> None:
    """Enforce ``jsonb_contracts[contract].required`` on a document.

    ``validation.schema.jsonb_contract_validation_required`` is true, and the
    storage boundary is the last place a malformed document can be stopped
    before it becomes an attested row that every later reader must special-case.
    """
    required = cached_model_card().jsonb_contracts[contract].required
    missing = [key for key in required if key not in document]
    if missing:
        raise ContractViolation(
            f"{what} does not satisfy jsonb_contracts.{contract}: missing "
            f"{missing}; the card requires {list(required)}"
        )


def _utcnow() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


# --------------------------------------------------------------------------
# Mutation policy
# --------------------------------------------------------------------------

#: Table -> the only columns an UPDATE may touch. Absent means fully append-only.
#: Mirrored exactly by the triggers in alembic/versions/001_agent_personalization.py;
#: this copy stops a bad write before it reaches the wire, that copy stops one
#: that never came through this module.
MUTABLE_COLUMNS: dict[str, frozenset[str]] = {
    "gc_user_profiles": frozenset(
        {
            "current_revision",
            "profile",
            "profile_hash",
            "effective_manifest_hash",
            "updated_at",
        }
    ),
    "gc_bridge_actions": frozenset({"status", "applied_at"}),
    "gc_vdf_attestations": frozenset({"verification_status"}),
}

#: SQLSTATE raised by the append-only triggers. Custom class so a genuine check
#: violation stays distinguishable from a history rewrite attempt.
APPEND_ONLY_SQLSTATE = "GC001"

#: Bridge lifecycle. Derived from ``observability.audit_events``: bridge.proposed,
#: bridge.applied, bridge.rejected, bridge.rolled_back.
BRIDGE_STATUSES: tuple[str, ...] = ("proposed", "applied", "rejected", "rolled-back")

#: Only these transitions exist. Anything else is a caller that lost track of
#: state, and ``bridge.apply`` twice must not apply twice.
BRIDGE_TRANSITIONS: dict[str, frozenset[str]] = {
    "proposed": frozenset({"applied", "rejected"}),
    "applied": frozenset({"rolled-back"}),
    "rejected": frozenset(),
    "rolled-back": frozenset(),
}

#: ``vdf.attested`` / ``vdf.verification_failed`` plus the pre-verification state.
VERIFICATION_STATUSES: tuple[str, ...] = ("unverified", "verified", "failed")


# --------------------------------------------------------------------------
# Storage backend
# --------------------------------------------------------------------------


class StorageBackend(abc.ABC):
    """The narrowest interface the repositories need.

    Deliberately without a ``delete``: erasure under
    ``retention.profile_revisions: tenant-configurable-with-user-deletion-override``
    is a separate, audited, out-of-band operation, and giving the ordinary write
    path a delete verb is how "append-only" quietly becomes "append-mostly".
    """

    #: Reported in every repository result, matching the repo's backend idiom.
    name: str = "abstract"

    @contextmanager
    def unit_of_work(self, tenant_id: Any) -> Iterator[None]:
        """Run a block atomically, scoped to one tenant."""
        raise NotImplementedError

    @abc.abstractmethod
    def insert(self, table: str, row: Mapping[str, Any]) -> None:
        ...

    @abc.abstractmethod
    def fetch_one(self, table: str, key: Mapping[str, Any]) -> dict[str, Any] | None:
        ...

    @abc.abstractmethod
    def fetch_all(
        self,
        table: str,
        where: Mapping[str, Any],
        *,
        order_by: Sequence[tuple[str, str]] = (),
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        ...

    @abc.abstractmethod
    def _apply_update(
        self,
        table: str,
        key: Mapping[str, Any],
        values: Mapping[str, Any],
        expected: Mapping[str, Any],
    ) -> int:
        ...

    def update(
        self,
        table: str,
        key: Mapping[str, Any],
        values: Mapping[str, Any],
        *,
        expected: Mapping[str, Any] | None = None,
    ) -> int:
        """Conditional update. Returns the number of rows changed (0 or 1).

        The mutability policy is checked before anything reaches storage so both
        backends fail identically, and so the failure names the column rather
        than surfacing as a trigger error from three layers down.
        """
        allowed = MUTABLE_COLUMNS.get(table)
        if allowed is None:
            raise AppendOnlyViolation(
                f"{table} is append-only: UPDATE is forbidden. To change what "
                "the history says, append a new row that supersedes the old one."
            )
        forbidden = sorted(set(values) - allowed)
        if forbidden:
            raise AppendOnlyViolation(
                f"{table}: columns {forbidden} are immutable once written; only "
                f"{sorted(allowed)} may be updated"
            )
        return self._apply_update(table, key, values, expected or {})


def _table(table: str) -> Any:
    try:
        return models.metadata.tables[table]
    except KeyError as exc:  # pragma: no cover - a typo in this module only
        raise KeyError(f"unknown table {table!r}") from exc


class InMemoryBackend(StorageBackend):
    """Reference backend that borrows PostgreSQL's rules from the metadata.

    Column set, NOT NULL, primary key and composite foreign keys are all read
    from ``models.metadata``, so this is not a parallel schema that can drift —
    it is the same schema, executed differently. Foreign keys follow PostgreSQL's
    default ``MATCH SIMPLE``: a key with any NULL column is not checked, which is
    what makes ``supersedes_revision IS NULL`` mean "undoes nothing".

    What it does not reproduce: CHECK constraints (the record dataclasses
    validate those ranges instead), row-level security as a *privilege* (tenancy
    is enforced by the tenant_id predicate that every key carries), and real
    concurrency. Those gaps are covered by the ``postgres``-marked tests.
    """

    name = "in-memory"

    def __init__(self) -> None:
        self._rows: dict[str, dict[tuple[Any, ...], dict[str, Any]]] = {
            name: {} for name in models.metadata.tables
        }
        self._seq = 0
        self._depth = 0
        self._tenant: _uuid.UUID | None = None
        self._snapshot: dict[str, dict[tuple[Any, ...], dict[str, Any]]] | None = None

    # -- transaction ------------------------------------------------------

    @contextmanager
    def unit_of_work(self, tenant_id: Any) -> Iterator[None]:
        tid = _as_uuid(tenant_id, "tenant_id")
        if self._depth:
            if self._tenant != tid:
                raise TenantScopeError(
                    f"unit of work is already open for tenant {self._tenant}; "
                    f"cannot nest a block for tenant {tid}. Close the outer "
                    "block first — cross-tenant work must not share a transaction."
                )
            self._depth += 1
            try:
                yield
            finally:
                self._depth -= 1
            return

        self._tenant = tid
        self._depth = 1
        self._snapshot = {
            name: dict(rows) for name, rows in self._rows.items()
        }
        try:
            yield
        except BaseException:
            # Roll back to the pre-block image. Rows are replaced wholesale by
            # ``_apply_update``, never mutated, so a shallow copy per table is a
            # complete undo.
            assert self._snapshot is not None
            self._rows = self._snapshot
            raise
        finally:
            self._depth -= 1
            self._snapshot = None
            self._tenant = None

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _pk(table: str, row: Mapping[str, Any]) -> tuple[Any, ...]:
        cols = [c.name for c in _table(table).primary_key.columns]
        return tuple(row[c] for c in cols)

    def _validate_row(self, table: str, row: Mapping[str, Any]) -> None:
        t = _table(table)
        known = set(t.c.keys())
        unknown = sorted(set(row) - known)
        if unknown:
            raise ValueError(f"{table}: unknown columns {unknown}")
        for column in t.c:
            if column.name not in row:
                raise ValueError(f"{table}: missing column {column.name!r}")
            if row[column.name] is None and not column.nullable:
                raise ValueError(f"{table}: column {column.name!r} is NOT NULL")

    def _check_foreign_keys(self, table: str, row: Mapping[str, Any]) -> None:
        for fk in _table(table).foreign_key_constraints:
            local = [element.parent.name for element in fk.elements]
            values = [row[name] for name in local]
            if any(v is None for v in values):
                continue  # MATCH SIMPLE
            target_table = fk.referred_table.name
            target_cols = [element.column.name for element in fk.elements]
            predicate = dict(zip(target_cols, values))
            if not self._scan(target_table, predicate):
                raise ForeignKeyViolation(
                    f"{table}: {dict(zip(local, values))} references "
                    f"{target_table}{predicate} which does not exist"
                )

    def _scan(
        self, table: str, where: Mapping[str, Any]
    ) -> list[tuple[int, dict[str, Any]]]:
        out = []
        for row in self._rows[table].values():
            if all(row.get(k) == v for k, v in where.items()):
                out.append((row["__seq__"], row))
        return out

    # -- operations -------------------------------------------------------

    def insert(self, table: str, row: Mapping[str, Any]) -> None:
        self._validate_row(table, row)
        key = self._pk(table, row)
        if key in self._rows[table]:
            raise DuplicateRow(f"{table}: primary key {key} already exists")
        self._check_foreign_keys(table, row)
        self._seq += 1
        # Deep copy on the way in as well as on the way out: a caller holding the
        # record it just wrote must not be able to reach into stored state through
        # it, which is exactly what PostgreSQL guarantees for free.
        stored = {key: copy.deepcopy(value) for key, value in row.items()}
        stored["__seq__"] = self._seq
        self._rows[table][key] = stored

    def fetch_one(self, table: str, key: Mapping[str, Any]) -> dict[str, Any] | None:
        rows = self._scan(table, key)
        if not rows:
            return None
        if len(rows) > 1:
            raise PersistenceError(
                f"{table}: {dict(key)} matched {len(rows)} rows; a fetch_one key "
                "must be a primary key"
            )
        return self._public(rows[0][1])

    def fetch_all(
        self,
        table: str,
        where: Mapping[str, Any],
        *,
        order_by: Sequence[tuple[str, str]] = (),
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        rows = self._scan(table, where)
        for column, direction in reversed(list(order_by)):
            rows.sort(key=lambda pair: pair[1][column], reverse=direction == "desc")
        if limit is not None:
            rows = rows[:limit]
        return [self._public(row) for _, row in rows]

    def _apply_update(
        self,
        table: str,
        key: Mapping[str, Any],
        values: Mapping[str, Any],
        expected: Mapping[str, Any],
    ) -> int:
        matches = self._scan(table, {**key, **expected})
        if not matches:
            return 0
        _, current = matches[0]
        updated = dict(current)
        updated.update(values)
        self._validate_row(table, self._public(updated))
        self._rows[table][self._pk(table, updated)] = updated
        return 1

    @staticmethod
    def _public(row: Mapping[str, Any]) -> dict[str, Any]:
        return {k: copy.deepcopy(v) for k, v in row.items() if k != "__seq__"}


class SqlAlchemyBackend(StorageBackend):
    """PostgreSQL backend.

    ``unit_of_work`` opens a transaction and sets ``app.tenant_id`` with
    ``set_config(..., is_local => true)`` so the row-level-security policies
    created by the migration are in force for exactly the duration of that
    transaction. The repositories *also* put ``tenant_id`` in every predicate:
    RLS is the backstop for code that forgets, not a substitute for code that
    remembers.
    """

    name = "postgresql"

    def __init__(self, engine: Any) -> None:
        self._engine = engine
        self._conn: Any = None
        self._depth = 0
        self._tenant: _uuid.UUID | None = None

    @contextmanager
    def unit_of_work(self, tenant_id: Any) -> Iterator[None]:
        import sqlalchemy as sa

        tid = _as_uuid(tenant_id, "tenant_id")
        if self._depth:
            if self._tenant != tid:
                raise TenantScopeError(
                    f"unit of work is already open for tenant {self._tenant}; "
                    f"cannot nest a block for tenant {tid}. One transaction "
                    "carries one app.tenant_id setting."
                )
            self._depth += 1
            try:
                yield
            finally:
                self._depth -= 1
            return

        conn = self._engine.connect()
        trans = conn.begin()
        self._conn = conn
        self._tenant = tid
        self._depth = 1
        try:
            conn.execute(
                sa.text("SELECT set_config('app.tenant_id', :tid, true)"),
                {"tid": str(tid)},
            )
            yield
        except BaseException:
            trans.rollback()
            raise
        else:
            trans.commit()
        finally:
            self._depth -= 1
            self._conn = None
            self._tenant = None
            conn.close()

    # -- helpers ----------------------------------------------------------

    def _require_conn(self) -> Any:
        if self._conn is None:
            raise TenantScopeError(
                "no unit of work is open; every PostgreSQL statement must run "
                "inside unit_of_work(tenant_id) so app.tenant_id is set and the "
                "row-level-security policy can see a tenant"
            )
        return self._conn

    @staticmethod
    def _to_db(table: str, row: Mapping[str, Any]) -> dict[str, Any]:
        import sqlalchemy as sa

        t = _table(table)
        out: dict[str, Any] = {}
        for name, value in row.items():
            column = t.c[name]
            if isinstance(column.type, sa.Numeric) and value is not None:
                # Decimal(str(x)) rather than Decimal(x): the binary expansion of
                # a float would reintroduce the digits STORAGE_PRECISION exists
                # to discard.
                out[name] = Decimal(str(round(float(value), STORAGE_PRECISION)))
            else:
                out[name] = value
        return out

    @staticmethod
    def _from_db(table: str, row: Mapping[str, Any]) -> dict[str, Any]:
        import sqlalchemy as sa

        t = _table(table)
        out: dict[str, Any] = {}
        for name, value in row.items():
            column = t.c[name]
            if isinstance(column.type, sa.Numeric) and value is not None:
                out[name] = round(float(value), STORAGE_PRECISION)
            else:
                out[name] = value
        return out

    @staticmethod
    def _translate(exc: Exception) -> Exception:
        """Map PostgreSQL SQLSTATEs onto this module's error vocabulary."""
        orig = getattr(exc, "orig", None)
        # psycopg3 exposes ``sqlstate``; psycopg2 calls the same thing ``pgcode``.
        sqlstate = getattr(orig, "sqlstate", None) or getattr(orig, "pgcode", None)
        if sqlstate == "23505":
            return DuplicateRow(str(exc))
        if sqlstate == "23503":
            return ForeignKeyViolation(str(exc))
        if sqlstate == APPEND_ONLY_SQLSTATE:
            return AppendOnlyViolation(str(exc))
        if sqlstate == "42501":
            return TenantScopeError(
                "row-level security refused the statement: app.tenant_id does "
                f"not match the row's tenant_id ({exc})"
            )
        return exc

    # -- operations -------------------------------------------------------

    def insert(self, table: str, row: Mapping[str, Any]) -> None:
        import sqlalchemy as sa

        conn = self._require_conn()
        try:
            conn.execute(sa.insert(_table(table)).values(self._to_db(table, row)))
        except sa.exc.DatabaseError as exc:
            raise self._translate(exc) from exc

    def fetch_one(self, table: str, key: Mapping[str, Any]) -> dict[str, Any] | None:
        rows = self.fetch_all(table, key, limit=2)
        if not rows:
            return None
        if len(rows) > 1:
            raise PersistenceError(
                f"{table}: {dict(key)} matched more than one row; a fetch_one "
                "key must be a primary key"
            )
        return rows[0]

    def fetch_all(
        self,
        table: str,
        where: Mapping[str, Any],
        *,
        order_by: Sequence[tuple[str, str]] = (),
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        import sqlalchemy as sa

        conn = self._require_conn()
        t = _table(table)
        stmt = sa.select(t)
        for name, value in where.items():
            stmt = stmt.where(t.c[name] == value)
        for name, direction in order_by:
            column = t.c[name]
            stmt = stmt.order_by(column.desc() if direction == "desc" else column.asc())
        if limit is not None:
            stmt = stmt.limit(limit)
        try:
            result = conn.execute(stmt)
        except sa.exc.DatabaseError as exc:
            raise self._translate(exc) from exc
        return [self._from_db(table, dict(r._mapping)) for r in result]

    def _apply_update(
        self,
        table: str,
        key: Mapping[str, Any],
        values: Mapping[str, Any],
        expected: Mapping[str, Any],
    ) -> int:
        import sqlalchemy as sa

        conn = self._require_conn()
        t = _table(table)
        stmt = sa.update(t)
        for name, value in {**key, **expected}.items():
            stmt = stmt.where(t.c[name] == value)
        try:
            result = conn.execute(stmt.values(self._to_db(table, values)))
        except sa.exc.DatabaseError as exc:
            raise self._translate(exc) from exc
        return int(result.rowcount)


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProfileHead:
    """The current profile of one user in one tenant."""

    tenant_id: _uuid.UUID
    user_id: _uuid.UUID
    current_revision: int
    profile: dict[str, Any]
    profile_hash: str
    effective_manifest_hash: str
    created_at: _dt.datetime
    updated_at: _dt.datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "tenant_id", _as_uuid(self.tenant_id, "tenant_id"))
        object.__setattr__(self, "user_id", _as_uuid(self.user_id, "user_id"))
        object.__setattr__(
            self,
            "current_revision",
            _as_revision(self.current_revision, "current_revision"),
        )
        object.__setattr__(self, "profile", _require_mapping(self.profile, "profile"))
        object.__setattr__(self, "created_at", _as_utc(self.created_at, "created_at"))
        object.__setattr__(self, "updated_at", _as_utc(self.updated_at, "updated_at"))

    def to_json(self) -> dict[str, Any]:
        return {
            "tenant_id": str(self.tenant_id),
            "user_id": str(self.user_id),
            "current_revision": self.current_revision,
            "profile": self.profile,
            "profile_hash": self.profile_hash,
            "effective_manifest_hash": self.effective_manifest_hash,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class ProfileRevision:
    """One immutable entry in a user's profile history."""

    tenant_id: _uuid.UUID
    user_id: _uuid.UUID
    revision: int
    source_event_ids: list[str]
    delta: dict[str, Any]
    full_snapshot: dict[str, Any]
    confidence_summary: dict[str, Any]
    supersedes_revision: int | None
    vdf_proof_id: _uuid.UUID
    created_at: _dt.datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "tenant_id", _as_uuid(self.tenant_id, "tenant_id"))
        object.__setattr__(self, "user_id", _as_uuid(self.user_id, "user_id"))
        object.__setattr__(self, "revision", _as_revision(self.revision, "revision"))
        object.__setattr__(
            self,
            "source_event_ids",
            _require_sequence(self.source_event_ids, "source_event_ids",
                              allow_empty=False),
        )
        object.__setattr__(self, "delta", _require_mapping(self.delta, "delta"))
        object.__setattr__(
            self, "full_snapshot", _require_mapping(self.full_snapshot, "full_snapshot")
        )
        object.__setattr__(
            self,
            "confidence_summary",
            _require_mapping(self.confidence_summary, "confidence_summary"),
        )
        if self.supersedes_revision is not None:
            superseded = _as_revision(self.supersedes_revision, "supersedes_revision")
            if superseded >= self.revision:
                raise ValueError(
                    f"supersedes_revision {superseded} must be older than "
                    f"revision {self.revision}; a rollback points backwards"
                )
            object.__setattr__(self, "supersedes_revision", superseded)
        object.__setattr__(
            self, "vdf_proof_id", _as_uuid(self.vdf_proof_id, "vdf_proof_id")
        )
        object.__setattr__(self, "created_at", _as_utc(self.created_at, "created_at"))
        _require_contract(self.full_snapshot, "user_profile", "full_snapshot")

    @property
    def is_rollback(self) -> bool:
        return self.supersedes_revision is not None

    def to_json(self) -> dict[str, Any]:
        return {
            "tenant_id": str(self.tenant_id),
            "user_id": str(self.user_id),
            "revision": self.revision,
            "source_event_ids": self.source_event_ids,
            "delta": self.delta,
            "full_snapshot": self.full_snapshot,
            "confidence_summary": self.confidence_summary,
            "supersedes_revision": self.supersedes_revision,
            "vdf_proof_id": str(self.vdf_proof_id),
            "created_at": self.created_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class GroupIntentRevision:
    """One revision of a group's aggregated intent, dissent included."""

    tenant_id: _uuid.UUID
    group_id: _uuid.UUID
    revision: int
    aggregate_intent: dict[str, Any]
    member_intents: dict[str, Any]
    decision_policy: dict[str, Any]
    dissent: list[Any]
    vdf_proof_id: _uuid.UUID
    created_at: _dt.datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "tenant_id", _as_uuid(self.tenant_id, "tenant_id"))
        object.__setattr__(self, "group_id", _as_uuid(self.group_id, "group_id"))
        object.__setattr__(self, "revision", _as_revision(self.revision, "revision"))
        object.__setattr__(
            self,
            "aggregate_intent",
            _require_mapping(self.aggregate_intent, "aggregate_intent"),
        )
        object.__setattr__(
            self,
            "member_intents",
            _require_mapping(self.member_intents, "member_intents"),
        )
        object.__setattr__(
            self,
            "decision_policy",
            _require_mapping(self.decision_policy, "decision_policy"),
        )
        # invariants.preserve_minority_positions: [] means "none observed",
        # None would mean "we did not look", and only the first is a fact.
        object.__setattr__(
            self, "dissent", _require_sequence(self.dissent, "dissent", allow_empty=True)
        )
        object.__setattr__(
            self, "vdf_proof_id", _as_uuid(self.vdf_proof_id, "vdf_proof_id")
        )
        object.__setattr__(self, "created_at", _as_utc(self.created_at, "created_at"))

    def to_json(self) -> dict[str, Any]:
        return {
            "tenant_id": str(self.tenant_id),
            "group_id": str(self.group_id),
            "revision": self.revision,
            "aggregate_intent": self.aggregate_intent,
            "member_intents": self.member_intents,
            "decision_policy": self.decision_policy,
            "dissent": self.dissent,
            "vdf_proof_id": str(self.vdf_proof_id),
            "created_at": self.created_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class TensionSnapshot:
    """A measured tension gradient bound to the intent revision it was measured against."""

    tenant_id: _uuid.UUID
    group_id: _uuid.UUID
    snapshot_id: _uuid.UUID
    group_intent_revision: int
    member_gradients: dict[str, Any]
    group_tension: float
    tension_class: str
    confidence: float
    topology_descriptor: dict[str, Any]
    vdf_proof_id: _uuid.UUID
    observed_at: _dt.datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "tenant_id", _as_uuid(self.tenant_id, "tenant_id"))
        object.__setattr__(self, "group_id", _as_uuid(self.group_id, "group_id"))
        object.__setattr__(
            self, "snapshot_id", _as_uuid(self.snapshot_id, "snapshot_id")
        )
        object.__setattr__(
            self,
            "group_intent_revision",
            _as_revision(self.group_intent_revision, "group_intent_revision"),
        )
        object.__setattr__(
            self,
            "member_gradients",
            _require_mapping(self.member_gradients, "member_gradients"),
        )
        object.__setattr__(
            self, "group_tension", _as_ratio(self.group_tension, "group_tension")
        )
        if self.tension_class not in TENSION_CLASSES:
            raise ValueError(
                f"tension_class {self.tension_class!r} is not one of "
                f"{list(TENSION_CLASSES)}"
            )
        object.__setattr__(self, "confidence", _as_ratio(self.confidence, "confidence"))
        object.__setattr__(
            self,
            "topology_descriptor",
            _require_mapping(self.topology_descriptor, "topology_descriptor"),
        )
        object.__setattr__(
            self, "vdf_proof_id", _as_uuid(self.vdf_proof_id, "vdf_proof_id")
        )
        object.__setattr__(
            self, "observed_at", _as_utc(self.observed_at, "observed_at")
        )
        _require_contract(self.as_contract_document(), "tension_snapshot", "snapshot")

    def as_contract_document(self) -> dict[str, Any]:
        """The ``jsonb_contracts.tension_snapshot`` view of this row."""
        return {
            "group_id": str(self.group_id),
            "group_intent_revision": self.group_intent_revision,
            "member_gradients": self.member_gradients,
            "group_tension": self.group_tension,
            "tension_class": self.tension_class,
            "confidence": self.confidence,
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "tenant_id": str(self.tenant_id),
            "snapshot_id": str(self.snapshot_id),
            "topology_descriptor": self.topology_descriptor,
            "vdf_proof_id": str(self.vdf_proof_id),
            "observed_at": self.observed_at.isoformat(),
            **self.as_contract_document(),
        }


@dataclass(frozen=True, slots=True)
class BridgeAction:
    """A proposed, applied, rejected, or rolled-back bridge."""

    tenant_id: _uuid.UUID
    bridge_id: _uuid.UUID
    group_id: _uuid.UUID
    bridge_type: str
    proposal: dict[str, Any]
    simulation: dict[str, Any]
    authorization: dict[str, Any]
    status: str
    rollback_of: _uuid.UUID | None
    vdf_proof_id: _uuid.UUID
    created_at: _dt.datetime
    applied_at: _dt.datetime | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "tenant_id", _as_uuid(self.tenant_id, "tenant_id"))
        object.__setattr__(self, "bridge_id", _as_uuid(self.bridge_id, "bridge_id"))
        object.__setattr__(self, "group_id", _as_uuid(self.group_id, "group_id"))
        if self.bridge_type not in BRIDGE_STRATEGIES:
            raise ValueError(
                f"bridge_type {self.bridge_type!r} is not one of "
                f"{list(BRIDGE_STRATEGIES)} (bridge_engine.ordered_strategies)"
            )
        object.__setattr__(self, "proposal", _require_mapping(self.proposal, "proposal"))
        object.__setattr__(
            self, "simulation", _require_mapping(self.simulation, "simulation")
        )
        object.__setattr__(
            self, "authorization", _require_mapping(self.authorization, "authorization")
        )
        if self.status not in BRIDGE_STATUSES:
            raise ValueError(
                f"status {self.status!r} is not one of {list(BRIDGE_STATUSES)}"
            )
        if self.rollback_of is not None:
            rollback_of = _as_uuid(self.rollback_of, "rollback_of")
            if rollback_of == self.bridge_id:
                raise ValueError(
                    "rollback_of must name a different bridge action; an action "
                    "cannot roll itself back"
                )
            object.__setattr__(self, "rollback_of", rollback_of)
        object.__setattr__(
            self, "vdf_proof_id", _as_uuid(self.vdf_proof_id, "vdf_proof_id")
        )
        object.__setattr__(self, "created_at", _as_utc(self.created_at, "created_at"))
        if self.applied_at is not None:
            object.__setattr__(
                self, "applied_at", _as_utc(self.applied_at, "applied_at")
            )
        if self.status == "applied" and self.applied_at is None:
            raise ValueError(
                "an applied bridge must record applied_at; the observed-versus-"
                "predicted comparison the rollback trigger needs is timed from it"
            )
        # simulation.required_before_apply + the application rule both need the
        # before/after estimates and the boundary/authorization verdicts to be
        # present on the stored row, not recomputed from the proposal later.
        for key in ("predicted_before", "predicted_after"):
            if key not in self.simulation:
                raise ContractViolation(
                    f"simulation is missing {key!r}; "
                    "simulation.estimate_before_and_after is required before a "
                    "bridge may be stored"
                )
        for key in ("boundary_violation", "authorized"):
            if key not in self.authorization:
                raise ContractViolation(
                    f"authorization is missing {key!r}; jsonb_contracts."
                    "bridge_action requires the boundary and authorization "
                    "verdicts to be recorded with the action"
                )

    def as_contract_document(self) -> dict[str, Any]:
        """The ``jsonb_contracts.bridge_action`` view of this row."""
        return {
            "bridge_id": str(self.bridge_id),
            "bridge_type": self.bridge_type,
            "predicted_before": self.simulation["predicted_before"],
            "predicted_after": self.simulation["predicted_after"],
            "boundary_violation": self.authorization["boundary_violation"],
            "authorized": self.authorization["authorized"],
            "status": self.status,
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "tenant_id": str(self.tenant_id),
            "group_id": str(self.group_id),
            "proposal": self.proposal,
            "simulation": self.simulation,
            "authorization": self.authorization,
            "rollback_of": str(self.rollback_of) if self.rollback_of else None,
            "vdf_proof_id": str(self.vdf_proof_id),
            "created_at": self.created_at.isoformat(),
            "applied_at": self.applied_at.isoformat() if self.applied_at else None,
            **self.as_contract_document(),
        }


@dataclass(frozen=True, slots=True)
class VdfAttestation:
    """A Rule 30 proof envelope bound to the event it attests."""

    tenant_id: _uuid.UUID
    proof_id: _uuid.UUID
    event_id: _uuid.UUID
    envelope: dict[str, Any]
    verification_status: str
    created_at: _dt.datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "tenant_id", _as_uuid(self.tenant_id, "tenant_id"))
        object.__setattr__(self, "proof_id", _as_uuid(self.proof_id, "proof_id"))
        object.__setattr__(self, "event_id", _as_uuid(self.event_id, "event_id"))
        object.__setattr__(self, "envelope", _require_mapping(self.envelope, "envelope"))
        if self.verification_status not in VERIFICATION_STATUSES:
            raise ValueError(
                f"verification_status {self.verification_status!r} is not one of "
                f"{list(VERIFICATION_STATUSES)}"
            )
        object.__setattr__(self, "created_at", _as_utc(self.created_at, "created_at"))
        required = cached_model_card().rule30_vdf.proof_envelope.required_fields
        missing = [key for key in required if key not in self.envelope]
        if missing:
            raise ContractViolation(
                f"envelope is missing {missing}; "
                f"rule30_vdf.proof_envelope.required_fields is {list(required)} "
                "and offline verification needs all of them"
            )

    def to_json(self) -> dict[str, Any]:
        return {
            "tenant_id": str(self.tenant_id),
            "proof_id": str(self.proof_id),
            "event_id": str(self.event_id),
            "envelope": self.envelope,
            "verification_status": self.verification_status,
            "created_at": self.created_at.isoformat(),
        }


# --------------------------------------------------------------------------
# Repositories
# --------------------------------------------------------------------------


class _Repository:
    """Shared plumbing: a backend, and the reported backend name."""

    def __init__(self, backend: StorageBackend) -> None:
        self._backend = backend

    @property
    def backend(self) -> str:
        return self._backend.name

    @property
    def storage(self) -> StorageBackend:
        return self._backend


class ProfileRepository(_Repository):
    """Append-only user profiles with an optimistically-concurrent head."""

    HEAD = "gc_user_profiles"
    HISTORY = "gc_profile_revisions"

    # -- reads ------------------------------------------------------------

    def get_profile(self, tenant_id: Any, user_id: Any) -> ProfileHead | None:
        tid, uid = _as_uuid(tenant_id, "tenant_id"), _as_uuid(user_id, "user_id")
        with self._backend.unit_of_work(tid):
            row = self._backend.fetch_one(
                self.HEAD, {"tenant_id": tid, "user_id": uid}
            )
        return ProfileHead(**row) if row else None

    def current_revision(self, tenant_id: Any, user_id: Any) -> int:
        """0 when the user has no profile yet — the value to pass as
        ``expected_revision`` for the very first write."""
        head = self.get_profile(tenant_id, user_id)
        return head.current_revision if head else 0

    def get_revision(
        self, tenant_id: Any, user_id: Any, revision: int
    ) -> ProfileRevision | None:
        tid, uid = _as_uuid(tenant_id, "tenant_id"), _as_uuid(user_id, "user_id")
        rev = _as_revision(revision, "revision")
        with self._backend.unit_of_work(tid):
            row = self._backend.fetch_one(
                self.HISTORY, {"tenant_id": tid, "user_id": uid, "revision": rev}
            )
        return ProfileRevision(**row) if row else None

    def list_revisions(
        self, tenant_id: Any, user_id: Any, *, limit: int | None = None
    ) -> list[ProfileRevision]:
        """Newest first, matching ``btree(tenant_id, user_id, revision desc)``."""
        tid, uid = _as_uuid(tenant_id, "tenant_id"), _as_uuid(user_id, "user_id")
        with self._backend.unit_of_work(tid):
            rows = self._backend.fetch_all(
                self.HISTORY,
                {"tenant_id": tid, "user_id": uid},
                order_by=(("revision", "desc"),),
                limit=limit,
            )
        return [ProfileRevision(**row) for row in rows]

    # -- writes -----------------------------------------------------------

    def append_revision(
        self,
        tenant_id: Any,
        user_id: Any,
        *,
        expected_revision: int,
        delta: Mapping[str, Any],
        full_snapshot: Mapping[str, Any],
        source_event_ids: Sequence[str],
        confidence_summary: Mapping[str, Any],
        vdf_proof_id: Any,
        profile_hash: str,
        effective_manifest_hash: str,
        supersedes_revision: int | None = None,
        now: _dt.datetime | None = None,
    ) -> ProfileRevision:
        """Append revision ``expected_revision + 1`` and advance the head.

        ``expected_revision`` is the revision the caller read. Pass ``0`` to
        create the profile. A mismatch raises :class:`RevisionConflict` — the
        write is refused, never merged, because merging two unrelated deltas is
        how an inference the user corrected comes back.
        """
        tid, uid = _as_uuid(tenant_id, "tenant_id"), _as_uuid(user_id, "user_id")
        expected = _as_revision(expected_revision, "expected_revision", minimum=0)
        stamp = _as_utc(now or _utcnow(), "now")

        with self._backend.unit_of_work(tid):
            head_row = self._backend.fetch_one(
                self.HEAD, {"tenant_id": tid, "user_id": uid}
            )
            current = int(head_row["current_revision"]) if head_row else 0
            if expected != current:
                raise self._conflict(tid, uid, expected, current)

            revision = ProfileRevision(
                tenant_id=tid,
                user_id=uid,
                revision=current + 1,
                source_event_ids=source_event_ids,
                delta=delta,
                full_snapshot=full_snapshot,
                confidence_summary=confidence_summary,
                supersedes_revision=supersedes_revision,
                vdf_proof_id=vdf_proof_id,
                created_at=stamp,
            )

            # Take the head first: it is the lock. Once the compare-and-swap
            # succeeds this transaction owns revision N+1, so the history insert
            # below cannot lose a race with a second writer.
            if head_row is None:
                try:
                    self._backend.insert(
                        self.HEAD,
                        {
                            "tenant_id": tid,
                            "user_id": uid,
                            "current_revision": revision.revision,
                            "profile": revision.full_snapshot,
                            "profile_hash": profile_hash,
                            "effective_manifest_hash": effective_manifest_hash,
                            "created_at": stamp,
                            "updated_at": stamp,
                        },
                    )
                except DuplicateRow as exc:
                    raise RevisionConflict(
                        f"profile {uid} in tenant {tid} was created concurrently "
                        "while this write believed it did not exist; re-read it "
                        "with get_profile() and retry with the revision it "
                        "reports",
                        table=self.HEAD,
                        tenant_id=tid,
                        entity=str(uid),
                        expected=0,
                        actual=None,
                    ) from exc
            else:
                changed = self._backend.update(
                    self.HEAD,
                    {"tenant_id": tid, "user_id": uid},
                    {
                        "current_revision": revision.revision,
                        "profile": revision.full_snapshot,
                        "profile_hash": profile_hash,
                        "effective_manifest_hash": effective_manifest_hash,
                        "updated_at": stamp,
                    },
                    expected={"current_revision": current},
                )
                if changed != 1:
                    raise self._conflict(tid, uid, expected, None)

            self._backend.insert(self.HISTORY, self._history_row(revision))

        return revision

    def rollback_to(
        self,
        tenant_id: Any,
        user_id: Any,
        target_revision: int,
        *,
        expected_revision: int,
        source_event_ids: Sequence[str],
        vdf_proof_id: Any,
        profile_hash: str,
        effective_manifest_hash: str,
        now: _dt.datetime | None = None,
    ) -> ProfileRevision:
        """Undo by appending, never by deleting.

        Writes a NEW revision whose ``full_snapshot`` is exactly the target's and
        whose ``supersedes_revision`` names the revision being undone (the head
        at the time of the call). ``user_controls.rollback`` and
        ``append_only_revision_history`` are only compatible this way: after a
        rollback the history says both what was believed and that it was undone.

        ``profile_hash`` and ``effective_manifest_hash`` are required rather than
        copied from the target: the manifest hash is a *recompile* output, and
        inventing one here would attest a compile that never ran.
        """
        tid, uid = _as_uuid(tenant_id, "tenant_id"), _as_uuid(user_id, "user_id")
        target = _as_revision(target_revision, "target_revision")
        expected = _as_revision(expected_revision, "expected_revision", minimum=1)

        with self._backend.unit_of_work(tid):
            head_row = self._backend.fetch_one(
                self.HEAD, {"tenant_id": tid, "user_id": uid}
            )
            if head_row is None:
                raise RevisionConflict(
                    f"cannot roll back profile {uid} in tenant {tid}: it has no "
                    "revisions. Create it with append_revision(expected_revision=0) "
                    "first.",
                    table=self.HEAD,
                    tenant_id=tid,
                    entity=str(uid),
                    expected=expected,
                    actual=0,
                )
            current = int(head_row["current_revision"])
            if expected != current:
                raise self._conflict(tid, uid, expected, current)
            if target > current:
                raise ValueError(
                    f"target_revision {target} is ahead of the current revision "
                    f"{current}; rollback restores a past state, it does not "
                    "invent a future one"
                )

            target_row = self._backend.fetch_one(
                self.HISTORY, {"tenant_id": tid, "user_id": uid, "revision": target}
            )
            if target_row is None:
                raise ValueError(
                    f"revision {target} of profile {uid} in tenant {tid} does not "
                    "exist; list_revisions() reports what can be rolled back to"
                )
            restored = ProfileRevision(**target_row)

            return self.append_revision(
                tid,
                uid,
                expected_revision=current,
                delta={
                    "rollback": {
                        "to_revision": target,
                        "undoes_revision": current,
                    }
                },
                full_snapshot=restored.full_snapshot,
                source_event_ids=source_event_ids,
                # The confidence that justified the restored state travels with
                # it; recomputing it would claim evidence the rollback never saw.
                confidence_summary=restored.confidence_summary,
                vdf_proof_id=vdf_proof_id,
                profile_hash=profile_hash,
                effective_manifest_hash=effective_manifest_hash,
                supersedes_revision=current,
                now=now,
            )

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _history_row(revision: ProfileRevision) -> dict[str, Any]:
        return {
            "tenant_id": revision.tenant_id,
            "user_id": revision.user_id,
            "revision": revision.revision,
            "source_event_ids": revision.source_event_ids,
            "delta": revision.delta,
            "full_snapshot": revision.full_snapshot,
            "confidence_summary": revision.confidence_summary,
            "supersedes_revision": revision.supersedes_revision,
            "vdf_proof_id": revision.vdf_proof_id,
            "created_at": revision.created_at,
        }

    def _conflict(
        self, tid: _uuid.UUID, uid: _uuid.UUID, expected: int, actual: int | None
    ) -> RevisionConflict:
        seen = "another writer won the compare-and-swap" if actual is None else (
            f"the stored head is at revision {actual}"
        )
        return RevisionConflict(
            f"profile {uid} in tenant {tid}: write expected revision "
            f"{expected} but {seen}. Re-read with get_profile(), rebase the "
            f"delta onto the current snapshot, and retry with "
            f"expected_revision set to the revision you just read.",
            table=self.HEAD,
            tenant_id=tid,
            entity=str(uid),
            expected=expected,
            actual=actual,
        )


class GroupIntentRepository(_Repository):
    """Append-only group intent with compare-and-swap on the revision number.

    There is no head table: the primary key ``(tenant_id, group_id, revision)``
    *is* the compare-and-swap. Two writers that both read revision 4 both try to
    insert 5, and exactly one insert survives — the loser gets
    :class:`RevisionConflict` instead of silently clobbering a consensus it never
    saw, which is what ``deployment.consistency.group_intent_revision:
    compare-and-swap`` asks for.
    """

    TABLE = "gc_group_intents"

    def latest(self, tenant_id: Any, group_id: Any) -> GroupIntentRevision | None:
        tid, gid = _as_uuid(tenant_id, "tenant_id"), _as_uuid(group_id, "group_id")
        with self._backend.unit_of_work(tid):
            rows = self._backend.fetch_all(
                self.TABLE,
                {"tenant_id": tid, "group_id": gid},
                order_by=(("revision", "desc"),),
                limit=1,
            )
        return GroupIntentRevision(**rows[0]) if rows else None

    def current_revision(self, tenant_id: Any, group_id: Any) -> int:
        latest = self.latest(tenant_id, group_id)
        return latest.revision if latest else 0

    def get(
        self, tenant_id: Any, group_id: Any, revision: int
    ) -> GroupIntentRevision | None:
        tid, gid = _as_uuid(tenant_id, "tenant_id"), _as_uuid(group_id, "group_id")
        rev = _as_revision(revision, "revision")
        with self._backend.unit_of_work(tid):
            row = self._backend.fetch_one(
                self.TABLE, {"tenant_id": tid, "group_id": gid, "revision": rev}
            )
        return GroupIntentRevision(**row) if row else None

    def list_revisions(
        self, tenant_id: Any, group_id: Any, *, limit: int | None = None
    ) -> list[GroupIntentRevision]:
        tid, gid = _as_uuid(tenant_id, "tenant_id"), _as_uuid(group_id, "group_id")
        with self._backend.unit_of_work(tid):
            rows = self._backend.fetch_all(
                self.TABLE,
                {"tenant_id": tid, "group_id": gid},
                order_by=(("revision", "desc"),),
                limit=limit,
            )
        return [GroupIntentRevision(**row) for row in rows]

    def append_intent(
        self,
        tenant_id: Any,
        group_id: Any,
        *,
        expected_revision: int,
        aggregate_intent: Mapping[str, Any],
        member_intents: Mapping[str, Any],
        decision_policy: Mapping[str, Any],
        dissent: Sequence[Any],
        vdf_proof_id: Any,
        now: _dt.datetime | None = None,
    ) -> GroupIntentRevision:
        tid, gid = _as_uuid(tenant_id, "tenant_id"), _as_uuid(group_id, "group_id")
        expected = _as_revision(expected_revision, "expected_revision", minimum=0)
        stamp = _as_utc(now or _utcnow(), "now")

        with self._backend.unit_of_work(tid):
            rows = self._backend.fetch_all(
                self.TABLE,
                {"tenant_id": tid, "group_id": gid},
                order_by=(("revision", "desc"),),
                limit=1,
            )
            current = int(rows[0]["revision"]) if rows else 0
            if expected != current:
                raise self._conflict(tid, gid, expected, current)

            intent = GroupIntentRevision(
                tenant_id=tid,
                group_id=gid,
                revision=current + 1,
                aggregate_intent=aggregate_intent,
                member_intents=member_intents,
                decision_policy=decision_policy,
                dissent=dissent,
                vdf_proof_id=vdf_proof_id,
                created_at=stamp,
            )
            try:
                self._backend.insert(
                    self.TABLE,
                    {
                        "tenant_id": intent.tenant_id,
                        "group_id": intent.group_id,
                        "revision": intent.revision,
                        "aggregate_intent": intent.aggregate_intent,
                        "member_intents": intent.member_intents,
                        "decision_policy": intent.decision_policy,
                        "dissent": intent.dissent,
                        "vdf_proof_id": intent.vdf_proof_id,
                        "created_at": intent.created_at,
                    },
                )
            except DuplicateRow as exc:
                raise self._conflict(tid, gid, expected, None) from exc
        return intent

    def _conflict(
        self, tid: _uuid.UUID, gid: _uuid.UUID, expected: int, actual: int | None
    ) -> RevisionConflict:
        seen = (
            "another writer inserted that revision first"
            if actual is None
            else f"the stored latest revision is {actual}"
        )
        return RevisionConflict(
            f"group {gid} in tenant {tid}: write expected revision {expected} "
            f"but {seen}. Re-read with latest(), re-aggregate against the "
            "current member intents, and retry with the revision you just read.",
            table=self.TABLE,
            tenant_id=tid,
            entity=str(gid),
            expected=expected,
            actual=actual,
        )


class TensionRepository(_Repository):
    """Immutable tension observations, keyed by a caller-supplied snapshot id.

    Re-recording an identical snapshot succeeds (at-least-once delivery with
    idempotent consumers); re-using the id for different content raises
    :class:`IdempotencyConflict` rather than rewriting an attested measurement.
    """

    TABLE = "gc_tension_snapshots"

    def record_snapshot(
        self,
        tenant_id: Any,
        group_id: Any,
        snapshot_id: Any,
        *,
        group_intent_revision: int,
        member_gradients: Mapping[str, Any],
        group_tension: float,
        tension_class: str,
        confidence: float,
        topology_descriptor: Mapping[str, Any],
        vdf_proof_id: Any,
        observed_at: _dt.datetime | None = None,
    ) -> TensionSnapshot:
        snapshot = TensionSnapshot(
            tenant_id=tenant_id,
            group_id=group_id,
            snapshot_id=snapshot_id,
            group_intent_revision=group_intent_revision,
            member_gradients=member_gradients,
            group_tension=group_tension,
            tension_class=tension_class,
            confidence=confidence,
            topology_descriptor=topology_descriptor,
            vdf_proof_id=vdf_proof_id,
            observed_at=observed_at or _utcnow(),
        )
        row = {
            "tenant_id": snapshot.tenant_id,
            "group_id": snapshot.group_id,
            "snapshot_id": snapshot.snapshot_id,
            "group_intent_revision": snapshot.group_intent_revision,
            "member_gradients": snapshot.member_gradients,
            "group_tension": snapshot.group_tension,
            "tension_class": snapshot.tension_class,
            "confidence": snapshot.confidence,
            "topology_descriptor": snapshot.topology_descriptor,
            "vdf_proof_id": snapshot.vdf_proof_id,
            "observed_at": snapshot.observed_at,
        }
        with self._backend.unit_of_work(snapshot.tenant_id):
            try:
                self._backend.insert(self.TABLE, row)
            except DuplicateRow as exc:
                existing = self._backend.fetch_one(
                    self.TABLE,
                    {
                        "tenant_id": snapshot.tenant_id,
                        "group_id": snapshot.group_id,
                        "snapshot_id": snapshot.snapshot_id,
                    },
                )
                if existing is not None and TensionSnapshot(**existing) == snapshot:
                    return snapshot
                raise IdempotencyConflict(
                    f"snapshot {snapshot.snapshot_id} already exists in tenant "
                    f"{snapshot.tenant_id} with different content; a snapshot id "
                    "identifies one measurement. Use a new snapshot_id for a new "
                    "measurement."
                ) from exc
        return snapshot

    def get(
        self, tenant_id: Any, group_id: Any, snapshot_id: Any
    ) -> TensionSnapshot | None:
        tid = _as_uuid(tenant_id, "tenant_id")
        key = {
            "tenant_id": tid,
            "group_id": _as_uuid(group_id, "group_id"),
            "snapshot_id": _as_uuid(snapshot_id, "snapshot_id"),
        }
        with self._backend.unit_of_work(tid):
            row = self._backend.fetch_one(self.TABLE, key)
        return TensionSnapshot(**row) if row else None

    def latest(self, tenant_id: Any, group_id: Any) -> TensionSnapshot | None:
        """Newest observation, matching ``btree(tenant_id, group_id, observed_at desc)``.

        ``snapshot_id`` breaks ties so two snapshots recorded in the same
        microsecond order identically on both backends.
        """
        rows = self.list_for_group(tenant_id, group_id, limit=1)
        return rows[0] if rows else None

    def list_for_group(
        self, tenant_id: Any, group_id: Any, *, limit: int | None = None
    ) -> list[TensionSnapshot]:
        tid, gid = _as_uuid(tenant_id, "tenant_id"), _as_uuid(group_id, "group_id")
        with self._backend.unit_of_work(tid):
            rows = self._backend.fetch_all(
                self.TABLE,
                {"tenant_id": tid, "group_id": gid},
                order_by=(("observed_at", "desc"), ("snapshot_id", "desc")),
                limit=limit,
            )
        return [TensionSnapshot(**row) for row in rows]

    def list_by_class(
        self, tenant_id: Any, tension_class: str, *, limit: int | None = None
    ) -> list[TensionSnapshot]:
        """Backs ``btree(tenant_id, tension_class)`` — the escalation queue read."""
        if tension_class not in TENSION_CLASSES:
            raise ValueError(
                f"tension_class {tension_class!r} is not one of {list(TENSION_CLASSES)}"
            )
        tid = _as_uuid(tenant_id, "tenant_id")
        with self._backend.unit_of_work(tid):
            rows = self._backend.fetch_all(
                self.TABLE,
                {"tenant_id": tid, "tension_class": tension_class},
                order_by=(("observed_at", "desc"), ("snapshot_id", "desc")),
                limit=limit,
            )
        return [TensionSnapshot(**row) for row in rows]


class BridgeRepository(_Repository):
    """Bridge lifecycle, with rollback as a new action rather than an erasure."""

    TABLE = "gc_bridge_actions"

    def propose(
        self,
        tenant_id: Any,
        bridge_id: Any,
        *,
        group_id: Any,
        bridge_type: str,
        proposal: Mapping[str, Any],
        simulation: Mapping[str, Any],
        authorization: Mapping[str, Any],
        vdf_proof_id: Any,
        now: _dt.datetime | None = None,
    ) -> BridgeAction:
        action = BridgeAction(
            tenant_id=tenant_id,
            bridge_id=bridge_id,
            group_id=group_id,
            bridge_type=bridge_type,
            proposal=proposal,
            simulation=simulation,
            authorization=authorization,
            status="proposed",
            rollback_of=None,
            vdf_proof_id=vdf_proof_id,
            created_at=now or _utcnow(),
            applied_at=None,
        )
        with self._backend.unit_of_work(action.tenant_id):
            self._insert(action)
        return action

    def get(self, tenant_id: Any, bridge_id: Any) -> BridgeAction | None:
        tid = _as_uuid(tenant_id, "tenant_id")
        with self._backend.unit_of_work(tid):
            row = self._backend.fetch_one(
                self.TABLE,
                {"tenant_id": tid, "bridge_id": _as_uuid(bridge_id, "bridge_id")},
            )
        return BridgeAction(**row) if row else None

    def list_for_group(
        self, tenant_id: Any, group_id: Any, *, limit: int | None = None
    ) -> list[BridgeAction]:
        tid, gid = _as_uuid(tenant_id, "tenant_id"), _as_uuid(group_id, "group_id")
        with self._backend.unit_of_work(tid):
            rows = self._backend.fetch_all(
                self.TABLE,
                {"tenant_id": tid, "group_id": gid},
                order_by=(("created_at", "desc"), ("bridge_id", "desc")),
                limit=limit,
            )
        return [BridgeAction(**row) for row in rows]

    def list_by_status(
        self, tenant_id: Any, status: str, *, limit: int | None = None
    ) -> list[BridgeAction]:
        if status not in BRIDGE_STATUSES:
            raise ValueError(f"status {status!r} is not one of {list(BRIDGE_STATUSES)}")
        tid = _as_uuid(tenant_id, "tenant_id")
        with self._backend.unit_of_work(tid):
            rows = self._backend.fetch_all(
                self.TABLE,
                {"tenant_id": tid, "status": status},
                order_by=(("created_at", "desc"), ("bridge_id", "desc")),
                limit=limit,
            )
        return [BridgeAction(**row) for row in rows]

    def mark_applied(
        self,
        tenant_id: Any,
        bridge_id: Any,
        *,
        expected_status: str = "proposed",
        applied_at: _dt.datetime | None = None,
    ) -> BridgeAction:
        return self._transition(
            tenant_id,
            bridge_id,
            new_status="applied",
            expected_status=expected_status,
            applied_at=_as_utc(applied_at or _utcnow(), "applied_at"),
        )

    def mark_rejected(
        self, tenant_id: Any, bridge_id: Any, *, expected_status: str = "proposed"
    ) -> BridgeAction:
        return self._transition(
            tenant_id, bridge_id, new_status="rejected",
            expected_status=expected_status, applied_at=None,
        )

    def rollback(
        self,
        tenant_id: Any,
        bridge_id: Any,
        *,
        new_bridge_id: Any,
        proposal: Mapping[str, Any],
        simulation: Mapping[str, Any],
        authorization: Mapping[str, Any],
        vdf_proof_id: Any,
        now: _dt.datetime | None = None,
    ) -> BridgeAction:
        """Append the undo, then mark the original rolled-back.

        The rollback is itself an applied bridge action — it changes the world —
        so it carries its own simulation and authorization. ``bridge_engine.
        rollback.trigger_conditions`` fire on evidence, and that evidence has to
        be as inspectable as the evidence for the original apply.
        """
        tid = _as_uuid(tenant_id, "tenant_id")
        original_id = _as_uuid(bridge_id, "bridge_id")
        stamp = _as_utc(now or _utcnow(), "now")

        with self._backend.unit_of_work(tid):
            original_row = self._backend.fetch_one(
                self.TABLE, {"tenant_id": tid, "bridge_id": original_id}
            )
            if original_row is None:
                raise ValueError(
                    f"bridge {original_id} does not exist in tenant {tid}; there "
                    "is nothing to roll back"
                )
            original = BridgeAction(**original_row)
            if original.status != "applied":
                raise StatusConflict(
                    f"bridge {original_id} is {original.status!r}; only an "
                    "applied bridge can be rolled back",
                    table=self.TABLE,
                    tenant_id=tid,
                    entity=str(original_id),
                    expected="applied",
                    actual=original.status,
                )

            undo = BridgeAction(
                tenant_id=tid,
                bridge_id=new_bridge_id,
                group_id=original.group_id,
                bridge_type=original.bridge_type,
                proposal=proposal,
                simulation=simulation,
                authorization=authorization,
                status="applied",
                rollback_of=original_id,
                vdf_proof_id=vdf_proof_id,
                created_at=stamp,
                applied_at=stamp,
            )
            self._insert(undo)
            changed = self._backend.update(
                self.TABLE,
                {"tenant_id": tid, "bridge_id": original_id},
                {"status": "rolled-back"},
                expected={"status": "applied"},
            )
            if changed != 1:
                raise StatusConflict(
                    f"bridge {original_id} changed state while it was being "
                    "rolled back; re-read it with get() and retry",
                    table=self.TABLE,
                    tenant_id=tid,
                    entity=str(original_id),
                    expected="applied",
                    actual=None,
                )
        return undo

    # -- helpers ----------------------------------------------------------

    def _insert(self, action: BridgeAction) -> None:
        self._backend.insert(
            self.TABLE,
            {
                "tenant_id": action.tenant_id,
                "bridge_id": action.bridge_id,
                "group_id": action.group_id,
                "bridge_type": action.bridge_type,
                "proposal": action.proposal,
                "simulation": action.simulation,
                "authorization": action.authorization,
                "status": action.status,
                "rollback_of": action.rollback_of,
                "vdf_proof_id": action.vdf_proof_id,
                "created_at": action.created_at,
                "applied_at": action.applied_at,
            },
        )

    def _transition(
        self,
        tenant_id: Any,
        bridge_id: Any,
        *,
        new_status: str,
        expected_status: str,
        applied_at: _dt.datetime | None,
    ) -> BridgeAction:
        tid = _as_uuid(tenant_id, "tenant_id")
        bid = _as_uuid(bridge_id, "bridge_id")
        if new_status not in BRIDGE_TRANSITIONS.get(expected_status, frozenset()):
            raise ValueError(
                f"{expected_status!r} -> {new_status!r} is not a bridge lifecycle "
                f"transition; from {expected_status!r} the only moves are "
                f"{sorted(BRIDGE_TRANSITIONS.get(expected_status, frozenset()))}"
            )
        values: dict[str, Any] = {"status": new_status}
        if applied_at is not None:
            values["applied_at"] = applied_at

        with self._backend.unit_of_work(tid):
            changed = self._backend.update(
                self.TABLE,
                {"tenant_id": tid, "bridge_id": bid},
                values,
                expected={"status": expected_status},
            )
            if changed != 1:
                row = self._backend.fetch_one(
                    self.TABLE, {"tenant_id": tid, "bridge_id": bid}
                )
                actual = row["status"] if row else None
                raise StatusConflict(
                    f"bridge {bid} in tenant {tid}: expected status "
                    f"{expected_status!r} but found {actual!r}. Re-read it with "
                    "get() and decide again from the state it is actually in.",
                    table=self.TABLE,
                    tenant_id=tid,
                    entity=str(bid),
                    expected=expected_status,
                    actual=actual,
                )
            row = self._backend.fetch_one(
                self.TABLE, {"tenant_id": tid, "bridge_id": bid}
            )
        assert row is not None
        return BridgeAction(**row)


class AttestationRepository(_Repository):
    """Rule 30 proof envelopes.

    ``verification_status`` is the only mutable column: verification is
    deterministic and offline, so re-running it must be able to record what it
    found without the envelope becoming editable — an editable envelope is an
    unverifiable one.
    """

    TABLE = "gc_vdf_attestations"

    def record(
        self,
        tenant_id: Any,
        proof_id: Any,
        *,
        event_id: Any,
        envelope: Mapping[str, Any],
        verification_status: str = "unverified",
        now: _dt.datetime | None = None,
    ) -> VdfAttestation:
        attestation = VdfAttestation(
            tenant_id=tenant_id,
            proof_id=proof_id,
            event_id=event_id,
            envelope=envelope,
            verification_status=verification_status,
            created_at=now or _utcnow(),
        )
        row = {
            "tenant_id": attestation.tenant_id,
            "proof_id": attestation.proof_id,
            "event_id": attestation.event_id,
            "envelope": attestation.envelope,
            "verification_status": attestation.verification_status,
            "created_at": attestation.created_at,
        }
        with self._backend.unit_of_work(attestation.tenant_id):
            try:
                self._backend.insert(self.TABLE, row)
            except DuplicateRow as exc:
                existing = self._backend.fetch_one(
                    self.TABLE,
                    {
                        "tenant_id": attestation.tenant_id,
                        "proof_id": attestation.proof_id,
                    },
                )
                if existing is not None and (
                    existing["envelope"] == attestation.envelope
                    and existing["event_id"] == attestation.event_id
                ):
                    return VdfAttestation(**existing)
                raise IdempotencyConflict(
                    f"proof {attestation.proof_id} already exists in tenant "
                    f"{attestation.tenant_id} attesting a different envelope or "
                    "event; a proof id identifies one attestation"
                ) from exc
        return attestation

    def get(self, tenant_id: Any, proof_id: Any) -> VdfAttestation | None:
        tid = _as_uuid(tenant_id, "tenant_id")
        with self._backend.unit_of_work(tid):
            row = self._backend.fetch_one(
                self.TABLE,
                {"tenant_id": tid, "proof_id": _as_uuid(proof_id, "proof_id")},
            )
        return VdfAttestation(**row) if row else None

    def by_event(self, tenant_id: Any, event_id: Any) -> list[VdfAttestation]:
        """Backs ``btree(tenant_id, event_id)`` — "what attests this event?"."""
        tid = _as_uuid(tenant_id, "tenant_id")
        with self._backend.unit_of_work(tid):
            rows = self._backend.fetch_all(
                self.TABLE,
                {"tenant_id": tid, "event_id": _as_uuid(event_id, "event_id")},
                order_by=(("created_at", "desc"), ("proof_id", "desc")),
            )
        return [VdfAttestation(**row) for row in rows]

    def set_verification_status(
        self,
        tenant_id: Any,
        proof_id: Any,
        *,
        status: str,
        expected_status: str,
    ) -> VdfAttestation:
        if status not in VERIFICATION_STATUSES:
            raise ValueError(
                f"status {status!r} is not one of {list(VERIFICATION_STATUSES)}"
            )
        tid = _as_uuid(tenant_id, "tenant_id")
        pid = _as_uuid(proof_id, "proof_id")
        with self._backend.unit_of_work(tid):
            changed = self._backend.update(
                self.TABLE,
                {"tenant_id": tid, "proof_id": pid},
                {"verification_status": status},
                expected={"verification_status": expected_status},
            )
            if changed != 1:
                row = self._backend.fetch_one(
                    self.TABLE, {"tenant_id": tid, "proof_id": pid}
                )
                actual = row["verification_status"] if row else None
                raise StatusConflict(
                    f"attestation {pid} in tenant {tid}: expected "
                    f"{expected_status!r} but found {actual!r}. Re-read it with "
                    "get() before recording a verification result.",
                    table=self.TABLE,
                    tenant_id=tid,
                    entity=str(pid),
                    expected=expected_status,
                    actual=actual,
                )
            row = self._backend.fetch_one(
                self.TABLE, {"tenant_id": tid, "proof_id": pid}
            )
        assert row is not None
        return VdfAttestation(**row)


__all__ = [
    "APPEND_ONLY_SQLSTATE",
    "AppendOnlyViolation",
    "AttestationRepository",
    "BRIDGE_STATUSES",
    "BRIDGE_TRANSITIONS",
    "BridgeAction",
    "BridgeRepository",
    "ContractViolation",
    "DuplicateRow",
    "ForeignKeyViolation",
    "GroupIntentRepository",
    "GroupIntentRevision",
    "IdempotencyConflict",
    "InMemoryBackend",
    "MUTABLE_COLUMNS",
    "PersistenceError",
    "ProfileHead",
    "ProfileRepository",
    "ProfileRevision",
    "RevisionConflict",
    "SqlAlchemyBackend",
    "StatusConflict",
    "StorageBackend",
    "TensionRepository",
    "TensionSnapshot",
    "TenantScopeError",
    "VERIFICATION_STATUSES",
    "VdfAttestation",
    "canonical_jsonb",
]
