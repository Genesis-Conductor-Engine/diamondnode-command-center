"""MCP tool surface: schema validation, default-deny authorization, idempotency.

Three properties this module exists to guarantee, all from the card's ``mcp``
section:

**Schemas are read from the card, never retyped.** ``_validate`` walks the JSON
Schema stored in ``card.mcp.tools[*].input_schema``. Retyping the schemas in
Python would create two sources of truth that drift silently — the card would
document a contract the server does not enforce.

**Authorization defaults to deny and cannot self-expand.** A grant is a frozen
set of permissions attached to a tenant-scoped :class:`CallerIdentity`. Nothing
in a request can add to it: :func:`authorize` only ever *reads* the grant, and
the grant is immutable, so there is no code path from "the profile learned
something" or "the group decided something" to "the caller may now call
``bridge.apply``". The permission a tool requires likewise comes from the card.

**Mutations are idempotent by key.** ``at-least-once`` delivery means a retry is
normal traffic, and every mutating tool writes an append-only revision. Without
a replay guard, one delivered-twice ``profile.correct`` would create two
revisions and the second would look like a genuine second opinion in the audit
record. The store returns the *original* result for a repeated key.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol

from src.model_card import cached_model_card
from src.model_card.types import McpTool
from src.mcp_surface import errors


class ToolInputError(ValueError):
    """The request did not satisfy the card's declared schema for this tool."""


# --------------------------------------------------------------------------
# identity and authorization
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CallerIdentity:
    """Who is calling, and what they were granted.

    ``grants`` is a frozenset so a handler cannot mutate it mid-call. That is
    the mechanical half of ``self_expansion: forbidden`` — there is no setter,
    so no amount of inference can widen it.
    """

    tenant_id: str
    subject: str
    grants: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not self.tenant_id:
            raise ValueError("tenant_id is required: MCP access is tenant-scoped")
        if not self.subject:
            raise ValueError("subject is required")
        object.__setattr__(self, "grants", frozenset(self.grants))
        unknown = self.grants - cached_model_card().mcp.permissions
        if unknown:
            # A grant naming a permission the card does not define is a
            # configuration error, not a new capability. Refusing it here stops
            # a typo from becoming a silently-inert (or worse, silently-honoured)
            # permission.
            raise ValueError(
                f"grant names permissions absent from the model card: "
                f"{sorted(unknown)}"
            )

    def may(self, permission: str) -> bool:
        return permission in self.grants


def tool_spec(name: str) -> McpTool:
    return cached_model_card().mcp.tool(name)


def authorize(caller: CallerIdentity, tool_name: str) -> None:
    """Default deny. Raises AUTHORIZATION_DENIED when the grant is absent."""
    spec = tool_spec(tool_name)
    if not caller.may(spec.permission):
        raise errors.authorization_denied(
            f"caller {caller.subject!r} lacks {spec.permission!r} required by "
            f"{tool_name!r}",
            required_permission=spec.permission,
            tool=tool_name,
        )


def assert_same_tenant(caller: CallerIdentity, tenant_id: str, what: str) -> None:
    """Cross-tenant access is denied even when the permission is held.

    A grant says *what* a caller may do, never *whose* data they may do it to.
    Row-level security enforces this in the database; this check keeps a
    cross-tenant request from reaching the database at all, so the denial is
    attributable to the caller rather than surfacing as an empty result.
    """
    if tenant_id != caller.tenant_id:
        raise errors.authorization_denied(
            f"caller is scoped to tenant {caller.tenant_id!r} and may not reach "
            f"{what} in tenant {tenant_id!r}",
            requested_tenant=tenant_id,
        )


# --------------------------------------------------------------------------
# schema validation against the card
# --------------------------------------------------------------------------


_TYPE_CHECKS: dict[str, Callable[[Any], bool]] = {
    "object": lambda v: isinstance(v, dict),
    # bool is a subclass of int in Python; accepting True as an integer would
    # let `revision: true` through and land a boolean in a bigint column.
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "string": lambda v: isinstance(v, str),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "null": lambda v: v is None,
}


def _type_ok(value: Any, declared: Any) -> bool:
    types = declared if isinstance(declared, list) else [declared]
    return any(_TYPE_CHECKS.get(t, lambda _: True)(value) for t in types)


def _check_uuid(name: str, value: Any) -> None:
    try:
        uuid.UUID(str(value))
    except (ValueError, AttributeError, TypeError):
        raise ToolInputError(f"{name!r} must be a UUID, got {value!r}") from None


def _validate_property(name: str, value: Any, schema: Mapping[str, Any]) -> None:
    declared = schema.get("type")
    if declared is not None and not _type_ok(value, declared):
        raise ToolInputError(
            f"{name!r} must be of type {declared}, got {type(value).__name__}"
        )
    if schema.get("format") == "uuid" and value is not None:
        _check_uuid(name, value)
    if "enum" in schema and value not in schema["enum"]:
        raise ToolInputError(
            f"{name!r} must be one of {schema['enum']}, got {value!r}"
        )
    if "minimum" in schema and isinstance(value, (int, float)) and (
        value < schema["minimum"]
    ):
        raise ToolInputError(f"{name!r} must be >= {schema['minimum']}, got {value}")
    if "maximum" in schema and isinstance(value, (int, float)) and (
        value > schema["maximum"]
    ):
        raise ToolInputError(f"{name!r} must be <= {schema['maximum']}, got {value}")
    if "minLength" in schema and isinstance(value, str) and (
        len(value) < schema["minLength"]
    ):
        raise ToolInputError(
            f"{name!r} must be at least {schema['minLength']} character(s)"
        )
    if "minItems" in schema and isinstance(value, list) and (
        len(value) < schema["minItems"]
    ):
        raise ToolInputError(f"{name!r} must have at least {schema['minItems']} item(s)")
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for i, item in enumerate(value):
            _validate_property(f"{name}[{i}]", item, schema["items"])


def validate_input(tool_name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Validate against the card's schema and apply its declared defaults."""
    schema = tool_spec(tool_name).input_schema
    if not isinstance(arguments, Mapping):
        raise ToolInputError(f"{tool_name}: arguments must be an object")

    properties: Mapping[str, Any] = schema.get("properties", {})
    unknown = set(arguments) - set(properties)
    if unknown:
        # Rejecting unknown fields keeps a typo'd ``scope`` from silently
        # falling back to a default that deletes more than the caller meant.
        raise ToolInputError(
            f"{tool_name}: unknown argument(s) {sorted(unknown)}; "
            f"expected {sorted(properties)}"
        )
    for name in schema.get("required", []):
        if name not in arguments:
            raise ToolInputError(f"{tool_name}: missing required argument {name!r}")

    resolved = dict(arguments)
    for name, prop in properties.items():
        if name in resolved:
            _validate_property(name, resolved[name], prop)
        elif "default" in prop:
            resolved[name] = prop["default"]
    return resolved


# --------------------------------------------------------------------------
# idempotency
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Replay:
    fingerprint: str
    result: Any


class IdempotencyStore:
    """Per-tenant replay cache for mutating tool calls.

    Keyed by ``(tenant, tool, Idempotency-Key)``. The stored fingerprint is a
    canonical rendering of the arguments: replaying a key with *different*
    arguments is a client bug, and returning the first call's result would hide
    it, so that case raises instead.
    """

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str, str], _Replay] = {}
        self._lock = threading.Lock()

    @staticmethod
    def fingerprint(arguments: Mapping[str, Any]) -> str:
        from src.attestation.vdf import input_hash

        return input_hash(dict(arguments))

    def lookup(
        self, tenant_id: str, tool: str, key: str, arguments: Mapping[str, Any]
    ) -> tuple[bool, Any]:
        fp = self.fingerprint(arguments)
        with self._lock:
            entry = self._entries.get((tenant_id, tool, key))
        if entry is None:
            return False, None
        if entry.fingerprint != fp:
            raise errors.ProblemDetails(
                "PROFILE_REVISION_CONFLICT",
                f"Idempotency-Key {key!r} was already used for {tool!r} with "
                "different arguments; reuse of a key must repeat the same request",
                extra={"tool": tool},
            )
        return True, entry.result

    def record(
        self,
        tenant_id: str,
        tool: str,
        key: str,
        arguments: Mapping[str, Any],
        result: Any,
    ) -> None:
        with self._lock:
            self._entries[(tenant_id, tool, key)] = _Replay(
                self.fingerprint(arguments), result
            )


# --------------------------------------------------------------------------
# service seam
# --------------------------------------------------------------------------


class ToolServices(Protocol):
    """What the tool layer needs from the rest of the system.

    A Protocol rather than concrete imports so the surface is testable against
    fakes and so a partially-available backend degrades to a clear
    ``not implemented`` for one tool instead of failing the whole server import.
    """

    def profile_inspect(self, tenant_id: str, user_id: str, revision: int | None) -> Any: ...
    def profile_explain(self, tenant_id: str, user_id: str, include_provenance: bool) -> Any: ...
    def profile_correct(self, tenant_id: str, user_id: str, patch: Mapping[str, Any], reason: str) -> Any: ...
    def profile_rollback(self, tenant_id: str, user_id: str, target_revision: int, reason: str) -> Any: ...
    def profile_export(self, tenant_id: str, user_id: str, fmt: str) -> Any: ...
    def profile_delete(self, tenant_id: str, user_id: str, scope: str) -> Any: ...
    def group_intent_evaluate(self, tenant_id: str, group_id: str, member_event_refs: list[str], decision_policy: Any) -> Any: ...
    def group_tension_evaluate(self, tenant_id: str, group_id: str, group_intent_revision: int) -> Any: ...
    def bridge_propose(self, tenant_id: str, group_id: str, tension_snapshot_id: str, maximum_candidates: int) -> Any: ...
    def bridge_apply(self, tenant_id: str, bridge_id: str, expected_snapshot_id: str | None) -> Any: ...
    def bridge_rollback(self, tenant_id: str, bridge_id: str, reason: str) -> Any: ...
    def agent_compile(self, tenant_id: str, user_id: str, session_id: str, group_id: str | None, requested_capabilities: list[str]) -> Any: ...
    def attestation_verify(self, tenant_id: str, proof_id: str) -> Any: ...


_DISPATCH: dict[str, Callable[[ToolServices, CallerIdentity, dict[str, Any]], Any]] = {
    "profile.inspect": lambda s, c, a: s.profile_inspect(
        c.tenant_id, a["user_id"], a.get("revision")
    ),
    "profile.explain": lambda s, c, a: s.profile_explain(
        c.tenant_id, a["user_id"], a.get("include_provenance", True)
    ),
    "profile.correct": lambda s, c, a: s.profile_correct(
        c.tenant_id, a["user_id"], a["patch"], a["reason"]
    ),
    "profile.rollback": lambda s, c, a: s.profile_rollback(
        c.tenant_id, a["user_id"], a["target_revision"], a["reason"]
    ),
    "profile.export": lambda s, c, a: s.profile_export(
        c.tenant_id, a["user_id"], a["format"]
    ),
    "profile.delete": lambda s, c, a: s.profile_delete(
        c.tenant_id, a["user_id"], a["scope"]
    ),
    "group.intent.evaluate": lambda s, c, a: s.group_intent_evaluate(
        c.tenant_id, a["group_id"], a["member_event_refs"], a.get("decision_policy")
    ),
    "group.tension.evaluate": lambda s, c, a: s.group_tension_evaluate(
        c.tenant_id, a["group_id"], a["group_intent_revision"]
    ),
    "bridge.propose": lambda s, c, a: s.bridge_propose(
        c.tenant_id, a["group_id"], a["tension_snapshot_id"],
        a.get("maximum_candidates", 3),
    ),
    "bridge.apply": lambda s, c, a: s.bridge_apply(
        c.tenant_id, a["bridge_id"], a.get("expected_snapshot_id")
    ),
    "bridge.rollback": lambda s, c, a: s.bridge_rollback(
        c.tenant_id, a["bridge_id"], a["reason"]
    ),
    "agent.compile": lambda s, c, a: s.agent_compile(
        c.tenant_id, a["user_id"], a["session_id"], a.get("group_id"),
        a.get("requested_capabilities", []),
    ),
    "attestation.verify": lambda s, c, a: s.attestation_verify(
        c.tenant_id, a["proof_id"]
    ),
}


def assert_dispatch_covers_card() -> None:
    """Every card tool must be dispatchable, and nothing beyond them."""
    declared = {t.name for t in cached_model_card().mcp.tools}
    implemented = set(_DISPATCH)
    if declared != implemented:
        raise AssertionError(
            "MCP dispatch drifted from the model card: "
            f"unimplemented={sorted(declared - implemented)} "
            f"undeclared={sorted(implemented - declared)}"
        )


@dataclass
class ToolRouter:
    """Validates, authorizes, de-duplicates, then dispatches."""

    services: ToolServices
    idempotency: IdempotencyStore = field(default_factory=IdempotencyStore)

    def call(
        self,
        tool_name: str,
        caller: CallerIdentity,
        arguments: Mapping[str, Any],
        *,
        idempotency_key: str | None = None,
    ) -> Any:
        spec = tool_spec(tool_name)

        # Authorization first: an unauthorized caller must not learn whether
        # their arguments were well-formed, and must not consume an
        # idempotency key.
        authorize(caller, tool_name)

        resolved = validate_input(tool_name, arguments)

        if spec.mutating:
            if not idempotency_key:
                raise errors.ProblemDetails(
                    "PROFILE_REVISION_CONFLICT",
                    f"{tool_name!r} is a mutation and requires an "
                    "Idempotency-Key header; delivery is at-least-once, so an "
                    "unkeyed retry would write a second revision",
                    extra={"tool": tool_name},
                )
            replayed, previous = self.idempotency.lookup(
                caller.tenant_id, tool_name, idempotency_key, resolved
            )
            if replayed:
                return previous

        result = _DISPATCH[tool_name](self.services, caller, resolved)

        if spec.mutating and idempotency_key:
            self.idempotency.record(
                caller.tenant_id, tool_name, idempotency_key, resolved, result
            )
        return result
