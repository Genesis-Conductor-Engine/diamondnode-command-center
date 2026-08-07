"""Default-deny, tenant scoping, and permission non-expansion.

``permission-non-expansion`` is one of the card's behavioral acceptance cases:
no inferred preference and no group intent may widen an MCP capability grant.
The surface makes that structural — a grant is a frozenset on an immutable
identity with no setter — so the tests here try to widen it anyway, through
every route a caller actually has.
"""

from __future__ import annotations

import dataclasses

import pytest

from src.mcp_surface import errors
from src.mcp_surface.server import build_server
from src.mcp_surface.tools import CallerIdentity, authorize, tool_spec
from src.model_card import cached_model_card

from tests.mcp_surface.test_tools import (
    BRIDGE,
    GROUP,
    PROOF,
    TENANT,
    USER,
    VALID_ARGS,
    RecordingServices,
)

OTHER_TENANT = "22222222-2222-2222-2222-222222222222"


@pytest.fixture
def card():
    return cached_model_card()


@pytest.fixture
def services():
    return RecordingServices()


@pytest.fixture
def srv(services):
    return build_server(services, services)


# --------------------------------------------------------------------------
# default deny
# --------------------------------------------------------------------------


def test_a_caller_with_no_grants_is_denied_every_tool(card, srv, services):
    caller = CallerIdentity(TENANT, "anon", frozenset())
    for tool in card.mcp.tools:
        out = srv.call_tool(
            tool.name, caller, dict(VALID_ARGS[tool.name]), idempotency_key="k"
        )
        assert not out["ok"], f"{tool.name} was allowed without any grant"
        assert out["problem"]["code"] == "AUTHORIZATION_DENIED"
    assert services.calls == [], "a denied call still reached the backend"


def test_a_caller_with_no_grants_is_denied_every_resource(card, srv):
    caller = CallerIdentity(TENANT, "anon", frozenset())
    for resource in card.mcp.resources:
        uri = (
            resource.uri_template.replace("{tenant_id}", TENANT)
            .replace("{user_id}", USER)
            .replace("{group_id}", GROUP)
            .replace("{bridge_id}", BRIDGE)
            .replace("{proof_id}", PROOF)
            .replace("{revision}", "3")
        )
        out = srv.read_resource(uri, caller)
        assert not out["ok"], f"{uri} was readable without a grant"
        assert out["problem"]["code"] == "AUTHORIZATION_DENIED"


def test_each_tool_requires_exactly_the_cards_permission(card):
    for tool in card.mcp.tools:
        holder = CallerIdentity(TENANT, "svc", frozenset({tool.permission}))
        authorize(holder, tool.name)  # must not raise

        others = card.mcp.permissions - {tool.permission}
        wrong = CallerIdentity(TENANT, "svc", frozenset(others))
        with pytest.raises(errors.ProblemDetails) as exc:
            authorize(wrong, tool.name)
        assert exc.value.code == "AUTHORIZATION_DENIED"


def test_export_is_a_separate_permission_from_read(card):
    """The card separates export from read to bound exfiltration."""
    reader = CallerIdentity(TENANT, "svc", frozenset({"profile:read"}))
    authorize(reader, "profile.inspect")
    with pytest.raises(errors.ProblemDetails):
        authorize(reader, "profile.export")
    assert tool_spec("profile.export").permission == "profile:export"


def test_delete_is_a_separate_permission_from_write(card):
    writer = CallerIdentity(TENANT, "svc", frozenset({"profile:write"}))
    authorize(writer, "profile.correct")
    with pytest.raises(errors.ProblemDetails):
        authorize(writer, "profile.delete")


def test_authorization_is_checked_before_input_validation(card, srv):
    """An unauthorized caller must not learn whether their input was valid.

    Validating first would turn the surface into an oracle for argument shapes
    and existing ids.
    """
    caller = CallerIdentity(TENANT, "anon", frozenset())
    out = srv.call_tool("profile.inspect", caller, {"garbage": True})
    assert out["problem"]["code"] == "AUTHORIZATION_DENIED"


# --------------------------------------------------------------------------
# tenant scoping
# --------------------------------------------------------------------------


def test_a_grant_does_not_cross_tenants(card, srv, services):
    """A grant says what a caller may do, never whose data they may touch."""
    caller = CallerIdentity(TENANT, "svc", frozenset(card.mcp.permissions))
    out = srv.read_resource(f"profile://{OTHER_TENANT}/{USER}/effective", caller)
    assert not out["ok"]
    assert out["problem"]["code"] == "AUTHORIZATION_DENIED"
    assert services.calls == []


def test_tools_act_only_within_the_callers_tenant(card, srv, services):
    """The tenant comes from the identity, never from the arguments.

    No card tool takes a tenant_id parameter, so there is nothing for a caller
    to forge; this test pins that property.
    """
    for tool in card.mcp.tools:
        assert "tenant_id" not in tool.input_schema.get("properties", {}), (
            f"{tool.name} accepts a caller-supplied tenant_id"
        )

    caller = CallerIdentity(TENANT, "svc", frozenset(card.mcp.permissions))
    srv.call_tool("profile.inspect", caller, {"user_id": USER})
    name, args = services.calls[-1]
    assert args[0] == TENANT


def test_identity_requires_a_tenant():
    with pytest.raises(ValueError, match="tenant_id is required"):
        CallerIdentity("", "svc", frozenset())
    with pytest.raises(ValueError, match="subject is required"):
        CallerIdentity(TENANT, "", frozenset())


# --------------------------------------------------------------------------
# non-expansion
# --------------------------------------------------------------------------


def test_a_grant_naming_an_undeclared_permission_is_refused():
    """A typo must not become a capability, inert or otherwise."""
    with pytest.raises(ValueError, match="absent from the model card"):
        CallerIdentity(TENANT, "svc", frozenset({"profile:superuser"}))


def test_the_grant_is_immutable(card):
    caller = CallerIdentity(TENANT, "svc", frozenset({"profile:read"}))
    assert isinstance(caller.grants, frozenset)
    with pytest.raises(AttributeError):
        caller.grants |= {"profile:write"}  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        caller.grants = frozenset(card.mcp.permissions)  # type: ignore[misc]


def test_profile_learning_cannot_widen_the_grant(card, srv, services):
    """Run the learning path, then re-check authorization.

    The realistic attack is not "mutate the frozenset" — it is "make the system
    infer something that causes a later call to be allowed". So: exercise
    profile.correct with a patch that asks for more capability, then confirm the
    caller still cannot reach a tool they never held.
    """
    caller = CallerIdentity(TENANT, "svc", frozenset({"profile:write", "profile:read"}))
    srv.call_tool(
        "profile.correct",
        caller,
        {
            "user_id": USER,
            "patch": {
                "grants": ["bridge:apply", "profile:delete"],
                "capabilities": ["bridge:apply"],
                "permissions": {"bridge:apply": True},
            },
            "reason": "attempted self-expansion",
        },
        idempotency_key="expand-1",
    )
    out = srv.call_tool(
        "bridge.apply", caller, {"bridge_id": BRIDGE}, idempotency_key="expand-2"
    )
    assert not out["ok"]
    assert out["problem"]["code"] == "AUTHORIZATION_DENIED"


def test_group_intent_cannot_widen_the_grant(card, srv):
    """Group agreement is level 4; it cannot reach the permission envelope."""
    caller = CallerIdentity(
        TENANT, "svc", frozenset({"group:intent:write", "profile:read"})
    )
    srv.call_tool(
        "group.intent.evaluate",
        caller,
        {
            "group_id": GROUP,
            "member_event_refs": ["evt-1", "evt-2"],
            "decision_policy": {"grant_everyone": ["bridge:apply", "profile:delete"]},
        },
        idempotency_key="grp-1",
    )
    for tool, args in (
        ("bridge.apply", {"bridge_id": BRIDGE}),
        ("profile.delete", {"user_id": USER, "scope": "all-profile-data"}),
    ):
        out = srv.call_tool(tool, caller, args, idempotency_key=f"k-{tool}")
        assert not out["ok"]
        assert out["problem"]["code"] == "AUTHORIZATION_DENIED"


def test_the_card_itself_forbids_self_expansion(card):
    assert card.mcp.server.authorization.self_expansion == "forbidden"
    assert card.mcp.server.authorization.default == "deny"
    assert card.invariants.mcp_permission_envelope_cannot_self_expand
    assert card.invariants.personalization_may_not_widen_permissions
    assert card.invariants.personalization_may_narrow_permissions


def test_no_module_in_the_catalogue_may_modify_permissions(card):
    """The structural half: every module declares it cannot touch permissions."""
    modules = card.modular_differentiation.modules
    assert modules
    for name, module in modules.items():
        assert not module.can_modify_tool_permissions, name
    assert (
        card.modular_differentiation.module_selection.permission_filter
        == "mandatory-final-stage"
    )
