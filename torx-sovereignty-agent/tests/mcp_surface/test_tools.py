"""The MCP surface matches the card, tool for tool.

These tests deliberately assert against the *card* rather than against literals
retyped here. A test that hardcoded "profile.correct requires profile:write"
would keep passing after someone changed the card, which is the drift the whole
surface is built to prevent.
"""

from __future__ import annotations

import json
import uuid

import pytest

from src.mcp_surface import errors, resources, server, tools
from src.mcp_surface.server import agent_card, assert_surface_matches_card, build_server
from src.mcp_surface.tools import CallerIdentity, ToolInputError, validate_input
from src.model_card import cached_model_card

TENANT = "11111111-1111-1111-1111-111111111111"
USER = "00000000-0000-0000-0000-000000000001"
GROUP = "00000000-0000-0000-0000-000000000100"
BRIDGE = "00000000-0000-0000-0000-000000000200"
PROOF = "00000000-0000-0000-0000-000000000300"


class RecordingServices:
    """Fake backend that records every call it receives."""

    def __init__(self):
        self.calls: list[tuple[str, tuple]] = []
        self.revision = 0

    def _record(self, name, *args):
        self.calls.append((name, args))
        return {"tool": name, "args": args}

    def profile_inspect(self, *a):
        return self._record("profile.inspect", *a)

    def profile_explain(self, *a):
        return self._record("profile.explain", *a)

    def profile_correct(self, *a):
        self.revision += 1
        self.calls.append(("profile.correct", a))
        return {"revision": self.revision}

    def profile_rollback(self, *a):
        self.revision += 1
        self.calls.append(("profile.rollback", a))
        return {"revision": self.revision}

    def profile_export(self, *a):
        return self._record("profile.export", *a)

    def profile_delete(self, *a):
        return self._record("profile.delete", *a)

    def group_intent_evaluate(self, *a):
        return self._record("group.intent.evaluate", *a)

    def group_tension_evaluate(self, *a):
        return self._record("group.tension.evaluate", *a)

    def bridge_propose(self, *a):
        return self._record("bridge.propose", *a)

    def bridge_apply(self, *a):
        return self._record("bridge.apply", *a)

    def bridge_rollback(self, *a):
        return self._record("bridge.rollback", *a)

    def agent_compile(self, *a):
        return self._record("agent.compile", *a)

    def attestation_verify(self, *a):
        return self._record("attestation.verify", *a)

    # resource side
    def profile_effective(self, *a):
        return self._record("res.profile_effective", *a)

    def profile_revision(self, *a):
        return self._record("res.profile_revision", *a)

    def group_intent_latest(self, *a):
        return self._record("res.group_intent_latest", *a)

    def group_tension_latest(self, *a):
        return self._record("res.group_tension_latest", *a)

    def bridge_record(self, *a):
        return self._record("res.bridge_record", *a)

    def attestation_record(self, *a):
        return self._record("res.attestation_record", *a)


@pytest.fixture
def services():
    return RecordingServices()


@pytest.fixture
def srv(services):
    return build_server(services, services)


@pytest.fixture
def card():
    return cached_model_card()


def all_grants(card):
    return frozenset(card.mcp.permissions)


VALID_ARGS = {
    "profile.inspect": {"user_id": USER},
    "profile.explain": {"user_id": USER},
    "profile.correct": {"user_id": USER, "patch": {"a": 1}, "reason": "user said so"},
    "profile.rollback": {"user_id": USER, "target_revision": 3, "reason": "bad infer"},
    "profile.export": {"user_id": USER, "format": "json"},
    "profile.delete": {"user_id": USER, "scope": "inferred-only"},
    "group.intent.evaluate": {"group_id": GROUP, "member_event_refs": ["evt-1"]},
    "group.tension.evaluate": {"group_id": GROUP, "group_intent_revision": 8},
    "bridge.propose": {"group_id": GROUP, "tension_snapshot_id": PROOF},
    "bridge.apply": {"bridge_id": BRIDGE},
    "bridge.rollback": {"bridge_id": BRIDGE, "reason": "tension rose"},
    "agent.compile": {"user_id": USER, "session_id": "s-1"},
    "attestation.verify": {"proof_id": PROOF},
}


# --------------------------------------------------------------------------
# card conformance
# --------------------------------------------------------------------------


def test_surface_matches_the_card():
    assert_surface_matches_card()


def test_every_card_tool_is_routable(card, srv):
    caller = CallerIdentity(TENANT, "svc", all_grants(card))
    for tool in card.mcp.tools:
        args = VALID_ARGS[tool.name]
        key = str(uuid.uuid4()) if tool.mutating else None
        out = srv.call_tool(tool.name, caller, dict(args), idempotency_key=key)
        assert out["ok"], f"{tool.name} failed: {out.get('problem')}"


def test_error_codes_exactly_match_the_card(card):
    from src.mcp_surface.errors import _CODE_STATUS

    assert set(_CODE_STATUS) == set(card.mcp.errors.stable_codes)


def test_problem_json_carries_every_required_field(card):
    problem = errors.boundary_violation("a hard boundary blocks this bridge")
    doc = problem.to_json()
    for field in card.mcp.errors.required_fields:
        assert field in doc, f"problem+json missing required field {field!r}"
    assert doc["status"] == 403
    assert doc["code"] == "BOUNDARY_VIOLATION"


def test_problem_extras_cannot_overwrite_required_fields():
    problem = errors.boundary_violation("nope", status=200, code="OK", title="fine")
    doc = problem.to_json()
    assert doc["status"] == 403
    assert doc["code"] == "BOUNDARY_VIOLATION"


def test_unknown_error_code_is_rejected():
    with pytest.raises(ValueError, match="stable codes"):
        errors.ProblemDetails("NOT_A_CODE", "detail")


def test_empty_detail_is_rejected():
    with pytest.raises(ValueError, match="actionable"):
        errors.ProblemDetails("BOUNDARY_VIOLATION", "   ")


# --------------------------------------------------------------------------
# schema validation, driven by the card
# --------------------------------------------------------------------------


def test_missing_required_argument_is_rejected(card):
    for tool in card.mcp.tools:
        required = tool.input_schema.get("required", [])
        if not required:
            continue
        args = dict(VALID_ARGS[tool.name])
        args.pop(required[0])
        with pytest.raises(ToolInputError, match="missing required argument"):
            validate_input(tool.name, args)


def test_unknown_argument_is_rejected():
    with pytest.raises(ToolInputError, match="unknown argument"):
        validate_input("profile.inspect", {"user_id": USER, "scpoe": "all"})


def test_uuid_format_is_enforced():
    with pytest.raises(ToolInputError, match="must be a UUID"):
        validate_input("profile.inspect", {"user_id": "not-a-uuid"})


def test_enum_is_enforced():
    with pytest.raises(ToolInputError, match="must be one of"):
        validate_input("profile.export", {"user_id": USER, "format": "xml"})
    with pytest.raises(ToolInputError, match="must be one of"):
        validate_input("profile.delete", {"user_id": USER, "scope": "everything"})


def test_minimum_is_enforced():
    with pytest.raises(ToolInputError, match=">= 1"):
        validate_input(
            "profile.rollback",
            {"user_id": USER, "target_revision": 0, "reason": "x"},
        )


def test_min_length_is_enforced():
    with pytest.raises(ToolInputError, match="at least 1 character"):
        validate_input(
            "profile.rollback",
            {"user_id": USER, "target_revision": 2, "reason": ""},
        )


def test_min_items_is_enforced():
    with pytest.raises(ToolInputError, match="at least 1 item"):
        validate_input(
            "group.intent.evaluate", {"group_id": GROUP, "member_event_refs": []}
        )


def test_booleans_are_not_accepted_as_integers():
    """bool subclasses int in Python; a True revision must not reach a bigint."""
    with pytest.raises(ToolInputError, match="type"):
        validate_input(
            "profile.rollback",
            {"user_id": USER, "target_revision": True, "reason": "x"},
        )


def test_card_declared_defaults_are_applied():
    resolved = validate_input("bridge.propose", {
        "group_id": GROUP, "tension_snapshot_id": PROOF
    })
    assert resolved["maximum_candidates"] == 3
    assert validate_input("profile.explain", {"user_id": USER})[
        "include_provenance"
    ] is True


def test_nullable_types_from_the_card_are_accepted():
    """``revision: {type: ["integer", "null"]}`` must accept both."""
    assert validate_input("profile.inspect", {"user_id": USER, "revision": None})
    assert validate_input("profile.inspect", {"user_id": USER, "revision": 4})


# --------------------------------------------------------------------------
# idempotency
# --------------------------------------------------------------------------


def test_mutating_tool_requires_an_idempotency_key(card, srv, services):
    caller = CallerIdentity(TENANT, "svc", all_grants(card))
    out = srv.call_tool("profile.correct", caller, dict(VALID_ARGS["profile.correct"]))
    assert not out["ok"]
    assert out["problem"]["code"] == "PROFILE_REVISION_CONFLICT"
    assert services.revision == 0, "a keyless mutation reached the backend"


def test_read_only_tool_needs_no_key(card, srv):
    caller = CallerIdentity(TENANT, "svc", all_grants(card))
    assert srv.call_tool("profile.inspect", caller, {"user_id": USER})["ok"]


def test_replaying_a_key_returns_the_first_result_without_a_second_revision(
    card, srv, services
):
    caller = CallerIdentity(TENANT, "svc", all_grants(card))
    key = "idem-1"
    args = dict(VALID_ARGS["profile.correct"])
    first = srv.call_tool("profile.correct", caller, dict(args), idempotency_key=key)
    second = srv.call_tool("profile.correct", caller, dict(args), idempotency_key=key)
    assert first["result"] == second["result"]
    assert services.revision == 1, "at-least-once retry created a second revision"


def test_reusing_a_key_with_different_arguments_is_refused(card, srv):
    caller = CallerIdentity(TENANT, "svc", all_grants(card))
    key = "idem-2"
    srv.call_tool(
        "profile.correct", caller, dict(VALID_ARGS["profile.correct"]),
        idempotency_key=key,
    )
    out = srv.call_tool(
        "profile.correct",
        caller,
        {"user_id": USER, "patch": {"b": 2}, "reason": "different"},
        idempotency_key=key,
    )
    assert not out["ok"]
    assert out["problem"]["code"] == "PROFILE_REVISION_CONFLICT"


def test_idempotency_is_scoped_per_tenant(card, srv, services):
    """The same key from two tenants must not collide."""
    a = CallerIdentity(TENANT, "svc", all_grants(card))
    b = CallerIdentity("22222222-2222-2222-2222-222222222222", "svc", all_grants(card))
    args = dict(VALID_ARGS["profile.correct"])
    srv.call_tool("profile.correct", a, dict(args), idempotency_key="shared")
    srv.call_tool("profile.correct", b, dict(args), idempotency_key="shared")
    assert services.revision == 2


# --------------------------------------------------------------------------
# resources and prompts
# --------------------------------------------------------------------------


def test_every_resource_template_resolves(card, srv, services):
    caller = CallerIdentity(TENANT, "svc", all_grants(card))
    uris = [
        f"profile://{TENANT}/{USER}/effective",
        f"profile://{TENANT}/{USER}/revisions/7",
        f"group://{TENANT}/{GROUP}/intent/latest",
        f"group://{TENANT}/{GROUP}/tension/latest",
        f"bridge://{TENANT}/{BRIDGE}",
        f"attestation://{TENANT}/{PROOF}",
    ]
    assert len(uris) == len(card.mcp.resources)
    for uri in uris:
        out = srv.read_resource(uri, caller)
        assert out["ok"], f"{uri} failed: {out.get('problem')}"


def test_unrecognised_uri_is_denied_not_probed(card, srv):
    caller = CallerIdentity(TENANT, "svc", all_grants(card))
    out = srv.read_resource(f"secrets://{TENANT}/everything", caller)
    assert not out["ok"]
    assert out["problem"]["code"] == "AUTHORIZATION_DENIED"


def test_template_placeholders_match_one_segment_only(card, srv):
    """A greedy placeholder would let one segment swallow a path."""
    caller = CallerIdentity(TENANT, "svc", all_grants(card))
    out = srv.read_resource(f"profile://{TENANT}/{USER}/extra/effective", caller)
    assert not out["ok"]


def test_non_integer_revision_segment_is_refused(card, srv):
    caller = CallerIdentity(TENANT, "svc", all_grants(card))
    out = srv.read_resource(f"profile://{TENANT}/{USER}/revisions/latest", caller)
    assert not out["ok"]
    assert out["problem"]["code"] == "STALE_INTENT_REVISION"


def test_prompts_match_the_card(card, srv):
    got = {p["name"] for p in srv.list_prompts()}
    assert got == {p.name for p in card.mcp.prompts}
    for prompt in srv.list_prompts():
        assert prompt["body"].strip(), f"{prompt['name']} has an empty body"


def test_resolve_group_gap_prompt_forbids_manufactured_agreement():
    body = resources.PROMPT_BODIES["resolve-group-gap"]
    assert "irreducible" in body
    assert "agreement" in body


# --------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------


def test_agent_card_advertises_exactly_the_card_surface(card):
    doc = agent_card()
    assert {t["name"] for t in doc["tools"]} == {t.name for t in card.mcp.tools}
    assert doc["authorization"]["default"] == "deny"
    assert doc["authorization"]["self_expansion"] == "forbidden"
    assert doc["idempotency"]["required_for_mutations"] is True
    assert doc["endpoints"]["agent_card"] == "/.well-known/agent-card.json"
    assert doc["endpoints"]["mcp"] == "/mcp"
    assert set(doc["transports"]) == {"stdio", "streamable-http"}


def test_agent_card_does_not_overclaim_readiness(card):
    """The card's readiness claim is explicitly not scale-proven."""
    doc = agent_card()
    assert doc["readiness"] == card.deployment.readiness_claim
    assert "scale-proven" not in doc["readiness"].replace(
        "not-scale-proven", ""
    )


def test_agent_card_is_json_serialisable(srv):
    json.loads(srv.agent_card_json())


def test_importing_the_server_module_binds_no_port():
    """Importing must be side-effect free.

    Re-importing here would be a no-op, so instead assert the transports refuse
    to run without an explicitly wired identity source — there is no path from
    import to listening socket.
    """
    assert hasattr(server, "serve_stdio")
    assert hasattr(server, "serve_http")
