"""MCP resources and prompts, driven by the card's URI templates.

Resources are read paths, so they carry their own permission and their own
tenant check. The tenant appears *in the URI*, which makes cross-tenant access
a thing a client can literally type — :func:`resolve` therefore compares the URI
tenant against the caller's identity before anything is fetched, rather than
relying on the store to return nothing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from src.mcp_surface import errors
from src.mcp_surface.tools import CallerIdentity, assert_same_tenant
from src.model_card import cached_model_card


class ResourceNotFound(errors.ProblemDetails):
    pass


def _template_to_regex(template: str) -> re.Pattern[str]:
    """Turn ``profile://{tenant_id}/{user_id}/effective`` into a matcher.

    Placeholders match a single path segment: a greedy ``.+`` would let
    ``{tenant_id}`` swallow ``a/b`` and silently accept a URI whose segments do
    not line up with the template.
    """
    pattern = re.escape(template)
    pattern = re.sub(r"\\\{(\w+)\\\}", r"(?P<\1>[^/]+)", pattern)
    return re.compile(f"^{pattern}$")


@dataclass(frozen=True, slots=True)
class ResourceRoute:
    uri_template: str
    permission: str
    matcher: re.Pattern[str]

    def match(self, uri: str) -> dict[str, str] | None:
        m = self.matcher.match(uri)
        return dict(m.groupdict()) if m else None


def routes() -> tuple[ResourceRoute, ...]:
    return tuple(
        ResourceRoute(r.uri_template, r.permission, _template_to_regex(r.uri_template))
        for r in cached_model_card().mcp.resources
    )


class ResourceServices(Protocol):
    def profile_effective(self, tenant_id: str, user_id: str) -> Any: ...
    def profile_revision(self, tenant_id: str, user_id: str, revision: int) -> Any: ...
    def group_intent_latest(self, tenant_id: str, group_id: str) -> Any: ...
    def group_tension_latest(self, tenant_id: str, group_id: str) -> Any: ...
    def bridge_record(self, tenant_id: str, bridge_id: str) -> Any: ...
    def attestation_record(self, tenant_id: str, proof_id: str) -> Any: ...


_READERS = {
    "profile://{tenant_id}/{user_id}/effective": lambda s, p: s.profile_effective(
        p["tenant_id"], p["user_id"]
    ),
    "profile://{tenant_id}/{user_id}/revisions/{revision}": (
        lambda s, p: s.profile_revision(
            p["tenant_id"], p["user_id"], _as_revision(p["revision"])
        )
    ),
    "group://{tenant_id}/{group_id}/intent/latest": lambda s, p: s.group_intent_latest(
        p["tenant_id"], p["group_id"]
    ),
    "group://{tenant_id}/{group_id}/tension/latest": (
        lambda s, p: s.group_tension_latest(p["tenant_id"], p["group_id"])
    ),
    "bridge://{tenant_id}/{bridge_id}": lambda s, p: s.bridge_record(
        p["tenant_id"], p["bridge_id"]
    ),
    "attestation://{tenant_id}/{proof_id}": lambda s, p: s.attestation_record(
        p["tenant_id"], p["proof_id"]
    ),
}


def _as_revision(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError:
        raise errors.ProblemDetails(
            "STALE_INTENT_REVISION", f"revision segment {raw!r} is not an integer"
        ) from None
    if value < 1:
        raise errors.ProblemDetails(
            "STALE_INTENT_REVISION", f"revision must be >= 1, got {value}"
        )
    return value


def assert_readers_cover_card() -> None:
    declared = {r.uri_template for r in cached_model_card().mcp.resources}
    implemented = set(_READERS)
    if declared != implemented:
        raise AssertionError(
            "resource readers drifted from the model card: "
            f"unimplemented={sorted(declared - implemented)} "
            f"undeclared={sorted(implemented - declared)}"
        )


def resolve(
    uri: str, caller: CallerIdentity, services: ResourceServices
) -> Any:
    """Match, authorize, tenant-check, then read."""
    for route in routes():
        params = route.match(uri)
        if params is None:
            continue
        if not caller.may(route.permission):
            raise errors.authorization_denied(
                f"caller {caller.subject!r} lacks {route.permission!r} required "
                f"by {route.uri_template}",
                required_permission=route.permission,
                uri=uri,
            )
        assert_same_tenant(caller, params["tenant_id"], uri)
        return _READERS[route.uri_template](services, params)
    raise errors.ProblemDetails(
        "AUTHORIZATION_DENIED",
        f"no resource template matches {uri!r}; the surface is default-deny, so "
        "an unrecognised URI is refused rather than probed",
        extra={"uri": uri},
    )


# --------------------------------------------------------------------------
# prompts
# --------------------------------------------------------------------------


PROMPT_BODIES: Mapping[str, str] = {
    "explain-personalization": (
        "Explain this user's effective agent. For each active module and each "
        "inferred preference, name the evidence events that activated it, the "
        "confidence, and the decay applied. State plainly which preferences are "
        "explicit (user-owned) and which are inferred (advisory and reversible). "
        "Do not present an inferred preference as a stated one."
    ),
    "resolve-group-gap": (
        "Render the current group gap. List the shared invariants, the hard "
        "constraints that cannot be traded, every recorded dissenting position "
        "with its holder, and the ranked minimum-cost bridge options with their "
        "predicted before/after tension per participant. If the conflict is "
        "irreducible, say so and stop — do not propose a compromise that "
        "crosses a hard boundary, and do not describe an unresolved state as "
        "agreement."
    ),
    "audit-bridge-decision": (
        "Audit this bridge decision. Show the precedence checks in order "
        "(hard boundary, legal/safety/tenant policy, tool authorization, group "
        "decision rule, inferred preference), the authorization grant consulted, "
        "the predicted and observed tension before and after, the preserved "
        "counterfactual, and the Rule 30 VDF proof id binding the revision."
    ),
}


def prompts() -> tuple[dict[str, str], ...]:
    card = cached_model_card()
    return tuple(
        {
            "name": p.name,
            "purpose": p.purpose,
            "body": PROMPT_BODIES[p.name],
        }
        for p in card.mcp.prompts
    )


def assert_prompts_cover_card() -> None:
    declared = {p.name for p in cached_model_card().mcp.prompts}
    implemented = set(PROMPT_BODIES)
    if declared != implemented:
        raise AssertionError(
            "prompt bodies drifted from the model card: "
            f"unimplemented={sorted(declared - implemented)} "
            f"undeclared={sorted(implemented - declared)}"
        )
