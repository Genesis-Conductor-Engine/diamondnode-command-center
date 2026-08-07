"""MCP server wiring: tool/resource/prompt registration and discovery.

Importing this module must not bind a port or start a loop — a test that imports
the surface to check a contract should not leave a listener behind, and neither
should a worker that only wants the agent card. Construction is explicit
(:func:`build_server`) and serving is explicit (:func:`serve_stdio`,
:func:`serve_http`).

The MCP SDK is an optional import. When it is absent the router, the agent card,
and every contract check still work — the transports are the only thing lost.
That keeps ``mcp_surface`` testable on a machine without the SDK and matches how
the rest of this system treats a heavy dependency.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from src.mcp_surface import errors, resources, tools
from src.mcp_surface.resources import ResourceServices
from src.mcp_surface.tools import CallerIdentity, ToolRouter, ToolServices
from src.model_card import cached_model_card

_sdk: dict[str, Any] = {"loaded": False, "ok": False, "error": None}


def _load_sdk() -> dict[str, Any]:
    if _sdk["loaded"]:
        return _sdk
    _sdk["loaded"] = True
    try:
        from mcp.server import Server  # type: ignore

        _sdk.update(ok=True, Server=Server)
    except Exception as exc:
        _sdk.update(ok=False, error=f"{type(exc).__name__}: {exc}")
    return _sdk


def sdk_available() -> bool:
    return bool(_load_sdk()["ok"])


def agent_card() -> dict[str, Any]:
    """The document served at ``/.well-known/agent-card.json``.

    Built from the model card so discovery cannot advertise a tool the server
    does not route, or a permission the authorization filter does not know.
    """
    card = cached_model_card()
    srv = card.mcp.server
    return {
        "schema_version": card.schema_version,
        "id": srv.id,
        "name": card.metadata.name,
        "version": card.metadata.version,
        "protocol_version": srv.protocol_version,
        "transports": list(srv.transports),
        "endpoints": {
            "mcp": srv.discovery.mcp_path,
            "agent_card": srv.discovery.agent_card_path,
        },
        "content_types": list(srv.content_types),
        "authentication": {
            "required": srv.authentication.required,
            "mechanisms": list(srv.authentication.mechanisms),
        },
        "authorization": {
            "model": srv.authorization.model,
            "default": srv.authorization.default,
            "self_expansion": srv.authorization.self_expansion,
        },
        "idempotency": {
            "required_for_mutations": srv.idempotency.required_for_mutations,
            "header": srv.idempotency.header,
        },
        "tools": [
            {
                "name": t.name,
                "mutating": t.mutating,
                "permission": t.permission,
                "input_schema": t.input_schema,
            }
            for t in card.mcp.tools
        ],
        "resources": [
            {"uri_template": r.uri_template, "permission": r.permission}
            for r in card.mcp.resources
        ],
        "prompts": [{"name": p.name, "purpose": p.purpose} for p in card.mcp.prompts],
        "errors": {
            "format": card.mcp.errors.format,
            "stable_codes": list(card.mcp.errors.stable_codes),
        },
        "readiness": card.deployment.readiness_claim,
    }


def assert_surface_matches_card() -> None:
    """One call that fails the build on any drift from the card."""
    errors.assert_codes_match_card()
    tools.assert_dispatch_covers_card()
    resources.assert_readers_cover_card()
    resources.assert_prompts_cover_card()


@dataclass
class SovereigntyMcpServer:
    """The surface, independent of transport."""

    router: ToolRouter
    resource_services: ResourceServices

    def __post_init__(self) -> None:
        assert_surface_matches_card()

    def call_tool(
        self,
        name: str,
        caller: CallerIdentity,
        arguments: dict[str, Any],
        *,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """Invoke a tool, rendering any refusal as problem+json.

        Refusals are returned rather than raised so a transport cannot
        accidentally turn a BOUNDARY_VIOLATION into a generic 500 that loses the
        stable code the client is meant to branch on.
        """
        try:
            result = self.router.call(
                name, caller, arguments, idempotency_key=idempotency_key
            )
        except errors.ProblemDetails as problem:
            return {"ok": False, "problem": problem.to_json()}
        except tools.ToolInputError as bad_input:
            return {
                "ok": False,
                "problem": errors.ProblemDetails(
                    "INSUFFICIENT_CONFIDENCE", str(bad_input), extra={"tool": name}
                ).to_json(),
            }
        return {"ok": True, "result": result}

    def read_resource(self, uri: str, caller: CallerIdentity) -> dict[str, Any]:
        try:
            return {
                "ok": True,
                "result": resources.resolve(uri, caller, self.resource_services),
            }
        except errors.ProblemDetails as problem:
            return {"ok": False, "problem": problem.to_json()}

    def list_prompts(self) -> tuple[dict[str, str], ...]:
        return resources.prompts()

    def agent_card_json(self) -> str:
        return json.dumps(agent_card(), indent=2, sort_keys=True)


def build_server(
    services: ToolServices, resource_services: ResourceServices
) -> SovereigntyMcpServer:
    return SovereigntyMcpServer(
        router=ToolRouter(services=services), resource_services=resource_services
    )


def _require_sdk() -> Any:
    state = _load_sdk()
    if not state["ok"]:
        raise RuntimeError(
            f"the MCP SDK is not importable ({state['error']}); install `mcp` to "
            "serve a transport. The router and agent card work without it."
        )
    return state["Server"]


def serve_stdio(server: SovereigntyMcpServer) -> None:  # pragma: no cover
    """Serve over stdio. Blocks; never called at import."""
    _require_sdk()
    raise NotImplementedError(
        "stdio transport requires an authenticated CallerIdentity source; wire "
        "the deployment's OIDC/service-identity resolver before serving"
    )


def serve_http(server: SovereigntyMcpServer, host: str, port: int) -> None:  # pragma: no cover
    """Serve streamable-http. Blocks; never called at import."""
    _require_sdk()
    raise NotImplementedError(
        "streamable-http transport requires an authenticated CallerIdentity "
        "source; wire the deployment's OIDC/service-identity resolver before "
        "serving"
    )
