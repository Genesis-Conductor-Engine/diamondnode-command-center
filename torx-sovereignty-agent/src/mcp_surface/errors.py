"""RFC 9457 ``application/problem+json`` errors for the MCP surface.

Named ``mcp_surface`` rather than ``mcp``: the installed MCP SDK owns the
top-level module name ``mcp``, and a local package shadowing it would break the
server import entirely.

The card fixes seven ``stable_codes``. They are stable in the contractual sense —
clients branch on them — so the code set is derived from the loaded card and
:func:`assert_codes_match_card` fails the build if the two ever drift. A code
that exists in the card but not here is an unimplemented contract; a code here
but not in the card is an undocumented one clients cannot rely on.

Errors carry a ``trace_id`` because every failure has to be joinable to the span
that produced it — the card lists ``trace_id`` as a required field, and a
boundary rejection nobody can trace back to its decision is not auditable.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping

from src.model_card import cached_model_card

PROBLEM_CONTENT_TYPE = "application/problem+json"
PROBLEM_BASE_URI = "https://genesisconductor.io/problems"

# HTTP status per code. Chosen so a client can act without parsing prose:
# 403 is "you may not", 409 is "your view of the world is stale, re-read", and
# 422 is "the request was understood and refused on its merits".
_CODE_STATUS: dict[str, tuple[int, str]] = {
    "BOUNDARY_VIOLATION": (403, "Hard boundary violation"),
    "INSUFFICIENT_CONFIDENCE": (422, "Insufficient confidence"),
    "AUTHORIZATION_DENIED": (403, "Authorization denied"),
    "STALE_INTENT_REVISION": (409, "Stale intent revision"),
    "BRIDGE_INCREASES_TENSION": (422, "Bridge increases tension"),
    "PROFILE_REVISION_CONFLICT": (409, "Profile revision conflict"),
    "VDF_VERIFICATION_FAILED": (422, "VDF verification failed"),
}


def card_codes() -> tuple[str, ...]:
    return tuple(cached_model_card().mcp.errors.stable_codes)


def assert_codes_match_card() -> None:
    """Fail loudly when the implemented code set drifts from the card."""
    declared = set(card_codes())
    implemented = set(_CODE_STATUS)
    if declared != implemented:
        missing = sorted(declared - implemented)
        extra = sorted(implemented - declared)
        raise AssertionError(
            "MCP error codes drifted from the model card: "
            f"unimplemented={missing} undocumented={extra}"
        )


@dataclass(frozen=True, slots=True)
class ProblemDetails(Exception):
    """A problem+json document that is also the raised exception.

    One object rather than an exception plus a serialiser: a handler that
    catches the error always has the exact document the client will receive, so
    the two cannot describe different failures.
    """

    code: str
    detail: str
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    instance: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.code not in _CODE_STATUS:
            raise ValueError(
                f"{self.code!r} is not one of the card's stable codes: "
                f"{sorted(_CODE_STATUS)}"
            )
        if not self.detail.strip():
            raise ValueError(
                f"{self.code}: detail must be a non-empty, actionable message"
            )
        Exception.__init__(self, f"{self.code}: {self.detail}")

    @property
    def status(self) -> int:
        return _CODE_STATUS[self.code][0]

    @property
    def title(self) -> str:
        return _CODE_STATUS[self.code][1]

    @property
    def type(self) -> str:
        return f"{PROBLEM_BASE_URI}/{self.code.lower().replace('_', '-')}"

    def to_json(self) -> dict[str, Any]:
        doc: dict[str, Any] = {
            "type": self.type,
            "title": self.title,
            "status": self.status,
            "code": self.code,
            "trace_id": self.trace_id,
            "detail": self.detail,
        }
        if self.instance:
            doc["instance"] = self.instance
        # Extras never overwrite a required field: a caller-supplied "status"
        # must not be able to turn a 403 into a 200 in the serialised document.
        required = set(doc)
        for k, v in self.extra.items():
            if k not in required:
                doc[k] = v
        return doc


def boundary_violation(detail: str, **extra: Any) -> ProblemDetails:
    return ProblemDetails("BOUNDARY_VIOLATION", detail, extra=extra)


def authorization_denied(detail: str, **extra: Any) -> ProblemDetails:
    return ProblemDetails("AUTHORIZATION_DENIED", detail, extra=extra)


def insufficient_confidence(detail: str, **extra: Any) -> ProblemDetails:
    return ProblemDetails("INSUFFICIENT_CONFIDENCE", detail, extra=extra)


def stale_intent_revision(detail: str, **extra: Any) -> ProblemDetails:
    return ProblemDetails("STALE_INTENT_REVISION", detail, extra=extra)


def bridge_increases_tension(detail: str, **extra: Any) -> ProblemDetails:
    return ProblemDetails("BRIDGE_INCREASES_TENSION", detail, extra=extra)


def profile_revision_conflict(detail: str, **extra: Any) -> ProblemDetails:
    return ProblemDetails("PROFILE_REVISION_CONFLICT", detail, extra=extra)


def vdf_verification_failed(detail: str, **extra: Any) -> ProblemDetails:
    return ProblemDetails("VDF_VERIFICATION_FAILED", detail, extra=extra)
