"""Spans carrying the card's ``observability.tracing.required_span_attributes``.

The card names nine attributes, and two of them are named ``*.id_hash`` rather
than ``*.id``. That spelling is the enforceable half of
``security_and_privacy.data_minimization``: a trace backend is a long-lived,
widely-readable, cross-tenant store, so a user identifier that reaches it must be
a pseudonym. Anyone who can join a raw ``user.id`` in a span to the same id in
the profile tables has reconstructed exactly the per-user behavioural record the
card promises not to export.

Making that a review rule would not hold. Instead it is structural:
:class:`SpanAttributes` **has no field that can hold a raw subject identifier**.
Its constructor takes hashes; the only way to obtain one is
:func:`hash_identifier`, which is applied by :meth:`SpanAttributes.for_subject`
on the way in and discards its input immediately. A raw id handed to the hash
field is rejected by ``__post_init__`` with an actionable message rather than
being hashed for you — silently accepting it would make the same mistake
invisible at the next call site that formats the id differently.

``tenant.id`` is deliberately *not* hashed: the card lists it unhashed, a tenant
is an organisation rather than a natural person, and tenant-scoped querying is
the operational reason traces exist. Data minimisation here is about the
individual.

**Backend.** ``opentelemetry-api`` is imported lazily and used when present.
Because the OTel API without a configured SDK is a documented no-op that discards
attributes, every span is also appended to a *bounded* in-process ring buffer;
that buffer is what the degraded-operation tests and the local HUD read.
:func:`backend_status` reports which path is live.
"""

from __future__ import annotations

import hashlib
import os
import re
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, Iterator, Mapping

from src.model_card.loader import cached_model_card
from src.model_card.types import AgentModelCard
from src.observability.metrics import (
    INSTRUMENTATION_SCOPE,
    MetricRegistry,
    default_registry,
)
from src.torx_layer.state import BRIDGE_STRATEGIES, TENSION_CLASSES

# Card attribute keys. Written out rather than derived from the card so a typo
# here is caught by verify_span_attributes() instead of producing a span that
# quietly disagrees with the contract.
ATTR_TENANT_ID = "tenant.id"
ATTR_USER_ID_HASH = "user.id_hash"
ATTR_GROUP_ID_HASH = "group.id_hash"
ATTR_PROFILE_REVISION = "profile.revision"
ATTR_GROUP_INTENT_REVISION = "group_intent.revision"
ATTR_TENSION_CLASS = "tension.class"
ATTR_BRIDGE_TYPE = "bridge.type"
ATTR_EFFECTIVE_MANIFEST_HASH = "effective_manifest.hash"
ATTR_VDF_PROOF_ID = "vdf.proof_id"

SPAN_ATTRIBUTE_KEYS: tuple[str, ...] = (
    ATTR_TENANT_ID,
    ATTR_USER_ID_HASH,
    ATTR_GROUP_ID_HASH,
    ATTR_PROFILE_REVISION,
    ATTR_GROUP_INTENT_REVISION,
    ATTR_TENSION_CLASS,
    ATTR_BRIDGE_TYPE,
    ATTR_EFFECTIVE_MANIFEST_HASH,
    ATTR_VDF_PROOF_ID,
)

#: 16-byte BLAKE2b digest: 128 bits is far past collision risk for a tenant's
#: user population, and short enough that traces stay readable.
_HASH_BYTES = 16
_HASH_RE = re.compile(r"^[0-9a-f]{32}$")
_MANIFEST_HASH_RE = re.compile(r"^[0-9a-f]{16,128}$")
_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)

#: Environment name of the hashing pepper. Without it the hash is a pure
#: function of the id, so anyone holding the id list can re-derive every
#: pseudonym. It is a deployment secret, not a per-process random value: the
#: pseudonym must be stable across processes or spans from two replicas of the
#: same request cannot be correlated.
ID_HASH_SALT_ENV = "TORX_ID_HASH_SALT"

#: Used when the pepper is unset. Hashing still removes the raw id from the
#: trace, which is the invariant; the pepper only raises the cost of a
#: dictionary attack, so its absence degrades privacy without breaking the
#: contract, and a deployment is expected to set it.
_DEFAULT_SALT = "gc.torx-contextual-sovereignty-agent/unpeppered"

#: Bound on the in-process span buffer. A ring buffer, because an unbounded
#: recorder in a long-running worker is a memory leak wearing a telemetry
#: costume.
SPAN_BUFFER_SIZE = 256


class SpanContractError(ValueError):
    """Raised when emitted span attributes do not match the card's list."""


# --------------------------------------------------------------------------
# identifier hashing
# --------------------------------------------------------------------------


def _salt() -> bytes:
    return os.environ.get(ID_HASH_SALT_ENV, _DEFAULT_SALT).encode("utf-8")


def hash_identifier(raw: str, *, scope: str = "") -> str:
    """Pseudonymise a subject identifier for export.

    ``scope`` (normally the tenant id) is mixed in so the same user in two
    tenants does not produce the same pseudonym; cross-tenant linkage would
    re-create the identifier graph the hash exists to break.

    Keyed BLAKE2b rather than a bare digest: the pepper is the key, so the
    construction is a MAC and length-extension or precomputation against a plain
    ``sha256(uuid)`` table does not apply.
    """
    if not isinstance(raw, str) or not raw:
        raise ValueError("hash_identifier requires a non-empty string identifier")
    digest = hashlib.blake2b(
        f"{scope}\x00{raw}".encode("utf-8"), digest_size=_HASH_BYTES, key=_salt()
    )
    return digest.hexdigest()


def _validate_hash(value: str, field: str) -> str:
    if value == "":
        return value
    if _UUID_RE.match(value):
        raise ValueError(
            f"{field} was given the raw identifier {value!r}. Raw subject ids "
            "must never reach a trace backend; build the attributes with "
            "SpanAttributes.for_subject(user_id=...) or hash_identifier() "
            "instead."
        )
    if not _HASH_RE.match(value):
        raise ValueError(
            f"{field} must be a 32-character lowercase hex digest from "
            f"hash_identifier(), got {value!r}"
        )
    return value


# --------------------------------------------------------------------------
# attributes
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SpanAttributes:
    """The nine card attributes, validated.

    Every field has a defined "not known yet" value — ``""`` for the string
    facets and ``0`` for the revisions (revisions are 1-based in an append-only
    history, so 0 is unambiguously "no revision"). The span always emits all nine
    keys with those defaults rather than omitting them, so a trace query can
    index a fixed key set instead of guessing which spans carry which facet.
    """

    tenant_id: str
    user_id_hash: str = ""
    group_id_hash: str = ""
    profile_revision: int = 0
    group_intent_revision: int = 0
    tension_class: str = ""
    bridge_type: str = ""
    effective_manifest_hash: str = ""
    vdf_proof_id: str = ""

    def __post_init__(self) -> None:
        if not self.tenant_id:
            raise ValueError(
                "tenant_id is required: an unscoped span cannot be routed to a "
                "tenant's trace store and would leak across row-level security"
            )
        _validate_hash(self.user_id_hash, "user_id_hash")
        _validate_hash(self.group_id_hash, "group_id_hash")
        for field in ("profile_revision", "group_intent_revision"):
            v = getattr(self, field)
            if not isinstance(v, int) or isinstance(v, bool) or v < 0:
                raise ValueError(f"{field} must be a non-negative int, got {v!r}")
        if self.tension_class and self.tension_class not in TENSION_CLASSES:
            raise ValueError(
                f"tension_class {self.tension_class!r} is not one of "
                f"{list(TENSION_CLASSES)}"
            )
        if self.bridge_type and self.bridge_type not in BRIDGE_STRATEGIES:
            raise ValueError(
                f"bridge_type {self.bridge_type!r} is not one of "
                f"{list(BRIDGE_STRATEGIES)}"
            )
        if self.effective_manifest_hash and not _MANIFEST_HASH_RE.match(
            self.effective_manifest_hash
        ):
            raise ValueError(
                "effective_manifest_hash must be a lowercase hex digest, got "
                f"{self.effective_manifest_hash!r}"
            )

    @classmethod
    def for_subject(
        cls,
        *,
        tenant_id: str,
        user_id: str | None = None,
        group_id: str | None = None,
        **rest: Any,
    ) -> SpanAttributes:
        """Build from raw ids, hashing them on the way in.

        The raw values are arguments only; nothing on the returned object can
        reproduce them.
        """
        return cls(
            tenant_id=tenant_id,
            user_id_hash=(
                hash_identifier(user_id, scope=tenant_id) if user_id else ""
            ),
            group_id_hash=(
                hash_identifier(group_id, scope=tenant_id) if group_id else ""
            ),
            **rest,
        )

    def evolve(self, **changes: Any) -> SpanAttributes:
        """A revalidated copy. Used when a facet becomes known mid-span."""
        return replace(self, **changes)

    def attributes(self) -> dict[str, Any]:
        """The nine card keys, ready to attach to a span."""
        return {
            ATTR_TENANT_ID: self.tenant_id,
            ATTR_USER_ID_HASH: self.user_id_hash,
            ATTR_GROUP_ID_HASH: self.group_id_hash,
            ATTR_PROFILE_REVISION: self.profile_revision,
            ATTR_GROUP_INTENT_REVISION: self.group_intent_revision,
            ATTR_TENSION_CLASS: self.tension_class,
            ATTR_BRIDGE_TYPE: self.bridge_type,
            ATTR_EFFECTIVE_MANIFEST_HASH: self.effective_manifest_hash,
            ATTR_VDF_PROOF_ID: self.vdf_proof_id,
        }

    @property
    def unset(self) -> tuple[str, ...]:
        """Card keys still at their "not known" default.

        Reported rather than enforced: a tension snapshot legitimately has no
        ``bridge.type`` yet, and refusing to emit that span would lose the only
        record of how the tension was classified.
        """
        return tuple(
            k for k, v in self.attributes().items() if v == "" or v == 0
        )


def verify_span_attributes(
    keys: Mapping[str, Any] | tuple[str, ...], card: AgentModelCard | None = None
) -> tuple[str, ...]:
    """Assert the emitted key set is exactly ``required_span_attributes``.

    Raises on drift in either direction: a card edit that adds an attribute must
    fail the build rather than produce spans that silently lack it, and an
    attribute this module emits but the card does not declare is an undocumented
    field flowing to the trace backend.
    """
    card = card if card is not None else cached_model_card()
    required = tuple(card.observability.tracing.required_span_attributes)
    emitted = tuple(keys) if not isinstance(keys, Mapping) else tuple(keys.keys())
    missing = [k for k in required if k not in emitted]
    extra = [k for k in emitted if k not in required]
    if missing or extra:
        raise SpanContractError(
            "span attributes do not match "
            "observability.tracing.required_span_attributes. "
            f"required but not emitted: {sorted(missing)}; "
            f"emitted but not required: {sorted(extra)}"
        )
    return required


# --------------------------------------------------------------------------
# optional OpenTelemetry backend
# --------------------------------------------------------------------------

_otel_state: dict[str, Any] = {"loaded": False, "ok": False, "error": None}


def _load_otel() -> dict[str, Any]:
    """Import ``opentelemetry.trace`` once, recording why it failed if it did."""
    if _otel_state["loaded"]:
        return _otel_state
    _otel_state["loaded"] = True
    try:
        from opentelemetry import trace as otel_trace

        _otel_state["trace"] = otel_trace
        _otel_state["ok"] = True
    except Exception as exc:  # pragma: no cover - exercised only without OTel
        _otel_state["error"] = f"{type(exc).__name__}: {exc}"
    return _otel_state


def otel_available() -> bool:
    return bool(_load_otel()["ok"])


def backend_status() -> dict[str, Any]:
    st = _load_otel()
    return {
        "otel_available": bool(st["ok"]),
        "error": st.get("error"),
        "backend": "opentelemetry" if st["ok"] else "in-process",
    }


# --------------------------------------------------------------------------
# in-process recorder
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RecordedSpan:
    """One finished span, as the in-process recorder saw it."""

    name: str
    attributes: Mapping[str, Any]
    duration_s: float
    error: str | None
    backend: str

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "attributes": dict(self.attributes),
            "duration_s": self.duration_s,
            "error": self.error,
            "backend": self.backend,
        }


class SpanRecorder:
    """Bounded ring buffer of finished spans."""

    def __init__(self, maxlen: int = SPAN_BUFFER_SIZE) -> None:
        if maxlen < 1:
            raise ValueError(f"span buffer size must be >= 1, got {maxlen}")
        self._buf: deque[RecordedSpan] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def append(self, span: RecordedSpan) -> None:
        with self._lock:
            self._buf.append(span)

    def spans(self) -> tuple[RecordedSpan, ...]:
        with self._lock:
            return tuple(self._buf)

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()


_recorder = SpanRecorder()


def recorded_spans() -> tuple[RecordedSpan, ...]:
    return _recorder.spans()


def reset_recorded_spans() -> None:
    _recorder.clear()


# --------------------------------------------------------------------------
# the span helper
# --------------------------------------------------------------------------


class SpanHandle:
    """Live span: attributes can still be narrowed while the block runs.

    ``bridge.type`` and ``vdf.proof_id`` are only known part-way through the
    operation they describe, so the handle re-validates and re-attaches rather
    than forcing the caller to open a second span (which would break the
    parent/child shape of the trace).
    """

    __slots__ = ("name", "_attrs", "_otel_span")

    def __init__(self, name: str, attrs: SpanAttributes, otel_span: Any) -> None:
        self.name = name
        self._attrs = attrs
        self._otel_span = otel_span

    @property
    def attrs(self) -> SpanAttributes:
        return self._attrs

    def update(self, **changes: Any) -> SpanAttributes:
        """Replace field values, revalidating. Raw ids are rejected here too."""
        self._attrs = self._attrs.evolve(**changes)
        if self._otel_span is not None:
            for key, value in self._attrs.attributes().items():
                self._otel_span.set_attribute(key, value)
        return self._attrs


@contextmanager
def span(
    name: str,
    attrs: SpanAttributes,
    *,
    metric: str | None = None,
    registry: MetricRegistry | None = None,
    card: AgentModelCard | None = None,
    recorder: SpanRecorder | None = None,
) -> Iterator[SpanHandle]:
    """Emit one span carrying every card-required attribute.

    ``metric`` optionally names a ``_seconds`` histogram to receive the span's
    duration, so a caller does not have to time the same block twice.
    """
    otel_span = None
    st = _load_otel()
    if st["ok"]:
        tracer = st["trace"].get_tracer(INSTRUMENTATION_SCOPE)
        otel_span = tracer.start_span(name)
        for key, value in attrs.attributes().items():
            otel_span.set_attribute(key, value)

    handle = SpanHandle(name, attrs, otel_span)
    sink = recorder if recorder is not None else _recorder
    backend = "opentelemetry" if otel_span is not None else "in-process"
    start = time.perf_counter()
    error: str | None = None
    try:
        yield handle
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        duration = time.perf_counter() - start
        emitted = handle.attrs.attributes()
        # Checked on the way out, so a card edit fails on the first span the
        # process emits rather than at some later reporting stage.
        verify_span_attributes(emitted, card)
        sink.append(
            RecordedSpan(
                name=name,
                attributes=dict(emitted),
                duration_s=round(duration, 9),
                error=error,
                backend=backend,
            )
        )
        if metric is not None:
            (registry if registry is not None else default_registry()).instrument(
                metric
            ).record(duration)
        if otel_span is not None:
            if error is not None:
                otel_span.set_attribute("error", True)
            otel_span.end()
