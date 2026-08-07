"""The card's ``observability.metrics``, declared once and enforced against it.

``observability.metrics`` is a *contract*, not a wish list: the degraded-operation
and no-false-consensus result cards are only auditable if the counters they name
actually exist and actually move. A metric that is listed in the card but never
registered is worse than a missing metric — the operator reads "no boundary
violations blocked" from an empty series and cannot distinguish it from "the
counter was never wired".

So this module inverts the usual direction. It does not export a hand-kept list
of names; it declares a *kind* (counter / gauge / histogram) for every name and
then checks that the kind table and the card's list are the same set. A card edit
that adds ``gc_something_total`` fails :func:`verify_metric_names` — and therefore
fails registry construction and the build — instead of being quietly
unimplemented.

**Backend.** ``opentelemetry-api`` is imported lazily; when present, every record
is mirrored into a real OTel instrument so a configured SDK exports it. The OTel
API *without* a configured SDK is a documented no-op (``NoOpMeter``), and the
degradation controller has to be able to read its own timings back in-process, so
the in-process accumulator is always the source of truth for
:meth:`MetricRegistry.snapshot`. ``registry.backend`` reports which path is live,
matching the ``backend`` field the TORX circuits and the consensus sampler
already publish.

Recorded values are rounded to :data:`~src.torx_layer.state.STORAGE_PRECISION`
decimals before accumulation, so a snapshot serialised into an audit record is
byte-stable whichever backend produced it.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Mapping

from src.model_card.loader import cached_model_card
from src.model_card.types import AgentModelCard
from src.torx_layer.state import STORAGE_PRECISION

#: Instrument kinds. Named rather than an enum because the strings land in JSON
#: snapshots and in the OTel instrument description.
COUNTER = "counter"
GAUGE = "gauge"
HISTOGRAM = "histogram"

_KINDS = (COUNTER, GAUGE, HISTOGRAM)

#: The declared kind of every metric in ``observability.metrics``.
#:
#: Kind is a semantic choice, not a formatting one: ``gc_group_tension_current``
#: is the *current* group tension, so summing it across a window is meaningless
#: and it must be a gauge; the ``_seconds`` and ``_delta`` families need their
#: distribution (a p99 compile time is the load-qualification gate, an average is
#: not), so they are histograms. Everything ``_total`` is a monotone counter.
METRIC_KINDS: Mapping[str, str] = {
    "gc_profile_revision_total": COUNTER,
    "gc_profile_mutation_rejected_total": COUNTER,
    "gc_group_tension_current": GAUGE,
    "gc_bridge_predicted_delta": HISTOGRAM,
    "gc_bridge_observed_delta": HISTOGRAM,
    "gc_boundary_violation_block_total": COUNTER,
    "gc_false_consensus_guard_total": COUNTER,
    "gc_effective_agent_compile_seconds": HISTOGRAM,
    "gc_torx_update_seconds": HISTOGRAM,
    "gc_topology_sidecar_seconds": HISTOGRAM,
    "gc_vdf_attestation_seconds": HISTOGRAM,
    "gc_mcp_tool_call_total": COUNTER,
}

#: OTel units. ``1`` is the OTel spelling for a dimensionless count.
METRIC_UNITS: Mapping[str, str] = {
    name: ("s" if name.endswith("_seconds") else "1")
    for name in METRIC_KINDS
}

METRIC_DESCRIPTIONS: Mapping[str, str] = {
    "gc_profile_revision_total": "append-only profile revisions created",
    "gc_profile_mutation_rejected_total": (
        "profile mutations refused by the evidence or permission guard"
    ),
    "gc_group_tension_current": "current aggregate group tension in [0, 1]",
    "gc_bridge_predicted_delta": "simulated tension change of a proposed bridge",
    "gc_bridge_observed_delta": "measured tension change after a bridge applied",
    "gc_boundary_violation_block_total": (
        "actions blocked by a hard individual boundary"
    ),
    "gc_false_consensus_guard_total": (
        "aggregations refused because consensus would have been manufactured"
    ),
    "gc_effective_agent_compile_seconds": "effective-agent manifest compile time",
    "gc_torx_update_seconds": "TORX factor-graph update time",
    "gc_topology_sidecar_seconds": "topology sidecar descriptor latency",
    "gc_vdf_attestation_seconds": "Rule 30 VDF attestation time",
    "gc_mcp_tool_call_total": "MCP tool invocations by tool and outcome",
}

#: Instrument namespace. Shared with the tracer so a span and its metrics carry
#: the same scope name in an exported trace.
INSTRUMENTATION_SCOPE = "gc.torx-contextual-sovereignty-agent"


def _round(x: float) -> float:
    return round(float(x), STORAGE_PRECISION)


# --------------------------------------------------------------------------
# card agreement
# --------------------------------------------------------------------------


class MetricContractError(ValueError):
    """Raised when the kind table and the card's metric list disagree."""


def verify_metric_names(card: AgentModelCard | None = None) -> tuple[str, ...]:
    """Assert :data:`METRIC_KINDS` covers ``observability.metrics`` exactly.

    Returns the card's list in card order. Raises rather than warning: a metric
    the card promises but the process does not export makes every dashboard
    built on it silently wrong, and a metric exported but not declared is an
    undocumented data flow out of the process.
    """
    card = card if card is not None else cached_model_card()
    declared = tuple(card.observability.metrics)
    missing = [n for n in declared if n not in METRIC_KINDS]
    extra = [n for n in METRIC_KINDS if n not in declared]
    if missing or extra:
        raise MetricContractError(
            "observability.metrics and METRIC_KINDS disagree. "
            f"declared in card but not implemented: {sorted(missing)}; "
            f"implemented but not in card: {sorted(extra)}. "
            "Add the metric to METRIC_KINDS (with its kind, unit and "
            "description) or remove it from the card."
        )
    bad_kinds = sorted(n for n, k in METRIC_KINDS.items() if k not in _KINDS)
    if bad_kinds:
        raise MetricContractError(
            f"metrics with an unknown kind: {bad_kinds}; kind must be one of "
            f"{list(_KINDS)}"
        )
    return declared


# --------------------------------------------------------------------------
# optional OpenTelemetry backend
# --------------------------------------------------------------------------

_otel_state: dict[str, Any] = {"loaded": False, "ok": False, "error": None}


def _load_otel() -> dict[str, Any]:
    """Import ``opentelemetry.metrics`` once, recording why it failed if it did.

    Same lazy-optional-backend idiom as ``torx_layer.circuits._load_torx``:
    importing must never raise at module scope, because observability is not
    allowed to be the reason the agent cannot start.
    """
    if _otel_state["loaded"]:
        return _otel_state
    _otel_state["loaded"] = True
    try:
        from opentelemetry import metrics as otel_metrics

        _otel_state["metrics"] = otel_metrics
        _otel_state["ok"] = True
    except Exception as exc:  # pragma: no cover - exercised only without OTel
        _otel_state["error"] = f"{type(exc).__name__}: {exc}"
    return _otel_state


def otel_available() -> bool:
    return bool(_load_otel()["ok"])


def backend_status() -> dict[str, Any]:
    """Which metrics path is live, for the HUD and for audit records."""
    st = _load_otel()
    version: str | None
    try:
        from importlib.metadata import version as _pkg_version

        version = _pkg_version("opentelemetry-api")
    except Exception:
        version = None
    return {
        "otel_available": bool(st["ok"]),
        "otel_version": version,
        "error": st.get("error"),
        "backend": "opentelemetry" if st["ok"] else "in-process",
    }


# --------------------------------------------------------------------------
# points and series
# --------------------------------------------------------------------------

AttrKey = tuple[tuple[str, str], ...]


def _attr_key(attributes: Mapping[str, Any]) -> AttrKey:
    """Canonical, order-independent identity for an attribute set.

    Values are stringified so ``{"outcome": 1}`` and ``{"outcome": "1"}`` are the
    same series — otherwise a caller that formats one call site differently
    splits a counter in half without any error.
    """
    return tuple(sorted((str(k), str(v)) for k, v in attributes.items()))


@dataclass(frozen=True, slots=True)
class MetricPoint:
    """An immutable read of one ``(name, attributes)`` series."""

    name: str
    kind: str
    attributes: AttrKey
    value: float
    count: int
    total: float
    minimum: float | None
    maximum: float | None

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "attributes": {k: v for k, v in self.attributes},
            "value": self.value,
            "count": self.count,
            "sum": self.total,
            "min": self.minimum,
            "max": self.maximum,
        }


class _Series:
    """Mutable accumulator behind one ``(name, attributes)`` pair."""

    __slots__ = ("kind", "value", "count", "total", "minimum", "maximum")

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.value = 0.0
        self.count = 0
        self.total = 0.0
        self.minimum: float | None = None
        self.maximum: float | None = None

    def update(self, value: float) -> None:
        v = _round(value)
        self.count += 1
        self.total = _round(self.total + v)
        self.minimum = v if self.minimum is None else min(self.minimum, v)
        self.maximum = v if self.maximum is None else max(self.maximum, v)
        if self.kind == GAUGE:
            self.value = v
        elif self.kind == COUNTER:
            self.value = self.total
        else:  # histogram: ``value`` reports the running mean
            self.value = _round(self.total / self.count)


# --------------------------------------------------------------------------
# instruments
# --------------------------------------------------------------------------


class Instrument:
    """One card metric, bound to a registry.

    The three record methods are deliberately *not* interchangeable: calling
    ``.set()`` on a counter would silently reinterpret a monotone total as a
    level, so the wrong method raises with the name of the right one.
    """

    __slots__ = ("name", "kind", "_registry", "_otel")

    def __init__(self, name: str, kind: str, registry: MetricRegistry, otel: Any) -> None:
        self.name = name
        self.kind = kind
        self._registry = registry
        self._otel = otel

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Instrument {self.name} {self.kind}>"

    def _require(self, kind: str, method: str) -> None:
        if self.kind != kind:
            right = {COUNTER: "add", GAUGE: "set", HISTOGRAM: "record"}[self.kind]
            raise ValueError(
                f"{self.name!r} is a {self.kind}; {method}() is for a {kind}. "
                f"Use {right}() instead."
            )

    def add(self, value: float = 1.0, /, **attributes: Any) -> None:
        """Increment a counter."""
        self._require(COUNTER, "add")
        if value < 0:
            raise ValueError(
                f"{self.name!r} is a counter and cannot decrease; got {value}"
            )
        self._registry._record(self.name, self.kind, float(value), attributes)
        if self._otel is not None:
            self._otel.add(float(value), attributes or None)

    def set(self, value: float, /, **attributes: Any) -> None:
        """Set a gauge to its current level."""
        self._require(GAUGE, "set")
        self._registry._record(self.name, self.kind, float(value), attributes)
        if self._otel is not None:
            self._otel.set(float(value), attributes or None)

    def record(self, value: float, /, **attributes: Any) -> None:
        """Record one observation into a histogram."""
        self._require(HISTOGRAM, "record")
        self._registry._record(self.name, self.kind, float(value), attributes)
        if self._otel is not None:
            self._otel.record(float(value), attributes or None)

    def observe(self, value: float, /, **attributes: Any) -> None:
        """Kind-agnostic record, for callers driving a metric by name."""
        {COUNTER: self.add, GAUGE: self.set, HISTOGRAM: self.record}[self.kind](
            value, **attributes
        )


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------


class MetricRegistry:
    """Every card metric, pre-created, with an in-process readback.

    Instruments are created eagerly at construction so that the set of exported
    series is fixed by the card rather than by which code paths happened to run;
    an operator looking at ``gc_boundary_violation_block_total`` sees ``0``, not
    a missing series.
    """

    def __init__(
        self,
        card: AgentModelCard | None = None,
        *,
        use_otel: bool | None = None,
    ) -> None:
        self._names = verify_metric_names(card)
        self._lock = threading.Lock()
        self._series: dict[tuple[str, AttrKey], _Series] = {}

        meter = None
        if use_otel is None:
            use_otel = otel_available()
        if use_otel:
            st = _load_otel()
            if not st["ok"]:
                raise ValueError(
                    "use_otel=True but opentelemetry-api is not importable: "
                    f"{st.get('error')}"
                )
            meter = st["metrics"].get_meter(INSTRUMENTATION_SCOPE)
        self._backend = "opentelemetry" if meter is not None else "in-process"

        self._instruments: dict[str, Instrument] = {}
        for name in self._names:
            kind = METRIC_KINDS[name]
            self._instruments[name] = Instrument(
                name, kind, self, self._make_otel(meter, name, kind)
            )

    # -- construction helpers ------------------------------------------------

    @staticmethod
    def _make_otel(meter: Any, name: str, kind: str) -> Any:
        if meter is None:
            return None
        unit = METRIC_UNITS[name]
        description = METRIC_DESCRIPTIONS[name]
        try:
            if kind == COUNTER:
                return meter.create_counter(name, unit=unit, description=description)
            if kind == GAUGE:
                return meter.create_gauge(name, unit=unit, description=description)
            return meter.create_histogram(name, unit=unit, description=description)
        except Exception:
            # An OTel version without ``create_gauge`` must not take the process
            # down; the in-process accumulator still carries the value.
            return None

    # -- introspection -------------------------------------------------------

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def names(self) -> tuple[str, ...]:
        """Metric names, in card order."""
        return self._names

    def kind_of(self, name: str) -> str:
        return METRIC_KINDS[self._checked(name)]

    def _checked(self, name: str) -> str:
        if name not in self._instruments:
            raise KeyError(
                f"unknown metric {name!r}; the card declares {list(self._names)}"
            )
        return name

    # -- instrument accessors ------------------------------------------------

    def instrument(self, name: str) -> Instrument:
        return self._instruments[self._checked(name)]

    def counter(self, name: str) -> Instrument:
        return self._typed(name, COUNTER)

    def gauge(self, name: str) -> Instrument:
        return self._typed(name, GAUGE)

    def histogram(self, name: str) -> Instrument:
        return self._typed(name, HISTOGRAM)

    def _typed(self, name: str, kind: str) -> Instrument:
        inst = self._instruments[self._checked(name)]
        if inst.kind != kind:
            raise ValueError(
                f"{name!r} is declared as a {inst.kind}, not a {kind}"
            )
        return inst

    # -- recording -----------------------------------------------------------

    def _record(
        self, name: str, kind: str, value: float, attributes: Mapping[str, Any]
    ) -> None:
        key = (name, _attr_key(attributes))
        with self._lock:
            series = self._series.get(key)
            if series is None:
                series = _Series(kind)
                self._series[key] = series
            series.update(value)

    @contextmanager
    def timed(self, name: str, /, **attributes: Any) -> Iterator[None]:
        """Time a block into a ``_seconds`` histogram.

        The observation is recorded on the way out **including** the failure
        path: a topology sidecar call that raised still took wall-clock time on
        the caller, and hiding it would make the degradation controller's own
        latency evidence unfalsifiable.
        """
        self._typed(name, HISTOGRAM)
        start = time.perf_counter()
        try:
            yield
        finally:
            self.instrument(name).record(time.perf_counter() - start, **attributes)

    # -- readback ------------------------------------------------------------

    def points(self, name: str | None = None) -> tuple[MetricPoint, ...]:
        """Every recorded series, or those of one metric, sorted for stability."""
        if name is not None:
            self._checked(name)
        with self._lock:
            items = [
                (k, s.kind, s.value, s.count, s.total, s.minimum, s.maximum)
                for k, s in self._series.items()
                if name is None or k[0] == name
            ]
        return tuple(
            MetricPoint(
                name=key[0],
                kind=kind,
                attributes=key[1],
                value=value,
                count=count,
                total=total,
                minimum=minimum,
                maximum=maximum,
            )
            for key, kind, value, count, total, minimum, maximum in sorted(
                items, key=lambda t: (t[0][0], t[0][1])
            )
        )

    def value(self, name: str, /, **attributes: Any) -> float:
        """Current value of one series; ``0.0`` when nothing was recorded."""
        key = (self._checked(name), _attr_key(attributes))
        with self._lock:
            series = self._series.get(key)
        return series.value if series is not None else 0.0

    def count(self, name: str, /, **attributes: Any) -> int:
        key = (self._checked(name), _attr_key(attributes))
        with self._lock:
            series = self._series.get(key)
        return series.count if series is not None else 0

    def snapshot(self) -> dict[str, Any]:
        """JSON-safe read of the whole registry, for an audit record or the HUD."""
        return {
            "backend": self._backend,
            "names": list(self._names),
            "points": [p.to_json() for p in self.points()],
        }

    def reset(self) -> None:
        """Drop accumulated series.

        Test-support only: metrics are process-local telemetry, not the
        append-only record, so clearing them loses no attested state.
        """
        with self._lock:
            self._series.clear()


# --------------------------------------------------------------------------
# process-wide default
# --------------------------------------------------------------------------

_default_lock = threading.Lock()
_default: MetricRegistry | None = None


def default_registry() -> MetricRegistry:
    """The process-wide registry, created on first use."""
    global _default
    with _default_lock:
        if _default is None:
            _default = MetricRegistry()
        return _default


def set_default_registry(registry: MetricRegistry | None) -> None:
    """Install (or clear) the process-wide registry."""
    global _default
    with _default_lock:
        _default = registry
