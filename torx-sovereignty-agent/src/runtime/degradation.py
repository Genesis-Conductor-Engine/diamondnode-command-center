"""Bounded fallback controller for ``runtime.degradation`` and the topology chain.

Two mechanisms live here, because the card ties them together.

**The ladder.** ``runtime.degradation.order`` is a *sequence*, not a set of
options: each rung sheds strictly more than the one before it, so the operator
reading rung 3 knows rungs 1 and 2 are also in force. The order is read from the
loaded card and never restated here — a card edit that inserts a rung must change
behaviour, not just documentation. Two properties make the ladder trustworthy
under load:

* *Monotone.* One observation of pressure advances at most one rung. Skipping
  straight to the bottom would shed capability the system did not need to shed,
  and the audit record would no longer show which rung actually restored health.
* *No flapping.* Recovery climbs back one rung at a time and only after a
  stability dwell — a consecutive-healthy count **and** a minimum time at the
  current rung. Degradation is cheap to enter and deliberately expensive to
  leave, because a controller that recovers on the first good sample oscillates
  exactly when the system is least able to absorb it.

**The protected set.** ``runtime.degradation.never_disable`` names
``authorization-filter``, ``hard-boundary-enforcement`` and
``dissent-preservation``. Those are not "not currently scheduled for disabling" —
they must be impossible to disable. So there is no code path that can clear
them: :meth:`DegradationRung.is_active` answers ``True`` for a protected control
*before* it consults anything else, :class:`DegradationRung` refuses to construct
if a rung's disabled set intersects the protected set, and the controller
validates its whole effect table against the protected set at construction. A
rung that switched off the authorization filter cannot be built, let alone
reached.

**The topology chain.** ``torx.topology_sidecar.critical_path`` is false and
``runtime.critical_path.excluded`` lists the sidecar, so a sidecar failure or
timeout must not reach the decision path. :class:`TopologyFallback` walks
``torx.topology_sidecar.fallback_order`` — last-valid descriptor, then a neutral
descriptor, then the non-topological TORX baseline — and always returns a usable
:class:`~src.torx_layer.state.PDit`. :func:`guarded_decision` shows the whole
contract in one call: the sidecar may hang, raise, or be switched off, and the
caller still receives a decision.

The sidecar probe runs on a *daemon* thread when a timeout is given. Abandoning a
hung worker is acceptable precisely because the card puts this work off the
critical path; a pooled worker would instead join at interpreter exit and let a
wedged sidecar hold the process open.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from types import EllipsisType
from typing import Any, Callable, Mapping, Sequence

from src.model_card.loader import cached_model_card
from src.model_card.types import AgentModelCard
from src.observability.metrics import MetricRegistry, default_registry
from src.torx_layer.circuits import DecisionInputs, DecisionMarginals, evaluate_decision
from src.torx_layer.state import STORAGE_PRECISION, PDit

# --------------------------------------------------------------------------
# capabilities the ladder may shed
# --------------------------------------------------------------------------

#: Every control a degradation step is allowed to switch off. Naming them
#: explicitly is what lets the controller reject a card step it does not
#: understand: an unknown step that quietly shed nothing would look like a
#: working ladder while providing no relief at all.
CAP_TOPOLOGY_RECOMPUTE = "topology-recompute"
CAP_TOPOLOGY_SIDECAR = "topology-sidecar"
CAP_TOPOLOGY_DESCRIPTOR_INPUT = "topology-descriptor-input"
CAP_AUTOMATIC_PROFILE_INFERENCE = "automatic-profile-inference"
CAP_PERSONALIZATION_LAYERS = "personalization-layer-application"
CAP_MODULAR_DIFFERENTIATION = "modular-differentiation"
CAP_BRIDGE_ENGINE = "bridge-engine"

SHEDDABLE_CAPABILITIES: tuple[str, ...] = (
    CAP_TOPOLOGY_RECOMPUTE,
    CAP_TOPOLOGY_SIDECAR,
    CAP_TOPOLOGY_DESCRIPTOR_INPUT,
    CAP_AUTOMATIC_PROFILE_INFERENCE,
    CAP_PERSONALIZATION_LAYERS,
    CAP_MODULAR_DIFFERENTIATION,
    CAP_BRIDGE_ENGINE,
)


@dataclass(frozen=True, slots=True)
class StepEffect:
    """What one card degradation step actually does.

    ``disables`` is cumulative by position, not by this record: rung *n* holds
    the union of the effects of steps 1..n.
    """

    disables: tuple[str, ...]
    refresh_multiplier: float
    summary: str

    def __post_init__(self) -> None:
        unknown = [c for c in self.disables if c not in SHEDDABLE_CAPABILITIES]
        if unknown:
            raise ValueError(
                f"step effect names capabilities outside SHEDDABLE_CAPABILITIES: "
                f"{unknown}"
            )
        if self.refresh_multiplier < 1.0:
            raise ValueError(
                "refresh_multiplier must be >= 1.0 (degrading may only slow the "
                f"sidecar down), got {self.refresh_multiplier}"
            )


#: Semantics of each step the card may list. The card owns the *order*; this
#: table owns what each named step means. A step in the card with no entry here
#: is a construction-time error, so the ladder can never contain a rung that
#: silently does nothing.
#:
#: The 4x refresh backoff on the first rung is chosen so the sidecar's
#: ``update_interval_ms`` default of 1000 becomes 4s — still inside the 20s
#: rolling window the card declares, so the descriptor keeps covering the same
#: evidence at a quarter of the cost.
STEP_EFFECTS: Mapping[str, StepEffect] = {
    "reduce-topology-refresh-frequency": StepEffect(
        disables=(),
        refresh_multiplier=4.0,
        summary="sidecar still recomputes, at a quarter of the rate",
    ),
    "use-last-valid-topology-descriptor": StepEffect(
        disables=(CAP_TOPOLOGY_RECOMPUTE,),
        refresh_multiplier=1.0,
        summary="serve the cached descriptor; stop recomputing",
    ),
    "disable-topology-sidecar": StepEffect(
        disables=(CAP_TOPOLOGY_SIDECAR,),
        refresh_multiplier=1.0,
        summary="stop consuming sidecar output; descriptor becomes neutral",
    ),
    "freeze-profile-learning": StepEffect(
        disables=(CAP_AUTOMATIC_PROFILE_INFERENCE,),
        refresh_multiplier=1.0,
        summary=(
            "no automatic profile inference; explicit user boundaries and "
            "corrections still apply"
        ),
    ),
    # Shedding the bridge engine looks aggressive but points the right way: with
    # no bridge proposals an unreconciled disagreement surfaces as unresolved,
    # which is what no_false_consensus requires. The failure mode of keeping a
    # degraded bridge engine is a manufactured agreement.
    "retain-base-agent-plus-hard-boundaries": StepEffect(
        disables=(
            CAP_TOPOLOGY_DESCRIPTOR_INPUT,
            CAP_PERSONALIZATION_LAYERS,
            CAP_MODULAR_DIFFERENTIATION,
            CAP_BRIDGE_ENGINE,
        ),
        refresh_multiplier=1.0,
        summary=(
            "immutable base agent plus hard boundaries only; conflicts surface "
            "as unresolved rather than bridged"
        ),
    ),
}


# --------------------------------------------------------------------------
# rungs
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DegradationRung:
    """One position on the ladder, including nominal (``index == 0``).

    The class refuses to exist in a state that violates ``never_disable``; that
    is the structural guarantee, not a runtime check the caller may skip.
    """

    index: int
    step: str
    summary: str
    disabled: tuple[str, ...]
    protected: tuple[str, ...]
    refresh_interval_ms: float | None
    reason: str = ""

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError(f"rung index must be >= 0, got {self.index}")
        if not self.protected:
            raise ValueError(
                "a rung with an empty protected set cannot enforce "
                "runtime.degradation.never_disable"
            )
        violated = sorted(set(self.disabled) & set(self.protected))
        if violated:
            raise ValueError(
                f"degradation rung {self.index} ({self.step!r}) would disable "
                f"never_disable controls {violated}; these must remain active at "
                "every rung including the last"
            )
        if self.index == 0 and self.step:
            raise ValueError("rung 0 is nominal and must have an empty step name")
        if self.index > 0 and not self.step:
            raise ValueError(f"rung {self.index} must name a card degradation step")
        if self.refresh_interval_ms is not None and self.refresh_interval_ms <= 0:
            raise ValueError(
                "refresh_interval_ms must be positive or None (no refresh), got "
                f"{self.refresh_interval_ms}"
            )

    @property
    def is_nominal(self) -> bool:
        return self.index == 0

    def is_active(self, control: str) -> bool:
        """Is ``control`` still running at this rung?

        Protected controls are answered first and unconditionally. There is no
        branch below this line that can return ``False`` for one of them.
        """
        if control in self.protected:
            return True
        if control in self.disabled:
            return False
        if control in SHEDDABLE_CAPABILITIES:
            return True
        raise ValueError(
            f"unknown control {control!r}; known controls are "
            f"{sorted(set(SHEDDABLE_CAPABILITIES) | set(self.protected))}"
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "step": self.step,
            "summary": self.summary,
            "disabled": list(self.disabled),
            "protected": list(self.protected),
            "refresh_interval_ms": self.refresh_interval_ms,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class DegradationTransition:
    """One recorded ladder move.

    In-memory telemetry, not the attested record: the durable, append-only
    history is the audit event stream. The deque is bounded so a long-lived
    worker cannot grow it without limit.
    """

    at: float
    from_index: int
    to_index: int
    direction: str
    source: str
    detail: str

    def to_json(self) -> dict[str, Any]:
        return {
            "at": round(self.at, STORAGE_PRECISION),
            "from_index": self.from_index,
            "to_index": self.to_index,
            "direction": self.direction,
            "source": self.source,
            "detail": self.detail,
        }


HISTORY_LIMIT = 512


# --------------------------------------------------------------------------
# controller
# --------------------------------------------------------------------------


class DegradationController:
    """Walks ``runtime.degradation.order`` under pressure and back out again."""

    def __init__(
        self,
        card: AgentModelCard | None = None,
        *,
        stability_dwell: int = 3,
        min_dwell_seconds: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
        history_limit: int = HISTORY_LIMIT,
    ) -> None:
        card = card if card is not None else cached_model_card()
        degradation = card.runtime.degradation

        self._order: tuple[str, ...] = tuple(degradation.order)
        self._protected: tuple[str, ...] = tuple(degradation.never_disable)
        if not self._order:
            raise ValueError("runtime.degradation.order is empty: there is no ladder")
        if not self._protected:
            raise ValueError(
                "runtime.degradation.never_disable is empty: nothing would be "
                "protected from the ladder"
            )

        unknown = [s for s in self._order if s not in STEP_EFFECTS]
        if unknown:
            raise ValueError(
                f"runtime.degradation.order names steps with no implementation: "
                f"{unknown}. Add a StepEffect for each before deploying — an "
                "unimplemented rung sheds nothing and gives no relief."
            )
        # The structural half of never_disable, checked once for the whole table
        # rather than per rung: no effect anywhere may name a protected control.
        for step in self._order:
            overlap = sorted(set(STEP_EFFECTS[step].disables) & set(self._protected))
            if overlap:
                raise ValueError(
                    f"degradation step {step!r} would disable {overlap}, which "
                    "runtime.degradation.never_disable forbids"
                )

        self._base_refresh_ms = float(
            card.torx.topology_sidecar.defaults.update_interval_ms
        )
        if stability_dwell < 1:
            raise ValueError(f"stability_dwell must be >= 1, got {stability_dwell}")
        if min_dwell_seconds < 0:
            raise ValueError(
                f"min_dwell_seconds must be >= 0, got {min_dwell_seconds}"
            )
        self._stability_dwell = int(stability_dwell)
        self._min_dwell_seconds = float(min_dwell_seconds)
        self._clock = clock

        self._lock = threading.Lock()
        self._index = 0
        self._healthy_streak = 0
        self._entered_at = clock()
        self._reason = ""
        self._history: deque[DegradationTransition] = deque(maxlen=history_limit)
        self._rungs = tuple(self._build_rung(i) for i in range(len(self._order) + 1))

    # -- construction --------------------------------------------------------

    def _build_rung(self, index: int, reason: str = "") -> DegradationRung:
        applied = self._order[:index]
        disabled: list[str] = []
        multiplier = 1.0
        for step in applied:
            effect = STEP_EFFECTS[step]
            multiplier = max(multiplier, effect.refresh_multiplier)
            for cap in effect.disables:
                if cap not in disabled:
                    disabled.append(cap)
        refresh: float | None = round(
            self._base_refresh_ms * multiplier, STORAGE_PRECISION
        )
        if CAP_TOPOLOGY_RECOMPUTE in disabled:
            refresh = None
        return DegradationRung(
            index=index,
            step=self._order[index - 1] if index else "",
            summary=(
                STEP_EFFECTS[self._order[index - 1]].summary
                if index
                else "nominal: no capability shed"
            ),
            disabled=tuple(disabled),
            protected=self._protected,
            refresh_interval_ms=refresh,
            reason=reason,
        )

    # -- introspection -------------------------------------------------------

    @property
    def order(self) -> tuple[str, ...]:
        """The card's ladder, in card order."""
        return self._order

    @property
    def protected(self) -> tuple[str, ...]:
        return self._protected

    @property
    def max_level(self) -> int:
        return len(self._order)

    @property
    def level(self) -> int:
        with self._lock:
            return self._index

    @property
    def rung(self) -> DegradationRung:
        with self._lock:
            index, reason = self._index, self._reason
        base = self._rungs[index]
        return base if base.reason == reason else _with_reason(base, reason)

    def rung_at(self, index: int) -> DegradationRung:
        """The rung the ladder would occupy at ``index``, without moving."""
        if not 0 <= index <= self.max_level:
            raise ValueError(
                f"rung index must be in [0, {self.max_level}], got {index}"
            )
        return self._rungs[index]

    def is_active(self, control: str) -> bool:
        return self.rung.is_active(control)

    @property
    def history(self) -> tuple[DegradationTransition, ...]:
        with self._lock:
            return tuple(self._history)

    # -- the ladder ----------------------------------------------------------

    def observe(
        self,
        under_pressure: bool,
        *,
        source: str = "",
        detail: str = "",
    ) -> DegradationRung:
        """Feed one health sample and return the resulting rung.

        Pressure escalates by exactly one rung per observation. Health recovers
        by one rung only once the dwell is satisfied, and any single pressure
        sample resets the healthy streak — so a flapping dependency parks the
        ladder rather than oscillating it.
        """
        now = self._clock()
        with self._lock:
            previous = self._index
            if under_pressure:
                self._healthy_streak = 0
                if self._index < self.max_level:
                    self._index += 1
                    self._entered_at = now
                    self._reason = detail or source or "pressure"
                    direction = "escalate"
                else:
                    direction = "hold"
            else:
                self._healthy_streak += 1
                dwell_elapsed = (now - self._entered_at) >= self._min_dwell_seconds
                if (
                    self._index > 0
                    and self._healthy_streak >= self._stability_dwell
                    and dwell_elapsed
                ):
                    self._index -= 1
                    self._entered_at = now
                    self._healthy_streak = 0
                    self._reason = detail or source or "recovered"
                    direction = "recover"
                else:
                    direction = "hold"
            if direction != "hold":
                self._history.append(
                    DegradationTransition(
                        at=now,
                        from_index=previous,
                        to_index=self._index,
                        direction=direction,
                        source=source,
                        detail=detail,
                    )
                )
            index, reason = self._index, self._reason
        return _with_reason(self._rungs[index], reason)

    def reset(self, *, source: str = "reset") -> DegradationRung:
        """Return to nominal in one move.

        Deliberately explicit and separate from :meth:`observe`: an operator
        clearing an incident is not the same evidence as a healthy sample, and
        conflating them would let a health probe skip the dwell.
        """
        now = self._clock()
        with self._lock:
            previous = self._index
            if previous:
                self._history.append(
                    DegradationTransition(
                        at=now,
                        from_index=previous,
                        to_index=0,
                        direction="reset",
                        source=source,
                        detail="",
                    )
                )
            self._index = 0
            self._healthy_streak = 0
            self._entered_at = now
            self._reason = ""
        return self._rungs[0]

    # -- topology coupling ---------------------------------------------------

    @property
    def topology_stage_floor(self) -> str:
        """First :data:`TOPOLOGY_STAGES` entry the resolver may return.

        Derived from the rung's shed capabilities rather than from its index, so
        it stays correct if the card reorders the ladder.
        """
        rung = self.rung
        if rung.is_active(CAP_TOPOLOGY_RECOMPUTE):
            return ""
        if rung.is_active(CAP_TOPOLOGY_SIDECAR):
            return STAGE_LAST_VALID
        if rung.is_active(CAP_TOPOLOGY_DESCRIPTOR_INPUT):
            return STAGE_NEUTRAL
        return STAGE_BASELINE

    def to_json(self) -> dict[str, Any]:
        rung = self.rung
        with self._lock:
            streak, entered = self._healthy_streak, self._entered_at
        return {
            "order": list(self._order),
            "level": rung.index,
            "max_level": self.max_level,
            "rung": rung.to_json(),
            "healthy_streak": streak,
            "stability_dwell": self._stability_dwell,
            "min_dwell_seconds": self._min_dwell_seconds,
            "entered_at": round(entered, STORAGE_PRECISION),
            "topology_stage_floor": self.topology_stage_floor,
            "history": [t.to_json() for t in self.history],
        }


def _with_reason(rung: DegradationRung, reason: str) -> DegradationRung:
    if rung.reason == reason:
        return rung
    return DegradationRung(
        index=rung.index,
        step=rung.step,
        summary=rung.summary,
        disabled=rung.disabled,
        protected=rung.protected,
        refresh_interval_ms=rung.refresh_interval_ms,
        reason=reason,
    )


# --------------------------------------------------------------------------
# topology descriptor fallback chain
# --------------------------------------------------------------------------

#: The card declares ``topology_descriptor`` as a pdit but does not enumerate its
#: outcomes, because the enumeration belongs to whoever must be able to build one
#: *without* the sidecar — that is this module. With
#: ``maximum_homology_dimension: 1`` the sidecar sees H0 (connected components)
#: and H1 (loops), so the bounded readings are: one component and no loop, more
#: than one component, at least one loop, and the outcome that asserts nothing.
TOPOLOGY_DESCRIPTOR_STATES: tuple[str, ...] = (
    "connected",
    "fragmented",
    "cyclic",
    "indeterminate",
)

#: Stage names. ``STAGE_FRESH`` is not part of the card's ``fallback_order`` —
#: it is the non-degraded case the chain exists to replace.
STAGE_FRESH = "sidecar-descriptor"
STAGE_LAST_VALID = "last-valid-descriptor"
STAGE_NEUTRAL = "neutral-descriptor"
STAGE_BASELINE = "non-topological-torx-baseline"

TOPOLOGY_STAGES: tuple[str, ...] = (STAGE_LAST_VALID, STAGE_NEUTRAL, STAGE_BASELINE)


def neutral_descriptor(
    outcomes: Sequence[str] = TOPOLOGY_DESCRIPTOR_STATES,
    label: str = "topology_descriptor",
) -> PDit:
    """A descriptor that provably carries no topological claim.

    Uniform, so ``PDit.margin`` is 0. The card's descriptor-invariance contract
    is ``C_F * epsilon < Delta_descriptor / 2``; at margin 0 no perturbation
    bound satisfies it, so a downstream consumer that checks the margin — as the
    stability result requires — cannot mistake this for a measurement.
    """
    return PDit.uniform(tuple(outcomes), label)


@dataclass(frozen=True, slots=True)
class TopologyResolution:
    """What the decision path got, and how degraded it was to get it."""

    descriptor: PDit
    stage: str
    informative: bool
    use_topology_edge: bool
    sidecar_attempted: bool
    sidecar_ok: bool
    error: str | None
    elapsed_s: float

    @property
    def degraded(self) -> bool:
        return self.stage != STAGE_FRESH

    def to_json(self) -> dict[str, Any]:
        return {
            "descriptor": self.descriptor.to_json(),
            "stage": self.stage,
            "informative": self.informative,
            "use_topology_edge": self.use_topology_edge,
            "sidecar_attempted": self.sidecar_attempted,
            "sidecar_ok": self.sidecar_ok,
            "degraded": self.degraded,
            "error": self.error,
            "elapsed_s": round(self.elapsed_s, STORAGE_PRECISION),
        }


class SidecarTimeout(TimeoutError):
    """The sidecar probe exceeded its budget and was abandoned."""


def _call_with_timeout(fn: Callable[[], Any], timeout_s: float | None) -> Any:
    """Run ``fn`` with a wall-clock budget, on a daemon thread when bounded.

    A hung probe is abandoned rather than joined. That is safe only because the
    card keeps this work off the critical path; the thread is a daemon so a
    wedged sidecar cannot hold the interpreter open at exit.
    """
    if timeout_s is None:
        return fn()
    if timeout_s <= 0:
        raise ValueError(f"timeout_s must be positive or None, got {timeout_s}")
    box: dict[str, Any] = {}

    def _run() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - relayed to the caller below
            box["error"] = exc

    worker = threading.Thread(target=_run, name="torx-topology-sidecar", daemon=True)
    worker.start()
    worker.join(timeout_s)
    if worker.is_alive():
        raise SidecarTimeout(f"topology sidecar exceeded {timeout_s}s budget")
    if "error" in box:
        raise box["error"]
    return box["value"]


class TopologyFallback:
    """Walks ``torx.topology_sidecar.fallback_order`` and always yields a PDit."""

    def __init__(
        self,
        card: AgentModelCard | None = None,
        *,
        outcomes: Sequence[str] = TOPOLOGY_DESCRIPTOR_STATES,
        timeout_s: float | None = 0.25,
        registry: MetricRegistry | None = None,
    ) -> None:
        card = card if card is not None else cached_model_card()
        self._order: tuple[str, ...] = tuple(
            card.torx.topology_sidecar.fallback_order
        )
        unknown = [s for s in self._order if s not in TOPOLOGY_STAGES]
        if unknown:
            raise ValueError(
                f"torx.topology_sidecar.fallback_order names unimplemented "
                f"stages {unknown}; known stages are {list(TOPOLOGY_STAGES)}"
            )
        if STAGE_BASELINE not in self._order:
            raise ValueError(
                "fallback_order must end in a non-topological baseline, "
                f"got {list(self._order)}"
            )
        self._outcomes = tuple(outcomes)
        if len(set(self._outcomes)) != len(self._outcomes) or not self._outcomes:
            raise ValueError(
                f"topology descriptor outcomes must be unique and non-empty, got "
                f"{list(self._outcomes)}"
            )
        self._enabled = bool(card.torx.topology_sidecar.enabled)
        self._timeout_s = timeout_s
        self._registry = registry
        self._lock = threading.Lock()
        self._last_valid: PDit | None = None

    # -- cache ---------------------------------------------------------------

    @property
    def fallback_order(self) -> tuple[str, ...]:
        return self._order

    @property
    def last_valid(self) -> PDit | None:
        with self._lock:
            return self._last_valid

    def remember(self, descriptor: PDit) -> PDit:
        """Validate and cache a descriptor produced by the sidecar."""
        checked = self._checked(descriptor)
        with self._lock:
            self._last_valid = checked
        return checked

    def _checked(self, descriptor: Any) -> PDit:
        if not isinstance(descriptor, PDit):
            raise ValueError(
                "topology sidecar must return a PDit over "
                f"{list(self._outcomes)}, got {type(descriptor).__name__}"
            )
        if tuple(descriptor.outcomes) != self._outcomes:
            raise ValueError(
                f"topology descriptor outcomes {list(descriptor.outcomes)} do not "
                f"match the configured {list(self._outcomes)}; a re-labelled "
                "descriptor would silently re-map stored snapshots"
            )
        return descriptor

    # -- the chain -----------------------------------------------------------

    def resolve(
        self,
        fetch: Callable[[], PDit] | None = None,
        *,
        floor: str = "",
        timeout_s: float | None | EllipsisType = ...,
    ) -> TopologyResolution:
        """Return a usable descriptor. Never raises for a sidecar failure.

        ``floor`` names the first fallback stage the caller is allowed to reach —
        normally :attr:`DegradationController.topology_stage_floor`. An empty
        floor permits a fresh probe.
        """
        budget = self._timeout_s if isinstance(timeout_s, EllipsisType) else timeout_s
        if floor and floor not in TOPOLOGY_STAGES:
            raise ValueError(
                f"unknown topology stage floor {floor!r}; expected one of "
                f"{list(TOPOLOGY_STAGES)} or '' for no floor"
            )

        started = time.perf_counter()
        attempted = False
        error: str | None = None
        may_probe = self._enabled and not floor and fetch is not None
        if may_probe:
            attempted = True
            try:
                descriptor = self._checked(_call_with_timeout(fetch, budget))
                with self._lock:
                    self._last_valid = descriptor
                return self._finish(
                    descriptor,
                    STAGE_FRESH,
                    informative=True,
                    use_edge=True,
                    attempted=True,
                    ok=True,
                    error=None,
                    started=started,
                )
            except BaseException as exc:  # noqa: BLE001 - the whole point: absorb it
                # A sidecar failure of any kind, including a timeout, must not
                # reach the decision path. It is recorded, not raised.
                error = f"{type(exc).__name__}: {exc}"

        for stage in self._order:
            if floor and TOPOLOGY_STAGES.index(stage) < TOPOLOGY_STAGES.index(floor):
                continue
            if stage == STAGE_LAST_VALID:
                cached = self.last_valid
                if cached is None:
                    continue
                return self._finish(
                    cached, stage, True, True, attempted, False, error, started
                )
            if stage == STAGE_NEUTRAL:
                return self._finish(
                    neutral_descriptor(self._outcomes),
                    stage,
                    False,
                    True,
                    attempted,
                    False,
                    error,
                    started,
                )
            # STAGE_BASELINE: TORX runs with no topology parent at all. A neutral
            # descriptor is still returned so a consumer that ignores
            # ``use_topology_edge`` degrades to the neutral case rather than
            # crashing on a missing input.
            return self._finish(
                neutral_descriptor(self._outcomes),
                stage,
                False,
                False,
                attempted,
                False,
                error,
                started,
            )

        # Unreachable while fallback_order contains the baseline, which the
        # constructor requires — but a return here beats an implicit None.
        return self._finish(
            neutral_descriptor(self._outcomes),
            STAGE_BASELINE,
            False,
            False,
            attempted,
            False,
            error,
            started,
        )

    def _finish(
        self,
        descriptor: PDit,
        stage: str,
        informative: bool,
        use_edge: bool,
        attempted: bool,
        ok: bool,
        error: str | None,
        started: float,
    ) -> TopologyResolution:
        elapsed = time.perf_counter() - started
        registry = self._registry if self._registry is not None else default_registry()
        registry.histogram("gc_topology_sidecar_seconds").record(elapsed, stage=stage)
        return TopologyResolution(
            descriptor=descriptor,
            stage=stage,
            informative=informative,
            use_topology_edge=use_edge,
            sidecar_attempted=attempted,
            sidecar_ok=ok,
            error=error,
            elapsed_s=elapsed,
        )


# --------------------------------------------------------------------------
# the two together
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GuardedDecision:
    """A decision plus the topology and ladder state it was made under."""

    decision: DecisionMarginals
    topology: TopologyResolution
    rung: DegradationRung

    def to_json(self) -> dict[str, Any]:
        return {
            "decision": {
                "backend": self.decision.backend,
                "bridge_viability": self.decision.bridge_viability.to_json(),
                "boundary_violation": self.decision.boundary_violation.to_json(),
                "action_authorized": self.decision.action_authorized.to_json(),
                "resolution_status": self.decision.resolution_status.to_json(),
            },
            "topology": self.topology.to_json(),
            "rung": self.rung.to_json(),
        }


def guarded_decision(
    inputs: DecisionInputs,
    *,
    fallback: TopologyFallback,
    controller: DegradationController | None = None,
    fetch: Callable[[], PDit] | None = None,
    timeout_s: float | None | EllipsisType = ...,
    registry: MetricRegistry | None = None,
) -> GuardedDecision:
    """Resolve topology, then decide. The sidecar cannot break either.

    When a ``controller`` is supplied it is fed pressure **only from an actual
    probe** — a resolution that returned the cached or neutral descriptor without
    attempting a probe is not evidence about the sidecar's health, and treating
    it as pressure would ratchet the ladder to the bottom and never let it back.
    Recovery from a rung at or below ``use-last-valid-topology-descriptor``
    therefore comes from an explicit health probe the caller schedules, which is
    also the only thing that can honestly clear the incident.
    """
    floor = controller.topology_stage_floor if controller is not None else ""
    resolution = fallback.resolve(fetch, floor=floor, timeout_s=timeout_s)
    if controller is not None and resolution.sidecar_attempted:
        controller.observe(
            not resolution.sidecar_ok,
            source="topology-sidecar",
            detail=resolution.error or "",
        )
    rung = controller.rung if controller is not None else _nominal_rung()

    reg = registry if registry is not None else default_registry()
    with reg.timed("gc_torx_update_seconds", stage=resolution.stage):
        decision = evaluate_decision(inputs)
    return GuardedDecision(decision=decision, topology=resolution, rung=rung)


def _nominal_rung(card: AgentModelCard | None = None) -> DegradationRung:
    """Rung 0 for callers that run without a controller."""
    card = card if card is not None else cached_model_card()
    return DegradationRung(
        index=0,
        step="",
        summary="nominal: no capability shed",
        disabled=(),
        protected=tuple(card.runtime.degradation.never_disable),
        refresh_interval_ms=float(
            card.torx.topology_sidecar.defaults.update_interval_ms
        ),
    )
