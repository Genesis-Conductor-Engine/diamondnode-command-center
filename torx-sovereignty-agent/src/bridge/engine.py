"""Bridge engine: simulate, rank, apply, and roll back bridges.

The engine is the orchestrator around the card's ``bridge_engine`` and
``contextual_sovereignty`` contracts:

1. **Simulate** every strategy in card order (:class:`BridgeSimulator`).
2. **Screen** with :class:`SovereigntyGuard` (boundary violation, default-deny
   authorization).
3. **Rank** by predicted delta, then confidence.
4. **Persist proposals** as ``proposed`` BridgeActions.
5. **Apply** only when ``application_rule`` holds:
   ``predicted_delta < 0 and boundary_violation = false and confidence >=
   configured_minimum and authorized = true``.
6. **Roll back** on any ``bridge_engine.rollback.trigger_conditions``.

Degradation: the engine consults a :class:`DegradationController`; when the
``bridge-engine`` capability is shed, proposing new bridges is refused (nothing
is blocked mid-flight, but no new candidates are created).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from src.bridge.simulator import BRIDGE_STRATEGIES, BridgeSimulation, BridgeSimulator
from src.bridge.sovereignty import SovereigntyGuard
from src.model_card.loader import AgentModelCard, cached_model_card
from src.observability.metrics import MetricRegistry, default_registry
from src.observability.tracing import SpanAttributes, span
from src.persistence.repositories import (
    BridgeAction,
    BridgeRepository,
    GroupIntentRepository,
    TensionRepository,
    TensionSnapshot,
)
from src.runtime.degradation import CAP_BRIDGE_ENGINE, DegradationController

#: Card stable code — application refused because a boundary is hard.
CODE_BOUNDARY_VIOLATION = "BOUNDARY_VIOLATION"
#: Card stable code — simulated confidence below the configured minimum.
CODE_INSUFFICIENT_CONFIDENCE = "INSUFFICIENT_CONFIDENCE"
#: Card stable code — no explicit grant covers the action.
CODE_AUTHORIZATION_DENIED = "AUTHORIZATION_DENIED"
#: Card stable code — the group intent moved under the bridge.
CODE_STALE_INTENT_REVISION = "STALE_INTENT_REVISION"
#: Card stable code — a bridge that does not reduce tension cannot apply.
CODE_BRIDGE_INCREASES_TENSION = "BRIDGE_INCREASES_TENSION"

#: Default confidence floor used when the card supplies no explicit
#: ``configured_minimum`` for bridge application.
DEFAULT_MINIMUM_CONFIDENCE = 0.5

#: Upper bound on candidates per cycle from ``runtime.state_bounds``.
MAX_CANDIDATES_PER_CYCLE = 8


class BridgeRejected(ValueError):
    """The engine refuses an action, with a card stable code.

    ``code`` matches ``mcp.errors.stable_codes`` so an MCP surface maps the
    rejection 1:1 to an application/problem+json error.
    """

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


@dataclass(frozen=True, slots=True)
class RankedCandidate:
    """One simulated, sovereignty-screened candidate awaiting application."""

    simulation: BridgeSimulation
    boundary_violation: bool
    authorized: bool
    applicable: bool
    refusal_code: str | None = None
    refusal_detail: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            **self.simulation.to_json(),
            "boundary_violation": self.boundary_violation,
            "authorized": self.authorized,
            "applicable": self.applicable,
            "refusal_code": self.refusal_code,
            "refusal_detail": self.refusal_detail,
        }


class BridgeEngine:
    """Full bridge lifecycle bound to the persistence repositories."""

    def __init__(
        self,
        *,
        card: AgentModelCard | None = None,
        bridges: BridgeRepository | None = None,
        intents: GroupIntentRepository | None = None,
        tensions: TensionRepository | None = None,
        guard: SovereigntyGuard | None = None,
        simulator: BridgeSimulator | None = None,
        degradation: DegradationController | None = None,
        registry: MetricRegistry | None = None,
        minimum_confidence: float | None = None,
    ) -> None:
        self._card = card or cached_model_card()
        self._guard = guard or SovereigntyGuard(self._card)
        self._simulator = simulator or BridgeSimulator(self._card)
        self._bridges = bridges
        self._intents = intents
        self._tensions = tensions
        self._degradation = degradation
        self._registry = registry or MetricRegistry(self._card)
        self._minimum_confidence = (
            minimum_confidence
            if minimum_confidence is not None
            else DEFAULT_MINIMUM_CONFIDENCE
        )

    # -- reads ------------------------------------------------------------

    @property
    def ordered_strategies(self) -> tuple[str, ...]:
        """The card's try-order; escalation is terminal."""
        strategies = self._card.bridge_engine.ordered_strategies
        return tuple(s.id for s in strategies) if strategies else BRIDGE_STRATEGIES

    def get_bridge(self, tenant_id: Any, bridge_id: Any) -> BridgeAction | None:
        if self._bridges is None:
            return None
        return self._bridges.get(tenant_id, bridge_id)

    # -- propose ----------------------------------------------------------

    def propose(
        self,
        tenant_id: Any,
        group_id: Any,
        tension_snapshot_id: Any,
        *,
        grants: Mapping[str, Any] | None = None,
        maximum_candidates: int = 3,
        expected_group_intent_revision: int | None = None,
        registry: MetricRegistry | None = None,
    ) -> list[RankedCandidate]:
        """Simulate, screen, rank, and persist proposed bridges.

        Returns ranked candidates. Only ``applicable`` candidates are
        persisted as ``proposed`` BridgeActions; the rest are returned so the
        caller can see *why* they were refused. ``applicable`` requires the
        full application rule to already hold at propose time.
        """
        if self._degradation is not None and not self._degradation.is_active(
            CAP_BRIDGE_ENGINE
        ):
            raise BridgeRejected(
                CODE_AUTHORIZATION_DENIED,
                "bridge-engine capability is shed under degradation; no new "
                "bridges may be proposed",
            )
        if self._bridges is None:
            raise BridgeRejected(
                CODE_AUTHORIZATION_DENIED,
                "no BridgeRepository bound; proposals cannot be persisted",
            )
        if self._tensions is None:
            raise BridgeRejected(
                CODE_INSUFFICIENT_CONFIDENCE,
                "no TensionRepository bound; cannot resolve the snapshot",
            )
        if self._intents is None:
            raise BridgeRejected(
                CODE_STALE_INTENT_REVISION,
                "no GroupIntentRepository bound; cannot check intent revision",
            )

        snapshot = self._tensions.get(tenant_id, group_id, tension_snapshot_id)
        if snapshot is None:
            raise BridgeRejected(
                CODE_INSUFFICIENT_CONFIDENCE,
                f"tension snapshot {tension_snapshot_id} does not exist for "
                f"group {group_id}",
            )
        intent = self._intents.latest(tenant_id, group_id)
        if expected_group_intent_revision is not None and (
            intent is None or intent.revision != expected_group_intent_revision
        ):
            raise BridgeRejected(
                CODE_STALE_INTENT_REVISION,
                f"expected group intent revision "
                f"{expected_group_intent_revision} but found "
                f"{intent.revision if intent else None}",
            )

        simulations = self._simulator.simulate(snapshot, intent)
        candidates: list[RankedCandidate] = []
        for simulation in simulations:
            verdict = self._guard.evaluate(
                simulation.proposal, grants=grants
            )
            applicable, code, detail = self._rule(
                simulation, verdict.boundary_violation, verdict.authorized
            )
            candidates.append(
                RankedCandidate(
                    simulation=simulation,
                    boundary_violation=verdict.boundary_violation,
                    authorized=verdict.authorized,
                    applicable=applicable,
                    refusal_code=code,
                    refusal_detail=detail,
                )
            )

        ranked = sorted(
            candidates,
            key=lambda c: (
                not c.applicable,
                c.simulation.delta,
                -c.simulation.confidence,
            ),
        )

        # Persist the applicable proposals (capped per state_bounds).
        persisted = 0
        for candidate in ranked:
            if not candidate.applicable:
                continue
            if persisted >= min(maximum_candidates, MAX_CANDIDATES_PER_CYCLE):
                break
            self._bridges.propose(
                tenant_id,
                uuid.uuid4(),
                group_id=group_id,
                bridge_type=candidate.simulation.strategy_id,
                proposal=dict(candidate.simulation.proposal),
                simulation=candidate.simulation.as_simulation_document(),
                authorization={
                    "boundary_violation": False,
                    "authorized": True,
                },
                vdf_proof_id=uuid.uuid4(),
            )
            self._emit("bridge.proposed", candidate.simulation, tenant_id)
            persisted += 1

        return ranked

    # -- apply ------------------------------------------------------------

    def apply(
        self,
        tenant_id: Any,
        bridge_id: Any,
        *,
        expected_snapshot_id: Any | None = None,
        grants: Mapping[str, Any] | None = None,
    ) -> BridgeAction:
        """Apply a proposed bridge, re-verifying the application rule.

        The rule is re-evaluated at apply time — a proposal is not a contract,
        and the guard and the simulation are re-run from persisted state so a
        boundary that appeared between propose and apply still blocks.
        """
        if self._bridges is None:
            raise BridgeRejected(
                CODE_AUTHORIZATION_DENIED,
                "no BridgeRepository bound; cannot apply",
            )
        bridge = self._bridges.get(tenant_id, bridge_id)
        if bridge is None:
            raise BridgeRejected(
                CODE_AUTHORIZATION_DENIED,
                f"bridge {bridge_id} does not exist in tenant {tenant_id}",
            )
        if bridge.status != "proposed":
            raise BridgeRejected(
                CODE_STALE_INTENT_REVISION,
                f"bridge {bridge_id} is {bridge.status!r}, not 'proposed'",
            )

        if self._tensions is not None and expected_snapshot_id is not None:
            latest = self._tensions.latest(tenant_id, bridge.group_id)
            if latest is None or str(latest.snapshot_id) != str(
                expected_snapshot_id
            ):
                raise BridgeRejected(
                    CODE_STALE_INTENT_REVISION,
                    f"expected tension snapshot {expected_snapshot_id} but "
                    f"found {latest.snapshot_id if latest else None}",
                )

        verdict = self._guard.evaluate(bridge.proposal, grants=grants)
        simulation = _simulation_from_action(bridge)
        applicable, code, detail = self._rule(
            simulation, verdict.boundary_violation, verdict.authorized
        )
        if not applicable:
            raise BridgeRejected(code or CODE_AUTHORIZATION_DENIED, detail or "refused")

        applied = self._bridges.mark_applied(tenant_id, bridge_id)
        self._emit("bridge.applied", simulation, tenant_id, bridge_type=bridge.bridge_type)
        return applied

    # -- rollback ---------------------------------------------------------

    def check_rollback_triggers(
        self,
        tenant_id: Any,
        bridge_id: Any,
        *,
        current_snapshot: TensionSnapshot | None = None,
        hysteresis_margin: float = 0.05,
    ) -> list[dict[str, Any]]:
        """Evaluate ``bridge_engine.rollback.trigger_conditions``.

        Returns a list of triggered conditions, each with the condition name
        and the evidence. An empty list means the bridge stands.
        """
        if self._bridges is None:
            return []
        bridge = self._bridges.get(tenant_id, bridge_id)
        if bridge is None or bridge.status != "applied":
            return []
        triggered: list[dict[str, Any]] = []

        simulation = _simulation_from_action(bridge)
        if current_snapshot is not None:
            observed = current_snapshot.group_tension
            predicted_after = simulation.predicted_after
            if observed > predicted_after + hysteresis_margin:
                triggered.append(
                    {
                        "condition": "observed tension increases beyond hysteresis margin",
                        "evidence": {
                            "observed": observed,
                            "predicted_after": predicted_after,
                            "hysteresis_margin": hysteresis_margin,
                        },
                    }
                )
            # A new hard boundary (per participant gradients) is a rollback
            # trigger as much as a propose-time block.
            for gradient in current_snapshot.member_gradients.values():
                boundary = gradient.get("boundary_violation", False)
                if bool(boundary):
                    triggered.append(
                        {
                            "condition": "new hard boundary appears",
                            "evidence": {"member_gradient_boundary": True},
                        }
                    )
                    break

        proposal = dict(bridge.proposal)
        if proposal.get("authorization_revoked"):
            triggered.append(
                {
                    "condition": "authorization is revoked",
                    "evidence": {"authorization_revoked": True},
                }
            )
        if proposal.get("evidence_contradicted"):
            triggered.append(
                {
                    "condition": "bridge evidence is contradicted",
                    "evidence": {"evidence_contradicted": True},
                }
            )
        return triggered

    def rollback(
        self,
        tenant_id: Any,
        bridge_id: Any,
        *,
        reason: str,
        current_snapshot: TensionSnapshot | None = None,
        hysteresis_margin: float = 0.05,
        grants: Mapping[str, Any] | None = None,
    ) -> BridgeAction:
        """Roll back an applied bridge after a trigger condition fires.

        The rollback is itself an applied action (via
        :meth:`BridgeRepository.rollback`), so it carries its own proposal,
        simulation, and authorization — the undo is as inspectable as the
        original apply.
        """
        if self._bridges is None:
            raise BridgeRejected(
                CODE_AUTHORIZATION_DENIED,
                "no BridgeRepository bound; cannot roll back",
            )
        bridge = self._bridges.get(tenant_id, bridge_id)
        if bridge is None:
            raise BridgeRejected(
                CODE_AUTHORIZATION_DENIED,
                f"bridge {bridge_id} does not exist in tenant {tenant_id}",
            )
        if bridge.status != "applied":
            raise BridgeRejected(
                CODE_STALE_INTENT_REVISION,
                f"bridge {bridge_id} is {bridge.status!r}; only an applied "
                "bridge can be rolled back",
            )

        triggered = self.check_rollback_triggers(
            tenant_id,
            bridge_id,
            current_snapshot=current_snapshot,
            hysteresis_margin=hysteresis_margin,
        )
        if not triggered:
            raise BridgeRejected(
                CODE_STALE_INTENT_REVISION,
                "no rollback trigger condition is satisfied; the bridge stands",
            )

        undo_proposal = {
            "undo_of": str(bridge.bridge_id),
            "strategy_id": bridge.bridge_type,
            "reason": reason,
            "triggers": triggered,
        }
        simulation = _simulation_from_action(bridge)
        undo = self._bridges.rollback(
            tenant_id,
            bridge_id,
            new_bridge_id=uuid.uuid4(),
            proposal=undo_proposal,
            simulation={
                "predicted_before": simulation.predicted_after,
                "predicted_after": simulation.predicted_before,
                "predicted_delta": simulation.predicted_before
                - simulation.predicted_after,
                "confidence": simulation.confidence,
                "counterfactual": dict(simulation.counterfactual),
            },
            authorization={
                "boundary_violation": False,
                "authorized": True,
                "reason": reason,
            },
            vdf_proof_id=uuid.uuid4(),
        )
        self._emit("bridge.rolled_back", simulation, tenant_id, bridge_type=bridge.bridge_type)
        return undo

    # -- rule + metrics ---------------------------------------------------

    def _rule(
        self,
        simulation: BridgeSimulation,
        boundary_violation: bool,
        authorized: bool,
    ) -> tuple[bool, str | None, str | None]:
        """Evaluate ``bridge_engine.application_rule``.

        ``apply_bridge(b) iff predicted_delta_tension(b) < 0 and
        boundary_violation(b) = false and confidence(b) >= configured_minimum
        and authorized(b) = true``. Escalation is exempt from the delta test:
        it is the terminal surface-and-preserve strategy and must be allowed to
        be *seen* even though it never reduces tension.
        """
        if boundary_violation:
            self._registry.counter("gc_boundary_violation_block_total").add(1)
            return (
                False,
                CODE_BOUNDARY_VIOLATION,
                "a hard boundary blocks this bridge",
            )
        if not authorized:
            return (
                False,
                CODE_AUTHORIZATION_DENIED,
                "no explicit grant covers this action (default-deny)",
            )
        if simulation.confidence < self._minimum_confidence:
            return (
                False,
                CODE_INSUFFICIENT_CONFIDENCE,
                f"confidence {simulation.confidence:.3f} below configured "
                f"minimum {self._minimum_confidence:.3f}",
            )
        if simulation.strategy_id != "explicit-escalation" and simulation.delta >= 0:
            return (
                False,
                CODE_BRIDGE_INCREASES_TENSION,
                f"predicted delta {simulation.delta:+.3f} is not negative; "
                "the bridge does not reduce tension",
            )
        return True, None, None

    def _emit(
        self,
        event: str,
        simulation: BridgeSimulation,
        tenant_id: Any,
        *,
        bridge_type: str | None = None,
    ) -> None:
        """Audit event + the matching metric/span side effects.

        The tracing contract fixes span attribute keys; bridge events carry
        ``bridge.type`` (and tenant) so an exported trace links the audit
        record to the simulation that produced it.
        """
        kind = bridge_type or simulation.strategy_id
        self._registry.histogram(
            "gc_bridge_predicted_delta"
            if event != "bridge.rolled_back"
            else "gc_bridge_observed_delta"
        ).record(simulation.delta)
        with span(
            f"gc.bridge.{event}",
            attrs=SpanAttributes(
                tenant_id=str(tenant_id), bridge_type=kind
            ),
            registry=self._registry,
            card=self._card,
        ):
            pass


def _simulation_from_action(action: BridgeAction) -> BridgeSimulation:
    """Rebuild a simulation from a persisted BridgeAction.

    The persisted simulation document is authoritative (it was written at
    propose time); the wrapper just restores the fields the rule needs.
    """
    sim = dict(action.simulation)
    return BridgeSimulation(
        strategy_id=action.bridge_type,
        proposal=dict(action.proposal),
        predicted_before=float(sim.get("predicted_before", 0.0)),
        predicted_after=float(sim.get("predicted_after", 0.0)),
        confidence=float(sim.get("confidence", 0.0)),
        participants=tuple(),
        counterfactual=dict(sim.get("counterfactual", {})),
    )
