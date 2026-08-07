"""Contextual sovereignty guard for the bridge engine.

``contextual_sovereignty`` (model card) fixes a 5-level precedence ladder:

    1. hard-individual-boundary    -> cannot-be-overridden
    2. legal-safety-tenant-policy  -> cannot-be-overridden
    3. explicit-tool-authorization -> caps-autonomous-action
    4. configured-group-decision-rule -> governs-ordinary-preferences
    5. automatically-inferred-preference -> advisory-and-reversible

The guard answers two questions for every proposed bridge:

* ``boundary_violation`` — does the proposal touch a hard boundary category
  (``consent``, ``safety``, ``legal``, ``privacy``, ``data-residency``,
  ``declared-non-negotiable``, ``permission-envelope``)?
* ``authorized`` — is there an explicit grant, and is that grant strong enough
  for the precedence class the proposal reaches?

Authorization is default-deny. Nothing is authorized by absence of a check;
level-3 ``explicit-tool-authorization`` is the ceiling for autonomous action,
and levels 1–2 cannot be overridden by any grant the bridge engine holds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from src.model_card.loader import AgentModelCard, cached_model_card
from src.model_card.types import ContextualSovereignty

# --------------------------------------------------------------------------
# precedence classes and their effects (card constants)
# --------------------------------------------------------------------------

CLASS_HARD_BOUNDARY = "hard-individual-boundary"
CLASS_TENANT_POLICY = "legal-safety-tenant-policy"
CLASS_TOOL_AUTHORIZATION = "explicit-tool-authorization"
CLASS_GROUP_RULE = "configured-group-decision-rule"
CLASS_INFERRED = "automatically-inferred-preference"

EFFECT_CANNOT_OVERRIDE = "cannot-be-overridden"
EFFECT_CAPS_AUTONOMOUS = "caps-autonomous-action"
EFFECT_GOVERNS_ORDINARY = "governs-ordinary-preferences"
EFFECT_ADVISORY = "advisory-and-reversible"

#: The card's hard boundary categories. A proposal whose effect field lands in
#: one of these is a boundary violation regardless of how confident it is.
HARD_BOUNDARY_CATEGORIES: tuple[str, ...] = (
    "consent",
    "safety",
    "legal",
    "privacy",
    "data-residency",
    "declared-non-negotiable",
    "permission-envelope",
)

#: Keys a proposal uses to declare *what* it reaches for. A value here that
#: names a hard boundary category flags the whole proposal.
_EFFECT_KEYS = (
    "effect_class",
    "class",
    "category",
    "effect",
    "boundary_category",
    "targets",
    "subject",
    "permission",
)


# --------------------------------------------------------------------------
# verdict
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SovereigntyVerdict:
    """The guard's answer for one proposed bridge."""

    boundary_violation: bool
    authorized: bool
    precedence_class: str
    precedence_level: int
    reason: str

    def to_json(self) -> dict[str, Any]:
        return {
            "boundary_violation": self.boundary_violation,
            "authorized": self.authorized,
            "precedence_class": self.precedence_class,
            "precedence_level": self.precedence_level,
            "reason": self.reason,
        }


# --------------------------------------------------------------------------
# guard
# --------------------------------------------------------------------------


class SovereigntyGuard:
    """Evaluate a bridge proposal against the sovereignty contract.

    A proposal is a mapping the bridge engine produces; the guard reads the
    effect fields and the explicit grants, and applies the precedence ladder
    without trusting either side.
    """

    def __init__(
        self,
        card: AgentModelCard | None = None,
        *,
        sovereignty: ContextualSovereignty | None = None,
    ) -> None:
        self._card = card or cached_model_card()
        self._sovereignty = sovereignty or self._card.contextual_sovereignty

    # -- card access ------------------------------------------------------

    @property
    def sovereignty(self) -> ContextualSovereignty:
        return self._sovereignty

    def level_of(self, cls: str) -> int:
        return self._sovereignty.level_of(cls)

    @property
    def hard_boundary_categories(self) -> tuple[str, ...]:
        categories = self._sovereignty.hard_boundary_categories
        return tuple(categories) if categories else HARD_BOUNDARY_CATEGORIES

    # -- proposal inspection ---------------------------------------------

    @staticmethod
    def _proposal_categories(proposal: Mapping[str, Any]) -> tuple[str, ...]:
        """Every category the proposal claims to reach, in a stable order."""
        found: set[str] = set()
        for key in _EFFECT_KEYS:
            if key not in proposal:
                continue
            value = proposal[key]
            if isinstance(value, str):
                found.add(value)
            elif isinstance(value, (list, tuple, set, frozenset)):
                for item in value:
                    if isinstance(item, str):
                        found.add(item)
        return tuple(sorted(found))

    def boundary_categories_flagged(
        self, proposal: Mapping[str, Any]
    ) -> tuple[str, ...]:
        """Hard boundary categories the proposal touches, empty if none."""
        hard = set(self.hard_boundary_categories)
        return tuple(
            sorted(c for c in self._proposal_categories(proposal) if c in hard)
        )

    def precedence_class_of(self, proposal: Mapping[str, Any]) -> str:
        """The highest (most binding) precedence class the proposal reaches.

        Defaults to level-5 advisory when the proposal declares nothing.
        """
        categories = self._proposal_categories(proposal)
        for level in sorted(
            (p for p in self._sovereignty.precedence), key=lambda p: p.level
        ):
            if level.cls in categories:
                return level.cls
        for category in categories:
            if category in self.hard_boundary_categories:
                return CLASS_HARD_BOUNDARY
        return CLASS_INFERRED

    # -- evaluation -------------------------------------------------------

    def evaluate(
        self,
        proposal: Mapping[str, Any],
        *,
        grants: Mapping[str, Any] | None = None,
    ) -> SovereigntyVerdict:
        """Return the verdict for one proposal.

        ``grants`` is the caller's evidence of authorization: a mapping whose
        keys are permission/scope names and whose values are grants (typically
        ``{"granted": True, "level": 3}``). Default-deny applies when a grant
        is absent.
        """
        grants = grants or {}

        boundary_categories = self.boundary_categories_flagged(proposal)
        if boundary_categories:
            return SovereigntyVerdict(
                boundary_violation=True,
                authorized=False,
                precedence_class=CLASS_HARD_BOUNDARY,
                precedence_level=1,
                reason=(
                    f"proposal touches hard boundary categories "
                    f"{list(boundary_categories)}; cannot-be-overridden"
                ),
            )

        precedence_class = self.precedence_class_of(proposal)
        precedence_level = self.level_of(precedence_class)

        # explicit-tool-authorization caps autonomous action: a grant may
        # authorize work at level 3, but never binds a level 1–2 class.
        if precedence_level <= 2:
            return SovereigntyVerdict(
                boundary_violation=False,
                authorized=False,
                precedence_class=precedence_class,
                precedence_level=precedence_level,
                reason=(
                    f"precedence level {precedence_level} "
                    f"({precedence_class}) is cannot-be-overridden; the bridge "
                    "engine holds no grant that binds it"
                ),
            )

        # Default-deny: only an explicit grant covers this scope.
        scopes = self._scopes(proposal)
        granted = [
            (scope, grants[scope])
            for scope in scopes
            if scope in grants and self._grant_is_active(grants[scope])
        ]
        if not granted:
            return SovereigntyVerdict(
                boundary_violation=False,
                authorized=False,
                precedence_class=precedence_class,
                precedence_level=precedence_level,
                reason=(
                    f"default-deny: no explicit grant covers {sorted(scopes)} "
                    f"({precedence_class})"
                ),
            )

        return SovereigntyVerdict(
            boundary_violation=False,
            authorized=True,
            precedence_class=precedence_class,
            precedence_level=precedence_level,
            reason=(
                f"explicit grant covers {sorted(scopes)} at precedence level "
                f"{precedence_level}"
            ),
        )

    @staticmethod
    def _scopes(proposal: Mapping[str, Any]) -> tuple[str, ...]:
        """The permission scopes a proposal exercises."""
        scopes: set[str] = set()
        raw = proposal.get("grant_scopes") or proposal.get("permissions") or proposal.get(
            "scopes"
        )
        if isinstance(raw, str):
            scopes.add(raw)
        elif isinstance(raw, (list, tuple, set, frozenset)):
            for item in raw:
                if isinstance(item, str):
                    scopes.add(item)
        scope = proposal.get("scope")
        if isinstance(scope, str):
            scopes.add(scope)
        if not scopes:
            # Proposals that reach level 4 (group rule) or level 5 (inferred)
            # still need a grant; fall back to a conventional scope name.
            scopes.add(proposal.get("bridge_type", "bridge"))
        return tuple(sorted(scopes))

    @staticmethod
    def _grant_is_active(grant: Any) -> bool:
        if isinstance(grant, Mapping):
            return bool(grant.get("granted", True)) and not bool(
                grant.get("revoked", False)
            )
        return bool(grant)


def evaluate_proposal(
    proposal: Mapping[str, Any],
    *,
    grants: Mapping[str, Any] | None = None,
    card: AgentModelCard | None = None,
) -> SovereigntyVerdict:
    """Module-level convenience wrapper around :class:`SovereigntyGuard`."""
    return SovereigntyGuard(card).evaluate(proposal, grants=grants)
