"""The modular-differentiation compiler: layers in, effective agent out.

``modular_differentiation.deterministic: true`` and
``base_agent.deterministic_merge`` together say something strong: given the same
inputs, this must produce byte-identical output, because the manifest's hash is a
required trace attribute (``observability.tracing.effective_manifest.hash``) and
is stored on ``gc_user_profiles.effective_manifest_hash``. Two workers compiling
the same user must agree, or the audit record cannot say which manifest an action
ran under.

Everything here follows from that:

* **The order is the card's.** ``base_agent.compile_order`` is read at run time
  and walked in sequence. A stage name the compiler does not implement is a
  refusal, not a skip — a silently ignored ``authorization_filter`` would be a
  compiler that publishes an unfiltered capability set.
* **The merge is total and stable.** Scalars: the later (higher-precedence)
  layer wins. Lists: stable deduplicating union, first occurrence keeps its
  position, so the result does not depend on set iteration order. Mappings
  recurse.
* **Hard boundaries reject rather than resolve.** ``hard_boundary_conflict:
  reject-compile``. Only ``immutable_base``, ``tenant_policy`` and
  ``hard_user_boundaries`` may write under ``hard_boundaries`` at all, and two of
  them disagreeing on the same leaf raises. A merge rule that let precedence
  settle it would mean a tenant policy could overwrite an individual's boundary
  simply by being merged later — the exact inversion
  ``contextual_sovereignty.precedence`` forbids.
* **The authorization filter is last and is an intersection.** ``granted =
  base ∩ every layer's constraint ∩ request``. Set intersection cannot produce an
  element that was not in ``base``, so ``permission-non-expansion`` is a property
  of the operation rather than of a check that might be skipped. Nothing before
  it can add a capability, and nothing after it runs.

**No timestamp is hashed.** A ``compiled_at`` field would make two identical
compiles produce different hashes and quietly destroy the determinism the card
claims. The card version *is* hashed: the same layers under a different contract
are a different effective agent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from src.model_card import AgentModelCard, cached_model_card

from .evidence import canonical_json, content_hash
from .inference import UserProfile

# --------------------------------------------------------------------------
# errors
# --------------------------------------------------------------------------


class CompileRejected(ValueError):
    """The compile cannot produce a manifest and must not produce a partial one."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


class HardBoundaryConflict(CompileRejected):
    """Two layers disagree about a hard boundary, or one tried to invent one."""

    def __init__(self, detail: str) -> None:
        super().__init__("BOUNDARY_VIOLATION", detail)


# --------------------------------------------------------------------------
# stage names
# --------------------------------------------------------------------------

STAGE_IMMUTABLE_BASE = "immutable_base"
STAGE_TENANT_POLICY = "tenant_policy"
STAGE_HARD_USER_BOUNDARIES = "hard_user_boundaries"
STAGE_AUTOMATIC_USER_PROFILE = "automatic_user_profile"
STAGE_GROUP_RELATIVE_OVERLAY = "group_relative_overlay"
STAGE_SESSION_CONTEXT = "session_context"
STAGE_MODULE_SELECTION = "module_selection"
STAGE_AUTHORIZATION_FILTER = "authorization_filter"

#: Stages that contribute documents to the deep merge, in card order.
MERGE_STAGES: tuple[str, ...] = (
    STAGE_IMMUTABLE_BASE,
    STAGE_TENANT_POLICY,
    STAGE_HARD_USER_BOUNDARIES,
    STAGE_AUTOMATIC_USER_PROFILE,
    STAGE_GROUP_RELATIVE_OVERLAY,
    STAGE_SESSION_CONTEXT,
)

#: Only these may write anything beneath ``hard_boundaries``. Levels 1 and 2 of
#: ``contextual_sovereignty.precedence`` are the individual and the tenant; a
#: group overlay or a session cannot mint a boundary for a user, and an inferred
#: preference certainly cannot.
BOUNDARY_DECLARING_STAGES: frozenset[str] = frozenset(
    {STAGE_IMMUTABLE_BASE, STAGE_TENANT_POLICY, STAGE_HARD_USER_BOUNDARIES}
)

#: The subtree the boundary rules protect.
HARD_BOUNDARY_KEY = "hard_boundaries"

#: Key inside ``hard_boundaries`` holding capabilities the user has taken away.
#: A list, and lists union — more revocation is always narrower, so two layers
#: contributing revocations is a merge, not a conflict.
REVOKED_CAPABILITIES_KEY = "revoked_capabilities"

#: Modules that run whatever the score says. The sovereignty guard enforces the
#: three ``runtime.degradation.never_disable`` controls; the attestor is required
#: by ``automatic_mutation.postconditions`` ("Rule 30 VDF proof envelope
#: emitted"). Leaving either to a score would make a never-disable control
#: disablable by scoring it 0.59.
MANDATORY_MODULES: tuple[str, ...] = (
    "contextual-sovereignty-guard",
    "rule30-vdf-attestor",
)

#: Where a layer puts per-module scores. It travels through the ordinary merge,
#: so a higher-precedence layer overrides one module's score without having to
#: restate the rest.
MODULE_SCORES_KEY = "module_scores"


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BaseContract:
    """``base_agent.contract_id`` with its grant — the ceiling for everything.

    ``mutation: forbidden-at-runtime``: this object is frozen and the compiler
    only ever reads it. ``capabilities`` is the widest set any compile can
    produce.
    """

    contract_id: str
    capabilities: frozenset[str]
    settings: Mapping[str, Any] = field(default_factory=dict)
    module_requirements: Mapping[str, frozenset[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.contract_id).strip():
            raise ValueError("base contract requires a contract_id")
        object.__setattr__(self, "capabilities", frozenset(self.capabilities))
        object.__setattr__(
            self,
            "module_requirements",
            {k: frozenset(v) for k, v in self.module_requirements.items()},
        )


@dataclass(frozen=True, slots=True)
class AgentLayer:
    """One compile-order layer.

    ``capabilities`` is a *constraint*, not a contribution: ``None`` means "this
    layer has no opinion", and a set means "narrow to at most this". There is no
    way to spell "add a capability", which is why no layer can.
    ``modules`` narrows the catalogue the same way.
    """

    name: str
    data: Mapping[str, Any] = field(default_factory=dict)
    capabilities: frozenset[str] | None = None
    modules: frozenset[str] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.data, Mapping):
            raise ValueError(f"layer {self.name!r}: data must be a mapping")
        if self.capabilities is not None:
            object.__setattr__(self, "capabilities", frozenset(self.capabilities))
        if self.modules is not None:
            object.__setattr__(self, "modules", frozenset(self.modules))


@dataclass(frozen=True, slots=True)
class CompileContext:
    """Everything ``modular_differentiation.inputs`` names, as typed layers."""

    base: BaseContract
    tenant_policy: AgentLayer | None = None
    hard_user_boundaries: AgentLayer | None = None
    automatic_user_profile: AgentLayer | None = None
    group_relative_overlay: AgentLayer | None = None
    session_context: AgentLayer | None = None
    requested_capabilities: frozenset[str] | None = None
    card: AgentModelCard | None = None

    def __post_init__(self) -> None:
        if self.requested_capabilities is not None:
            object.__setattr__(
                self, "requested_capabilities", frozenset(self.requested_capabilities)
            )

    def layer_for(self, stage: str) -> AgentLayer | None:
        if stage == STAGE_IMMUTABLE_BASE:
            return AgentLayer(
                STAGE_IMMUTABLE_BASE,
                self.base.settings,
                capabilities=self.base.capabilities,
            )
        return {
            STAGE_TENANT_POLICY: self.tenant_policy,
            STAGE_HARD_USER_BOUNDARIES: self.hard_user_boundaries,
            STAGE_AUTOMATIC_USER_PROFILE: self.automatic_user_profile,
            STAGE_GROUP_RELATIVE_OVERLAY: self.group_relative_overlay,
            STAGE_SESSION_CONTEXT: self.session_context,
        }.get(stage)


# --------------------------------------------------------------------------
# deterministic deep merge
# --------------------------------------------------------------------------


def _list_union(existing: Sequence[Any], incoming: Sequence[Any]) -> list[Any]:
    """Stable deduplicating union.

    Order is "everything already there, then whatever is new, in arrival order".
    Membership is tested on the canonical encoding so unhashable entries (nested
    mappings) deduplicate too — a set would raise on them, and a linear ``in``
    test would compare ``{"a": 1, "b": 2}`` and ``{"b": 2, "a": 1}`` as different.
    """
    out: list[Any] = []
    seen: set[str] = set()
    for item in list(existing) + list(incoming):
        marker = canonical_json(item)
        if marker in seen:
            continue
        seen.add(marker)
        out.append(item)
    return out


class _Missing:
    """Sentinel distinguishing "absent" from "present and set to None"."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<missing>"


_MISSING = _Missing()


def _same(a: Any, b: Any) -> bool:
    """Equality that tolerates float representation, for boundary comparison."""
    if isinstance(a, float) or isinstance(b, float):
        try:
            return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-9)
        except (TypeError, ValueError):
            return False
    return a == b


def _is_protected(path: tuple[str, ...]) -> bool:
    return HARD_BOUNDARY_KEY in path


def _claim_subtree(
    owners: dict[tuple[str, ...], str], path: tuple[str, ...], value: Any, stage: str
) -> None:
    """Record ``stage`` as the author of every leaf under ``path``.

    A layer that writes the whole ``hard_boundaries`` mapping in one assignment
    still authored each boundary inside it. Without this, the conflict raised
    against a *later* layer could not name who set the original value, and an
    operator reading the error would have to diff the layers by hand.
    """
    owners[path] = stage
    if isinstance(value, Mapping):
        for key, sub in value.items():
            _claim_subtree(owners, (*path, str(key)), sub, stage)


def deep_merge(
    existing: Mapping[str, Any],
    incoming: Mapping[str, Any],
    *,
    stage: str,
    owners: dict[tuple[str, ...], str],
    path: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Ordered deep merge of ``incoming`` (higher precedence) onto ``existing``.

    ``owners`` records which stage last wrote each protected leaf, so a conflict
    message can name both sides. It is mutated as the merge proceeds; on a
    rejected compile the caller discards the whole result, so partial mutation
    never escapes.
    """
    merged: dict[str, Any] = dict(existing)
    for key, new_value in incoming.items():
        key = str(key)
        here = (*path, key)
        protected = _is_protected(here)
        if protected and stage not in BOUNDARY_DECLARING_STAGES:
            raise HardBoundaryConflict(
                f"layer {stage!r} writes {'.'.join(here)}: hard boundaries are "
                "declared by the individual and by tenant policy only "
                "(precedence levels 1 and 2)"
            )
        old_value = merged.get(key, _MISSING)
        if isinstance(old_value, Mapping) and isinstance(new_value, Mapping):
            merged[key] = deep_merge(
                old_value, new_value, stage=stage, owners=owners, path=here
            )
            continue
        if isinstance(old_value, (list, tuple)) and isinstance(new_value, (list, tuple)):
            merged[key] = _list_union(old_value, new_value)
            if protected:
                owners[here] = stage
            continue
        if protected and old_value is not _MISSING and not _same(old_value, new_value):
            raise HardBoundaryConflict(
                f"{'.'.join(here)} is a hard boundary set to {old_value!r} by "
                f"{owners.get(here, 'an earlier layer')!r}; layer {stage!r} "
                f"would change it to {new_value!r}. A hard boundary cannot be "
                "overridden, so the compile is rejected rather than resolved"
            )
        merged[key] = new_value
        if protected:
            _claim_subtree(owners, here, new_value, stage)
    return merged


# --------------------------------------------------------------------------
# module selection
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModuleActivation:
    """One selected module and why it is on."""

    name: str
    score: float
    mandatory: bool
    activation: str  # "rule" | "score"

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "score": round(self.score, 9),
            "mandatory": self.mandatory,
            "activation": self.activation,
        }


def select_modules(
    catalog: Sequence[str],
    scores: Mapping[str, float],
    *,
    threshold: float,
    maximum: int,
    mandatory: Iterable[str] = (),
    excluded: Iterable[str] = (),
) -> tuple[ModuleActivation, ...]:
    """``rule-plus-score`` selection.

    The *rule* half decides membership questions a score cannot: a module the
    card requires is on at any score, and a module whose precondition is absent
    (an unavailable topology sidecar) is off at any score. The *score* half then
    ranks whatever remains against ``activation_threshold``.

    Ties break on name so two workers with identical inputs produce identical
    manifests; ``conflict_resolution`` reads
    "higher-precedence-constraint-then-higher-confidence", which is exactly
    (mandatory, then score) — the name is the third key only to make the order
    total.
    """
    if maximum < 1:
        raise CompileRejected(
            "MODULE_SELECTION_INVALID", f"max_active_modules must be >= 1, got {maximum}"
        )
    mandatory_set = frozenset(mandatory)
    excluded_set = frozenset(excluded)
    forced_and_excluded = mandatory_set & excluded_set
    if forced_and_excluded:
        raise CompileRejected(
            "MODULE_SELECTION_INVALID",
            f"modules are both mandatory and excluded: {sorted(forced_and_excluded)}",
        )
    candidates: list[ModuleActivation] = []
    for name in catalog:
        if name in excluded_set:
            continue
        raw = scores.get(name, 0.0)
        try:
            score = float(raw)
        except (TypeError, ValueError):
            raise CompileRejected(
                "MODULE_SELECTION_INVALID",
                f"module {name!r} has a non-numeric score {raw!r}",
            ) from None
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise CompileRejected(
                "MODULE_SELECTION_INVALID",
                f"module {name!r} score {score!r} is outside the card's "
                "score_range [0.0, 1.0]",
            )
        if name in mandatory_set:
            candidates.append(ModuleActivation(name, score, True, "rule"))
        elif score >= threshold:
            candidates.append(ModuleActivation(name, score, False, "score"))
    candidates.sort(key=lambda m: (not m.mandatory, -m.score, m.name))
    if len(candidates) > maximum:
        dropped_mandatory = [m.name for m in candidates[maximum:] if m.mandatory]
        if dropped_mandatory:
            raise CompileRejected(
                "MODULE_SELECTION_INVALID",
                f"max_active_modules={maximum} cannot hold the mandatory modules "
                f"{sorted(dropped_mandatory)}",
            )
        candidates = candidates[:maximum]
    return tuple(candidates)


def _module_rules(
    doc: Mapping[str, Any], catalog: Sequence[str]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The rule half: which modules are forced on and which are ruled out."""
    mandatory = tuple(m for m in MANDATORY_MODULES if m in catalog)
    excluded: list[str] = []
    if doc.get("topology_sidecar_available") is False:
        # ``topology_sidecar.fallback_order`` ends at a non-topological baseline,
        # and ``runtime.degradation`` disables the sidecar before it touches
        # anything else. A scored-in sidecar with no sidecar available would sit
        # in the manifest claiming a descriptor it cannot produce.
        excluded.append("topology-stability-sidecar")
    for name in doc.get("disabled_modules", ()) or ():
        excluded.append(str(name))
    return mandatory, tuple(dict.fromkeys(excluded))


# --------------------------------------------------------------------------
# authorization filter
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AuthorizationResult:
    granted: frozenset[str]
    denied: frozenset[str]
    narrowed_by: tuple[str, ...]
    dropped_modules: tuple[str, ...]


def authorization_filter(
    base_capabilities: frozenset[str],
    constraints: Sequence[tuple[str, frozenset[str]]],
    *,
    requested: frozenset[str] | None,
    revoked: frozenset[str],
    universe: frozenset[str],
    modules: Sequence[ModuleActivation],
    module_requirements: Mapping[str, frozenset[str]],
) -> tuple[AuthorizationResult, tuple[ModuleActivation, ...]]:
    """The mandatory final stage: intersect, never union.

    Every operation here removes elements. ``granted`` therefore cannot contain a
    capability absent from ``base_capabilities`` regardless of what any layer
    asked for — the closing assertion is a tripwire for a future edit that
    replaces one of these with a ``|``, not a check the correctness depends on.
    """
    granted = frozenset(base_capabilities)
    narrowed_by: list[str] = []
    for stage, caps in constraints:
        before = granted
        granted &= caps
        if granted != before:
            narrowed_by.append(stage)
    if requested is not None:
        before = granted
        granted &= requested
        if granted != before:
            narrowed_by.append("requested_capabilities")
    if revoked:
        before = granted
        granted -= revoked
        if granted != before:
            narrowed_by.append("hard_user_boundaries.revoked_capabilities")
    # A grant naming a permission the card does not define is a configuration
    # error, not a new capability; drop it rather than publish it.
    granted &= universe

    if not granted <= base_capabilities:
        raise CompileRejected(  # pragma: no cover - structurally unreachable
            "AUTHORIZATION_DENIED",
            "authorization filter produced capabilities outside the base grant",
        )

    kept: list[ModuleActivation] = []
    dropped: list[str] = []
    for module in modules:
        needs = frozenset(module_requirements.get(module.name, ()))
        if needs <= granted:
            kept.append(module)
            continue
        if module.mandatory:
            raise CompileRejected(
                "AUTHORIZATION_DENIED",
                f"module {module.name!r} is mandatory but requires ungranted "
                f"capabilities {sorted(needs - granted)}",
            )
        dropped.append(module.name)
    denied = frozenset(requested - granted) if requested is not None else frozenset()
    return AuthorizationResult(
        granted=granted,
        denied=denied,
        narrowed_by=tuple(narrowed_by),
        dropped_modules=tuple(sorted(dropped)),
    ), tuple(kept)


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, eq=False)
class EffectiveAgentManifest:
    """A compiled effective agent, identified by the hash of its own content.

    Equality and hashing are the content hash, so the manifest can be used as a
    dict key and two compiles can be compared without walking the document.
    ``__post_init__`` recomputes the hash, which makes a manifest whose hash does
    not match its content unconstructible — the property
    ``validation.behavioral_acceptance.vdf-binding`` depends on downstream.
    """

    content: Mapping[str, Any]
    manifest_hash: str

    def __post_init__(self) -> None:
        recomputed = content_hash(self.content)
        if self.manifest_hash != recomputed:
            raise ValueError(
                f"manifest hash {self.manifest_hash!r} does not match its content "
                f"({recomputed!r})"
            )

    @classmethod
    def from_content(cls, content: Mapping[str, Any]) -> EffectiveAgentManifest:
        return cls(content=dict(content), manifest_hash=content_hash(content))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EffectiveAgentManifest):
            return NotImplemented
        return self.manifest_hash == other.manifest_hash

    def __hash__(self) -> int:
        return hash(self.manifest_hash)

    @property
    def granted_capabilities(self) -> frozenset[str]:
        return frozenset(self.content["granted_capabilities"])

    @property
    def active_modules(self) -> tuple[str, ...]:
        return tuple(m["name"] for m in self.content["active_modules"])

    @property
    def hard_boundaries(self) -> Mapping[str, Any]:
        return self.content["settings"].get(HARD_BOUNDARY_KEY, {})

    def setting(self, *path: str, default: Any = None) -> Any:
        node: Any = self.content["settings"]
        for key in path:
            if not isinstance(node, Mapping) or key not in node:
                return default
            node = node[key]
        return node

    def canonical_json(self) -> str:
        return canonical_json(self.content)

    def to_json(self) -> dict[str, Any]:
        """The manifest plus its hash.

        The hash is added *outside* the hashed content, the same way
        ``rule30_vdf.canonical_input.excluded_fields`` keeps ``vdf_proof`` out of
        the bytes it attests.
        """
        return {**dict(self.content), "manifest_hash": self.manifest_hash}


# --------------------------------------------------------------------------
# compile
# --------------------------------------------------------------------------


def compile_effective_agent(context: CompileContext) -> EffectiveAgentManifest:
    """Compile the effective agent for one user, in the card's order."""
    card = context.card or cached_model_card()
    merge_spec = card.base_agent.deterministic_merge
    if merge_spec.strategy != "ordered-deep-merge":
        raise CompileRejected(
            "COMPILE_CONTRACT_MISMATCH",
            f"this compiler implements ordered-deep-merge; the card asks for "
            f"{merge_spec.strategy!r}",
        )
    order = tuple(card.base_agent.compile_order)
    if not order or order[-1] != STAGE_AUTHORIZATION_FILTER:
        raise CompileRejected(
            "COMPILE_CONTRACT_MISMATCH",
            "the authorization filter is a mandatory final stage; "
            f"compile_order ends with {order[-1] if order else None!r}",
        )
    universe = card.mcp.permissions
    unknown = context.base.capabilities - universe
    if unknown:
        raise CompileRejected(
            "COMPILE_CONTRACT_MISMATCH",
            f"base contract grants capabilities the card does not define: "
            f"{sorted(unknown)}",
        )

    selection_spec = card.modular_differentiation.module_selection
    if selection_spec.permission_filter != "mandatory-final-stage":
        raise CompileRejected(
            "COMPILE_CONTRACT_MISMATCH",
            "module_selection.permission_filter must be mandatory-final-stage",
        )
    catalog = tuple(sorted(card.modular_differentiation.modules))
    max_modules = min(
        selection_spec.max_active_modules,
        card.runtime.state_bounds.active_modules_per_effective_agent,
    )

    doc: dict[str, Any] = {}
    owners: dict[tuple[str, ...], str] = {}
    constraints: list[tuple[str, frozenset[str]]] = []
    applied: list[str] = []
    modules: tuple[ModuleActivation, ...] = ()
    authorization: AuthorizationResult | None = None

    for stage in order:
        if stage in MERGE_STAGES:
            layer = context.layer_for(stage)
            if layer is None:
                continue
            doc = deep_merge(doc, layer.data, stage=stage, owners=owners)
            if layer.capabilities is not None:
                constraints.append((stage, layer.capabilities))
            if layer.modules is not None:
                catalog = tuple(c for c in catalog if c in layer.modules)
            applied.append(stage)
        elif stage == STAGE_MODULE_SELECTION:
            mandatory, excluded = _module_rules(doc, catalog)
            scores = doc.get(MODULE_SCORES_KEY, {}) or {}
            if not isinstance(scores, Mapping):
                raise CompileRejected(
                    "MODULE_SELECTION_INVALID",
                    f"{MODULE_SCORES_KEY} must be a mapping, got "
                    f"{type(scores).__name__}",
                )
            modules = select_modules(
                catalog,
                scores,
                threshold=selection_spec.activation_threshold,
                maximum=max_modules,
                mandatory=mandatory,
                excluded=excluded,
            )
            applied.append(stage)
        elif stage == STAGE_AUTHORIZATION_FILTER:
            if STAGE_MODULE_SELECTION in order and STAGE_MODULE_SELECTION not in applied:
                raise CompileRejected(
                    "COMPILE_CONTRACT_MISMATCH",
                    "authorization_filter reached before module_selection",
                )
            boundaries = doc.get(HARD_BOUNDARY_KEY, {}) or {}
            revoked = frozenset(
                str(c) for c in (boundaries.get(REVOKED_CAPABILITIES_KEY, ()) or ())
            )
            authorization, modules = authorization_filter(
                context.base.capabilities,
                constraints,
                requested=context.requested_capabilities,
                revoked=revoked,
                universe=universe,
                modules=modules,
                module_requirements=context.base.module_requirements,
            )
            applied.append(stage)
        else:
            raise CompileRejected(
                "COMPILE_CONTRACT_MISMATCH",
                f"compile_order names a stage this compiler does not implement: "
                f"{stage!r}",
            )

    if authorization is None:  # pragma: no cover - guarded by the order check
        raise CompileRejected(
            "COMPILE_CONTRACT_MISMATCH", "the authorization filter did not run"
        )

    content = {
        "contract_id": context.base.contract_id,
        "card_id": card.metadata.id,
        "card_version": card.metadata.version,
        "compile_order": list(order),
        "layers_applied": applied,
        "settings": doc,
        "hard_boundaries": doc.get(HARD_BOUNDARY_KEY, {}),
        "granted_capabilities": sorted(authorization.granted),
        "denied_capabilities": sorted(authorization.denied),
        "narrowed_by": list(authorization.narrowed_by),
        "active_modules": [m.to_json() for m in modules],
        "deauthorized_modules": list(authorization.dropped_modules),
        "contested_features": sorted(
            str(f) for f in (doc.get("contested_features", ()) or ())
        ),
        "unresolved_contradictions": int(doc.get("unresolved_contradictions", 0) or 0),
    }
    return EffectiveAgentManifest.from_content(content)


# --------------------------------------------------------------------------
# profile -> layers
# --------------------------------------------------------------------------


def layers_from_profile(
    profile: UserProfile,
) -> tuple[AgentLayer, AgentLayer]:
    """Split a profile into its ``hard_user_boundaries`` and profile layers.

    Two layers, not one, because the card's compile order puts them at different
    precedence levels: the boundaries land above tenant policy in effect (they
    cannot be overridden by anything later), while the inferred preferences land
    below the group overlay and the session, where they are meant to be
    overridable — ``automatically-inferred-preference`` is precedence level 5,
    "advisory-and-reversible".

    Contested features and unresolved contradictions travel with the profile
    layer so the compiled manifest reports them. A manifest that presented a
    contested preference as settled would be manufacturing agreement between two
    pieces of evidence, which is the same failure
    ``no_false_consensus`` names between two people.
    """
    boundaries = dict(profile.hard_boundaries())
    if profile.revoked_capabilities:
        boundaries[REVOKED_CAPABILITIES_KEY] = sorted(profile.revoked_capabilities)
    preferences: dict[str, dict[str, Any]] = {}
    for state in profile.features.values():
        if state.domain in (HARD_BOUNDARY_KEY,):
            continue
        preferences.setdefault(state.domain, {})[state.name] = state.rendered_value()
    boundary_layer = AgentLayer(
        STAGE_HARD_USER_BOUNDARIES,
        {HARD_BOUNDARY_KEY: boundaries} if boundaries else {},
    )
    profile_layer = AgentLayer(
        STAGE_AUTOMATIC_USER_PROFILE,
        {
            "preferences": preferences,
            "profile_revision": profile.revision,
            "contested_features": list(profile.contested_features),
            "unresolved_contradictions": len(profile.unresolved_contradictions),
        },
    )
    return boundary_layer, profile_layer
