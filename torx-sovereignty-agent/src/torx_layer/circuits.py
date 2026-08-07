"""TORX probabilistic circuits for the model card's factor graph.

This module is the actual binding to the Extropic ``torx`` kernel
(``extro-torx``). It expresses the decision half of
``torx.factor_graph.directed_edges`` as real ``torx.psc`` circuits:

===================================  =========================================
card node                            realisation
===================================  =========================================
``boundary_violation``   (pbit)      ``PNOT`` on a binary site
``action_authorized``    (pbit)      ``PNOT`` on a binary site
``bridge_viability``     (pbit)      ``PNOT`` + structural veto (``PCSWAP``)
``tension_class``        (pdit)      7-ary site driven by ``PditShift``
``bridge_candidate``     (pdit)      7-ary site driven by ``PditShift``
``resolution_status``    (pdit)      marginal of the joint decision density
``individual/group_intent``, ``tension_gradient`` (pmode)
                                     ``HybridPCircuit`` + affine-Gaussian
                                     conditioning (see :mod:`.kernels`)
===================================  =========================================

**Why the veto is structural, not probabilistic.** Every torx gate is
parameterised by ``p = sigmoid(theta)``. If the hard-boundary veto were encoded
as "multiply viability by a small number", a boundary violation at ``p = 0.98``
would still leave 2% of the probability mass on a viable bridge, and
``cannot-be-overridden`` would be a strong preference rather than a rule.
Instead the veto is wired as a **deterministic** controlled-SWAP
(``theta = +DETERMINISTIC``, so ``p = 1``) that exchanges the viability site with
a site pinned at 0. The wiring, not the parameter, is what forbids the action —
so no choice of estimate can produce a viable bridge under a violated boundary.
:func:`veto_is_structural` is the property test of exactly that claim.

Every entry point degrades: when ``torx`` or JAX is unavailable the same
semantics are computed by an exact pure-Python enumeration over the (tiny) state
space. That is the card's ``non-topological-torx-baseline`` rung, and it is exact
rather than approximate because the discrete circuits here have at most a few
hundred basis states. :func:`backend_status` reports which path ran, and the
value lands in ``TorxDecisionState.backend`` so an audit record always says which
kernel produced the decision.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Sequence

from .state import (
    BRIDGE_STRATEGIES,
    RESOLUTION_STATES,
    TENSION_CLASSES,
    PBit,
    PDit,
)

# theta magnitude at which sigmoid(theta) is 1.0 (or 0.0) in float32. float32
# sigmoid saturates well before this; 40 is comfortably past saturation while
# staying finite, so the gate matrix contains exact 0/1 entries with no NaN risk
# from an inf-minus-inf inside the kernel.
DETERMINISTIC = 40.0

# Site indices of the decision circuit. Fixed and shared with the fallback so
# both paths describe the same circuit.
SITE_BOUNDARY_VIOLATION = 0
SITE_ACTION_AUTHORIZED = 1
SITE_BRIDGE_VIABILITY = 2
SITE_VETO = 3  # work site: violation OR NOT authorized
SITE_ZERO = 4  # work site: pinned at 0, the source of the veto's reset
DECISION_SITES = 5

_torx_state: dict[str, Any] = {"loaded": False, "ok": False, "error": None}


def _load_torx() -> dict[str, Any]:
    """Import ``torx``/JAX once, recording why it failed if it did.

    Mirrors the lazy-optional-backend idiom the rest of the node uses
    (``thrml_daemon._load_backends``): import never raises at module scope, so a
    machine without the kernels can still run the guard, the persistence layer,
    and the MCP surface.
    """
    if _torx_state["loaded"]:
        return _torx_state
    _torx_state["loaded"] = True
    try:
        import jax  # noqa: F401
        import jax.numpy as jnp
        from torx.psc import (
            DiscretePCircuit,
            PCopy,
            PCSWAP,
            PDEMUX,
            PNOT,
            POR,
            StateVectorSimulator,
        )

        _torx_state.update(
            ok=True,
            jnp=jnp,
            DiscretePCircuit=DiscretePCircuit,
            StateVectorSimulator=StateVectorSimulator,
            PNOT=PNOT,
            POR=POR,
            PCopy=PCopy,
            PCSWAP=PCSWAP,
            PDEMUX=PDEMUX,
            backend=jax.default_backend(),
        )
    except Exception as exc:  # ImportError, JAX plugin failure, missing CUDA...
        _torx_state.update(ok=False, error=f"{type(exc).__name__}: {exc}")
    return _torx_state


def torx_available() -> bool:
    """True when the real torx kernel can run."""
    if os.environ.get("TORX_FORCE_FALLBACK") == "1":
        return False
    return bool(_load_torx()["ok"])


def backend_status() -> dict[str, Any]:
    """Which TORX path is live, for the HUD and for audit records."""
    st = _load_torx()
    forced = os.environ.get("TORX_FORCE_FALLBACK") == "1"
    try:
        from importlib.metadata import version

        torx_version = version("extro-torx")
    except Exception:
        torx_version = None
    return {
        "torx_available": bool(st["ok"]) and not forced,
        "torx_version": torx_version,
        "jax_backend": st.get("backend"),
        "forced_fallback": forced,
        "error": st.get("error"),
        "engine": _engine_label(),
    }


def _engine_label() -> str:
    """Engine identity string, e.g. ``torx-0.0.1/jax-cpu``.

    Same shape as the sampler's ``engine_label`` so the existing WebGPU
    self-observer validator (which allows the ``thrml`` and ``torx`` engine
    families and requires a resolvable version) accepts it. An unresolvable
    version yields a non-numeric label and fails that validator closed rather
    than publishing an unverifiable engine claim.
    """
    st = _load_torx()
    if not st["ok"] or os.environ.get("TORX_FORCE_FALLBACK") == "1":
        return "torx-fallback/python-exact"
    try:
        from importlib.metadata import version

        ver = version("extro-torx")
    except Exception:
        ver = "unknown"
    return f"torx-{ver}/jax-{st.get('backend', 'cpu')}"


def logit(p: float, *, cap: float = DETERMINISTIC) -> float:
    """Inverse sigmoid, saturating at ``±cap`` instead of returning ``±inf``.

    ``p`` of exactly 0 or 1 is a legitimate input (a certain estimate), and an
    infinite theta would propagate NaN through the gate matrix.
    """
    p = min(max(float(p), 0.0), 1.0)
    if p <= 0.0:
        return -cap
    if p >= 1.0:
        return cap
    return max(min(math.log(p / (1.0 - p)), cap), -cap)


def sigmoid(theta: float) -> float:
    if theta >= 0:
        return 1.0 / (1.0 + math.exp(-theta))
    e = math.exp(theta)
    return e / (1.0 + e)


# --------------------------------------------------------------------------
# decision circuit
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DecisionInputs:
    """Estimates feeding the decision circuit.

    ``p_boundary_violation`` and ``p_action_authorized`` come from the
    sovereignty guard's structured checks; ``p_bridge_viability`` is the bridge
    simulator's belief that the candidate reduces tension. All three are
    *estimates* — the circuit's job is to combine them under the structural veto.
    """

    p_bridge_viability: float
    p_boundary_violation: float
    p_action_authorized: float

    def __post_init__(self) -> None:
        for name in (
            "p_bridge_viability",
            "p_boundary_violation",
            "p_action_authorized",
        ):
            v = getattr(self, name)
            if not math.isfinite(v) or not 0.0 <= v <= 1.0:
                raise ValueError(f"{name} must be a probability in [0, 1], got {v!r}")


@dataclass(frozen=True, slots=True)
class DecisionMarginals:
    """Marginals read back out of the decision circuit."""

    bridge_viability: PBit
    boundary_violation: PBit
    action_authorized: PBit
    resolution_status: PDit
    backend: str


def _decision_gates():
    """The gate list, shared by the torx path and documented for the fallback.

    Order matters — this is a circuit, not a set:

    1. ``PNOT(0)``            sample ``boundary_violation``
    2. ``PNOT(1)``            sample ``action_authorized``
    3. ``PNOT(2)``            sample raw ``bridge_viability``
    4. ``PCopy([1, 3])``      veto work site := authorized                (p=1)
    5. ``PNOT(3)``            veto work site := NOT authorized            (p=1)
    6. ``PCopy([0, 4])``      zero work site := boundary_violation        (p=1)
    7. ``POR([3, 4])``        veto := NOT authorized OR violation;
                              site 4 is reset to 0 by ``POR`` itself      (p=1)
    8. ``PCSWAP([3, 2, 4])``  if veto: swap viability with the 0 site     (p=1)

    Steps 4-8 all run at ``theta = DETERMINISTIC``: they are the *wiring* of the
    factor graph, not estimates. Step 7 is what makes step 8 safe — ``POR``
    leaves its second operand at 0, which is precisely the constant the
    controlled swap needs.
    """
    st = _load_torx()
    PNOT, POR, PCopy, PCSWAP = st["PNOT"], st["POR"], st["PCopy"], st["PCSWAP"]
    return [
        PNOT(SITE_BOUNDARY_VIOLATION),
        PNOT(SITE_ACTION_AUTHORIZED),
        PNOT(SITE_BRIDGE_VIABILITY),
        PCopy([SITE_ACTION_AUTHORIZED, SITE_VETO]),
        PNOT(SITE_VETO),
        PCopy([SITE_BOUNDARY_VIOLATION, SITE_ZERO]),
        POR([SITE_VETO, SITE_ZERO]),
        PCSWAP([SITE_VETO, SITE_BRIDGE_VIABILITY, SITE_ZERO]),
    ]


def _decision_thetas(inputs: DecisionInputs):
    st = _load_torx()
    jnp = st["jnp"]
    return [
        jnp.array([logit(inputs.p_boundary_violation)]),
        jnp.array([logit(inputs.p_action_authorized)]),
        jnp.array([logit(inputs.p_bridge_viability)]),
        jnp.array([DETERMINISTIC]),
        jnp.array([DETERMINISTIC]),
        jnp.array([DETERMINISTIC]),
        jnp.array([DETERMINISTIC]),
        jnp.array([DETERMINISTIC]),
    ]


def _resolution_from_joint(
    p_viable: float, p_violation: float, p_authorized: float
) -> PDit:
    """Map the decision pbits onto ``resolution_status``.

    ``blocked`` and ``unresolved`` are kept distinct on purpose: *blocked* is a
    sovereign refusal (a boundary or an authorization gap) and must surface as
    such, while *unresolved* is the no-false-consensus outcome where nothing was
    forbidden but no bridge helped. Collapsing them would let a refusal read as
    a mere failure to find an option.
    """
    p_blocked = min(1.0, p_violation + (1.0 - p_authorized) - p_violation * (1.0 - p_authorized))
    p_open = 1.0 - p_blocked
    p_resolved = p_open * p_viable
    p_unresolved = p_open * (1.0 - p_viable)
    # A viable-but-uncertain bridge lands partially resolved rather than
    # resolved; the split is the viability entropy, so a 50/50 bridge cannot
    # report a clean resolution.
    uncertainty = 4.0 * p_viable * (1.0 - p_viable)  # 0 at certainty, 1 at p=0.5
    p_partial = p_resolved * uncertainty
    p_resolved -= p_partial
    return PDit(
        RESOLUTION_STATES,
        (p_resolved, p_partial, p_unresolved, p_blocked),
        label="resolution_status",
    )


def _decision_fallback(inputs: DecisionInputs) -> DecisionMarginals:
    """Exact enumeration of the same circuit, without torx or JAX.

    The three sampled pbits are independent by construction and steps 4-8 are
    deterministic, so the marginals are closed-form: viability survives exactly
    when neither the boundary is violated nor authorization is missing.
    """
    p_v = inputs.p_bridge_viability
    p_b = inputs.p_boundary_violation
    p_a = inputs.p_action_authorized
    p_no_veto = (1.0 - p_b) * p_a
    p_viable_final = p_v * p_no_veto
    return DecisionMarginals(
        bridge_viability=PBit(p_viable_final, "bridge_viability"),
        boundary_violation=PBit(p_b, "boundary_violation"),
        action_authorized=PBit(p_a, "action_authorized"),
        resolution_status=_resolution_from_joint(p_v, p_b, p_a),
        backend="torx-fallback/python-exact",
    )


_decision_jit: Any = None


def _build_decision_jit():
    """Compile the decision circuit once and reuse it.

    The circuit *structure* never changes — five sites, eight gates, fixed
    wiring — only the three estimate thetas do. Rebuilding the circuit per call
    made JAX re-trace ``density``'s ``fori_loop`` every time, costing ~2.7s a
    decision on this node. The card puts the TORX update on the critical path
    with a p99 decision budget (``gc_torx_update_seconds``,
    ``TORX-update-p99-within-decision-budget``), so per-call retracing is a
    correctness-of-deployment problem, not just slow tests.

    The structure is closed over as a Python constant, so it is static to the
    trace; only the theta vector is an argument.
    """
    global _decision_jit
    if _decision_jit is not None:
        return _decision_jit
    st = _load_torx()
    import jax

    jnp = st["jnp"]
    circuit = st["DiscretePCircuit"](_decision_gates(), reps=1)
    sim = st["StateVectorSimulator"]()
    n_gates = len(circuit.gates)
    # All five sites start at 0, so the initial distribution is the point mass
    # on |00000> — index 0 of the flattened state vector.
    x0 = jnp.zeros((2**DECISION_SITES,)).at[0].set(1.0)

    @jax.jit
    def run(theta_stack):
        thetas = [theta_stack[i] for i in range(n_gates)]
        built = sim.build_circuit(circuit, thetas)
        return sim.expval_all(built, x0)

    _decision_jit = run
    return run


def evaluate_decision(inputs: DecisionInputs) -> DecisionMarginals:
    """Run the decision circuit and return its marginals.

    Uses the real torx state-vector simulator when available; the pure-Python
    enumeration otherwise. Both paths agree to within float32 precision, which
    ``tests/torx_layer/test_circuits.py`` asserts directly.
    """
    if not torx_available():
        return _decision_fallback(inputs)
    st = _load_torx()
    try:
        jnp = st["jnp"]
        run = _build_decision_jit()
        expvals = run(jnp.stack(_decision_thetas(inputs)))
        p_viable = float(expvals[SITE_BRIDGE_VIABILITY])
        p_violation = float(expvals[SITE_BOUNDARY_VIOLATION])
        p_authorized = float(expvals[SITE_ACTION_AUTHORIZED])
    except Exception:
        # A kernel failure must not take the decision path down; degrade to the
        # exact baseline and say so in ``backend``.
        return _decision_fallback(inputs)

    # ``resolution_status`` is derived from the *raw* viability and the vetoes so
    # that "a good bridge that was blocked" is distinguishable from "a bad
    # bridge"; the circuit's post-veto viability is reported separately.
    return DecisionMarginals(
        bridge_viability=PBit(p_viable, "bridge_viability"),
        boundary_violation=PBit(p_violation, "boundary_violation"),
        action_authorized=PBit(p_authorized, "action_authorized"),
        resolution_status=_resolution_from_joint(
            inputs.p_bridge_viability, p_violation, p_authorized
        ),
        backend=_engine_label(),
    )


def veto_is_structural(
    p_bridge_viability: float, *, violated: bool = True, authorized: bool = True
) -> bool:
    """Assert the veto holds for *any* viability estimate.

    Exposed as a function (rather than living only in a test) because the
    sovereignty guard calls it as a self-check before trusting the circuit: if a
    kernel upgrade ever changed ``PCSWAP`` semantics, the guard fails closed
    instead of quietly permitting a vetoed bridge.
    """
    out = evaluate_decision(
        DecisionInputs(
            p_bridge_viability=p_bridge_viability,
            p_boundary_violation=1.0 if violated else 0.0,
            p_action_authorized=1.0 if authorized else 0.0,
        )
    )
    return out.bridge_viability.is_certainly_false()


# --------------------------------------------------------------------------
# pdit circuits: tension class and bridge candidate
# --------------------------------------------------------------------------


def _categorical_fallback(scores: Sequence[float], outcomes: Sequence[str], label: str) -> PDit:
    return PDit.from_scores(
        {o: float(s) for o, s in zip(outcomes, scores)}, label=label
    )


def evaluate_categorical(
    scores: Sequence[float], outcomes: Sequence[str], *, label: str = ""
) -> PDit:
    """Turn per-outcome scores into a pdit, computed on the torx kernel.

    **Encoding.** The k-ary outcome is realised as a *one-hot register of k
    binary torx sites* rather than a single k-ary site. This is deliberate: the
    single-site pdit gates torx exposes (``PditShift``, ``PditCycle``) are
    cyclic shifts of the whole distribution, so a chain of them starting from a
    point mass can only reach Poisson-binomial shapes over the ring — it cannot
    represent an arbitrary categorical, and using it would silently return a
    distribution that is not the one the scores describe.

    **Circuit.** A stick-breaking chain of ``PDEMUX`` gates. ``PDEMUX([i, i+1])``
    moves a token from site ``i`` to site ``i + 1`` with probability
    ``p = sigmoid(theta)`` and otherwise leaves it at ``i`` (resetting ``i + 1``
    either way). Starting with the token at site 0 and choosing

        p_i = 1 - target_i / remaining_i

    puts the token on site ``i`` with probability exactly ``target_i``. Because
    the register is one-hot, ``expval_all`` reads the categorical straight off
    the marginals. The construction is exact for any categorical, and the
    ``target`` it reproduces is the softmax of ``scores`` — the same value the
    fallback computes, so both paths agree to float precision.
    """
    if len(scores) != len(outcomes):
        raise ValueError(f"{len(scores)} scores for {len(outcomes)} outcomes")
    target_pdit = _categorical_fallback(scores, outcomes, label)
    if not torx_available() or len(outcomes) < 2:
        return target_pdit
    try:
        st = _load_torx()
        jnp = st["jnp"]
        k = len(outcomes)
        target = target_pdit.probs
        # Stick-breaking move probabilities: P(token leaves site i | it arrived).
        remaining = 1.0
        thetas = []
        for i in range(k - 1):
            p_move = 1.0 - (target[i] / remaining) if remaining > 0 else 0.0
            thetas.append(logit(p_move))
            remaining = max(remaining - target[i], 0.0)
        probs = [float(v) for v in _build_categorical_jit(k)(jnp.asarray(thetas))]
        total = math.fsum(probs)
        if not math.isfinite(total) or total <= 0:
            return target_pdit
        return PDit(tuple(outcomes), tuple(probs), label)
    except Exception:
        return target_pdit


#: Compiled stick-breaking circuits keyed by outcome count. Same reason as the
#: decision circuit: the structure depends only on ``k``, and re-tracing it on
#: every classification would put the pdit construction outside the card's
#: decision budget.
_categorical_jit_cache: dict[int, Any] = {}


def _build_categorical_jit(k: int):
    if k in _categorical_jit_cache:
        return _categorical_jit_cache[k]
    st = _load_torx()
    import jax

    jnp = st["jnp"]
    PDEMUX = st["PDEMUX"]
    circuit = st["DiscretePCircuit"]([PDEMUX([i, i + 1]) for i in range(k - 1)], reps=1)
    sim = st["StateVectorSimulator"]()
    # Token starts on site 0: |1 0 0 ... 0>. Site 0 is the most significant axis
    # of the reshaped state, so that basis index is 2**(k-1).
    x0 = jnp.zeros((2**k,)).at[2 ** (k - 1)].set(1.0)

    @jax.jit
    def run(theta_vector):
        thetas = [theta_vector[i : i + 1] for i in range(k - 1)]
        return sim.expval_all(sim.build_circuit(circuit, thetas), x0)

    _categorical_jit_cache[k] = run
    return run


def tension_class_pdit(scores: Sequence[float]) -> PDit:
    """Categorical over ``tension_gradient.classes``."""
    return evaluate_categorical(scores, TENSION_CLASSES, label="tension_class")


def bridge_candidate_pdit(scores: Sequence[float]) -> PDit:
    """Categorical over ``bridge_engine.ordered_strategies``."""
    return evaluate_categorical(scores, BRIDGE_STRATEGIES, label="bridge_candidate")
