"""Group-intent consensus as a thrml Ising energy-based model.

``individual_and_group_intent.group_intent`` says
``aggregation: robust-constraint-aware-consensus`` and, explicitly,
``simple_average_forbidden: true``. This module is why that clause is
enforceable rather than aspirational.

**The model.** One spin per *ordinary preference proposition*. Member ``i``
holds a position ``p_ij`` in ``[-1, 1]`` on proposition ``j`` with weight
``w_i``:

    h_j  = sum_i w_i p_ij                        (weighted support)
    J_jk = sum_i w_i p_ij p_jk / sum_i w_i       (co-movement across members)
    H(s) = -sum_j h_j s_j  -  sum_{j<k} J_jk s_j s_k

The group position is the vector of magnetisations ``m_j = <s_j>`` under the
Boltzmann distribution at inverse temperature ``beta``, sampled with thrml's
block-Gibbs sampler over a greedy graph colouring (the same block construction
the node's existing ``thrml_ebm_sampler`` uses).

**Why this is not an average.** With ``J = 0`` the marginals reduce to
``tanh(beta h_j)`` — a monotone rescaling of the weighted mean, i.e. the average
answer. The coupling term is what makes it different: it rewards *coherent
packages*. When members who back proposition A also back proposition B, the EBM
will not hand back a majority for A together with a majority against B if no
actual coalition holds that combination. A per-proposition average happily
returns exactly that incoherent bundle, and a group asked to act on it would
find no member who agrees with the "consensus".
``tests/kernels/test_thrml_consensus.py::test_coupling_beats_per_item_average``
exhibits a concrete such case.

**Hard constraints never enter the EBM.** They are a non-overridable union
applied by the sovereignty guard at precedence level 1. Sampling them would make
them negotiable — a sufficiently strong coupling could flip one — which is
precisely what ``cannot-be-overridden`` forbids. Only ordinary preferences are
sampled here.

**Dissent is an output, not a residual.** Every member whose own position
opposes the sampled consensus is recorded with the magnitude of the
disagreement, satisfying ``preserve_minority_positions`` and
``dissent: preserved-as-first-class-state``.
"""

from __future__ import annotations

import itertools
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .energy_gate import GateDecision, energy_gate

# Sampling defaults. Small by design: ``state_bounds.group_member_count_soft_limit``
# is 256 members and propositions are the *ordinary preferences under discussion*,
# not the whole catalogue, so these graphs are tens of nodes, not thousands.
DEFAULT_BETA = 1.0
DEFAULT_SAMPLES = 256
DEFAULT_WARMUP = 128
DEFAULT_STEPS_PER_SAMPLE = 2
DEFAULT_SEED = 7

# Above this many propositions the exact fallback stops enumerating and uses
# mean-field iteration instead. 2**20 states is about a second of pure Python;
# beyond that the sampler (or mean field) is the right tool anyway.
EXACT_ENUMERATION_LIMIT = 20

_thrml_state: dict[str, Any] = {"loaded": False, "ok": False, "error": None}


def _load_thrml() -> dict[str, Any]:
    if _thrml_state["loaded"]:
        return _thrml_state
    _thrml_state["loaded"] = True
    try:
        import jax
        import jax.numpy as jnp
        from thrml import Block, SamplingSchedule, SpinNode, sample_states
        from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init

        _thrml_state.update(
            ok=True,
            jax=jax,
            jnp=jnp,
            Block=Block,
            SamplingSchedule=SamplingSchedule,
            SpinNode=SpinNode,
            sample_states=sample_states,
            IsingEBM=IsingEBM,
            IsingSamplingProgram=IsingSamplingProgram,
            hinton_init=hinton_init,
            backend=jax.default_backend(),
        )
    except Exception as exc:
        _thrml_state.update(ok=False, error=f"{type(exc).__name__}: {exc}")
    return _thrml_state


def thrml_available() -> bool:
    if os.environ.get("THRML_FORCE_FALLBACK") == "1":
        return False
    return bool(_load_thrml()["ok"])


def engine_label() -> str:
    """``thrml-<version>/jax-<backend>``, matching the node's HUD contract."""
    st = _load_thrml()
    if not st["ok"] or os.environ.get("THRML_FORCE_FALLBACK") == "1":
        return "thrml-fallback/python-exact"
    try:
        from importlib.metadata import version

        ver = version("thrml")
    except Exception:
        ver = "unknown"
    return f"thrml-{ver}/jax-{st.get('backend', 'cpu')}"


# --------------------------------------------------------------------------
# inputs / outputs
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MemberPosition:
    """One member's stance on the ordinary preferences under discussion.

    ``positions`` maps proposition id -> stance in ``[-1, 1]``; 0 means "no
    stated position", which is distinct from "neutral after argument" only in
    ``participation`` (used for the participation-imbalance dimension).
    """

    member_id: str
    positions: Mapping[str, float]
    weight: float = 1.0
    participation: float = 1.0

    def __post_init__(self) -> None:
        if self.weight < 0 or not math.isfinite(self.weight):
            raise ValueError(
                f"member {self.member_id!r}: weight must be finite and >= 0, "
                f"got {self.weight}"
            )
        for prop, v in self.positions.items():
            if not math.isfinite(v) or not -1.0 <= v <= 1.0:
                raise ValueError(
                    f"member {self.member_id!r}: position on {prop!r} must be "
                    f"in [-1, 1], got {v}"
                )


@dataclass(frozen=True, slots=True)
class DissentRecord:
    """A preserved minority position — first-class stored state."""

    member_id: str
    proposition: str
    member_position: float
    group_position: float
    magnitude: float

    def to_json(self) -> dict[str, Any]:
        return {
            "member_id": self.member_id,
            "proposition": self.proposition,
            "member_position": round(self.member_position, 9),
            "group_position": round(self.group_position, 9),
            "magnitude": round(self.magnitude, 9),
        }


@dataclass(frozen=True, slots=True)
class ConsensusResult:
    """Output of the EBM aggregation."""

    propositions: tuple[str, ...]
    magnetization: tuple[float, ...]
    decision: tuple[int, ...]
    confidence: tuple[float, ...]
    dissent: tuple[DissentRecord, ...]
    mean_energy: float
    backend: str
    gate: GateDecision | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def position(self, proposition: str) -> float:
        return self.magnetization[self.propositions.index(proposition)]

    def as_mapping(self) -> dict[str, float]:
        return dict(zip(self.propositions, self.magnetization))

    @property
    def overall_confidence(self) -> float:
        """Evidence-weighted confidence, per ``components.confidence``.

        The mean of ``|m_j|``: a group that is decisive on every proposition
        scores 1, and one sitting near the decision boundary scores near 0. A
        low value is what stops a bridge from being applied on a coin flip.
        """
        if not self.confidence:
            return 0.0
        return round(math.fsum(self.confidence) / len(self.confidence), 9)

    def to_json(self) -> dict[str, Any]:
        return {
            "propositions": list(self.propositions),
            "magnetization": [round(v, 9) for v in self.magnetization],
            "decision": list(self.decision),
            "confidence": [round(v, 9) for v in self.confidence],
            "overall_confidence": self.overall_confidence,
            "dissent": [d.to_json() for d in self.dissent],
            "mean_energy": round(self.mean_energy, 9),
            "backend": self.backend,
            "gate": self.gate.to_json() if self.gate else None,
            "diagnostics": dict(self.diagnostics),
        }


# --------------------------------------------------------------------------
# model construction
# --------------------------------------------------------------------------


def build_ising(
    members: Sequence[MemberPosition], propositions: Sequence[str]
) -> tuple[list[float], list[tuple[int, int, float]]]:
    """Return ``(biases, edges)`` for the consensus Ising model.

    Weights are normalised so the energy scale does not depend on how many
    members happen to be present — otherwise a large group would sample
    effectively colder (sharper, more overconfident) than a small one purely
    because of headcount.
    """
    n = len(propositions)
    index = {p: i for i, p in enumerate(propositions)}
    total_w = math.fsum(m.weight for m in members)
    if total_w <= 0:
        raise ValueError("member weights sum to zero; cannot aggregate")

    biases = [0.0] * n
    for m in members:
        for prop, val in m.positions.items():
            if prop in index:
                biases[index[prop]] += m.weight * val / total_w

    edges: list[tuple[int, int, float]] = []
    for a, b in itertools.combinations(range(n), 2):
        pa, pb = propositions[a], propositions[b]
        coupling = math.fsum(
            m.weight * m.positions.get(pa, 0.0) * m.positions.get(pb, 0.0)
            for m in members
        ) / total_w
        if abs(coupling) > 1e-12:
            edges.append((a, b, coupling))
    return biases, edges


def greedy_coloring(n_nodes: int, edges: Sequence[tuple[int, int, float]]) -> list[list[int]]:
    """Colour classes with no intra-class edge — valid block-Gibbs blocks.

    Same construction as the node's existing thrml sampler: spins inside one
    colour class are conditionally independent given the others, so a whole
    class can be resampled in one vectorised step.
    """
    adj: list[set[int]] = [set() for _ in range(n_nodes)]
    for i, j, _ in edges:
        adj[i].add(j)
        adj[j].add(i)
    color = [-1] * n_nodes
    for v in range(n_nodes):
        used = {color[u] for u in adj[v] if color[u] >= 0}
        c = 0
        while c in used:
            c += 1
        color[v] = c
    classes: dict[int, list[int]] = {}
    for v, c in enumerate(color):
        classes.setdefault(c, []).append(v)
    return [classes[c] for c in sorted(classes)]


def ising_energy(
    spins: Sequence[int], biases: Sequence[float], edges: Sequence[tuple[int, int, float]]
) -> float:
    """``H(s) = -sum_j h_j s_j - sum_{j<k} J_jk s_j s_k``."""
    e = -math.fsum(b * s for b, s in zip(biases, spins))
    e -= math.fsum(w * spins[i] * spins[j] for i, j, w in edges)
    return e


# --------------------------------------------------------------------------
# samplers
# --------------------------------------------------------------------------


def _sample_thrml(
    biases: Sequence[float],
    edges: Sequence[tuple[int, int, float]],
    *,
    beta: float,
    samples: int,
    warmup: int,
    steps_per_sample: int,
    seed: int,
) -> tuple[list[float], float, str, dict[str, Any]]:
    st = _load_thrml()
    jax, jnp = st["jax"], st["jnp"]
    Block, SpinNode = st["Block"], st["SpinNode"]
    SamplingSchedule, sample_states = st["SamplingSchedule"], st["sample_states"]
    IsingEBM, IsingSamplingProgram = st["IsingEBM"], st["IsingSamplingProgram"]

    n = len(biases)
    nodes = [SpinNode() for _ in range(n)]
    edge_pairs = [(nodes[i], nodes[j]) for i, j, _ in edges]
    weights = jnp.asarray([w for _, _, w in edges]) if edges else jnp.zeros((0,))
    bias_arr = jnp.asarray(list(biases))
    model = IsingEBM(nodes, edge_pairs, bias_arr, weights, jnp.asarray(float(beta)))

    blocks = [Block([nodes[v] for v in cls]) for cls in greedy_coloring(n, edges)]
    program = IsingSamplingProgram(model, blocks, [])
    schedule = SamplingSchedule(warmup, samples, steps_per_sample)

    key = jax.random.key(seed)
    k_init, k_samp = jax.random.split(key)
    try:
        init = st["hinton_init"](k_init, model, blocks, ())
    except Exception:
        init = [
            jax.random.bernoulli(k_init, 0.5, (len(b.nodes),)) for b in blocks
        ]

    t0 = time.perf_counter()
    out = sample_states(k_samp, program, schedule, init, [], [Block(nodes)])
    spins = jnp.where(jnp.asarray(out[0]), 1.0, -1.0)  # [n_samples, n]
    elapsed = time.perf_counter() - t0

    magnetization = [float(v) for v in jnp.mean(spins, axis=0)]
    if edges:
        src = jnp.asarray([i for i, _, _ in edges])
        dst = jnp.asarray([j for _, j, _ in edges])
        coupling_energy = (spins[:, src] * spins[:, dst]) @ weights
    else:
        coupling_energy = jnp.zeros((spins.shape[0],))
    energy = -(coupling_energy + spins @ bias_arr)
    return (
        magnetization,
        float(jnp.mean(energy)),
        engine_label(),
        {
            "n_blocks": len(blocks),
            "n_samples": int(spins.shape[0]),
            "elapsed_s": round(elapsed, 4),
            "samples_per_s": round(spins.shape[0] / max(elapsed, 1e-9), 1),
        },
    )


def _sample_exact(
    biases: Sequence[float], edges: Sequence[tuple[int, int, float]], *, beta: float
) -> tuple[list[float], float, str, dict[str, Any]]:
    """Exact Boltzmann marginals by enumeration — the reference implementation.

    Used as the fallback for small models and as the oracle the sampler is
    checked against in tests. Exact rather than approximate, so a degraded run
    is not a *worse* answer, only a slower one.
    """
    n = len(biases)
    log_z_terms = []
    for mask in range(1 << n):
        spins = [1 if (mask >> k) & 1 else -1 for k in range(n)]
        log_z_terms.append(-beta * ising_energy(spins, biases, edges))
    peak = max(log_z_terms)
    weights = [math.exp(t - peak) for t in log_z_terms]
    z = math.fsum(weights)
    mag = [0.0] * n
    mean_energy = 0.0
    for mask, w in enumerate(weights):
        p = w / z
        spins = [1 if (mask >> k) & 1 else -1 for k in range(n)]
        for k in range(n):
            mag[k] += p * spins[k]
        mean_energy += p * ising_energy(spins, biases, edges)
    return mag, mean_energy, "thrml-fallback/python-exact", {"method": "enumeration"}


def _sample_mean_field(
    biases: Sequence[float],
    edges: Sequence[tuple[int, int, float]],
    *,
    beta: float,
    iterations: int = 200,
    damping: float = 0.5,
) -> tuple[list[float], float, str, dict[str, Any]]:
    """Damped mean-field marginals for models too large to enumerate.

    ``m_j = tanh(beta (h_j + sum_k J_jk m_k))``, damped to avoid the oscillation
    that undamped iteration shows on frustrated (negatively coupled) graphs.
    Approximate, and labelled as such in ``backend`` so an audit record never
    presents it as an exact result.
    """
    n = len(biases)
    adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for i, j, w in edges:
        adj[i].append((j, w))
        adj[j].append((i, w))
    m = [math.tanh(beta * b) for b in biases]
    for _ in range(iterations):
        delta = 0.0
        for j in range(n):
            field_j = biases[j] + math.fsum(w * m[k] for k, w in adj[j])
            new = math.tanh(beta * field_j)
            blended = damping * m[j] + (1.0 - damping) * new
            delta = max(delta, abs(blended - m[j]))
            m[j] = blended
        if delta < 1e-10:
            break
    energy = -math.fsum(b * mj for b, mj in zip(biases, m)) - math.fsum(
        w * m[i] * m[j] for i, j, w in edges
    )
    return m, energy, "thrml-fallback/python-meanfield", {"method": "mean-field"}


# --------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------


def aggregate_consensus(
    members: Sequence[MemberPosition],
    propositions: Sequence[str],
    *,
    beta: float = DEFAULT_BETA,
    samples: int = DEFAULT_SAMPLES,
    warmup: int = DEFAULT_WARMUP,
    steps_per_sample: int = DEFAULT_STEPS_PER_SAMPLE,
    seed: int = DEFAULT_SEED,
    dissent_threshold: float = 0.05,
    check_gate: bool = True,
) -> ConsensusResult:
    """Aggregate ordinary preferences into a group position.

    Raises on an empty member list rather than returning a neutral consensus: a
    group with no members has no intent, and returning "all zero" would let a
    downstream bridge act on a fabricated agreement.
    """
    if not members:
        raise ValueError("cannot aggregate consensus over zero members")
    props = tuple(propositions)
    if not props:
        raise ValueError("cannot aggregate consensus over zero propositions")
    if len(set(props)) != len(props):
        raise ValueError(f"propositions must be unique, got {props}")
    if beta <= 0:
        raise ValueError(f"beta must be > 0, got {beta}")

    biases, edges = build_ising(members, props)

    gate = energy_gate() if check_gate else None
    n = len(props)
    used_gpu_path = False
    if thrml_available() and (gate is None or gate.gpu_ok or n > EXACT_ENUMERATION_LIMIT):
        try:
            if gate is not None and not gate.gpu_ok:
                # Outside the governor envelope: run thrml on CPU rather than
                # not at all. JAX_PLATFORMS is read at import, so this only
                # takes effect for a process that has not initialised CUDA; the
                # sampler is small enough that CPU is the right answer anyway.
                os.environ.setdefault("JAX_PLATFORMS", "cpu")
            mag, energy, backend, diag = _sample_thrml(
                biases,
                edges,
                beta=beta,
                samples=samples,
                warmup=warmup,
                steps_per_sample=steps_per_sample,
                seed=seed,
            )
            used_gpu_path = True
        except Exception as exc:
            mag, energy, backend, diag = _fallback_sample(biases, edges, beta)
            diag = {**diag, "thrml_error": f"{type(exc).__name__}: {exc}"}
    else:
        mag, energy, backend, diag = _fallback_sample(biases, edges, beta)

    decision = tuple(1 if v >= 0 else -1 for v in mag)
    confidence = tuple(round(abs(v), 9) for v in mag)

    dissent: list[DissentRecord] = []
    for member in members:
        for k, prop in enumerate(props):
            stance = member.positions.get(prop, 0.0)
            if stance == 0.0:
                continue
            # Dissent is disagreement in *sign* against the group position, and
            # only when the group actually took one: near-zero magnetisation is
            # an undecided group, not a majority to dissent from.
            if abs(mag[k]) < dissent_threshold:
                continue
            if (stance > 0) != (mag[k] > 0):
                dissent.append(
                    DissentRecord(
                        member_id=member.member_id,
                        proposition=prop,
                        member_position=stance,
                        group_position=mag[k],
                        magnitude=abs(stance) * abs(mag[k]),
                    )
                )

    return ConsensusResult(
        propositions=props,
        magnetization=tuple(round(v, 9) for v in mag),
        decision=decision,
        confidence=confidence,
        dissent=tuple(dissent),
        mean_energy=energy,
        backend=backend,
        gate=gate,
        diagnostics={
            **diag,
            "n_propositions": n,
            "n_members": len(members),
            "n_edges": len(edges),
            "beta": beta,
            "sampled": used_gpu_path,
        },
    )


def _fallback_sample(biases, edges, beta):
    if len(biases) <= EXACT_ENUMERATION_LIMIT:
        return _sample_exact(biases, edges, beta=beta)
    return _sample_mean_field(biases, edges, beta=beta)


def weighted_average(
    members: Sequence[MemberPosition], propositions: Sequence[str]
) -> dict[str, float]:
    """The forbidden aggregation, implemented for comparison only.

    Kept in the module so the test that proves the EBM differs from it has a
    real reference, and so a reader can see exactly what
    ``simple_average_forbidden`` is ruling out. Nothing in the decision path
    calls this.
    """
    total_w = math.fsum(m.weight for m in members)
    if total_w <= 0:
        raise ValueError("member weights sum to zero")
    return {
        p: math.fsum(m.weight * m.positions.get(p, 0.0) for m in members) / total_w
        for p in propositions
    }
