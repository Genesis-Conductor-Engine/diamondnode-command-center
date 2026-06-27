#!/usr/bin/env python3
"""
Procedural Truth Verifier — double loopback protocol for Genesis Conductor.

Eight-step verification pipeline:
  1. plan          — k-graph plan vector from knowledge graph adjacency
  2. dispatch      — dual-path loopback probes (/q-mem/ vs /v1/)
  3. execute       — collect response traces and latency fingerprints
  4. eigen         — Laplacian eigen projection (JAX-ready; numpy fallback)
  5. rule30_vdf    — Rule 30 cellular automaton VDF seal
  6. thermo        — optional thermo attestation hook
  7. automaton     — emergence score vs plan isomorphism
  8. truth         — crystal_score aggregation and evt emission

Designed for Rψ pipeline and Tunnel-Through workflow integration.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.error import URLError
from urllib.request import Request, urlopen

try:
    import numpy as np

    _NUMPY = True
except ImportError:
    np = None  # type: ignore
    _NUMPY = False

# ── Constants ────────────────────────────────────────────────────────────────

SCHEMA_VERSION = "evt-1.0"
RECORD_TYPE = "procedural_truth"
CRYSTAL_TARGET = 0.85
RULE30_MASK = 0x7FFFFFFF


@dataclass
class LoopbackEndpoint:
    name: str
    url: str
    path: str = "/health"


@dataclass
class VerifierConfig:
    qmem_base: str = "http://127.0.0.1:8082"
    ollama_base: str = "http://127.0.0.1:11434"
    qmem_path: str = "/health"
    ollama_path: str = "/api/tags"
    rule30_steps: int = 64
    crystal_target: float = CRYSTAL_TARGET
    node_id: str = "diamondnode"
    agent_id: str = "procedural_truth_verifier"
    timeout_s: float = 5.0
    plan_nodes: Optional[List[str]] = None


@dataclass
class LoopbackTrace:
    endpoint: str
    url: str
    status: int
    latency_ms: float
    body_hash: str
    body_preview: str
    ok: bool


@dataclass
class VerificationResult:
    passed: bool
    crystal_score: Dict[str, Any]
    evt: Dict[str, Any]
    metrics: Dict[str, Any]


# ── Rule 30 VDF ──────────────────────────────────────────────────────────────

def rule30_step(state: int, width: int = 31) -> int:
    """Single Rule 30 CA step on a bit-packed row."""
    next_state = 0
    for i in range(width):
        left = (state >> ((i + 1) % width)) & 1
        center = (state >> i) & 1
        right = (state >> ((i - 1) % width)) & 1
        new_bit = left ^ (center | right)
        next_state |= (new_bit & 1) << i
    return next_state & RULE30_MASK


def rule30_vdf(seed: int, steps: int) -> Tuple[int, List[int]]:
    """Advance Rule 30 for `steps` iterations; return final state and trace tail."""
    state = seed & RULE30_MASK
    trace: List[int] = []
    for _ in range(steps):
        state = rule30_step(state)
        trace.append(state)
    return state, trace[-8:] if len(trace) >= 8 else trace


def seed_from_traces(traces: Sequence[LoopbackTrace]) -> int:
    """Derive Rule 30 seed from loopback trace hashes."""
    material = "|".join(t.body_hash for t in traces)
    digest = hashlib.sha256(material.encode()).hexdigest()
    return int(digest[:8], 16) & RULE30_MASK


# ── Graph / Eigen ────────────────────────────────────────────────────────────

def _default_plan_graph() -> Tuple[List[str], List[Tuple[int, int, float]]]:
    """Bootstrap k-graph plan: procedural truth pipeline topology."""
    nodes = [
        "plan", "dispatch", "execute", "eigen",
        "rule30_vdf", "thermo", "automaton", "truth",
    ]
    edges = [
        (0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0),
        (4, 5, 0.8), (5, 6, 0.8), (6, 7, 1.0),
        (0, 7, 0.3),  # double loopback closure
        (2, 6, 0.5),  # execute ↔ automaton cross-link
    ]
    return nodes, edges


def build_laplacian(
    n: int, edges: Sequence[Tuple[int, int, float]]
) -> Any:
    """Build normalized graph Laplacian."""
    if _NUMPY:
        adj = np.zeros((n, n), dtype=np.float64)
        for i, j, w in edges:
            adj[i, j] += w
            adj[j, i] += w
        deg = np.diag(adj.sum(axis=1))
        return deg - adj

    # Pure-Python fallback
    adj = [[0.0] * n for _ in range(n)]
    for i, j, w in edges:
        adj[i][j] += w
        adj[j][i] += w
    lap = [[0.0] * n for _ in range(n)]
    for i in range(n):
        deg = sum(adj[i])
        lap[i][i] = deg
        for j in range(n):
            lap[i][j] -= adj[i][j]
    return lap


def eigen_project(
    laplacian: Any, signal: Sequence[float], k: int = 3
) -> Dict[str, Any]:
    """Project signal onto top-k Laplacian eigenvectors."""
    n = len(signal)
    k = min(k, n - 1) if n > 1 else 1

    if _NUMPY:
        eigvals, eigvecs = np.linalg.eigh(laplacian)
        idx = np.argsort(eigvals)[:k]
        basis = eigvecs[:, idx]
        sig = np.array(signal, dtype=np.float64)
        coeffs = basis.T @ sig
        projected = basis @ coeffs
        delta = float(np.linalg.norm(sig - projected))
        return {
            "eigenvalues": [float(eigvals[i]) for i in idx],
            "coefficients": [float(c) for c in coeffs],
            "loopback_delta_eigen": delta,
            "projection_norm": float(np.linalg.norm(projected)),
        }

    # Power-iteration fallback for smallest eigenvector only
    v = [1.0 / math.sqrt(n)] * n
    for _ in range(32):
        w = [0.0] * n
        for i in range(n):
            for j in range(n):
                w[i] += laplacian[i][j] * v[j]
        norm = math.sqrt(sum(x * x for x in w)) or 1.0
        v = [x / norm for x in w]
    coeffs = [sum(v[i] * signal[i] for i in range(n))]
    delta = math.sqrt(sum((signal[i] - coeffs[0] * v[i]) ** 2 for i in range(n)))
    return {
        "eigenvalues": [0.0],
        "coefficients": coeffs,
        "loopback_delta_eigen": delta,
        "projection_norm": abs(coeffs[0]),
    }


# ── Loopback probes ──────────────────────────────────────────────────────────

def probe_endpoint(
    name: str, base: str, path: str, timeout_s: float
) -> LoopbackTrace:
    url = f"{base.rstrip('/')}{path}"
    t0 = time.perf_counter()
    status = 0
    body = b""
    ok = False
    try:
        req = Request(url, headers={"User-Agent": "procedural-truth-verifier/1.0"})
        with urlopen(req, timeout=timeout_s) as resp:
            status = resp.status
            body = resp.read(4096)
            ok = 200 <= status < 300
    except (URLError, OSError, TimeoutError) as exc:
        body = str(exc).encode()[:256]
        status = 0
    latency_ms = (time.perf_counter() - t0) * 1000.0
    body_hash = hashlib.sha256(body).hexdigest()[:16]
    preview = body[:120].decode("utf-8", errors="replace")
    return LoopbackTrace(
        endpoint=name,
        url=url,
        status=status,
        latency_ms=round(latency_ms, 2),
        body_hash=body_hash,
        body_preview=preview,
        ok=ok,
    )


def dispatch_loopback(config: VerifierConfig) -> List[LoopbackTrace]:
    return [
        probe_endpoint("q-mem", config.qmem_base, config.qmem_path, config.timeout_s),
        probe_endpoint("ollama_v1", config.ollama_base, config.ollama_path, config.timeout_s),
    ]


# ── Automaton emergence ──────────────────────────────────────────────────────

def automaton_emergence_score(
    plan_vector: Sequence[float],
    trace_vector: Sequence[float],
    vdf_final: int,
) -> float:
    """Isomorphism score between plan and observed trace + VDF seal."""
    if not plan_vector or not trace_vector:
        return 0.0
    n = min(len(plan_vector), len(trace_vector))
    dot = sum(plan_vector[i] * trace_vector[i] for i in range(n))
    norm_p = math.sqrt(sum(x * x for x in plan_vector[:n])) or 1.0
    norm_t = math.sqrt(sum(x * x for x in trace_vector[:n])) or 1.0
    cosine = dot / (norm_p * norm_t)
    vdf_factor = (vdf_final & 0xFFFF) / 65535.0
    return max(0.0, min(1.0, 0.7 * cosine + 0.3 * vdf_factor))


def trace_to_signal(traces: Sequence[LoopbackTrace], n: int) -> List[float]:
    """Encode loopback traces as a fixed-length signal vector."""
    raw = []
    for t in traces:
        raw.append(1.0 if t.ok else 0.0)
        raw.append(min(t.latency_ms / 1000.0, 1.0))
        raw.append(int(t.body_hash[:4], 16) / 65535.0)
    while len(raw) < n:
        raw.append(0.0)
    return raw[:n]


# ── Crystal score ────────────────────────────────────────────────────────────

def compute_crystal_score(
    loopback_ok: bool,
    eigen_delta: float,
    emergence: float,
    target: float = CRYSTAL_TARGET,
) -> Dict[str, Any]:
    """Weighted crystal score; pass when composite >= target."""
    loopback_component = 1.0 if loopback_ok else 0.0
    eigen_component = max(0.0, 1.0 - min(eigen_delta, 1.0))
    weights = {"loopback": 0.35, "eigen": 0.30, "emergence": 0.35}
    value = (
        weights["loopback"] * loopback_component
        + weights["eigen"] * eigen_component
        + weights["emergence"] * emergence
    )
    passed = value >= target
    return {
        "metric": "procedural_truth_composite",
        "value": round(value, 4),
        "target": target,
        "passed": passed,
        "weight": 0.15,
        "components": {
            "loopback": round(loopback_component, 4),
            "eigen": round(eigen_component, 4),
            "emergence": round(emergence, 4),
        },
    }


# ── Main verifier ────────────────────────────────────────────────────────────

class ProceduralTruthVerifier:
    """Double loopback procedural truth verification engine."""

    def __init__(self, config: Optional[VerifierConfig] = None):
        self.config = config or VerifierConfig()

    def verify(
        self,
        *,
        source: str = "manual",
        seismic_run_id: Optional[str] = None,
        thermo_evt: Optional[Dict[str, Any]] = None,
    ) -> VerificationResult:
        nodes, edges = _default_plan_graph()
        n = len(nodes)
        plan_signal = [1.0] * n  # uniform plan prior

        # Steps 1-3: plan, dispatch, execute
        traces = dispatch_loopback(self.config)
        trace_signal = trace_to_signal(traces, n)
        loopback_ok = all(t.ok for t in traces)

        # Step 4: eigen projection
        lap = build_laplacian(n, edges)
        eigen = eigen_project(lap, plan_signal, k=3)
        eigen_trace = eigen_project(lap, trace_signal, k=3)
        loopback_delta = abs(
            eigen["loopback_delta_eigen"] - eigen_trace["loopback_delta_eigen"]
        )

        # Step 5: Rule 30 VDF seal
        seed = seed_from_traces(traces)
        vdf_final, vdf_tail = rule30_vdf(seed, self.config.rule30_steps)

        # Step 7: automaton emergence
        emergence = automaton_emergence_score(plan_signal, trace_signal, vdf_final)

        # Step 6: thermo hook (optional)
        thermo_eff = None
        if thermo_evt:
            thermo_eff = thermo_evt.get("landauer", {}).get("efficiency")

        # Step 8: truth / crystal
        crystal = compute_crystal_score(
            loopback_ok, loopback_delta, emergence, self.config.crystal_target
        )
        if thermo_eff is not None and thermo_eff > 0:
            crystal["components"]["thermo"] = round(min(thermo_eff, 1.0), 4)

        evt_id = f"evt_procedural_truth_{uuid.uuid4().hex[:8]}"
        ts = datetime.now(timezone.utc).isoformat()

        metrics = {
            "loopback_traces": [
                {
                    "endpoint": t.endpoint,
                    "url": t.url,
                    "status": t.status,
                    "latency_ms": t.latency_ms,
                    "body_hash": t.body_hash,
                    "ok": t.ok,
                }
                for t in traces
            ],
            "eigen_plan": eigen,
            "eigen_trace": eigen_trace,
            "loopback_delta_eigen": round(loopback_delta, 6),
            "rule30_seed": seed,
            "rule30_final": vdf_final,
            "rule30_tail": vdf_tail,
            "automaton_emergence": round(emergence, 4),
            "plan_nodes": nodes,
            "numpy_available": _NUMPY,
        }

        evt: Dict[str, Any] = {
            "evt_id": evt_id,
            "schema_version": SCHEMA_VERSION,
            "record_type": RECORD_TYPE,
            "timestamp": ts,
            "node_id": self.config.node_id,
            "agent_id": self.config.agent_id,
            "source": source,
            "seismic_run_id": seismic_run_id,
            "protocol": {
                "name": "double_loopback",
                "version": "1.0.0",
                "steps": [
                    "plan", "dispatch", "execute", "eigen",
                    "rule30_vdf", "thermo", "automaton", "truth",
                ],
            },
            "metrics": metrics,
            "crystal_score": crystal,
            "apex_goal": "procedural_truth : loopback ≅ plan → crystal ≥ 0.85",
            "provenance": {
                "pipeline": "procedural_truth_verifier_v1.0",
                "qmem_base": self.config.qmem_base,
                "ollama_base": self.config.ollama_base,
            },
        }
        if thermo_evt:
            evt["thermo_attestation"] = {
                "evt_id": thermo_evt.get("evt_id"),
                "efficiency": thermo_eff,
            }

        return VerificationResult(
            passed=crystal["passed"],
            crystal_score=crystal,
            evt=evt,
            metrics=metrics,
        )


def verify_procedural_truth(
    config: Optional[VerifierConfig] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Convenience entry point — returns evt dict."""
    result = ProceduralTruthVerifier(config).verify(**kwargs)
    return result.evt


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Procedural Truth Verifier CLI")
    parser.add_argument("--qmem-base", default="http://127.0.0.1:8082")
    parser.add_argument("--ollama-base", default="http://127.0.0.1:11434")
    parser.add_argument("--source", default="cli")
    parser.add_argument("--out", default=None, help="Write evt JSON to file")
    parser.add_argument("--json", action="store_true", help="Print evt JSON to stdout")
    args = parser.parse_args()

    cfg = VerifierConfig(qmem_base=args.qmem_base, ollama_base=args.ollama_base)
    result = ProceduralTruthVerifier(cfg).verify(source=args.source)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result.evt, f, indent=2)

    if args.json or not args.out:
        print(json.dumps(result.evt, indent=2))

    raise SystemExit(0 if result.passed else 1)


if __name__ == "__main__":
    main()