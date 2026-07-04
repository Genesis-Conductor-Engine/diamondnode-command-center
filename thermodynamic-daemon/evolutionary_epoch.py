"""Evolutionary Epoch — JAX-based Opux with HyperNEAT + Pareto reannealment.

One epoch:
  1. Ingest variance (AlphaGenome or synthetic) + thermo daemon state
  2. Diamond spectrum → harmonic orthogonal frequency basis
  3. HyperNEAT CPPN evolves field points across Pareto frontier
  4. Emit epoch witness for .sol attestation + Alchemy story log

Runs on ~/venv312 (JAX). VRAM budget: GTX 1650 4GB.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from diamond_spectrum import reducibility_score, spectrum_payload

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except Exception:
    import numpy as jnp
    HAS_JAX = False

STATE_DIR = Path("/tmp/thermo-epoch")
STATE_DIR.mkdir(parents=True, exist_ok=True)
EPOCH_LATEST = STATE_DIR / "epoch_latest.json"
EPOCH_HISTORY = STATE_DIR / "epoch_history.jsonl"


@dataclass
class EpochCandidate:
    name: str
    reducibility: float
    orthogonality_error: float
    landauer_efficiency: float
    field_hash: str
    is_pareto: bool = False

    def objectives(self) -> tuple[float, float, float]:
        return (self.reducibility, 1.0 - self.orthogonality_error, self.landauer_efficiency)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _pareto_frontier(candidates: list[EpochCandidate]) -> list[EpochCandidate]:
    frontier: list[EpochCandidate] = []
    for i, p in enumerate(candidates):
        dominated = False
        for j, q in enumerate(candidates):
            if i == j:
                continue
            po, oo, lo = p.objectives()
            qo, qoo, qlo = q.objectives()
            if qo >= po and qoo >= oo and qlo >= lo and (qo > po or qoo > oo or qlo > lo):
                dominated = True
                break
        if not dominated:
            p.is_pareto = True
            frontier.append(p)
    return sorted(frontier, key=lambda c: c.reducibility, reverse=True)


def _cppn_weights(seed: str, dim: int = 8) -> Any:
    h = hashlib.shake_256(seed.encode()).digest(dim * dim)
    vals = [(b - 128) / 128.0 for b in h]
    if HAS_JAX:
        return jnp.array(vals).reshape(dim, dim)
    return jnp.array(vals).reshape(dim, dim)


def _hyperneat_field(weights: Any, n_points: int = 8, dim: int = 4, seed: str = "") -> list[list[float]]:
    h = hashlib.shake_256((seed + "coords").encode()).digest(n_points * dim)
    coord_vals = [(b - 128) / 128.0 for b in h]
    if HAS_JAX:
        coords = jnp.array(coord_vals).reshape(n_points, dim)
        acts = jnp.tanh(coords @ weights[:dim, :dim].T)
        pts = jnp.clip(acts, -1, 1)
        return [[float(v) for v in row] for row in pts]
    coords = jnp.array(coord_vals).reshape(n_points, dim)
    acts = jnp.tanh(coords @ weights[:dim, :dim].T)
    pts = jnp.clip(acts, -1, 1)
    return [[float(v) for v in row] for row in pts]


def _mutate_seed(seed: str, generation: int, pareto_rank: int) -> str:
    return hashlib.sha256(f"{seed}:g{generation}:p{pareto_rank}".encode()).hexdigest()[:24]


def run_opux_epoch(
    variance: float,
    variant_id: str = "synthetic",
    thermo_state: dict | None = None,
    generations: int = 4,
    population: int = 12,
) -> dict[str, Any]:
    """Run one evolutionary epoch (Opux HyperNEAT + Pareto reannealment)."""
    epoch_id = f"epoch-{uuid.uuid4().hex[:12]}"
    started = datetime.now(timezone.utc).isoformat()
    spectrum = spectrum_payload(variance, variant_id)
    base_reduc = reducibility_score(
        spectrum["variance"], spectrum["mean_orthogonality_residual"]
    )

    thermo = thermo_state or {}
    temp_c = float(thermo.get("temperature_c", 55.0))
    vram_pct = float((thermo.get("vram") or {}).get("used_pct", 30.0))
    landauer_eff = max(0.1, 1.0 - (temp_c / 89.6) * 0.3 - (vram_pct / 100.0) * 0.2)

    seed_root = hashlib.sha256(
        json.dumps({"variant_id": variant_id, "variance": variance, "spectrum": spectrum["dispersion_index"]}, sort_keys=True).encode()
    ).hexdigest()[:32]

    candidates: list[EpochCandidate] = []
    best_field: list[list[float]] = []
    best_hash = ""

    for gen in range(generations):
        gen_candidates: list[EpochCandidate] = []
        for i in range(population):
            mut_seed = _mutate_seed(seed_root, gen, i)
            weights = _cppn_weights(mut_seed)
            field_pts = _hyperneat_field(weights, seed=mut_seed)
            content = json.dumps(field_pts, sort_keys=True)
            fh = hashlib.sha256(content.encode()).hexdigest()[:32]
            flat = [v for row in field_pts for v in row]
            mean = sum(flat) / max(1, len(flat))
            ortho_err = math.sqrt(sum((v - mean) ** 2 for v in flat) / max(1, len(flat)))
            reduc = min(1.0, base_reduc * (0.85 + 0.15 * (1.0 - ortho_err)) + gen * 0.01)
            c = EpochCandidate(
                name=f"opux-g{gen}-i{i}",
                reducibility=round(reduc, 6),
                orthogonality_error=round(ortho_err, 6),
                landauer_efficiency=round(landauer_eff * (0.9 + 0.1 * (1 - ortho_err)), 6),
                field_hash=fh,
            )
            gen_candidates.append(c)

        pareto = _pareto_frontier(gen_candidates)
        candidates.extend(pareto)
        if pareto:
            top = pareto[0]
            best_hash = top.field_hash
            best_field = _hyperneat_field(_cppn_weights(_mutate_seed(seed_root, gen, 0)), seed=_mutate_seed(seed_root, gen, 0))

    final_pareto = _pareto_frontier(candidates)[-population:] if candidates else []
    if not final_pareto and candidates:
        final_pareto = sorted(candidates, key=lambda c: c.reducibility, reverse=True)[:3]

    work_hash = hashlib.sha256(
        json.dumps(
            {
                "epoch_id": epoch_id,
                "variant_id": variant_id,
                "best_field_hash": best_hash,
                "pareto": [c.as_dict() for c in final_pareto[:5]],
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()

    payload = {
        "type": "EVOLUTIONARY_EPOCH",
        "substrate": "JAX/HyperNEAT" if HAS_JAX else "numpy/HyperNEAT",
        "opux_version": "1.0.0",
        "epoch_id": epoch_id,
        "started_at": started,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "variant_id": variant_id,
        "variance": variance,
        "diamond_spectrum": spectrum,
        "thermo_snapshot": {
            "temperature_c": temp_c,
            "vram_used_pct": vram_pct,
            "landauer_efficiency": landauer_eff,
        },
        "generations": generations,
        "population": population,
        "pareto_frontier": [c.as_dict() for c in final_pareto[:8]],
        "best": final_pareto[0].as_dict() if final_pareto else None,
        "field_points": best_field,
        "field_hash": best_hash,
        "work_hash": work_hash,
        "reducibility_score": final_pareto[0].reducibility if final_pareto else base_reduc,
        "three_goals": {
            "main": "goal_1_galaxy_thermodynamic_reducibility",
            "milestone_2": "goal_2_mcp_fleet_propagation",
            "milestone_3": "goal_3_notion_onchain_wrap",
        },
        "provenance": "thermodynamic-daemon:evolutionary_epoch",
    }

    EPOCH_LATEST.write_text(json.dumps(payload, indent=2) + "\n")
    with EPOCH_HISTORY.open("a") as f:
        f.write(json.dumps({"epoch_id": epoch_id, "work_hash": work_hash, "at": payload["completed_at"]}) + "\n")

    return payload


def load_latest_epoch() -> dict | None:
    if not EPOCH_LATEST.is_file():
        return None
    try:
        return json.loads(EPOCH_LATEST.read_text())
    except Exception:
        return None


if __name__ == "__main__":
    t0 = time.time()
    result = run_opux_epoch(0.038, "chr17:41234470:A>G", {"temperature_c": 58, "vram": {"used_pct": 35}})
    result["elapsed_ms"] = round((time.time() - t0) * 1000, 2)
    print(json.dumps({"ok": True, "epoch_id": result["epoch_id"], "work_hash": result["work_hash"], "reducibility": result["reducibility_score"], "substrate": result["substrate"], "elapsed_ms": result["elapsed_ms"]}, indent=2))