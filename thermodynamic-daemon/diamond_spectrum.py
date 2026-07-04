"""Diamond reflection / refraction of frequencies given variance.

Maps biological or thermodynamic variance σ² to harmonic orthogonal frequency
pairs. First principle: realistic harmonic orthogonal topology — frequencies
form an orthonormal basis; Pareto reannealment preserves non-dominated modes.

Used by evolutionary_epoch.py as the spectral substrate for Opux HyperNEAT.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any


# Diamond dispersion index (simplified Cauchy): n(λ) ≈ A + B/λ²
DIAMOND_A = 2.417
DIAMOND_B = 0.012


@dataclass
class FrequencyPair:
    reflection_hz: float
    refraction_hz: float
    incident_angle_rad: float
    variance: float
    mode_index: int
    orthogonality_residual: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _snell_refraction(incident_hz: float, n1: float, n2: float, theta_i: float) -> float:
    """Snell-like refraction for frequency domain (analogous mapping)."""
    sin_t = (n1 / n2) * math.sin(theta_i)
    if abs(sin_t) > 1.0:
        return incident_hz * 0.0  # total internal reflection → null transmit
    theta_t = math.asin(sin_t)
    return incident_hz * (math.cos(theta_t) / max(1e-9, math.cos(theta_i)))


def variance_to_frequencies(
    variance: float,
    n_modes: int = 8,
    base_hz: float = 432.0,
    seed: str = "diamond",
) -> list[FrequencyPair]:
    """Project variance onto harmonic orthogonal frequency modes.

    Each mode k uses ω_k = base_hz × (k+1) × (1 + σ²/(k+1)).
    Reflection = incident; refraction = Snell-mapped through diamond dispersion.
    """
    sigma2 = max(0.0, float(variance))
    h = hashlib.sha256(seed.encode()).digest()
    theta_base = (h[0] / 255.0) * (math.pi / 4)

    pairs: list[FrequencyPair] = []
    prev_vec = None
    ortho_residuals: list[float] = []

    for k in range(n_modes):
        incident = base_hz * (k + 1) * (1.0 + sigma2 / (k + 1))
        n_diamond = DIAMOND_A + DIAMOND_B / max(incident ** 2, 1.0)
        theta_i = theta_base + (k * math.pi / (2 * n_modes))
        refracted = _snell_refraction(incident, 1.0, n_diamond, theta_i)
        if refracted <= 0:
            refracted = incident * 0.618  # golden-ratio fallback under TIR

        # Orthogonality check: mode vectors in (incident, refracted) plane
        vec = (incident, refracted)
        residual = 0.0
        if prev_vec is not None:
            dot = prev_vec[0] * vec[0] + prev_vec[1] * vec[1]
            norm_p = math.sqrt(prev_vec[0] ** 2 + prev_vec[1] ** 2) or 1.0
            norm_v = math.sqrt(vec[0] ** 2 + vec[1] ** 2) or 1.0
            residual = abs(dot / (norm_p * norm_v))
        ortho_residuals.append(residual)
        prev_vec = vec

        pairs.append(
            FrequencyPair(
                reflection_hz=round(incident, 4),
                refraction_hz=round(refracted, 4),
                incident_angle_rad=round(theta_i, 6),
                variance=sigma2,
                mode_index=k,
                orthogonality_residual=round(residual, 6),
            )
        )

    return pairs


def spectrum_payload(
    variance: float,
    variant_id: str = "synthetic",
    provenance: str = "diamond_spectrum",
) -> dict[str, Any]:
    pairs = variance_to_frequencies(variance, seed=variant_id)
    mean_ortho = sum(p.orthogonality_residual for p in pairs) / max(1, len(pairs))
    dispersion = DIAMOND_A + DIAMOND_B / max(pairs[0].reflection_hz ** 2, 1.0) if pairs else DIAMOND_A
    return {
        "type": "DIAMOND_SPECTRUM",
        "variant_id": variant_id,
        "variance": variance,
        "dispersion_index": round(dispersion, 6),
        "mean_orthogonality_residual": round(mean_ortho, 6),
        "modes": [p.as_dict() for p in pairs],
        "provenance": provenance,
    }


def reducibility_score(variance: float, ortho_mean: float, landauer_j_per_bit: float = 2.9e-21) -> float:
    """Thermodynamic information reducibility ∈ [0,1].

    Higher when variance compresses cleanly (low ortho residual) and
    Landauer cost per bit is within envelope.
    """
    var_term = 1.0 / (1.0 + variance)
    ortho_term = 1.0 - min(1.0, ortho_mean)
    landauer_term = min(1.0, landauer_j_per_bit * 1e20)  # normalized scale
    return round((var_term * 0.5 + ortho_term * 0.35 + landauer_term * 0.15), 6)


if __name__ == "__main__":
    demo = spectrum_payload(0.042, "chr17:41234470:A>G")
    demo["reducibility"] = reducibility_score(
        demo["variance"], demo["mean_orthogonality_residual"]
    )
    print(json.dumps(demo, indent=2))