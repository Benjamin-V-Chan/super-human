"""Simplified accelerated-life estimate from a joint torque series.

This is the v0 of `fatigue/model.py` from STRESS_TEST_PLAN.md — the honest,
closed-form slice used by the unified demo. It does NOT yet run FEA or a Kt
surrogate; it uses a nominal section-modulus stress with a fixed stress-
concentration factor. The full plan replaces `KT_FILLET` with an FEA-calibrated
surrogate. Treat the output as a *sim estimate*, not a durability guarantee.

Pipeline: torque(t) -> nominal bending stress sigma = M / Z at the joint
cross-section -> local stress = Kt * sigma -> one ADL actuation == one stress
cycle of amplitude sigma_a -> Basquin S-N gives cycles-to-failure N -> divide by
the usage rate to get years. Below the material endurance limit -> "infinite"
life (returned as math.inf, displayed as ">= target").

The material is a swappable `fatigue.materials.Material` (default PA12-CF FDM,
which reproduces the original hard-coded constants). The shared `section_modulus`
/ `cycles_to_failure` helpers are reused by `fatigue.recommend` so a proposed
hardware change is scored through the *exact same* math as the baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from prosthesis_rl.contracts import DesignParams
from prosthesis_rl.fatigue.materials import DEFAULT_MATERIAL, Material, get_material

# Geometry / stress-concentration defaults (material lives in fatigue.materials).
KT_FILLET = 1.8             # stress-concentration factor at the joint fillet (placeholder)
JOINT_RADIUS_M = 0.011      # effective solid radius of the loaded joint section

# Back-compat aliases: the baseline material's constants, kept so any external
# reference to the old module constants still resolves to the same numbers.
SIGMA_F_PRIME_PA = DEFAULT_MATERIAL.sigma_f_prime_pa
BASQUIN_B = DEFAULT_MATERIAL.basquin_b
ENDURANCE_LIMIT_PA = DEFAULT_MATERIAL.endurance_limit_pa
FDM_KNOCKDOWN = DEFAULT_MATERIAL.process_knockdown


@dataclass
class LifespanEstimate:
    predicted_years: float            # math.inf when below the endurance limit
    peak_stress_mpa: float            # local peak stress at the fillet
    amplitude_mpa: float              # stress amplitude used for S-N
    cycles_to_failure: float
    usage_cycles_per_day: int
    below_endurance_limit: bool
    material: str = "PA12-CF"

    @property
    def display_years(self) -> str:
        return display_years(self.predicted_years)


def display_years(years: float) -> str:
    """Human-readable service-life string (shared by estimate + recommender)."""
    if math.isinf(years):
        return ">=100 yr (below fatigue limit)"
    if years >= 100:
        return ">=100 yr"
    if years >= 10:
        return f"{years:.0f} yr"
    if years >= 1:
        return f"{years:.1f} yr"
    if years * 12 >= 1:
        return f"{years * 12:.1f} mo"
    return f"{years * 365:.0f} days"


def section_modulus(radius_m: float) -> float:
    """Elastic section modulus of a solid circular cross-section: Z = pi r^3 / 4."""
    return math.pi * radius_m ** 3 / 4.0


# Back-compat private alias (was `_section_modulus`).
_section_modulus = section_modulus


def cycles_to_failure(sigma_amp_pa: float, material: Material = DEFAULT_MATERIAL) -> float:
    """Basquin cycles-to-failure for a stress amplitude; math.inf below the limit.

    Basquin: sigma_a = sigma_f' (2N)^b  ->  N = 0.5 (sigma_a / sigma_f')^(1/b),
    where sigma_f' already carries the manufacturing knockdown. This is the single
    place the S-N curve is evaluated, so baseline and recommender stay consistent.
    """
    if sigma_amp_pa <= material.endurance_limit_pa or sigma_amp_pa <= 0:
        return math.inf
    return 0.5 * (sigma_amp_pa / material.effective_sigma_f_pa) ** (1.0 / material.basquin_b)


def years_from_amplitude(
    sigma_amp_pa: float,
    *,
    material: Material = DEFAULT_MATERIAL,
    usage_cycles_per_day: int = 300,
) -> float:
    """Service-life years for a stress amplitude (one ADL motion == one cycle)."""
    n = cycles_to_failure(sigma_amp_pa, material)
    if math.isinf(n):
        return math.inf
    return n / (usage_cycles_per_day * 365.0)


def estimate_lifespan(
    torque_series: np.ndarray,
    design: DesignParams,
    *,
    usage_cycles_per_day: int = 300,
    kt: float = KT_FILLET,
    radius_m: float = JOINT_RADIUS_M,
    material: str | Material = DEFAULT_MATERIAL,
) -> LifespanEstimate:
    """Estimate service life from the most-loaded joint's torque over one ADL motion.

    torque_series: 1-D array of torque (N*m) for the critical joint over the rollout.
    material: a `fatigue.materials.Material` or its key (default PA12-CF FDM).
    """
    mat = get_material(material)
    tau = np.abs(np.asarray(torque_series, dtype=float))
    peak_moment = float(tau.max()) if tau.size else 0.0
    span = float(tau.max() - tau.min()) if tau.size else 0.0

    z = section_modulus(radius_m)
    sigma_local_peak = kt * peak_moment / z                 # Pa
    sigma_amp = 0.5 * kt * span / z                          # Pa (range/2)

    n_cycles = cycles_to_failure(sigma_amp, mat)
    years = (math.inf if math.isinf(n_cycles)
             else n_cycles / (usage_cycles_per_day * 365.0))

    return LifespanEstimate(
        predicted_years=years,
        peak_stress_mpa=sigma_local_peak / 1e6,
        amplitude_mpa=sigma_amp / 1e6,
        cycles_to_failure=n_cycles,
        usage_cycles_per_day=usage_cycles_per_day,
        below_endurance_limit=math.isinf(n_cycles),
        material=mat.name,
    )
