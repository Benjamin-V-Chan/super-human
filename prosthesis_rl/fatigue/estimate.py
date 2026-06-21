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
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from prosthesis_rl.contracts import DesignParams

# --- Material: Nylon PA12-CF (STRESS_TEST_PLAN default), with an FDM knockdown.
# Rough literature-order values; the real DB lives in fatigue/materials.py.
SIGMA_F_PRIME_PA = 90e6      # fatigue strength coefficient (Pa)
BASQUIN_B = -0.10            # fatigue strength exponent
ENDURANCE_LIMIT_PA = 18e6   # below this amplitude -> effectively infinite life
FDM_KNOCKDOWN = 0.60        # inter-layer adhesion penalty for FDM prints
KT_FILLET = 1.8             # stress-concentration factor at the joint fillet (placeholder)

JOINT_RADIUS_M = 0.011      # effective solid radius of the loaded joint section


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
        if math.isinf(self.predicted_years):
            return ">=100 yr (below fatigue limit)"
        if self.predicted_years >= 100:
            return ">=100 yr"
        return f"{self.predicted_years:.1f} yr"


def _section_modulus(radius_m: float) -> float:
    """Elastic section modulus of a solid circular cross-section: Z = pi r^3 / 4."""
    return math.pi * radius_m ** 3 / 4.0


def estimate_lifespan(
    torque_series: np.ndarray,
    design: DesignParams,
    *,
    usage_cycles_per_day: int = 300,
    kt: float = KT_FILLET,
    radius_m: float = JOINT_RADIUS_M,
) -> LifespanEstimate:
    """Estimate service life from the most-loaded joint's torque over one ADL motion.

    torque_series: 1-D array of torque (N*m) for the critical joint over the rollout.
    """
    tau = np.abs(np.asarray(torque_series, dtype=float))
    peak_moment = float(tau.max()) if tau.size else 0.0
    span = float(tau.max() - tau.min()) if tau.size else 0.0

    z = _section_modulus(radius_m)
    sigma_local_peak = kt * peak_moment / z                 # Pa
    sigma_amp = 0.5 * kt * span / z                          # Pa (range/2)

    sigma_f = SIGMA_F_PRIME_PA * FDM_KNOCKDOWN

    if sigma_amp <= ENDURANCE_LIMIT_PA or sigma_amp <= 0:
        return LifespanEstimate(
            predicted_years=math.inf,
            peak_stress_mpa=sigma_local_peak / 1e6,
            amplitude_mpa=sigma_amp / 1e6,
            cycles_to_failure=math.inf,
            usage_cycles_per_day=usage_cycles_per_day,
            below_endurance_limit=True,
        )

    # Basquin: sigma_a = sigma_f' (2N)^b  ->  N = 0.5 (sigma_a / sigma_f')^(1/b)
    n_cycles = 0.5 * (sigma_amp / sigma_f) ** (1.0 / BASQUIN_B)
    years = n_cycles / (usage_cycles_per_day * 365.0)

    return LifespanEstimate(
        predicted_years=years,
        peak_stress_mpa=sigma_local_peak / 1e6,
        amplitude_mpa=sigma_amp / 1e6,
        cycles_to_failure=n_cycles,
        usage_cycles_per_day=usage_cycles_per_day,
        below_endurance_limit=False,
    )
