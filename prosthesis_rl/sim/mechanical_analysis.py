"""Beam-theory mechanical analysis for prosthetic designs.

Computes per-component bending stress, safety factor, mass, and generates
wall-thickness suggestions. Uses hollow-cylinder beam theory — not FEA, but
gives first-order estimates suitable for design-iteration gating.

    report = MechanicalAnalysis.run(params, components, load_cases)
    print(report.worst_safety_factor, report.total_mass_g)
"""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any

from prosthesis_rl.contracts import ComponentSpec, DesignParams, MechanicalReport

__all__ = ["MechanicalAnalysis"]

# ── Material data ──────────────────────────────────────────────────────────────

_YIELD_MPa: dict[str, float] = {
    "Ti-6Al-4V": 880.0,
    "CFRP-PA12": 210.0,
    "PA12":      50.0,
    "PETG":      45.0,
}

_DENSITY_kg_m3: dict[str, float] = {
    "Ti-6Al-4V": 4430.0,
    "CFRP-PA12": 1450.0,
    "PA12":      1010.0,
    "PETG":      1270.0,
}

# Stress concentration factors at joint interfaces
_Kt: dict[str, float] = {
    "socket":          1.3,
    "socket_pylon":    1.5,
    "upper_pylon":     1.2,
    "elbow_unit":      2.0,   # hinge joint — highest concentration
    "forearm_pylon":   1.2,
    "wrist_flex_unit": 1.8,
    "wrist_rot_unit":  1.8,
    "terminal_device": 1.4,
    "td_vo_hook":      1.3,
    "td_pinch_prehensor": 1.4,
    "td_passive_hand": 1.2,
    "td_myoelectric_hand": 1.3,
}

# Fatigue constants (aluminum-proxy Basquin model for each material)
_FATIGUE_b: dict[str, float] = {
    "Ti-6Al-4V": -0.085,
    "CFRP-PA12": -0.10,
    "PA12":      -0.12,
    "PETG":      -0.13,
}
_FATIGUE_sigma_f: dict[str, float] = {
    "Ti-6Al-4V": 1200.0,
    "CFRP-PA12": 320.0,
    "PA12":      80.0,
    "PETG":      70.0,
}

_CYCLES_PER_DAY = 300.0  # ADL cycles


class MechanicalAnalysis:
    """Beam-theory mechanical analysis of a prosthetic arm design."""

    # Weight budgets (g)
    _WEIGHT_BUDGET = {
        "above_elbow": 900.0,
        "below_elbow": 650.0,
        "default": 900.0,
    }

    @classmethod
    def run(
        cls,
        params: DesignParams,
        components: list[ComponentSpec],
        load_cases: list[dict[str, Any]] | None = None,
    ) -> MechanicalReport:
        """Analyze design, return MechanicalReport.

        load_cases: list of {"name": str, "peak_torque_nm": float, "location": str}
        If omitted, load cases are estimated from link geometry (1 N/cm peak force).
        """
        if load_cases is None:
            load_cases = cls._estimate_load_cases(params)

        component_results: list[dict[str, Any]] = []
        total_mass_g = 0.0
        worst_sf = float("inf")
        peak_stress = 0.0
        min_life_years = float("inf")

        # Build component map by name for cross-reference
        comp_map = {c.name: c for c in components}

        # Distal mass accumulation (approximate, sum from tip inward)
        # This drives bending moment at each cross-section.
        total_weight_n = sum(c.mass_g * 9.81 / 1000.0 for c in components)

        cumulative_distal_weight_n = 0.0
        reversed_comps = list(reversed(components))

        for comp in reversed_comps:
            cumulative_distal_weight_n += comp.mass_g * 9.81 / 1000.0

        # Compute per-component stress (proximal → distal, bending at each section)
        cum_distal_n = total_weight_n  # start with full weight at root
        for comp in components:
            comp_mass_n = comp.mass_g * 9.81 / 1000.0
            cum_distal_n -= comp_mass_n / 2   # midpoint of this component

            length_m = comp.length_mm / 1000.0
            od_m = comp.outer_radius_mm / 1000.0
            wt_m = comp.wall_thickness_mm / 1000.0
            ro = od_m / 2
            ri = max(0.0, ro - wt_m)

            # Bending moment at proximal face from distal distributed load
            # Peak torque from load cases targeting this component
            peak_torque = 0.0
            for lc in load_cases:
                if lc.get("location", "") in comp.name or not lc.get("location"):
                    peak_torque = max(peak_torque, float(lc.get("peak_torque_nm", 0.0)))
            # Cantilever bending from distal weight
            M_Nm = max(peak_torque, cum_distal_n * length_m)

            # Section modulus for hollow cylinder
            if ro <= ri:
                Z_m3 = 1e-6  # degenerate — avoid div zero
            else:
                I = math.pi * (ro**4 - ri**4) / 4
                Z_m3 = I / ro if ro > 0 else 1e-6

            sigma_bending = M_Nm / Z_m3 / 1e6  # MPa

            # Stress concentration
            Kt = _Kt.get(comp.name, 1.5)
            von_mises_mpa = sigma_bending * Kt

            # Safety factor
            yield_mpa = _YIELD_MPa.get(comp.material, 50.0)
            sf = yield_mpa / max(von_mises_mpa, 0.01)

            # Fatigue life estimate (Basquin)
            sigma_f = _FATIGUE_sigma_f.get(comp.material, 80.0)
            b = _FATIGUE_b.get(comp.material, -0.12)
            if von_mises_mpa > 0:
                life_cycles = sigma_f / (von_mises_mpa * Kt) ** (1 / b) if b != 0 else 1e9
                life_years = life_cycles / (_CYCLES_PER_DAY * 365.0)
            else:
                life_years = 99.0

            # Mass (hollow cylinder)
            density = _DENSITY_kg_m3.get(comp.material, 1010.0)
            vol_m3 = math.pi * (ro**2 - ri**2) * length_m
            mass_g = vol_m3 * density * 1000.0
            # Use actual comp mass if it's a terminal device (not a simple cylinder)
            if comp.component_type == "terminal_device":
                mass_g = comp.mass_g

            total_mass_g += mass_g
            worst_sf = min(worst_sf, sf)
            peak_stress = max(peak_stress, von_mises_mpa)
            min_life_years = min(min_life_years, max(0.0, life_years))

            component_results.append({
                "name": comp.name,
                "material": comp.material,
                "mass_g": round(mass_g, 1),
                "bending_moment_nm": round(M_Nm, 3),
                "stress_mpa": round(von_mises_mpa, 2),
                "safety_factor": round(min(sf, 99.9), 2),
                "life_years": round(min(life_years, 999.0), 2),
                "wall_thickness_mm": comp.wall_thickness_mm,
                "ok": sf >= 2.5,
            })

            cum_distal_n -= comp_mass_n / 2

        # Weight budget check
        n_links = len(params.links)
        budget_key = "above_elbow" if n_links >= 5 else "below_elbow"
        budget_g = cls._WEIGHT_BUDGET[budget_key]
        weight_ok = total_mass_g <= budget_g

        # Worst safety factor (handle edge case of empty components)
        if worst_sf == float("inf"):
            worst_sf = 0.0

        # Suggestions
        suggestions: list[str] = []
        for cr in component_results:
            if cr["safety_factor"] < 2.5:
                suggestions.append(
                    f"Increase wall thickness at '{cr['name']}' (current FoS={cr['safety_factor']:.2f} < 2.5). "
                    f"Try +0.5 mm wall thickness."
                )
            elif cr["safety_factor"] > 6.0:
                suggestions.append(
                    f"Over-engineered '{cr['name']}' (FoS={cr['safety_factor']:.2f}). "
                    f"Reduce wall thickness by 0.5 mm to save weight."
                )
        if not weight_ok:
            suggestions.append(
                f"Total mass {total_mass_g:.0f} g exceeds {budget_g:.0f} g budget. "
                f"Switch pylons to CFRP-PA12 or reduce wall thickness."
            )

        return MechanicalReport(
            components=component_results,
            total_mass_g=round(total_mass_g, 1),
            worst_safety_factor=round(worst_sf, 2),
            peak_stress_mpa=round(peak_stress, 2),
            predicted_life_years=round(min(min_life_years, 999.0), 2),
            weight_budget_ok=weight_ok,
            suggestions=suggestions,
        )

    @staticmethod
    def _estimate_load_cases(params: DesignParams) -> list[dict[str, Any]]:
        """Estimate load cases from geometry (1 N/cm end-effector force)."""
        total_len_m = sum(lk.length for lk in params.links)
        # Peak torque at elbow = F_ee × forearm_length
        F_N = max(5.0, total_len_m * 20.0)   # ~20 N/m arm weight proxy
        forearm_len = sum(
            lk.length for lk in params.links
            if any(j.name in {"elbow", "wrist", "wrist_flex", "wrist_rot", "td_grip", "pinch", "gripper_open", "myo_grip"}
                   for j in lk.joints)
        )
        return [
            {"name": "reach_far", "peak_torque_nm": F_N * forearm_len,   "location": "elbow_unit"},
            {"name": "grip_load", "peak_torque_nm": F_N * 0.06,          "location": "wrist_flex_unit"},
            {"name": "carry",     "peak_torque_nm": F_N * total_len_m,   "location": "socket"},
        ]

    @staticmethod
    def optimize_wall_thickness(
        comp: ComponentSpec,
        target_sf: float = 3.0,
        M_Nm: float = 5.0,
    ) -> float:
        """Binary search for wall thickness that achieves target_sf."""
        Kt = _Kt.get(comp.name, 1.5)
        yield_mpa = _YIELD_MPa.get(comp.material, 50.0)
        ro = comp.outer_radius_mm / 1000.0 / 2
        target_stress = yield_mpa / target_sf / Kt

        lo, hi = 0.5e-3, ro * 0.9
        for _ in range(30):
            wt = (lo + hi) / 2
            ri = max(0.0, ro - wt)
            if ro <= ri:
                break
            I = math.pi * (ro**4 - ri**4) / 4
            Z = I / ro
            stress = M_Nm / Z / 1e6
            if stress < target_stress:
                hi = wt
            else:
                lo = wt

        return round((lo + hi) / 2 * 1000, 2)   # mm
