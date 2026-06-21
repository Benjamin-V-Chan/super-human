"""Material DB, the material-swappable fatigue estimate, and the design recommender.

Pure-math (no MuJoCo / no PPO), so this runs everywhere and fast. It pins the
properties that make the durability dashboard trustworthy: the baseline material
is internally consistent, the S-N math is monotonic in the right direction, and
the recommender is honest in both the fatigue-limited and over-built regimes.
"""

from __future__ import annotations

import math

import numpy as np

from prosthesis_rl.contracts import DesignParams
from prosthesis_rl.fatigue.estimate import (
    JOINT_RADIUS_M,
    KT_FILLET,
    cycles_to_failure,
    display_years,
    estimate_lifespan,
    section_modulus,
    years_from_amplitude,
)
from prosthesis_rl.fatigue.materials import (
    DEFAULT_MATERIAL,
    MATERIALS,
    all_materials,
    get_material,
)
from prosthesis_rl.fatigue.recommend import (
    StressState,
    multiplier_display,
    recommend,
)

MPa = 1e6


# ── materials ────────────────────────────────────────────────────────────────
def test_material_db_internally_consistent():
    for m in all_materials():
        assert -0.13 <= m.basquin_b <= -0.04, m.key          # plausible Basquin exponent
        assert 0 < m.endurance_limit_pa < m.sigma_f_prime_pa, m.key
        assert 0.3 <= m.process_knockdown <= 1.0, m.key
        assert m.density_kg_m3 > 0 and m.rel_cost > 0, m.key
    # metals dominate printed plastics in effective fatigue strength
    al = get_material("al_6061_t6").effective_sigma_f_pa
    pa = get_material("pa12_cf_fdm").effective_sigma_f_pa
    assert al > 5 * pa
    # the carbon-filled FDM baseline is the strongest *printed* option
    printed = [m for m in all_materials() if m.process in ("FDM", "SLS/MJF")]
    assert max(printed, key=lambda m: m.effective_sigma_f_pa).key == "pa12_cf_fdm"


def test_get_material_resolves_and_errors():
    assert get_material(None) is DEFAULT_MATERIAL
    assert get_material("ti_6al_4v") is MATERIALS["ti_6al_4v"]
    assert get_material(DEFAULT_MATERIAL) is DEFAULT_MATERIAL
    try:
        get_material("unobtanium")
        assert False, "expected KeyError"
    except KeyError:
        pass


# ── estimate ─────────────────────────────────────────────────────────────────
def test_section_modulus_and_cycles_monotonic():
    assert section_modulus(0.02) > section_modulus(0.01)
    base = get_material("pa12_cf_fdm")
    # higher amplitude -> fewer cycles; below endurance -> infinite
    assert cycles_to_failure(60 * MPa, base) < cycles_to_failure(40 * MPa, base)
    assert math.isinf(cycles_to_failure(base.endurance_limit_pa * 0.5, base))


def test_estimate_material_swap_changes_life():
    design = DesignParams()
    tau = np.concatenate([np.linspace(0, 40, 60), np.linspace(40, 0, 60)])  # N·m, heavy
    weak = estimate_lifespan(tau, design, material="petg_cf_fdm")
    strong = estimate_lifespan(tau, design, material="ti_6al_4v")
    # same load, stronger material -> at least as many years (titanium: infinite)
    assert (math.isinf(strong.predicted_years)
            or strong.predicted_years >= weak.predicted_years)
    assert strong.material == get_material("ti_6al_4v").name


def test_display_years_buckets():
    assert display_years(math.inf).startswith(">=")
    assert "yr" in display_years(50)
    assert "mo" in display_years(0.2)
    assert "days" in display_years(0.01)


# ── recommender ──────────────────────────────────────────────────────────────
def _state_from_amp(amp_mpa: float, material="pa12_cf_fdm", usage=300) -> StressState:
    mat = get_material(material)
    years = years_from_amplitude(amp_mpa * MPa, material=mat, usage_cycles_per_day=usage)
    return StressState(
        task_id="open_drawer", critical_joint="shoulder_flex",
        amplitude_pa=amp_mpa * MPa, peak_pa=2 * amp_mpa * MPa,
        kt=KT_FILLET, radius_m=JOINT_RADIUS_M, material_key=material,
        usage_cycles_per_day=usage, baseline_years=years)


def test_recommend_fatigue_limited_all_help_and_ranked():
    state = _state_from_amp(80.0)                  # well above the 30 MPa endurance
    plan = recommend(state)
    assert not plan.over_built
    assert plan.load_margin < 1.0
    assert plan.recommendations and plan.recommended_path is not None
    # every recommendation is a real improvement (>= baseline), and at least one
    # reaches infinite life
    for r in plan.recommendations:
        assert math.isinf(r.improved_years) or r.improved_years >= state.baseline_years
    assert any(r.reaches_infinite_life for r in plan.recommendations)
    # ranked: infinite-life first, then by multiplier (non-increasing among finite)
    fin = [r.multiplier for r in plan.recommendations if not r.reaches_infinite_life]
    assert fin == sorted(fin, reverse=True)
    # the levers we expect are present
    ids = {r.id for r in plan.recommendations}
    assert {"thicken_joint", "fillet_radius", "material_metal", "control_torque"} <= ids


def test_recommend_is_honest_no_better_printed_plastic():
    # the carbon-filled FDM baseline already tops printed options, so there should
    # be no "switch printed material" recommendation (only metal helps).
    plan = recommend(_state_from_amp(80.0))
    assert "material_plastic" not in {r.id for r in plan.recommendations}


def test_recommend_over_built_lightens():
    state = _state_from_amp(5.0)                    # below the 30 MPa endurance
    plan = recommend(state)
    assert plan.over_built and plan.load_margin > 1.0
    ids = [r.id for r in plan.recommendations]
    assert "slim_joint" in ids                      # propose shedding mass
    assert ids[-1] == "keep_design"                 # always offer the honest "leave it"


def test_thicker_radius_and_lower_kt_extend_life():
    state = _state_from_amp(80.0)
    plan = recommend(state)
    thick = next(r for r in plan.recommendations if r.id == "thicken_joint")
    fillet = next(r for r in plan.recommendations if r.id == "fillet_radius")
    assert thick.improved_years > state.baseline_years or math.isinf(thick.improved_years)
    assert fillet.improved_years > state.baseline_years or math.isinf(fillet.improved_years)


def test_recommend_ignores_stale_baseline_years():
    # Even with a wrong/zero baseline_years, the recommender must re-derive the
    # baseline from the model and never surface a strictly-worse material.
    mat = get_material("pa12_cf_fdm")
    amp_mpa = 200.0                                  # heavily fatigue-limited
    true_base = years_from_amplitude(amp_mpa * MPa, material=mat, usage_cycles_per_day=300)
    bad = StressState(
        task_id="t", critical_joint="shoulder_flex",
        amplitude_pa=amp_mpa * MPa, peak_pa=2 * amp_mpa * MPa,
        kt=KT_FILLET, radius_m=JOINT_RADIUS_M, material_key="pa12_cf_fdm",
        usage_cycles_per_day=300, baseline_years=0.0)    # <- stale/zero
    plan = recommend(bad)
    assert "material_plastic" not in {r.id for r in plan.recommendations}
    for r in plan.recommendations:
        assert math.isinf(r.improved_years) or r.improved_years >= true_base * 0.999


def test_multiplier_display_cases():
    assert multiplier_display(1.0, math.inf) == "∞"
    assert multiplier_display(math.inf, math.inf) == "—"     # both infinite = no change
    assert multiplier_display(1.0, 12.0) == "12×"
    assert multiplier_display(0.0, 5.0) == "—"


def test_plan_serialises_json_safe():
    import json
    for amp in (5.0, 80.0):
        d = recommend(_state_from_amp(amp)).to_dict()
        json.dumps(d)                               # must not raise (no inf/nan)
        assert "recommendations" in d and "headline" in d
        for r in d["recommendations"]:
            assert r["improved_years"] is None or isinstance(r["improved_years"], (int, float))
