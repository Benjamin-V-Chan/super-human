"""Durability recommender: turn a stress-test verdict into ranked, *quantified* fixes.

The stress test says "the {critical_joint} on task {worst_case} fails in N years".
This module answers the user's real question — *what should I change?* — by taking
that one limiting load and re-running the **same** fatigue model under each
candidate hardware/process/control change, so every recommendation comes with a
proven before→after lifespan, not vibes.

The leverage comes from the S-N exponent. With Basquin b≈-0.10, cycles-to-failure
N ∝ σ_amp^(1/b) = σ_amp^-10, and σ_amp ∝ Kt / r³, so:

  * thicken the joint section a little          -> N ∝ r^30   (huge)
  * round the fillet (drop Kt)                  -> N ∝ Kt^-10
  * stiffer material / better process (σ_f')    -> N ∝ σ_f'^10
  * shave peak torque (control / lighter links) -> N ∝ τ^-10

so small, cheap changes move lifespan by orders of magnitude — and dropping the
amplitude below the material's endurance limit removes fatigue failure entirely.

    from prosthesis_rl.fatigue.recommend import recommend, StressState
    plan = recommend(state)        # state from rl.stress_test
    for r in plan.recommendations: print(r.title, r.multiplier_display)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from prosthesis_rl.fatigue.estimate import (
    KT_FILLET,
    JOINT_RADIUS_M,
    display_years,
    years_from_amplitude,
)
from prosthesis_rl.fatigue.materials import (
    DEFAULT_MATERIAL_KEY,
    Material,
    all_materials,
    get_material,
)

MPa = 1e6


def multiplier_display(base: float, improved: float) -> str:
    """'12×' / '∞' / '>1000×' / '—' — how much longer the part lasts."""
    if math.isinf(improved):
        return "—" if math.isinf(base) else "∞"   # both infinite = no lifespan change
    if base <= 0 or not math.isfinite(base):
        return "—"
    m = improved / base
    if m >= 1000:
        return ">1000×"
    if m >= 10:
        return f"{m:.0f}×"
    if m >= 1.05:
        return f"{m:.1f}×"
    return "≈1×"


def _safe_mult(base: float, improved: float) -> float:
    if base <= 0 or not math.isfinite(base):
        return math.inf if math.isinf(improved) else 0.0
    return improved / base


@dataclass
class StressState:
    """The limiting load the recommender works from (from rl.stress_test)."""

    task_id: str
    critical_joint: str
    amplitude_pa: float          # σ_amp at the fillet, baseline
    peak_pa: float               # σ_peak at the fillet, baseline
    kt: float = KT_FILLET
    radius_m: float = JOINT_RADIUS_M
    material_key: str = DEFAULT_MATERIAL_KEY
    usage_cycles_per_day: int = 300
    baseline_years: float = 0.0


@dataclass
class Recommendation:
    id: str
    category: str               # geometry | manufacturing | material | control | kinematics
    title: str
    action: str                 # the concrete spec change
    rationale: str
    baseline_years: float
    improved_years: float
    multiplier: float           # improved/baseline (math.inf when fatigue is eliminated)
    reaches_infinite_life: bool
    tradeoff: str
    effort: str                 # low | medium | high
    confidence: str             # low | medium | high

    @property
    def multiplier_display(self) -> str:
        return multiplier_display(self.baseline_years, self.improved_years)

    @property
    def baseline_display(self) -> str:
        return display_years(self.baseline_years)

    @property
    def improved_display(self) -> str:
        return display_years(self.improved_years)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "category": self.category, "title": self.title,
            "action": self.action, "rationale": self.rationale,
            "baseline_years": _json_years(self.baseline_years),
            "improved_years": _json_years(self.improved_years),
            "baseline_display": display_years(self.baseline_years),
            "improved_display": display_years(self.improved_years),
            "multiplier": None if not math.isfinite(self.multiplier) else round(self.multiplier, 2),
            "multiplier_display": self.multiplier_display,
            "reaches_infinite_life": self.reaches_infinite_life,
            "tradeoff": self.tradeoff, "effort": self.effort, "confidence": self.confidence,
        }


@dataclass
class ImprovementPlan:
    task_id: str
    critical_joint: str
    baseline_years: float
    recommendations: list[Recommendation] = field(default_factory=list)
    recommended_path: Recommendation | None = None   # cheapest stack that fixes it
    over_built: bool = False          # load already below the endurance limit
    load_margin: float = 1.0          # endurance_limit / amplitude (how much spare load)
    headline: str = ""                # one-line verdict for the dashboard

    @property
    def baseline_display(self) -> str:
        return display_years(self.baseline_years)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "critical_joint": self.critical_joint,
            "baseline_years": _json_years(self.baseline_years),
            "baseline_display": display_years(self.baseline_years),
            "over_built": self.over_built,
            "load_margin": (None if not math.isfinite(self.load_margin)
                            else round(self.load_margin, 2)),
            "headline": self.headline,
            "recommendations": [r.to_dict() for r in self.recommendations],
            "recommended_path": self.recommended_path.to_dict() if self.recommended_path else None,
        }


def _json_years(y: float):
    return None if math.isinf(y) else round(y, 4)


# --------------------------------------------------------------------------- #
# Core: recompute amplitude under a change, then run the SAME S-N math.        #
# σ_amp scales as (Kt'/Kt) · (r/r')³ · torque_scale; material only enters the  #
# cycles count (effective σ_f' + endurance limit).                            #
# --------------------------------------------------------------------------- #
def _years_under(
    state: StressState, *, kt: float | None = None, radius_m: float | None = None,
    material: str | Material | None = None, torque_scale: float = 1.0,
) -> tuple[float, float]:
    kt = state.kt if kt is None else kt
    r = state.radius_m if radius_m is None else radius_m
    mat = get_material(material if material is not None else state.material_key)
    amp = (state.amplitude_pa * (kt / state.kt) * (state.radius_m / r) ** 3 * torque_scale)
    years = years_from_amplitude(amp, material=mat, usage_cycles_per_day=state.usage_cycles_per_day)
    return years, amp


def _best_alt_material(state: StressState, *, processes: tuple[str, ...]):
    """Longest-lived material in `processes` that beats the baseline; else None.

    Returns (Material, improved_years). Honest by construction: if nothing in the
    allowed processes improves on the current material, there is no recommendation.
    """
    # Compare against the baseline recomputed through the SAME model, never the
    # externally-supplied baseline_years (which could be stale or zero) — so we can
    # never surface a strictly-worse material as an "improvement".
    base, _ = _years_under(state)
    best = None
    for m in all_materials():
        if m.key == state.material_key or m.process not in processes:
            continue
        yrs, _ = _years_under(state, material=m.key)
        better = math.isinf(yrs) or (math.isfinite(base) and yrs > base * 1.05)
        if not better:
            continue
        cmp = math.inf if math.isinf(yrs) else yrs
        if best is None or cmp > best[0]:
            best = (cmp, m, yrs)
    return (best[1], best[2]) if best else None


def _mk(state: StressState, *, id, category, title, action, rationale, improved_years,
        tradeoff, effort, confidence) -> Recommendation:
    return Recommendation(
        id=id, category=category, title=title, action=action, rationale=rationale,
        baseline_years=state.baseline_years, improved_years=improved_years,
        multiplier=_safe_mult(state.baseline_years, improved_years),
        reaches_infinite_life=math.isinf(improved_years),
        tradeoff=tradeoff, effort=effort, confidence=confidence,
    )


def recommend(state: StressState, *, target_years: float = 10.0) -> ImprovementPlan:
    """Recommend concrete changes for the limiting joint.

    Two regimes, both honest:
      * fatigue-limited (load above the endurance limit) -> rank lifespan *gains*
        from thickening / filleting / metal / gentler control;
      * over-built (load already below the limit, infinite life) -> recommend
        *lightening / cheaper* changes that keep it safe, and report the margin.
    """
    mat = get_material(state.material_key)
    # Re-derive the baseline through the model so every reported before→after is
    # self-consistent, regardless of what baseline_years the caller passed.
    state = replace(state, baseline_years=_years_under(state)[0])
    over_built = state.amplitude_pa <= mat.endurance_limit_pa
    margin = (math.inf if state.amplitude_pa <= 0
              else mat.endurance_limit_pa / state.amplitude_pa)

    if over_built:
        recs = _lighten(state, mat)
        path = recs[0] if recs else None
        headline = (f"Over-built: the {state.critical_joint} runs at "
                    f"{1/margin*100:.0f}% of its fatigue limit — "
                    f"{margin:.1f}× spare load, life is effectively unlimited.")
    else:
        recs = _improve(state, mat, target_years=target_years)
        path = _cheapest_fix(state, target_years=target_years)
        headline = (f"Fatigue-limited: the {state.critical_joint} sets a "
                    f"{display_years(state.baseline_years)} life on {state.task_id}.")

    plan = ImprovementPlan(
        task_id=state.task_id, critical_joint=state.critical_joint,
        baseline_years=state.baseline_years, recommendations=recs,
        recommended_path=path, over_built=over_built, load_margin=margin,
        headline=headline)
    return plan


def _improve(state: StressState, mat: Material, *, target_years: float) -> list[Recommendation]:
    """Fatigue-limited regime: ranked lifespan gains."""
    base = state.baseline_years
    recs: list[Recommendation] = []

    # 1) GEOMETRY — thicken the loaded section. Prefer the radius that drops the
    # amplitude below the endurance limit (infinite life) if it's a modest bump;
    # else size it to the target. r for infinite: amp·(r0/r)³ = endurance.
    r0 = state.radius_m
    if state.amplitude_pa > mat.endurance_limit_pa:
        r_inf = r0 * (state.amplitude_pa / mat.endurance_limit_pa) ** (1 / 3)
    else:
        r_inf = r0
    if r_inf <= 1.6 * r0:
        r1 = r_inf
    else:  # endurance out of reach by radius alone — size to target_years
        ratio = max(target_years / base, 1.0) if base > 0 else 100.0
        r1 = r0 * ratio ** (-mat.basquin_b / 3.0)
    yrs, _ = _years_under(state, radius_m=r1)
    recs.append(_mk(
        state, id="thicken_joint", category="geometry",
        title=f"Thicken the {state.critical_joint} section",
        action=f"Increase joint section radius {r0*1000:.1f} → {r1*1000:.1f} mm "
               f"(+{(r1/r0-1)*100:.0f}%)",
        rationale="N ∝ r³⁰ — a few mm on the loaded section is the single biggest lever.",
        improved_years=yrs,
        tradeoff=f"+{((r1/r0)**2-1)*100:.0f}% mass at that joint; trivial to print.",
        effort="low", confidence="high"))

    # 2) MANUFACTURING — round the fillet to cut the stress concentration.
    kt1 = 1.4
    yrs, _ = _years_under(state, kt=kt1)
    recs.append(_mk(
        state, id="fillet_radius", category="manufacturing",
        title="Add a generous fillet at the joint root",
        action=f"Round the fillet to drop the stress-concentration Kt {state.kt:.1f} → {kt1:.1f}",
        rationale="N ∝ Kt⁻¹⁰ — a sharp internal corner is throwing away most of the life.",
        improved_years=yrs,
        tradeoff="Essentially free — a CAD change, no mass/cost penalty.",
        effort="low", confidence="high"))

    # 3) MATERIAL (printable) — only if some *printed* material actually beats the
    # baseline. A carbon-filled FDM nylon already tops the printed options, so this
    # is usually empty — and saying so honestly is the point.
    plastic = _best_alt_material(state, processes=("FDM", "SLS/MJF"))
    if plastic is not None:
        m, yrs = plastic
        recs.append(_mk(
            state, id="material_plastic", category="manufacturing",
            title=f"Print the joint in {m.name}",
            action=f"Switch the loaded part to {m.name} "
                   f"(effective σ_f' {mat.effective_sigma_f_mpa:.0f} → {m.effective_sigma_f_mpa:.0f} MPa)",
            rationale="A higher effective fatigue strength lifts the whole S-N curve (N ∝ σ_f'¹⁰).",
            improved_years=yrs,
            tradeoff=f"~{m.rel_cost / mat.rel_cost:.1f}× part cost.",
            effort="medium", confidence="medium"))

    # 4) MATERIAL (metal) — jump to a machined metal joint, which usually removes
    # fatigue failure outright (its endurance limit sits above this load).
    metal = _best_alt_material(state, processes=("CNC", "wrought"))
    if metal is not None:
        m, yrs = metal
        dens = m.density_kg_m3 / mat.density_kg_m3
        recs.append(_mk(
            state, id="material_metal", category="material",
            title=f"Machine the limiting joint from {m.name}",
            action=f"Swap the {state.critical_joint} part to {m.name} "
                   f"(effective σ_f' {mat.effective_sigma_f_mpa:.0f} → {m.effective_sigma_f_mpa:.0f} MPa)",
            rationale="N ∝ σ_f'¹⁰; the metal's endurance limit sits above this load, so it never fatigues.",
            improved_years=yrs,
            tradeoff=f"{dens:.1f}× denser part + CNC cost (Ti is the lighter, pricier alt).",
            effort="high", confidence="medium"))

    # 5) CONTROL — retrain the policy to shave peak torque (the env already has an
    # energy penalty; weighting it up trades a little speed for lower loads).
    ts = 0.85
    yrs, _ = _years_under(state, torque_scale=ts)
    recs.append(_mk(
        state, id="control_torque", category="control",
        title="Retrain for a gentler trajectory",
        action=f"Raise the energy/torque penalty to cut peak joint torque ~{(1-ts)*100:.0f}%",
        rationale="N ∝ τ⁻¹⁰ — even a small reduction in peak load multiplies life, no hardware change.",
        improved_years=yrs,
        tradeoff="May slightly slow the motion or lower success; needs a retrain.",
        effort="medium", confidence="medium"))

    # Rank by proven gain (infinite-life first, then multiplier).
    recs.sort(key=lambda r: (r.reaches_infinite_life, r.multiplier), reverse=True)
    return recs


def _lighten(state: StressState, mat: Material) -> list[Recommendation]:
    """Over-built regime: keep the (infinite) life, cut mass and cost.

    The load sits below the endurance limit, so life is usage-independent — the
    spare capacity is best spent on a lighter, cheaper part, not more years.
    """
    recs: list[Recommendation] = []
    end = mat.endurance_limit_pa
    amp = state.amplitude_pa
    r0 = state.radius_m

    # 1) Slim the section until the amplitude reaches ~80% of the endurance limit
    # (keep a margin for impact/static loads the fatigue model doesn't see).
    if amp > 0:
        r_min = r0 * (amp / (0.8 * end)) ** (1 / 3)
        r_min = max(r_min, 0.5 * r0)          # don't shrink past half the section
        if r_min < 0.97 * r0:
            yrs, _ = _years_under(state, radius_m=r_min)
            recs.append(_mk(
                state, id="slim_joint", category="geometry",
                title=f"Slim down the over-built {state.critical_joint} section",
                action=f"Reduce joint section radius {r0*1000:.1f} → {r_min*1000:.1f} mm "
                       f"({(r_min/r0-1)*100:.0f}%)",
                rationale="The load is well under the endurance limit, so the section "
                          "carries more material than fatigue needs.",
                improved_years=yrs,
                tradeoff=f"−{(1-(r_min/r0)**2)*100:.0f}% mass at that joint; keep a margin "
                         "for impact/static loads not in this model.",
                effort="low", confidence="medium"))

    # 2) Drop to a cheaper printable material that still clears the endurance limit.
    cheaper = _cheapest_safe_material(state, mat)
    if cheaper is not None:
        m, yrs = cheaper
        recs.append(_mk(
            state, id="cheaper_material", category="material",
            title=f"Switch to cheaper {m.name}",
            action=f"Print the part in {m.name} "
                   f"(~{m.rel_cost / mat.rel_cost:.2f}× the cost) — still below its fatigue limit",
            rationale="With this much margin a lower-grade material stays in infinite-life territory.",
            improved_years=yrs,
            tradeoff="Lower toughness/stiffness; re-check non-fatigue loads.",
            effort="low", confidence="low"))

    # Always end with the honest "leave it" option so the panel is never empty.
    recs.append(_mk(
        state, id="keep_design", category="plan",
        title="Or leave it — it already lasts",
        action="No change needed for fatigue; spend the budget elsewhere (reach, grip, mass).",
        rationale="Fatigue is not the limiting factor for this joint at this load.",
        improved_years=state.baseline_years,
        tradeoff="—", effort="low", confidence="high"))
    return recs


def _cheapest_safe_material(state: StressState, mat: Material):
    """Cheapest material that is cheaper than the baseline and still infinite-life."""
    best = None
    for m in all_materials():
        if m.key == state.material_key or m.rel_cost >= mat.rel_cost:
            continue
        yrs, amp = _years_under(state, material=m.key)
        if not math.isinf(yrs):            # must stay below ITS endurance limit
            continue
        if best is None or m.rel_cost < best[0]:
            best = (m.rel_cost, m, yrs)
    return (best[1], best[2]) if best else None


# Cost-ranked stacks: the first one that removes fatigue failure (or, failing
# that, the longest-lived) becomes the recommended path. Cheap, low-risk changes
# first so we don't over-prescribe a titanium joint when a fillet would do.
# Cost-ascending stacks of cheap, low-risk levers (geometry/fillet/control need no
# new material). The metal stack is the heavy hammer of last resort; "__metal__"
# is resolved to the best available machined metal at runtime.
_COMBOS = [
    ("fillet", "round the fillet (Kt→1.4)", 1, dict(kt=1.4)),
    ("fillet_control", "round the fillet + retrain for ~15% lower torque", 2,
     dict(kt=1.4, torque_scale=0.85)),
    ("fillet_thicken", "round the fillet + thicken the joint ~20%", 3,
     dict(kt=1.4, radius_mult=1.2)),
    ("fillet_thicken_control", "fillet + 20% thicker joint + ~15% lower torque", 4,
     dict(kt=1.4, radius_mult=1.2, torque_scale=0.85)),
    ("metal", "machine the joint from metal", 6, dict(material="__metal__")),
]


def _cheapest_fix(state: StressState, *, target_years: float) -> Recommendation | None:
    # _COMBOS is cost-ascending, so the first stack that clears the target is the
    # cheapest one; if none clears it, fall back to the longest-lived stack.
    metal = _best_alt_material(state, processes=("CNC", "wrought"))
    metal_key = metal[0].key if metal else None
    best = None                        # (years_for_compare, cid, desc, yrs)
    for cid, desc, _cost, mods in _COMBOS:
        material = mods.get("material")
        if material == "__metal__":
            if metal_key is None:
                continue               # no metal available/better — skip the stack
            material = metal_key
        yrs, _ = _years_under(
            state, kt=mods.get("kt"), material=material,
            radius_m=state.radius_m * mods.get("radius_mult", 1.0),
            torque_scale=mods.get("torque_scale", 1.0))
        cmp_years = math.inf if math.isinf(yrs) else yrs
        if best is None or cmp_years > best[0]:
            best = (cmp_years, cid, desc, yrs)
        if math.isinf(yrs) or yrs >= target_years:   # cheapest hitting stack wins
            best = (cmp_years, cid, desc, yrs)
            break
    if best is None:
        return None
    _, cid, desc, yrs = best
    return _mk(
        state, id=f"path_{cid}", category="plan",
        title="Recommended path",
        action=f"Cheapest change that gets {state.task_id} to a long service life: {desc}.",
        rationale="Stacks the lowest-cost, lowest-risk levers until the load drops "
                  "below the endurance limit (or clears the target).",
        improved_years=yrs,
        tradeoff="Balanced for cost/effort over a single heavy change.",
        effort="low", confidence="medium")
