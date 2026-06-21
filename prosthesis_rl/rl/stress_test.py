"""Stress-test a trained policy across the ADL battery: lifespan + task success.

This is the harness the demo's "stress test it" story rides on. It takes one
trained PPO policy and runs it through every scenario in the ADL battery (tie
shoe, drink, drawer, …). For each task it:

  * pins the goal to the task-defining waypoint (``eval_waypoint``) and rolls the
    deterministic policy out, recording per-joint actuator torque every step;
  * picks the most-loaded joint and feeds its torque series to the fatigue model
    (`fatigue.estimate.estimate_lifespan`) — one ADL motion == one stress cycle —
    to get a predicted service life in years;
  * checks **task success**: did the hand finish within the waypoint's tolerance.

Aggregated, this answers two questions at once: does the arm *do* each daily task,
and which task is the **limiting load** that decides how long the hardware lasts.
Change the design (joint radius, material, fillet Kt in `fatigue.estimate`) and
re-run to see a hardware improvement move the worst-case lifespan number.

    from prosthesis_rl.rl.stress_test import stress_test_battery
    report = stress_test_battery(None, "assets/policies/scenario_ppo")
    print(report.summary_table())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from prosthesis_rl.contracts import DesignParams, ScenarioSpec
from prosthesis_rl.fatigue.estimate import JOINT_RADIUS_M, KT_FILLET, estimate_lifespan
from prosthesis_rl.fatigue.materials import DEFAULT_MATERIAL_KEY
from prosthesis_rl.fatigue.recommend import ImprovementPlan, StressState, recommend


@dataclass
class ScenarioResult:
    """One task's stress-test outcome: did it work, and how long will it last."""

    task_id: str
    action: str
    posture: str
    # task success
    success: bool
    final_distance_cm: float
    tolerance_cm: float
    # hardware / lifespan (from the most-loaded joint)
    critical_joint: str
    peak_torque_nm: float
    torque_span_nm: float
    peak_stress_mpa: float
    amplitude_mpa: float
    predicted_years: float
    display_years: str
    below_endurance_limit: bool
    steps: int
    # parameters the fatigue estimate used (so the recommender can re-run it)
    kt: float = KT_FILLET
    radius_m: float = JOINT_RADIUS_M
    material_key: str = DEFAULT_MATERIAL_KEY
    # downsampled critical-joint torque (N·m) over the motion, for the dashboard chart
    torque_trace: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        if not np.isfinite(self.predicted_years):
            d["predicted_years"] = None      # JSON has no inf; display_years carries it
        return d


@dataclass
class StressTestReport:
    """The battery result: per-task rows plus the aggregate stress-test verdict."""

    results: list[ScenarioResult] = field(default_factory=list)
    usage_cycles_per_day: int = 300
    material: str = "PA12-CF"
    load_factor: float = 1.0

    # ---- aggregates -------------------------------------------------------- #
    @property
    def success_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.success for r in self.results) / len(self.results)

    @property
    def worst_case(self) -> ScenarioResult | None:
        """The task with the shortest predicted life — the limiting load."""
        if not self.results:
            return None
        return min(self.results, key=lambda r: r.predicted_years)

    @property
    def min_predicted_years(self) -> float:
        wc = self.worst_case
        return wc.predicted_years if wc else float("inf")

    def improvement_plan(self, *, target_years: float = 10.0) -> ImprovementPlan | None:
        """Ranked, quantified fixes for the limiting load (the worst-case task)."""
        wc = self.worst_case
        if wc is None:
            return None
        state = StressState(
            task_id=wc.task_id, critical_joint=wc.critical_joint,
            amplitude_pa=wc.amplitude_mpa * 1e6, peak_pa=wc.peak_stress_mpa * 1e6,
            kt=wc.kt, radius_m=wc.radius_m, material_key=wc.material_key,
            usage_cycles_per_day=self.usage_cycles_per_day,
            baseline_years=wc.predicted_years,
        )
        return recommend(state, target_years=target_years)

    def to_dict(self) -> dict:
        wc = self.worst_case
        plan = self.improvement_plan()
        return {
            "usage_cycles_per_day": self.usage_cycles_per_day,
            "material": self.material,
            "load_factor": self.load_factor,
            "success_rate": self.success_rate,
            "min_predicted_years": (None if not np.isfinite(self.min_predicted_years)
                                    else self.min_predicted_years),
            "worst_case_task": wc.task_id if wc else None,
            "worst_case_display": wc.display_years if wc else None,
            "results": [r.to_dict() for r in self.results],
            "improvement_plan": plan.to_dict() if plan else None,
        }

    def to_json(self, indent: int | None = 2) -> str:
        import json

        return json.dumps(self.to_dict(), indent=indent)

    def summary_table(self) -> str:
        """A human-readable table for the CLI / demo."""
        head = (f"{'TASK':<16}{'SUCCESS':<9}{'FINAL':>8}{'CRIT JOINT':>14}"
                f"{'PEAK τ':>9}{'σ_amp':>9}{'LIFESPAN':>14}")
        rows = [head, "-" * len(head)]
        for r in self.results:
            ok = "HIT ✅" if r.success else "miss ❌"
            rows.append(
                f"{r.task_id:<16}{ok:<9}{r.final_distance_cm:>6.1f}cm"
                f"{r.critical_joint:>14}{r.peak_torque_nm:>7.2f}Nm"
                f"{r.amplitude_mpa:>7.1f}MPa{r.display_years:>14}"
            )
        wc = self.worst_case
        rows.append("-" * len(head))
        rows.append(f"success rate: {self.success_rate*100:.0f}%   "
                    f"({sum(r.success for r in self.results)}/{len(self.results)} tasks)")
        if wc is not None:
            rows.append(f"limiting load: {wc.task_id} via {wc.critical_joint} "
                        f"-> {wc.display_years}  "
                        f"(@ {self.usage_cycles_per_day} cycles/day, {self.material})")
        plan = self.improvement_plan()
        if plan is not None and plan.recommendations:
            rows.append("")
            rows.append(f"RECOMMENDATIONS for {plan.task_id}:")
            rows.append(f"  {plan.headline}")
            for rec in plan.recommendations[:4]:
                rows.append(f"  • {rec.action}")
                rows.append(f"      -> {rec.baseline_display} → {rec.improved_display}  "
                            f"({rec.multiplier_display})   [{rec.effort} effort, {rec.category}]")
            if plan.recommended_path is not None:
                rp = plan.recommended_path
                rows.append(f"  ★ recommended: {rp.action}")
                rows.append(f"      -> {rp.improved_display} ({rp.multiplier_display})")
        return "\n".join(rows)


def _downsample(series, n: int) -> list[float]:
    """Evenly subsample a 1-D series to at most n points (for the dashboard chart)."""
    arr = np.asarray(series, dtype=float).ravel()
    if arr.size <= n:
        return [float(x) for x in arr]
    idx = np.linspace(0, arr.size - 1, n).round().astype(int)
    return [float(x) for x in arr[idx]]


def rollout_torque(env, model, *, seed: int = 0):
    """Roll the deterministic policy through one episode; return (torque, info).

    torque is a [steps, dof] array of per-joint actuator force; info is the final
    step info (carries the distance/success for the pinned goal).
    """
    obs, _ = env.reset(seed=seed)
    series: list[np.ndarray] = []
    info: dict = {"distance": float("inf"), "success": 0.0}
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _reward, term, trunc, info = env.step(action)
        series.append(np.asarray(info["torque"], dtype=float))
        done = bool(term or trunc)
    torque = np.array(series, dtype=float) if series else np.zeros((1, env.dof))
    return torque, info


def stress_test_scenario(
    spec: ScenarioSpec,
    model,
    *,
    design: DesignParams,
    mesh_dir=None,
    usage_cycles_per_day: int = 300,
    snap_samples: int = 4000,
    seed: int = 0,
    material: str = DEFAULT_MATERIAL_KEY,
    load_factor: float = 1.0,
) -> ScenarioResult:
    """Run the policy through one scenario and score success + predicted lifespan.

    `load_factor` scales the measured joint torque before the fatigue estimate — a
    payload / duty-margin knob (e.g. 3.0 ≈ carrying a heavy load) so the same arm
    can be stress-tested under harder use than the unladen reach it was trained on.
    """
    from prosthesis_rl.rl.scenario_env import ScenarioReachEnv

    primary = spec.primary_waypoint()
    idx = spec.waypoints.index(primary)
    env = ScenarioReachEnv(spec, design, mesh_dir=mesh_dir, eval_waypoint=idx,
                           add_markers=False, snap_samples=snap_samples, seed=seed)
    torque, info = rollout_torque(env, model, seed=seed)

    # Most-loaded joint over the motion -> its (load-scaled) torque series drives
    # the fatigue estimate. Scaling is uniform, so it doesn't change which joint wins.
    peak_per_joint = np.max(np.abs(torque), axis=0)
    j = int(np.argmax(peak_per_joint))
    crit_series = torque[:, j] * float(load_factor)
    est = estimate_lifespan(crit_series, design, usage_cycles_per_day=usage_cycles_per_day,
                            material=material)

    final_d = float(info.get("distance", float("inf")))
    return ScenarioResult(
        task_id=spec.task_id,
        action=spec.primary_action,
        posture=spec.posture,
        success=final_d <= primary.tolerance_m,
        final_distance_cm=final_d * 100.0,
        tolerance_cm=primary.tolerance_m * 100.0,
        critical_joint=design.joint_names[j],
        peak_torque_nm=float(np.max(np.abs(crit_series))),
        torque_span_nm=float(crit_series.max() - crit_series.min()),
        peak_stress_mpa=est.peak_stress_mpa,
        amplitude_mpa=est.amplitude_mpa,
        predicted_years=est.predicted_years,
        display_years=est.display_years,
        below_endurance_limit=est.below_endurance_limit,
        steps=int(torque.shape[0]),
        material_key=material if isinstance(material, str) else DEFAULT_MATERIAL_KEY,
        torque_trace=_downsample(crit_series, 60),
    )


def stress_test_battery(
    scenarios,
    policy_path,
    *,
    design: DesignParams | None = None,
    mesh_dir=None,
    usage_cycles_per_day: int = 300,
    snap_samples: int = 4000,
    seed: int = 0,
    material: str = DEFAULT_MATERIAL_KEY,
    load_factor: float = 1.0,
) -> StressTestReport:
    """Run a saved policy through the ADL battery and aggregate the verdict.

    `scenarios=None` uses the full built-in battery. `scenarios` may also be a
    list of ScenarioSpec / action strings (same coercion as the trainer).
    """
    from stable_baselines3 import PPO

    from prosthesis_rl.fatigue.materials import get_material
    from prosthesis_rl.rl.train import _resolve_scenarios

    design = design or DesignParams()
    specs = _resolve_scenarios(scenarios, reach=sum(l.length for l in design.links))
    model = PPO.load(str(policy_path))

    report = StressTestReport(usage_cycles_per_day=usage_cycles_per_day,
                              material=get_material(material).name,
                              load_factor=load_factor)
    for i, spec in enumerate(specs):
        report.results.append(
            stress_test_scenario(spec, model, design=design, mesh_dir=mesh_dir,
                                 usage_cycles_per_day=usage_cycles_per_day,
                                 snap_samples=snap_samples, seed=seed + i,
                                 material=material, load_factor=load_factor)
        )
    return report
