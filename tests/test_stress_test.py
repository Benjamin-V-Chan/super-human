"""Stress-test harness: torque capture -> fatigue -> per-task lifespan + success.

The rollout/fatigue/report logic is exercised with a stub policy so the suite
stays fast (no PPO training); a tiny real train+battery run is covered live, not
in CI. Needs MuJoCo; skips cleanly when the optional `sim` extra is absent.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from prosthesis_rl.agents.scenario import ScenarioAgent, library_keys, library_scenarios
from prosthesis_rl.contracts import DesignParams

mujoco = pytest.importorskip("mujoco")

from prosthesis_rl.rl.scenario_env import ScenarioReachEnv  # noqa: E402
from prosthesis_rl.rl.stress_test import (  # noqa: E402
    ScenarioResult,
    StressTestReport,
    rollout_torque,
    stress_test_scenario,
)


class _StubPolicy:
    """Stands in for a trained PPO policy: returns a fixed action each step."""

    def __init__(self, dof: int, action=None) -> None:
        self.dof = dof
        self._a = (np.zeros(dof, dtype=np.float32) if action is None
                   else np.asarray(action, dtype=np.float32))

    def predict(self, obs, deterministic=True):
        return self._a, None


def _shoe_env(eval_waypoint=None) -> ScenarioReachEnv:
    sc = ScenarioAgent().for_action("tie my shoe")
    return ScenarioReachEnv(sc, DesignParams(), mesh_dir=None, snap_samples=800,
                            eval_waypoint=eval_waypoint)


def test_library_scenarios_covers_every_key():
    specs = library_scenarios()
    assert [s.task_id for s in specs] == library_keys()
    assert all(s.waypoints for s in specs)            # each has something to reach
    assert all(not s.validate() for s in specs)       # and is a valid spec


def test_eval_waypoint_pins_the_goal():
    sc = ScenarioAgent().for_action("tie my shoe")
    idx = sc.waypoints.index(sc.primary_waypoint())
    env = _shoe_env(eval_waypoint=idx)
    # Every reset puts the goal exactly on the pinned waypoint (no jitter).
    for seed in (0, 1, 7):
        env.reset(seed=seed)
        assert np.allclose(env.target, env.waypoint_targets[idx])


def test_rollout_torque_shape_and_info():
    env = _shoe_env(eval_waypoint=0)
    torque, info = rollout_torque(env, _StubPolicy(env.dof), seed=0)
    assert torque.ndim == 2 and torque.shape[1] == env.dof
    assert torque.shape[0] >= 1
    assert "distance" in info


def test_stress_test_scenario_reports_success_and_lifespan():
    sc = ScenarioAgent().for_action("tie my shoe")
    design = DesignParams()
    # A stub that drives toward the snapped primary config so the task succeeds.
    idx = sc.waypoints.index(sc.primary_waypoint())
    env = _shoe_env(eval_waypoint=idx)
    q_goal = env.waypoint_configs[idx]
    action = np.clip((q_goal - env.mid) / env.half, -1.0, 1.0).astype(np.float32)

    res = stress_test_scenario(sc, _StubPolicy(design.dof, action), design=design,
                               mesh_dir=None, snap_samples=800)
    assert isinstance(res, ScenarioResult)
    assert res.task_id == "tie_shoe"
    assert res.critical_joint in design.joint_names
    assert res.peak_torque_nm >= 0.0
    # lifespan is a real number or +inf (below the fatigue limit)
    assert res.predicted_years > 0
    assert (math.isinf(res.predicted_years)) == res.below_endurance_limit
    assert isinstance(res.success, bool)


def test_report_aggregates_worst_case_and_serialises():
    def _mk(task, years, ok):
        return ScenarioResult(
            task_id=task, action=task, posture="seated", success=ok,
            final_distance_cm=2.0, tolerance_cm=4.0, critical_joint="elbow",
            peak_torque_nm=1.0, torque_span_nm=2.0, peak_stress_mpa=10.0, amplitude_mpa=5.0,
            predicted_years=years, display_years=f"{years} yr",
            below_endurance_limit=False, steps=10)

    rep = StressTestReport(results=[
        _mk("a", 12.0, True), _mk("b", 3.5, True), _mk("c", 40.0, False)])
    assert rep.success_rate == pytest.approx(2 / 3)
    assert rep.worst_case.task_id == "b"            # shortest life is the limiter
    assert rep.min_predicted_years == 3.5

    import json
    d = json.loads(rep.to_json())
    assert d["worst_case_task"] == "b"
    assert len(d["results"]) == 3
    assert "limiting load: b" in rep.summary_table()


def test_report_handles_infinite_lifespan_in_json():
    res = ScenarioResult(
        task_id="x", action="x", posture="seated", success=True,
        final_distance_cm=1.0, tolerance_cm=4.0, critical_joint="wrist",
        peak_torque_nm=0.2, torque_span_nm=0.4, peak_stress_mpa=2.0, amplitude_mpa=1.0,
        predicted_years=math.inf, display_years=">=100 yr (below fatigue limit)",
        below_endurance_limit=True, steps=5)
    rep = StressTestReport(results=[res])
    import json
    d = json.loads(rep.to_json())          # math.inf is not valid JSON -> None
    assert d["results"][0]["predicted_years"] is None
    assert d["min_predicted_years"] is None
