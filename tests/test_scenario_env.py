"""ScenarioReachEnv + Gizmo injector: scene assembles and the goals are reachable.

These need MuJoCo; the suite skips cleanly when the optional `sim` extra is
absent (same convention as the rest of the sim tests).
"""

from __future__ import annotations

import numpy as np
import pytest

from prosthesis_rl.agents.scenario import ScenarioAgent
from prosthesis_rl.contracts import DesignParams, SceneObject, ScenarioSpec, TaskWaypoint
from prosthesis_rl.sim import gizmo_asset

mujoco = pytest.importorskip("mujoco")


def _shoe_scenario() -> ScenarioSpec:
    return ScenarioAgent().for_action("tie my shoe")


def test_fallback_box_injection_loads_in_mujoco():
    design = DesignParams()
    from prosthesis_rl.sim.mjcf_builder import build_mjcf

    xml = build_mjcf(design, mount_pos=(0.0, 0.25, 0.70))
    obj = SceneObject(name="shoe", prompt="a shoe", pos=(0.0, 0.30, 0.07))
    xml2 = gizmo_asset.inject_objects(xml, [obj])
    model = mujoco.MjModel.from_xml_string(xml2, {})
    # the object body is present...
    assert any(model.body(i).name == "obj_shoe" for i in range(model.nbody))
    # ...and visual-only by default (adds no extra contact geom over the bare arm)
    bare = mujoco.MjModel.from_xml_string(xml, {})
    assert model.ngeom == bare.ngeom + 1


def test_collidable_object_flag_adds_contact_geom():
    from prosthesis_rl.sim.mjcf_builder import build_mjcf

    xml = build_mjcf(DesignParams(), mount_pos=(0.0, -0.40, 1.00))
    obj = SceneObject(name="box", prompt="a box", pos=(0.0, 0.20, 0.90))
    solid = gizmo_asset.inject_objects(xml, [obj], collide=True)
    model = mujoco.MjModel.from_xml_string(solid, {})
    gid = model.geom("obj_box_geom").id
    assert model.geom_contype[gid] != 0  # collidable, unlike the visual default


def test_scenario_env_builds_and_snaps_reachable():
    from prosthesis_rl.rl.scenario_env import ScenarioReachEnv

    sc = _shoe_scenario()
    env = ScenarioReachEnv(sc, DesignParams(), mesh_dir=None, snap_samples=4000)
    # every waypoint snapped onto the reachable (and body-collision-free) manifold
    assert len(env.waypoint_targets) == len(sc.waypoints)
    assert max(env.waypoint_residual) < 0.05
    # the solid wearer is present so the arm can't phase through it
    assert len(env.body_geoms) > 0
    # the scene carries the shoe object body
    assert any(env.model.body(i).name == "obj_shoe" for i in range(env.model.nbody))


def test_scenario_env_body_is_collidable_and_starts_clear():
    """The wearer is solid (so the arm can't phase through) and neutral is penetration-free."""
    import mujoco

    from prosthesis_rl.rl.scenario_env import ScenarioReachEnv

    env = ScenarioReachEnv(_shoe_scenario(), DesignParams(), mesh_dir=None, snap_samples=2500)
    m, d = env.model, env.data
    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
    arm, body = env.arm_geoms, env.body_geoms
    arm_body = sum(
        1 for c in range(d.ncon)
        if ((d.contact[c].geom1 in arm) ^ (d.contact[c].geom2 in arm))
        and (d.contact[c].geom1 in body or d.contact[c].geom2 in body)
    )
    assert arm_body == 0  # neutral pose does not start inside the wearer


def test_scenario_env_targets_come_from_waypoints():
    from prosthesis_rl.rl.scenario_env import ScenarioReachEnv

    sc = _shoe_scenario()
    env = ScenarioReachEnv(sc, DesignParams(), mesh_dir=None, snap_samples=4000)
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    # the episode target equals one of the snapped waypoint targets (within jitter)
    nearest = min(np.linalg.norm(env.target - t) for t in env.waypoint_targets)
    assert nearest <= env.scenario.waypoints[0].tolerance_m + env._jitter + 1e-6
    # and a step runs
    obs, reward, term, trunc, info = env.step(env.action_space.sample())
    assert "distance" in info


def test_scripted_drive_reaches_the_task_point():
    """Commanding the snapped joint config puts the hand on the task point."""
    from prosthesis_rl.rl.scenario_env import ScenarioReachEnv
    from prosthesis_rl.sim.mjcf_builder import EE_SITE

    sc = _shoe_scenario()
    env = ScenarioReachEnv(sc, DesignParams(), mesh_dir=None, snap_samples=2500)
    model, data = env.model, env.data
    ndof = env.dof
    ee_id = model.site(EE_SITE).id
    idx = sc.waypoints.index(sc.primary_waypoint())
    q_goal, target = env.waypoint_configs[idx], env.waypoint_targets[idx]

    mujoco.mj_resetData(model, data)
    q = data.qpos[:ndof].copy()
    for _ in range(120):
        q += np.clip(q_goal - q, -0.05, 0.05)
        data.ctrl[:ndof] = q
        for _ in range(8):
            mujoco.mj_step(model, data)
    dist = float(np.linalg.norm(np.array(data.site_xpos[ee_id]) - target))
    assert dist < 0.05  # within the 5 cm success band
