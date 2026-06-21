"""Roll out a trained policy on the real arm, returning the same metrics/torque
log as the scripted controller — so the demo can swap scripted IK for the learned
policy with no other changes.

`build_obs` / `map_action` are the single source of truth for the policy's
observation and action encoding; `rl.env.ReachEnv` imports them too, so a saved
policy always sees the same encoding it was trained on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from prosthesis_rl.contracts import DesignParams
from prosthesis_rl.sim.control import REACH_SUCCESS_M, ReachMetrics, TorqueLog
from prosthesis_rl.sim.mjcf_builder import EE_SITE, joint_ranges

CONTROL_HZ = 25.0  # must match rl.env.ReachEnv default


def build_obs(q, qd, ee, target, mid, half) -> np.ndarray:
    """Observation vector: [joint pos (norm), joint vel (clipped), ee→target, target]."""
    q, qd = np.asarray(q, dtype=float), np.asarray(qd, dtype=float)
    ee, target = np.asarray(ee, dtype=float), np.asarray(target, dtype=float)
    mid, half = np.asarray(mid, dtype=float), np.asarray(half, dtype=float)
    if q.shape != qd.shape or q.shape != mid.shape or q.shape != half.shape:
        raise ValueError("q, qd, mid, and half must have matching shapes")
    if ee.shape != (3,) or target.shape != (3,):
        raise ValueError("ee and target must be three-dimensional vectors")
    qn = (q - mid) / np.where(half > 1e-6, half, 1.0)
    return np.concatenate([qn, np.clip(qd, -10, 10), ee - target, target]).astype(np.float32)


def map_action(a, mid, half, lo, hi) -> np.ndarray:
    """Map a policy action in [-1, 1] per DoF to an absolute joint position target."""
    a = np.asarray(a, dtype=float)
    mid, half = np.asarray(mid, dtype=float), np.asarray(half, dtype=float)
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    if a.shape != mid.shape or a.shape != half.shape or a.shape != lo.shape or a.shape != hi.shape:
        raise ValueError("action and joint-range vectors must have matching shapes")
    a = np.clip(a, -1.0, 1.0)
    return np.clip(mid + a * half, lo, hi)


def load_policy(path: str | Path):
    """Load a saved stable-baselines3 PPO policy."""
    from stable_baselines3 import PPO

    return PPO.load(str(path))


def run_policy_reach(
    model,
    data,
    design: DesignParams,
    target,
    policy,
    *,
    seconds: float = 5.0,
    fps: int = 30,
    control_hz: float = CONTROL_HZ,
    max_joint_rate: float = 3.0,
    frame_cb=None,
) -> tuple[ReachMetrics, TorqueLog]:
    """Drive the arm to `target` with a trained policy; mirror run_reach's outputs.

    The policy outputs absolute joint-position targets; we rate-limit the commanded
    setpoint to `max_joint_rate` rad/s (real actuators are velocity-bounded) so the
    position servo doesn't slam to a full-range jump and spike joint torque. This
    keeps the actuation — and the torque trace the lifespan model reads — physical.
    """
    import mujoco

    joints = design.joint_names
    qadr = np.array([model.joint(n).qposadr[0] for n in joints], dtype=int)
    dadr = np.array([model.joint(n).dofadr[0] for n in joints], dtype=int)
    ranges = joint_ranges(design)
    lo = np.array([ranges[n][0] for n in joints])
    hi = np.array([ranges[n][1] for n in joints])
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    ee_id = model.site(EE_SITE).id
    target = np.asarray(target, dtype=float)
    dof = len(joints)

    arm_bodies = {"mount", *(link.name for link in design.links)}
    arm_geoms = {
        g for g in range(model.ngeom)
        if model.body(model.geom_bodyid[g]).name in arm_bodies
    }

    if seconds <= 0 or fps <= 0 or control_hz <= 0 or max_joint_rate <= 0:
        raise ValueError("seconds, fps, control_hz, and max_joint_rate must be positive")

    dt = model.opt.timestep
    n_steps = max(1, round(seconds / dt))

    log = TorqueLog(dt=dt, joints=tuple(joints))
    m = ReachMetrics()
    self_contacts = 0

    def distance() -> float:
        return float(np.linalg.norm(target - data.site_xpos[ee_id]))

    cmd = data.qpos[qadr].copy()                  # rate-limited commanded setpoint
    max_delta = max_joint_rate / float(control_hz)
    next_control = 0.0
    next_frame = 0.0
    for step in range(n_steps):
        now = step * dt
        if now + 1e-12 >= next_control:
            q = data.qpos[qadr]
            qd = data.qvel[dadr]
            ee = np.array(data.site_xpos[ee_id], dtype=float)
            obs = build_obs(q, qd, ee, target, mid, half)
            action, _ = policy.predict(obs, deterministic=True)
            desired = map_action(action, mid, half, lo, hi)
            cmd = cmd + np.clip(desired - cmd, -max_delta, max_delta)
            data.ctrl[:dof] = np.clip(cmd, lo, hi)
            next_control += 1.0 / control_hz

        mujoco.mj_step(model, data)
        tau = data.actuator_force[:dof].copy()
        m.energy += float(np.sum(np.abs(tau * data.qvel[dadr]))) * dt
        log.torque.append([float(x) for x in tau])
        qpos = data.qpos[qadr]
        over = np.maximum(lo - qpos, 0) + np.maximum(qpos - hi, 0)
        m.rom_violation += float(np.sum(over)) * dt
        for c in range(data.ncon):
            g1, g2 = data.contact[c].geom1, data.contact[c].geom2
            if g1 in arm_geoms and g2 in arm_geoms:
                self_contacts += 1
        m.min_distance = min(m.min_distance, distance())
        if frame_cb is not None and now + dt + 1e-12 >= next_frame:
            frame_cb(data)
            next_frame += 1.0 / fps

    m.final_distance = distance()
    m.reach_success = 1.0 if m.final_distance <= REACH_SUCCESS_M else 0.0
    m.self_collision = 1.0 if self_contacts > 0 else 0.0
    return m, log
