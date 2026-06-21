"""Scripted damped-least-squares IK reach controller + a reach rollout.

This is the v1 scripted controller from STRESS_TEST_PLAN.md (Phase 1 allows
scripted/IK control before any learned `.pt` policy). It drives the arm's `ee`
site to a 3D target through the position actuators, so the motion respects
contacts and produces real actuator torques (which Phase 2 fatigue reads).

    ctrl = ReachController(model, design, target)
    metrics, torque_log = run_reach(model, data, ctrl, seconds=4.0, fps=30,
                                    frame_cb=render_one_frame)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from prosthesis_rl.contracts import DesignParams
from prosthesis_rl.sim.mjcf_builder import EE_SITE, joint_ranges

REACH_SUCCESS_M = 0.05  # ee within 5 cm of target == success


def sample_reachable_targets(
    model, design: DesignParams, n: int = 4, *, seed: int = 0,
) -> list[np.ndarray]:
    """Deterministic FK-sampled reach targets that are reachable by construction.

    Forward-samples joint configs, runs FK, and keeps forward (in front of the
    shoulder), table-height end-effector poses well away from the neutral pose —
    so a scripted/learned controller actually has to move to hit them.
    """
    import mujoco

    ee_id = model.site(EE_SITE).id
    joints = design.joint_names
    qadr = np.array([model.joint(j).qposadr[0] for j in joints], dtype=int)
    ranges = joint_ranges(design)
    lo = np.array([ranges[j][0] for j in joints])
    hi = np.array([ranges[j][1] for j in joints])
    mount_y = float(model.body("mount").pos[1])

    data = mujoco.MjData(model)
    data.qpos[qadr] = np.clip(np.zeros(len(joints)), lo, hi)
    mujoco.mj_forward(model, data)
    neutral = np.array(data.site_xpos[ee_id], dtype=float)

    rng = np.random.default_rng(seed)
    out: list[np.ndarray] = []
    for _ in range(2000):
        if len(out) >= n:
            break
        data.qpos[qadr] = rng.uniform(lo, hi)
        mujoco.mj_forward(model, data)
        ee = np.array(data.site_xpos[ee_id], dtype=float)
        if ee[2] <= 0.25 or np.linalg.norm(ee - neutral) < 0.15 or ee[1] <= mount_y:
            continue
        out.append(ee)
    while len(out) < n:  # fallback so callers always get n targets
        out.append(neutral + np.array([0.0, 0.30, 0.10]))
    return out


class ReachController:
    """Closed-loop DLS velocity IK feeding the position actuators."""

    def __init__(
        self,
        model,
        design: DesignParams,
        target: np.ndarray | tuple[float, float, float],
        *,
        damping: float = 0.10,
        gain: float = 0.6,
        max_step: float = 0.04,
    ) -> None:
        import mujoco

        self._mj = mujoco
        self.model = model
        self.target = np.asarray(target, dtype=float)
        self.damping = damping
        self.gain = gain
        self.max_step = max_step

        self.ee_id = model.site(EE_SITE).id
        # Joint set comes from the design's declared chain (any DoF count), in
        # actuator order — not a fixed module constant.
        self.joints = design.joint_names
        self.dof = len(self.joints)
        self.qadr = np.array([model.joint(n).qposadr[0] for n in self.joints], dtype=int)
        self.dadr = np.array([model.joint(n).dofadr[0] for n in self.joints], dtype=int)
        ranges = joint_ranges(design)
        self.lo = np.array([ranges[n][0] for n in self.joints])
        self.hi = np.array([ranges[n][1] for n in self.joints])
        # Geom ids belonging to the arm (mount + every link body) for self-collision.
        self.arm_body_names = {"mount", *(link.name for link in design.links)}
        self.arm_geom_ids = {
            g for g in range(model.ngeom)
            if model.body(model.geom_bodyid[g]).name in self.arm_body_names
        }
        self._jacp = np.zeros((3, model.nv))
        self._q_des: np.ndarray | None = None  # integrated setpoint (leads the arm)

    def update(self, data) -> None:
        """Advance the IK setpoint and write it to `data.ctrl` (caller steps).

        The setpoint is integrated independently of the actual (lagging) joint
        angles so it marches all the way to the IK solution and the position
        servo drags the arm up against gravity, rather than settling sagged.
        """
        if self._q_des is None:
            self._q_des = data.qpos[self.qadr].copy()
        self._mj.mj_jacSite(self.model, data, self._jacp, None, self.ee_id)
        J = self._jacp[:, self.dadr]                       # 3 x nDoF(arm)
        err = self.target - data.site_xpos[self.ee_id]      # 3
        JJt = J @ J.T + (self.damping ** 2) * np.eye(3)
        dq = J.T @ np.linalg.solve(JJt, err)                # nDoF
        n = np.linalg.norm(dq)
        if n > self.max_step:
            dq *= self.max_step / n
        self._q_des = np.clip(self._q_des + self.gain * dq, self.lo, self.hi)
        data.ctrl[:] = self._q_des

    def ee_pos(self, data) -> np.ndarray:
        return np.asarray(data.site_xpos[self.ee_id], dtype=float)

    def distance(self, data) -> float:
        return float(np.linalg.norm(self.target - self.ee_pos(data)))


@dataclass
class ReachMetrics:
    reach_success: float = 0.0
    final_distance: float = 1.0
    min_distance: float = 1.0
    energy: float = 0.0
    rom_violation: float = 0.0
    self_collision: float = 0.0


@dataclass
class TorqueLog:
    """Per-joint actuator torque time series (seed of the Phase-2 SignalLog)."""

    dt: float
    joints: tuple[str, ...] = ()
    torque: list[list[float]] = field(default_factory=list)  # [step][joint]

    def as_array(self) -> np.ndarray:
        return np.asarray(self.torque, dtype=float)


def run_reach(
    model,
    data,
    controller: ReachController,
    *,
    seconds: float = 4.0,
    fps: int = 30,
    frame_cb=None,
) -> tuple[ReachMetrics, TorqueLog]:
    """Step physics to `seconds`, driving the reach. Calls `frame_cb(data)` at fps.

    Returns reach metrics and the per-joint torque log.
    """
    import mujoco

    dt = model.opt.timestep
    substeps = max(1, round((1.0 / fps) / dt))
    n_frames = int(seconds * fps)
    arm_geoms = controller.arm_geom_ids

    log = TorqueLog(dt=dt, joints=tuple(controller.joints))
    m = ReachMetrics()
    self_contacts = 0

    for _ in range(n_frames):
        for _ in range(substeps):
            controller.update(data)
            mujoco.mj_step(model, data)
            tau = data.actuator_force[: controller.dof].copy()
            qd = data.qvel[controller.dadr]
            m.energy += float(np.sum(np.abs(tau * qd))) * dt
            log.torque.append([float(x) for x in tau])
            # ROM overshoot beyond compiled ranges (should be ~0 with clamped ctrl).
            q = data.qpos[controller.qadr]
            over = np.maximum(controller.lo - q, 0) + np.maximum(q - controller.hi, 0)
            m.rom_violation += float(np.sum(over)) * dt
            for c in range(data.ncon):
                g1, g2 = data.contact[c].geom1, data.contact[c].geom2
                if g1 in arm_geoms and g2 in arm_geoms:
                    self_contacts += 1
        d = controller.distance(data)
        m.min_distance = min(m.min_distance, d)
        if frame_cb is not None:
            frame_cb(data)

    m.final_distance = controller.distance(data)
    m.reach_success = 1.0 if m.final_distance <= REACH_SUCCESS_M else 0.0
    m.self_collision = 1.0 if self_contacts > 0 else 0.0
    return m, log
