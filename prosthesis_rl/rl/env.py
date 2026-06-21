"""Gymnasium reach env over the real MuJoCo arm — what PPO trains on.

The policy learns joint-space trajectories that drive the design's end-effector
to a randomized, reachable ADL target. The arm comes straight from
`sim.mjcf_builder` (the same model the demo renders), skinned with the per-link
CAD meshes, so a policy trained here transfers to the unified demo unchanged.

    from prosthesis_rl.rl.env import ReachEnv
    env = ReachEnv(mesh_dir="assets/stl/candidate")
    obs, _ = env.reset(seed=0)
    obs, reward, term, trunc, info = env.step(env.action_space.sample())

Observation (float32): [joint pos (norm), joint vel (scaled), ee→target (3), target (3)].
Action (float32, [-1, 1] per DoF): absolute joint position target, mapped to each
joint's range and fed to the position actuators.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ModuleNotFoundError as exc:  # pragma: no cover - clear hint if dep missing
    raise ModuleNotFoundError(
        "rl.env needs gymnasium — `pip install gymnasium stable-baselines3 torch`"
    ) from exc

from prosthesis_rl.contracts import DesignParams
from prosthesis_rl.sim.mjcf_builder import EE_SITE, build_mjcf, joint_ranges

REACH_SUCCESS_M = 0.05  # ee within 5 cm of target == success (matches sim.control)


class ReachEnv(gym.Env):
    """Single-arm reach task with reachable, FK-sampled goals."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        design: DesignParams | None = None,
        *,
        mesh_dir: str | Path | None = None,
        mount_pos: tuple[float, float, float] = (0.0, -0.40, 1.00),
        control_hz: float = 25.0,
        max_steps: int = 150,
        seed: int | None = None,
        xml_transform: Callable[[str], str] | None = None,
        goal_sampler: Callable[[np.random.Generator, np.ndarray], np.ndarray] | None = None,
        human_collide: bool = False,
        body_collision_penalty: float = 0.5,
    ) -> None:
        super().__init__()
        import mujoco

        self._mj = mujoco
        self.design = design or DesignParams()
        self.mount_pos = np.asarray(mount_pos, dtype=float)
        self.max_steps = int(max_steps)
        # Scenario hooks (both default to the original random-reach behaviour):
        #   xml_transform — rewrite the built MJCF (e.g. inject task objects);
        #   goal_sampler  — choose the episode target (e.g. task waypoints).
        self._goal_sampler = goal_sampler
        self._body_pen = float(body_collision_penalty)

        xml = build_mjcf(self.design, mount_pos=mount_pos, mesh_dir=mesh_dir,
                         human_collide=human_collide)
        if xml_transform is not None:
            xml = xml_transform(xml)
        self.model = mujoco.MjModel.from_xml_string(xml, {})
        self.data = mujoco.MjData(self.model)
        self.substeps = max(1, round((1.0 / control_hz) / self.model.opt.timestep))

        self.joints = self.design.joint_names
        self.dof = len(self.joints)
        self.qadr = np.array([self.model.joint(n).qposadr[0] for n in self.joints], dtype=int)
        self.dadr = np.array([self.model.joint(n).dofadr[0] for n in self.joints], dtype=int)
        ranges = joint_ranges(self.design)
        self.lo = np.array([ranges[n][0] for n in self.joints])
        self.hi = np.array([ranges[n][1] for n in self.joints])
        self.mid = 0.5 * (self.lo + self.hi)
        self.half = 0.5 * (self.hi - self.lo)
        self.ee_id = self.model.site(EE_SITE).id

        # Arm geoms for self-collision (mount + every link body).
        arm_bodies = {"mount", *(link.name for link in self.design.links)}
        self.arm_geoms = {
            g for g in range(self.model.ngeom)
            if self.model.body(self.model.geom_bodyid[g]).name in arm_bodies
        }
        # Body/scene geoms the arm must not phase through: the solid wearer plus
        # any injected task objects (obj_*). Empty unless the wearer is collidable.
        self.body_geoms = {
            g for g in range(self.model.ngeom)
            if (self.model.body(self.model.geom_bodyid[g]).name == "human"
                or self.model.body(self.model.geom_bodyid[g]).name.startswith("obj_"))
            and (self.model.geom_contype[g] or self.model.geom_conaffinity[g])
        }

        high = np.full(2 * self.dof + 6, 50.0, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.dof,), dtype=np.float32)

        self.rng = np.random.default_rng(seed)
        self.target = np.zeros(3)
        self._steps = 0
        self._prev_dist = 0.0

    # ── helpers ──────────────────────────────────────────────────────────────

    def _set_arm(self, q: np.ndarray) -> None:
        self.data.qpos[self.qadr] = q
        self.data.qvel[self.dadr] = 0.0
        self._mj.mj_forward(self.model, self.data)

    def _ee(self) -> np.ndarray:
        # np.array (not asarray) so we never alias MuJoCo's live data buffer.
        return np.array(self.data.site_xpos[self.ee_id], dtype=float)

    def _sample_reachable_target(self, neutral_ee: np.ndarray) -> np.ndarray:
        """FK-sample a real joint config so the goal is reachable by construction.

        Rejects goals too close to the neutral pose (no motion to learn) and
        favours forward reaches (+y, in front of the shoulder) — the ADL motion.
        """
        for _ in range(100):
            q = self.rng.uniform(self.lo, self.hi)
            self._set_arm(q)
            ee = self._ee()
            if not (0.25 < ee[2] < self.mount_pos[2] + 0.10):   # table-height band
                continue
            if np.linalg.norm(ee - neutral_ee) < 0.15:          # require real motion
                continue
            if ee[1] <= self.mount_pos[1]:                      # in front of the body
                continue
            return ee
        return neutral_ee + np.array([0.0, 0.30, 0.10])         # forward fallback

    def _obs(self) -> np.ndarray:
        from prosthesis_rl.rl.rollout import build_obs

        return build_obs(self.data.qpos[self.qadr], self.data.qvel[self.dadr],
                         self._ee(), self.target, self.mid, self.half)

    # ── gym API ──────────────────────────────────────────────────────────────

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._mj.mj_resetData(self.model, self.data)
        # Neutral start pose, then a reachable target sampled elsewhere in the ROM.
        neutral = np.clip(np.zeros(self.dof), self.lo, self.hi)
        self._set_arm(neutral)
        neutral_ee = self._ee().copy()
        if self._goal_sampler is not None:
            self.target = np.asarray(self._goal_sampler(self.rng, neutral_ee), dtype=float)
        else:
            self.target = self._sample_reachable_target(neutral_ee)
        self._set_arm(neutral)
        self.data.ctrl[: self.dof] = neutral
        self._steps = 0
        self._prev_dist = float(np.linalg.norm(self._ee() - self.target))
        return self._obs(), {}

    def step(self, action: np.ndarray):
        from prosthesis_rl.rl.rollout import map_action

        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        self.data.ctrl[: self.dof] = map_action(a, self.mid, self.half, self.lo, self.hi)

        energy = 0.0
        self_collision = False
        body_collision = False
        dt = self.model.opt.timestep
        for _ in range(self.substeps):
            self._mj.mj_step(self.model, self.data)
            tau = self.data.actuator_force[: self.dof]
            energy += float(np.sum(np.abs(tau * self.data.qvel[self.dadr]))) * dt
            for c in range(self.data.ncon):
                con = self.data.contact[c]
                if con.geom1 in self.arm_geoms and con.geom2 in self.arm_geoms:
                    self_collision = True
                # Arm touching the solid wearer/objects — the phasing we penalize.
                elif ((con.geom1 in self.arm_geoms) ^ (con.geom2 in self.arm_geoms)) and (
                        con.geom1 in self.body_geoms or con.geom2 in self.body_geoms):
                    body_collision = True

        dist = float(np.linalg.norm(self._ee() - self.target))
        success = dist < REACH_SUCCESS_M

        # Potential-based shaping toward the goal + a proximity bonus that sharpens
        # the final approach, with small control/energy/time costs.
        reward = 10.0 * (self._prev_dist - dist)
        reward += 0.5 * float(np.exp(-(dist / 0.05) ** 2))    # peaks near the goal
        reward -= 0.01 * float(np.sum(a ** 2))
        reward -= 0.0005 * energy
        reward -= 0.002                                       # mild time pressure
        if self_collision:
            reward -= 0.25
        if body_collision:
            reward -= self._body_pen          # discourage reaching through the wearer
        if success:
            reward += 10.0
        self._prev_dist = dist

        self._steps += 1
        terminated = success
        truncated = self._steps >= self.max_steps
        info = {"distance": dist, "success": float(success),
                "energy": energy, "self_collision": float(self_collision),
                "body_collision": float(body_collision),
                # Per-joint actuator torque at the end of this control step — the
                # signal the fatigue model integrates over a stress-test rollout.
                "torque": np.array(self.data.actuator_force[: self.dof], dtype=float)}
        return self._obs(), float(reward), terminated, truncated, info
