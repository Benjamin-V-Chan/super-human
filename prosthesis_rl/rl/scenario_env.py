"""Scenario reach env — train the arm on a real ADL task, not a random dot.

`ScenarioReachEnv` is a thin specialisation of `ReachEnv` that swaps the two
pieces that made the old task busywork:

  * the **scene**: the posture mount and the task objects from a `ScenarioSpec`
    are baked into the MJCF (via `sim.gizmo_asset`), so the arm is dropped into an
    actual situation (a shoe on the floor, a bottle on a table);
  * the **goal**: episode targets are the scenario's *waypoints* (weighted toward
    the task-completing one), each snapped onto the arm's reachable manifold, so
    the policy learns to put the hand where the task needs it.

Why snap? A shoulder-fixed arm can only reach a manifold of points; a Cartesian
waypoint authored by the library or an LLM often sits just off it (too close to
the body, past full extension). `sim.control.nearest_reachable` projects each
waypoint onto what this specific arm can actually do, so every goal is hittable
to within ~1 cm — and the markers in the viewer show the real, reachable point.

    from prosthesis_rl.agents.scenario import ScenarioAgent
    from prosthesis_rl.rl.scenario_env import ScenarioReachEnv
    scenario = ScenarioAgent().for_action("tie my shoe")
    env = ScenarioReachEnv(scenario, mesh_dir="webdemo/assets/scenes/arm_links")
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from prosthesis_rl.contracts import DesignParams, ScenarioSpec
from prosthesis_rl.contracts.scenario import clamp_to_reach
from prosthesis_rl.rl.env import ReachEnv
from prosthesis_rl.sim.gizmo_asset import inject_objects, waypoint_markers
from prosthesis_rl.sim.mjcf_builder import build_mjcf


class ScenarioReachEnv(ReachEnv):
    """Reach env whose scene and goals come from a ScenarioSpec."""

    def __init__(
        self,
        scenario: ScenarioSpec,
        design: DesignParams | None = None,
        *,
        mesh_dir: str | Path | None = None,
        control_hz: float = 25.0,
        max_steps: int = 150,
        seed: int | None = None,
        goal_jitter_m: float = 0.015,
        snap_samples: int = 4000,
        add_human: bool = True,
        add_markers: bool = True,
        eval_waypoint: int | None = None,
        human_collide: bool = True,
        target_band: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None,
    ) -> None:
        import mujoco

        design = design or DesignParams()
        self.scenario = scenario
        self.reach = float(sum(link.length for link in design.links))
        self._jitter = float(goal_jitter_m)
        # Domain randomization: when set, every episode's goal is drawn uniformly
        # from this world-space box (the spread of randomized task targets) instead
        # of the fixed waypoints — so one policy generalises across placements.
        self._band = (np.asarray(target_band[0], float), np.asarray(target_band[1], float)) \
            if target_band is not None else None
        mount = tuple(float(x) for x in scenario.mount_pos)

        # Pre-snap every waypoint onto this arm's reachable manifold, using a
        # throwaway bare-arm model (snapping needs FK, which needs a model). The
        # wearer is solid here too, so snapped goal poses never sit inside the body.
        from prosthesis_rl.sim.control import nearest_reachable

        bare = mujoco.MjModel.from_xml_string(
            build_mjcf(design, mount_pos=mount, mesh_dir=mesh_dir,
                       human_collide=human_collide), {})
        self.waypoint_targets: list[np.ndarray] = []
        self.waypoint_configs: list[np.ndarray] = []
        self.waypoint_residual: list[float] = []
        for i, wp in enumerate(scenario.waypoints):
            seed_pos = clamp_to_reach(mount, tuple(wp.pos), self.reach)
            ee, q, resid = nearest_reachable(bare, design, seed_pos,
                                             n=snap_samples, seed=1000 + i)
            self.waypoint_targets.append(ee)
            self.waypoint_configs.append(q)
            self.waypoint_residual.append(resid)

        self._wp_pos = np.array(self.waypoint_targets, dtype=float)
        w = np.array([max(wp.weight, 1e-3) for wp in scenario.waypoints], dtype=float)
        self._wp_w = w / w.sum()

        def transform(xml: str) -> str:
            xml = inject_objects(xml, scenario.objects)            # visual context
            if add_markers:
                xml = _splice_markers(xml, self.waypoint_targets)  # at reachable pts
            return xml

        # eval_waypoint pins every episode's goal to one waypoint (no weighting, no
        # jitter) so a stress-test rollout produces the repeatable load for *that*
        # task point; left None for training, which samples the weighted waypoints.
        self._eval_waypoint = (None if eval_waypoint is None
                               else int(eval_waypoint) % len(self._wp_pos))

        def sample_goal(rng: np.random.Generator, neutral_ee: np.ndarray) -> np.ndarray:
            if self._eval_waypoint is not None:
                return self._wp_pos[self._eval_waypoint]
            if self._band is not None:                              # randomized task target
                pos = rng.uniform(self._band[0], self._band[1])
                return np.asarray(clamp_to_reach(mount, tuple(pos), self.reach), dtype=float)
            i = rng.choice(len(self._wp_pos), p=self._wp_w)
            return self._wp_pos[i] + rng.uniform(-self._jitter, self._jitter, size=3)

        super().__init__(
            design,
            mesh_dir=mesh_dir,
            mount_pos=mount,
            control_hz=control_hz,
            max_steps=max_steps,
            seed=seed,
            xml_transform=transform,
            goal_sampler=sample_goal,
            human_collide=human_collide,
        )

    def primary_target(self) -> np.ndarray:
        """The snapped, reachable position of the most task-defining waypoint."""
        primary = self.scenario.primary_waypoint()
        idx = self.scenario.waypoints.index(primary)
        return self.waypoint_targets[idx]


def _splice_markers(xml: str, targets: list[np.ndarray]) -> str:
    from prosthesis_rl.contracts import TaskWaypoint

    markers = waypoint_markers(
        [TaskWaypoint(name=f"t{i}", pos=tuple(t), weight=2.0 if i == len(targets) - 1 else 1.0)
         for i, t in enumerate(targets)]
    )
    idx = xml.rfind("</worldbody>")
    return xml[:idx] + markers + "\n  " + xml[idx:]
