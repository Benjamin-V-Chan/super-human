"""Scenario contracts: what the policy is dropped into and asked to *do*.

The reach env used to train against a random FK-sampled point — the arm just
learned to put its end-effector at an arbitrary floating dot. That is busywork:
the geometry has nothing to do with any real Activity of Daily Living.

A `ScenarioSpec` replaces that with a concrete *task scene*: a posture (where the
body/shoulder sits), one or more **objects** placed in the world (a shoe on the
floor, a bottle on a table, a drawer at the waist), and the **waypoints** the
hand must actually reach to perform the task (down by the laces, at the cap, on
the handle). It is produced by `agents.scenario.ScenarioAgent` from the
Gemini/perception problem identification + the ADL task, and consumed by
`rl.scenario_env.ScenarioReachEnv` (training targets) and
`sim.gizmo_asset` (scene geometry, optionally Gizmo-generated).

Positions are world-frame metres, +y forward (in front of the body), +z up — the
same convention as `sim.mjcf_builder`.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

Vec3 = tuple[float, float, float]

# Canonical reach of the default 4-DoF arm (upper 0.30 + fore 0.26 + gripper 0.06).
# A shoulder-fixed arm can only touch points inside a sphere of this radius about
# the mount; waypoints are clamped onto it so every authored or LLM-proposed goal
# is reachable by construction.
DEFAULT_REACH_M = 0.62


def clamp_to_reach(mount: Vec3, pos: Vec3, reach: float = DEFAULT_REACH_M,
                   frac: float = 0.92) -> Vec3:
    """Pull `pos` onto the reachable sphere around `mount` if it sits outside it.

    Points already within `frac * reach` of the mount are returned unchanged;
    farther points are moved radially inward to exactly `frac * reach`, keeping
    their direction (so the hand still travels toward the object, just not past
    where the arm physically ends).
    """
    mx, my, mz = mount
    dx, dy, dz = pos[0] - mx, pos[1] - my, pos[2] - mz
    dist = math.sqrt(dx * dx + dy * dy + dz * dz)
    limit = frac * reach
    if dist <= limit or dist < 1e-9:
        return (float(pos[0]), float(pos[1]), float(pos[2]))
    s = limit / dist
    return (mx + dx * s, my + dy * s, mz + dz * s)


@dataclass
class TaskWaypoint:
    """A point in the world the hand must reach to advance the task.

    `weight` biases how often the goal sampler picks this waypoint as the episode
    target; the terminal/grasp waypoint usually gets the highest weight so the
    policy spends most of its budget learning the part that completes the task.
    """

    name: str
    pos: Vec3
    tolerance_m: float = 0.05
    dwell_s: float = 0.0
    weight: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TaskWaypoint":
        return cls(
            name=str(d.get("name", "waypoint")),
            pos=tuple(float(x) for x in d["pos"]),  # type: ignore[arg-type]
            tolerance_m=float(d.get("tolerance_m", 0.05)),
            dwell_s=float(d.get("dwell_s", 0.0)),
            weight=float(d.get("weight", 1.0)),
        )


@dataclass
class SceneObject:
    """A task object placed in the scene.

    `prompt` is the natural-language description handed to the Gizmo API to bake a
    physics-ready articulated MJCF (`scripts/recon/gizmo_assets.py`). `mjcf_dir`,
    once a bake lands in `assets/objects/<name>/`, points the injector at the real
    asset; until then the injector renders `fallback` (a coloured primitive box)
    so the scene is always complete and trainable without waiting on Gizmo.
    """

    name: str                       # slug -> assets/objects/<name>/
    prompt: str                     # Gizmo generation prompt
    pos: Vec3 = (0.0, 0.30, 0.10)
    rgba: tuple[float, float, float, float] = (0.80, 0.30, 0.25, 1.0)
    fallback_half: Vec3 = (0.06, 0.10, 0.04)  # half-extents of the fallback box
    mjcf_dir: str = ""              # set when a real Gizmo bake exists

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SceneObject":
        return cls(
            name=str(d["name"]),
            prompt=str(d.get("prompt", "")),
            pos=tuple(float(x) for x in d.get("pos", (0.0, 0.30, 0.10))),  # type: ignore[arg-type]
            rgba=tuple(float(x) for x in d.get("rgba", (0.80, 0.30, 0.25, 1.0))),  # type: ignore[arg-type]
            fallback_half=tuple(float(x) for x in d.get("fallback_half", (0.06, 0.10, 0.04))),  # type: ignore[arg-type]
            mjcf_dir=str(d.get("mjcf_dir", "")),
        )


# Posture -> world pose of the shoulder mount. "Bending down" is modelled by
# dropping and leaning the shoulder forward so a floor object comes within the
# arm's reach (a shoulder-fixed arm can't otherwise touch the floor). The
# scenario agent picks the posture; the env clamps any waypoint onto the
# reachable sphere as a safety net.
POSTURE_MOUNTS: dict[str, Vec3] = {
    "seated":       (0.0, -0.40, 1.00),   # default upright seat, hand works in front
    "table":        (0.0, -0.30, 0.95),   # leaned to a tabletop
    "lap":          (0.0, -0.25, 0.85),   # working in the lap
    "bent_forward": (0.0,  0.10, 0.80),   # hinged at the hip toward the knees
    "floor_reach":  (0.0,  0.25, 0.70),   # crouched, shoulder over the feet (clears floor)
}
DEFAULT_POSTURE = "seated"


@dataclass
class ScenarioSpec:
    """One trainable ADL task scene: posture + objects + reach waypoints."""

    task_id: str = ""
    primary_action: str = ""
    description: str = ""
    posture: str = DEFAULT_POSTURE
    mount_pos: Vec3 = POSTURE_MOUNTS[DEFAULT_POSTURE]
    objects: list[SceneObject] = field(default_factory=list)
    waypoints: list[TaskWaypoint] = field(default_factory=list)
    success_condition: str = ""
    source: str = ""               # "library:<key>" | "llm:<model>" | "fallback..."

    def validate(self) -> list[str]:
        problems: list[str] = []
        if not self.task_id:
            problems.append("task_id is required")
        if not self.waypoints:
            problems.append("at least one waypoint is required (nothing to reach)")
        if len(self.mount_pos) != 3:
            problems.append("mount_pos must be a 3-vector")
        return problems

    def primary_waypoint(self) -> TaskWaypoint:
        """The most task-defining waypoint (highest weight, else the last)."""
        if not self.waypoints:
            raise ValueError("scenario has no waypoints")
        return max(self.waypoints, key=lambda w: (w.weight, w.dwell_s))

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "primary_action": self.primary_action,
            "description": self.description,
            "posture": self.posture,
            "mount_pos": list(self.mount_pos),
            "objects": [o.to_dict() for o in self.objects],
            "waypoints": [w.to_dict() for w in self.waypoints],
            "success_condition": self.success_condition,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ScenarioSpec":
        posture = str(d.get("posture", DEFAULT_POSTURE))
        mount = d.get("mount_pos") or POSTURE_MOUNTS.get(posture, POSTURE_MOUNTS[DEFAULT_POSTURE])
        return cls(
            task_id=str(d.get("task_id", "")),
            primary_action=str(d.get("primary_action", "")),
            description=str(d.get("description", "")),
            posture=posture,
            mount_pos=tuple(float(x) for x in mount),  # type: ignore[arg-type]
            objects=[SceneObject.from_dict(o) for o in d.get("objects", [])],
            waypoints=[TaskWaypoint.from_dict(w) for w in d.get("waypoints", [])],
            success_condition=str(d.get("success_condition", "")),
            source=str(d.get("source", "")),
        )

    def to_json(self, indent: int | None = None) -> str:
        import json

        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, text: str) -> "ScenarioSpec":
        import json

        return cls.from_dict(json.loads(text))
