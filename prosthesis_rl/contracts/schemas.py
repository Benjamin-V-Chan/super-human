from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any


@dataclass
class Constraints:
    rom: dict[str, float] = field(default_factory=dict)
    residual_strength: dict[str, float] = field(default_factory=dict)
    grip_capacity: float = 0.0


@dataclass
class ProblemSpec:
    tasks: list[dict[str, Any]] = field(default_factory=list)
    constraints: Constraints = field(default_factory=Constraints)


@dataclass
class DesignParams:
    upper_arm_len: float = 0.30
    forearm_len: float = 0.26
    joint_stiffness: float = 1.0
    grip_width: float = 0.08
    joint_limits: dict[str, tuple[float, float]] = field(default_factory=dict)


@dataclass
class RewardBreakdown:
    success: float = 0.0
    energy_penalty: float = 0.0
    rom_penalty: float = 0.0
    collision_penalty: float = 0.0

    @property
    def scalar(self) -> float:
        return (
            self.success
            - self.energy_penalty
            - self.rom_penalty
            - self.collision_penalty
        )


@dataclass
class SimFeedback:
    reward: float
    breakdown: RewardBreakdown = field(default_factory=RewardBreakdown)
    metrics: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_json(self, indent: int | None = None) -> str:
        return json.dumps(asdict(self), indent=indent)


@dataclass
class LinkSpec:
    name: str
    length_m: float
    mass_kg: float


@dataclass
class JointSpec:
    name: str
    type: str  # "hinge" | "ball"
    limits_rad: tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))


@dataclass
class ActuatorSpec:
    joint: str
    torque_limit_nm: float


@dataclass
class MorphologySpec:
    """Simulated limb morphology produced by the design agent."""
    mount_frame: str
    links: list[LinkSpec] = field(default_factory=list)
    joints: list[JointSpec] = field(default_factory=list)
    actuators: list[ActuatorSpec] = field(default_factory=list)


@dataclass
class EvalResult:
    """Per-candidate evaluation summary from the simulation verifier."""
    task_id: str = ""
    num_rollouts: int = 0
    success_rate: float = 0.0
    mean_reward: float = 0.0
    mean_energy: float = 0.0
    collision_rate: float = 0.0
    video_path: str = ""
