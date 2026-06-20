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
