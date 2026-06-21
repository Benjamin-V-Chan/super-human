from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
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
    # Perception findings the design agent optimizes around (defaults keep the
    # contract backward-compatible).
    primary_action: str = ""  # specific observed action, e.g. "drinking from a bottle"
    affected_side: str = ""  # "left" | "right" — limb that needs the prosthesis
    residual_side: str = ""  # "left" | "right" — compensating limb


# ── Explicit kinematics ──────────────────────────────────────────────────────
# The design agent declares its degrees of freedom here, instead of the MJCF
# builder assuming a fixed arm. A `DesignParams.links` chain (proximal→distal)
# is the single source of truth for DoF count, joint axes/ranges, and which
# per-link mesh skins each moving body.


@dataclass(frozen=True)
class JointDef:
    """One actuated degree of freedom moving a link relative to its parent."""

    name: str
    axis: tuple[float, float, float]
    range_deg: tuple[float, float]
    type: str = "hinge"  # "hinge" | "slide"


@dataclass(frozen=True)
class LinkDef:
    """A body in the serial chain: its geometry + the DoF that move it.

    Frame convention (matches sim.mjcf_builder): the link's origin sits at its
    proximal joint and the link extends along local -Z by `length`. `mesh` is the
    per-link STL filename emitted by cad.bridge; None falls back to a primitive.
    """

    name: str
    length: float
    radius: float
    joints: tuple[JointDef, ...] = ()
    mesh: str | None = None
    rgba: tuple[float, float, float, float] = (0.70, 0.72, 0.78, 1.0)


def _as_degrees(lo: float, hi: float) -> tuple[float, float]:
    """Interpret a joint range as degrees, converting if it was given in radians.

    The design agent emits degrees, but PRD examples use radians (e.g. 2.531).
    Anything within ±2π is treated as radians and converted up to degrees so the
    rest of the contract stores a single unit.
    """
    if max(abs(lo), abs(hi)) <= 2.0 * math.pi:
        return math.degrees(lo), math.degrees(hi)
    return float(lo), float(hi)


def default_arm_chain(
    upper_arm_len: float,
    forearm_len: float,
    grip_width: float,
    joint_limits: dict[str, tuple[float, float]] | None,
) -> tuple[LinkDef, ...]:
    """The repo's canonical 4-DoF assistive arm, built from scalar params.

    Reproduces the previously-hardcoded topology so existing DesignParams keep
    working: upper_arm (shoulder flex + abduct) → forearm (elbow) → gripper
    (wrist). `joint_limits` overrides elbow/wrist ranges (degrees or radians).
    """
    lim = joint_limits or {}
    elbow = _as_degrees(*lim["elbow"]) if "elbow" in lim else (0.0, 130.0)
    wrist = _as_degrees(*lim["wrist"]) if "wrist" in lim else (-60.0, 60.0)
    return (
        LinkDef(
            name="upper_arm", length=float(upper_arm_len), radius=0.025,
            joints=(
                JointDef("shoulder_flex", (0, 1, 0), (-90.0, 120.0)),
                JointDef("shoulder_abduct", (1, 0, 0), (-60.0, 90.0)),
            ),
        ),
        LinkDef(
            name="forearm", length=float(forearm_len), radius=0.022,
            joints=(JointDef("elbow", (0, 1, 0), elbow),),
        ),
        LinkDef(
            name="gripper", length=0.06, radius=max(0.015, float(grip_width) / 2),
            joints=(JointDef("wrist", (1, 0, 0), wrist),),
            rgba=(0.85, 0.6, 0.2, 1.0),
        ),
    )


@dataclass
class DesignParams:
    upper_arm_len: float = 0.30
    forearm_len: float = 0.26
    joint_stiffness: float = 1.0
    grip_width: float = 0.08
    joint_limits: dict[str, tuple[float, float]] = field(default_factory=dict)
    # Explicit kinematic chain (proximal→distal). Empty => derive the canonical
    # 4-DoF arm from the scalar params above (backwards compatible).
    links: tuple[LinkDef, ...] = ()
    # Which shoulder the prosthesis mounts on — set by the design agent from
    # ProblemSpec.affected_side. Default "torso_right" is backward-compatible.
    mount_frame: str = "torso_right"

    def __post_init__(self) -> None:
        if not self.links:
            self.links = default_arm_chain(
                self.upper_arm_len, self.forearm_len,
                self.grip_width, self.joint_limits,
            )

    @property
    def dof(self) -> int:
        """Total actuated degrees of freedom across the chain."""
        return sum(len(link.joints) for link in self.links)

    @property
    def joint_names(self) -> list[str]:
        """Joint names in actuator/qpos order (proximal→distal)."""
        return [j.name for link in self.links for j in link.joints]


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
