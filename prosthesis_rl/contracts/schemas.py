from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
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
    # Anthropometrics of the intact (residual) arm, in meters — the prosthesis is
    # sized to mirror the contralateral limb. Keys: upper_arm_len, forearm_len,
    # hand_length, grip_span.
    residual_anthropometrics: dict[str, float] = field(default_factory=dict)


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


@dataclass
class OrchestrationAttempt:
    """One design, CAD export, and verification pass."""

    index: int
    design: DesignParams
    control_hints: dict[str, float]
    artifact_path: str
    feedback: SimFeedback


@dataclass
class OrchestrationResult:
    """Inspectable outcome of an iterative design run."""

    problem: ProblemSpec
    attempts: list[OrchestrationAttempt]
    best_attempt_index: int
    stop_reason: str

    def __post_init__(self) -> None:
        if not self.attempts:
            raise ValueError("an orchestration result requires at least one attempt")
        if not 0 <= self.best_attempt_index < len(self.attempts):
            raise ValueError("best_attempt_index must reference an attempt")

    @property
    def best_attempt(self) -> OrchestrationAttempt:
        return self.attempts[self.best_attempt_index]

    @property
    def reward(self) -> float:
        """Compatibility shortcut for score-only callers."""

        return self.best_attempt.feedback.reward

    def to_json(self, indent: int | None = None) -> str:
        return json.dumps(asdict(self), indent=indent)


# ── End-to-end pipeline contracts ─────────────────────────────────────────────
# The runtime loop is: TaskSpec (intake) → DesignParams + SimSpec (build) →
# PolicyArtifact (control) → EvalResult (evaluation). These mirror
# docs/TECHNICAL_PLAN.md and are the single source of truth every stage agrees
# on. Note: the plan's "MorphologySpec" is realized by `DesignParams` — its
# `links` (LinkDef chain) IS the morphology, so there is no separate class.


class _JsonContract:
    """Shared JSON (de)serialization for the flat pipeline contracts."""

    def to_json(self, indent: int | None = None) -> str:
        return json.dumps(asdict(self), indent=indent)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        """Build from a dict, ignoring unknown keys (tolerant of extra fields)."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def from_json(cls, text: str):
        return cls.from_dict(json.loads(text))


@dataclass
class TaskSpec(_JsonContract):
    """A concrete, measurable ADL task — the output of task intake.

    Uncertain clip/prompt details belong in `assumptions`, not in `goal`.
    """

    task_id: str = ""
    goal: str = ""
    objects: list[str] = field(default_factory=list)
    success_condition: str = ""
    episode_seconds: float = 8.0
    assumptions: list[str] = field(default_factory=list)

    def validate(self) -> list[str]:
        """Return a list of contract violations; empty means valid."""
        problems: list[str] = []
        if not self.task_id:
            problems.append("task_id is required")
        if not self.goal:
            problems.append("goal is required")
        if not self.success_condition:
            problems.append("success_condition is required (must be measurable)")
        if self.episode_seconds <= 0:
            problems.append("episode_seconds must be > 0")
        return problems


@dataclass
class SimSpec(_JsonContract):
    """Scene, timing, reward, and observation config for one runnable env."""

    scene: str = ""
    physics_hz: int = 100
    control_hz: int = 20
    initial_state_seed: int = 0
    reward_terms: list[str] = field(
        default_factory=lambda: ["success", "distance", "energy", "collision", "joint_limit"]
    )
    observations: list[str] = field(
        default_factory=lambda: ["joint_pos", "joint_vel", "target_pose", "end_effector_pose"]
    )

    def validate(self) -> list[str]:
        problems: list[str] = []
        if not self.scene:
            problems.append("scene is required")
        if self.physics_hz <= 0:
            problems.append("physics_hz must be > 0")
        if self.control_hz <= 0:
            problems.append("control_hz must be > 0")
        if self.control_hz > self.physics_hz:
            problems.append("control_hz must not exceed physics_hz")
        if not self.reward_terms:
            problems.append("at least one reward term is required")
        return problems


@dataclass
class PolicyArtifact(_JsonContract):
    """A loadable controller: scripted/IK config or an RL checkpoint."""

    kind: str = "scripted_ik"  # "scripted_ik" | "rl_checkpoint" | ...
    path: str = ""
    inputs: list[str] = field(default_factory=lambda: ["observation"])
    outputs: list[str] = field(default_factory=lambda: ["joint_targets"])
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> list[str]:
        problems: list[str] = []
        if not self.kind:
            problems.append("kind is required")
        elif self.kind not in {"scripted_ik", "rl_checkpoint"}:
            problems.append(f"unsupported policy kind: {self.kind}")
        if not self.path:
            problems.append("path is required")
        if not self.inputs:
            problems.append("at least one policy input is required")
        if not self.outputs:
            problems.append("at least one policy output is required")
        return problems


@dataclass
class EvalResult(_JsonContract):
    """Per-candidate evaluation summary produced by the verifier/evaluator."""

    task_id: str = ""
    num_rollouts: int = 0
    success_rate: float = 0.0
    mean_reward: float = 0.0
    mean_energy: float = 0.0
    collision_rate: float = 0.0
    video_path: str = ""

    def validate(self) -> list[str]:
        problems: list[str] = []
        if not self.task_id:
            problems.append("task_id is required")
        if self.num_rollouts < 1:
            problems.append("num_rollouts must be at least 1")
        if not 0.0 <= self.success_rate <= 1.0:
            problems.append("success_rate must be between 0 and 1")
        if self.mean_energy < 0.0:
            problems.append("mean_energy cannot be negative")
        if not 0.0 <= self.collision_rate <= 1.0:
            problems.append("collision_rate must be between 0 and 1")
        return problems
