"""Morphology design agent — Benji's owned module.

Responsibilities: ProblemSpec + sim feedback -> MorphologySpec candidates,
spatial/reachability validation, candidate comparison.

MorphologySpec / EvalResult dataclasses live here until the coordinator
(Vihaan) adds them to prosthesis_rl/contracts/schemas.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from prosthesis_rl.contracts import ProblemSpec, SimFeedback


# ── Pending contract additions (propose to coordinator) ───────────────────────

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
    """Simulated limb morphology. Migrate to contracts once coordinator adds it."""
    mount_frame: str
    links: list[LinkSpec] = field(default_factory=list)
    joints: list[JointSpec] = field(default_factory=list)
    actuators: list[ActuatorSpec] = field(default_factory=list)


@dataclass
class EvalResult:
    """Per-candidate evaluation summary. Migrate to contracts once coordinator adds it."""
    task_id: str = ""
    num_rollouts: int = 0
    success_rate: float = 0.0
    mean_reward: float = 0.0
    mean_energy: float = 0.0
    collision_rate: float = 0.0
    video_path: str = ""


# ── Constants ─────────────────────────────────────────────────────────────────

_VALID_JOINT_TYPES = {"hinge", "ball"}
_DEFAULT_UPPER_ARM_M = 0.30
_DEFAULT_FOREARM_M = 0.26
_DEFAULT_UPPER_MASS_KG = 0.80
_DEFAULT_FOREARM_MASS_KG = 0.60
_DEFAULT_SHOULDER_TORQUE_NM = 20.0
_DEFAULT_ELBOW_TORQUE_NM = 15.0


# ── Agent ─────────────────────────────────────────────────────────────────────

class DesignAgent:
    """ProblemSpec + sim feedback -> MorphologySpec candidates with spatial validation."""

    def propose(
        self,
        problem: ProblemSpec,
        feedback: SimFeedback | None = None,
    ) -> tuple[MorphologySpec, dict[str, float]]:
        """Return a single validated MorphologySpec informed by problem and feedback."""
        upper_m = _DEFAULT_UPPER_ARM_M
        forearm_m = _DEFAULT_FOREARM_M
        shoulder_lim = (0.0, math.radians(180.0))
        elbow_lim = (0.0, math.radians(145.0))

        if feedback is not None:
            if feedback.breakdown.rom_penalty > 0.1:
                shoulder_lim = (0.0, math.radians(200.0))
                elbow_lim = (0.0, math.radians(160.0))
            if feedback.breakdown.collision_penalty > 0.1:
                upper_m = max(0.20, upper_m - 0.02)
                forearm_m = max(0.18, forearm_m - 0.02)
            if feedback.reward < 0.25:
                forearm_m = min(0.34, forearm_m + 0.02)

        mount = self._mount_frame(problem)
        spec = self._build_spec(upper_m, forearm_m, shoulder_lim, elbow_lim, mount)
        errors = self.validate(spec, self._task_reach_m(problem))
        if errors:
            raise ValueError(f"Proposed morphology failed validation: {errors}")

        control_hints: dict[str, float] = {"ik_weight": 1.0, "grip_force_target": 0.35}
        return spec, control_hints

    def propose_candidates(
        self,
        problem: ProblemSpec,
        feedback: SimFeedback | None = None,
        n: int = 2,
    ) -> list[tuple[MorphologySpec, dict[str, float]]]:
        """Generate n morphology candidates with systematic dimensional variation."""
        base, base_hints = self.propose(problem, feedback)
        candidates: list[tuple[MorphologySpec, dict[str, float]]] = [(base, base_hints)]

        variations = [
            # longer / more ROM
            (0.32, 0.28, (0.0, math.radians(190.0)), (0.0, math.radians(155.0))),
            # shorter / compact
            (0.26, 0.22, (0.0, math.radians(160.0)), (0.0, math.radians(130.0))),
            # extended reach
            (0.34, 0.30, (0.0, math.radians(180.0)), (0.0, math.radians(145.0))),
        ]
        reach = self._task_reach_m(problem)
        mount = self._mount_frame(problem)
        for upper_m, forearm_m, sh_lim, el_lim in variations[: n - 1]:
            spec = self._build_spec(upper_m, forearm_m, sh_lim, el_lim, mount)
            if not self.validate(spec, reach):
                hints: dict[str, float] = {"ik_weight": 1.0, "grip_force_target": 0.35}
                candidates.append((spec, hints))
            if len(candidates) >= n:
                break

        return candidates

    def validate(self, spec: MorphologySpec, task_reach_m: float = 0.0) -> list[str]:
        """Return a list of validation errors; empty means the spec is valid.

        Checks: positive link dimensions, valid joint types and limits,
        actuator coverage, and whether total reach meets the task requirement.
        """
        errors: list[str] = []
        joint_names = {j.name for j in spec.joints}
        actuated = {a.joint for a in spec.actuators}

        for link in spec.links:
            if link.length_m <= 0:
                errors.append(
                    f"Link '{link.name}' has non-positive length {link.length_m}"
                )
            if link.mass_kg <= 0:
                errors.append(
                    f"Link '{link.name}' has non-positive mass {link.mass_kg}"
                )

        for joint in spec.joints:
            if joint.type not in _VALID_JOINT_TYPES:
                errors.append(
                    f"Joint '{joint.name}' has invalid type '{joint.type}'"
                )
            lo, hi = joint.limits_rad
            if lo >= hi:
                errors.append(
                    f"Joint '{joint.name}' limits [{lo:.3f}, {hi:.3f}] invalid (lower >= upper)"
                )

        for act in spec.actuators:
            if act.joint not in joint_names:
                errors.append(f"Actuator references unknown joint '{act.joint}'")
            if act.torque_limit_nm <= 0:
                errors.append(
                    f"Actuator for '{act.joint}' has non-positive torque {act.torque_limit_nm}"
                )

        for jname in joint_names:
            if jname not in actuated:
                errors.append(f"Joint '{jname}' has no actuator")

        total_reach = sum(link.length_m for link in spec.links)
        if total_reach < task_reach_m:
            errors.append(
                f"Total arm reach {total_reach:.3f} m < required {task_reach_m:.3f} m"
            )

        return errors

    def compare(
        self,
        candidates: list[MorphologySpec],
        eval_results: list[EvalResult],
    ) -> tuple[int, str]:
        """Pick best candidate by (mean_reward, success_rate, -collision_rate).

        Returns (index_of_best, rationale_string).
        """
        if not candidates or not eval_results:
            raise ValueError("Need at least one candidate and one eval result")
        if len(candidates) != len(eval_results):
            raise ValueError("candidates and eval_results must have the same length")

        best_i = 0
        best = eval_results[0]
        for i, r in enumerate(eval_results[1:], 1):
            if (r.mean_reward, r.success_rate, -r.collision_rate) > (
                best.mean_reward,
                best.success_rate,
                -best.collision_rate,
            ):
                best_i, best = i, r

        rationale = (
            f"Candidate {best_i} selected: mean_reward={best.mean_reward:.3f}, "
            f"success_rate={best.success_rate:.2%}, collision_rate={best.collision_rate:.2%}"
        )
        return best_i, rationale

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _task_reach_m(problem: ProblemSpec) -> float:
        """Extract minimum reach (m) from constraints.rom, default 0.5 m."""
        return float(problem.constraints.rom.get("reach_m", 0.5))

    @staticmethod
    def _mount_frame(problem: ProblemSpec) -> str:
        """Mount the prosthesis on the affected side from perception."""
        side = getattr(problem, "affected_side", "") or "right"
        return f"torso_{side}" if side in {"left", "right"} else "torso_right"

    @staticmethod
    def _build_spec(
        upper_m: float,
        forearm_m: float,
        shoulder_lim: tuple[float, float],
        elbow_lim: tuple[float, float],
        mount_frame: str = "torso_right",
    ) -> MorphologySpec:
        return MorphologySpec(
            mount_frame=mount_frame,
            links=[
                LinkSpec("upper", upper_m, _DEFAULT_UPPER_MASS_KG),
                LinkSpec("forearm", forearm_m, _DEFAULT_FOREARM_MASS_KG),
            ],
            joints=[
                JointSpec("shoulder_flexion", "hinge", shoulder_lim),
                JointSpec("elbow_flexion", "hinge", elbow_lim),
            ],
            actuators=[
                ActuatorSpec("shoulder_flexion", _DEFAULT_SHOULDER_TORQUE_NM),
                ActuatorSpec("elbow_flexion", _DEFAULT_ELBOW_TORQUE_NM),
            ],
        )
