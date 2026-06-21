"""Morphology design agent — Benji's owned module.

Responsibilities: ProblemSpec + sim feedback -> DesignParams candidates with
explicit kinematic chains, spatial/reachability validation, and candidate ranking.

DesignParams.links (LinkDef chain) IS the MorphologySpec — the coordinator
added JointDef/LinkDef/default_arm_chain to contracts so no local duplicates.
EvalResult now lives in contracts too; re-exported here for back-compat.
"""

from __future__ import annotations

from prosthesis_rl.contracts import (
    DesignParams,
    EvalResult,
    JointDef,
    LinkDef,
    ProblemSpec,
    SimFeedback,
)

__all__ = ["DesignAgent", "EvalResult"]


# ── Agent ─────────────────────────────────────────────────────────────────────

class DesignAgent:
    """ProblemSpec + sim feedback -> DesignParams candidates with spatial validation.

    Returns DesignParams with an explicit links chain so the MJCF builder,
    verifier, and RL env all see the same kinematic tree.
    """

    def propose(
        self,
        problem: ProblemSpec,
        feedback: SimFeedback | None = None,
    ) -> tuple[DesignParams, dict[str, float]]:
        """Return a validated DesignParams informed by problem constraints and feedback."""
        upper_m = 0.30
        forearm_m = 0.26
        grip_w = max(0.06, min(0.12, problem.constraints.grip_capacity * 0.15 + 0.06))

        elbow_lo, elbow_hi = 0.0, 130.0    # degrees — matches default_arm_chain
        wrist_lo, wrist_hi = -60.0, 60.0
        stiffness = 1.0

        if feedback is not None:
            if feedback.breakdown.rom_penalty > 0.1:
                elbow_hi = min(160.0, elbow_hi + 20.0)
                wrist_hi = min(80.0, wrist_hi + 10.0)
                wrist_lo = max(-80.0, wrist_lo - 10.0)
            if feedback.breakdown.collision_penalty > 0.1:
                upper_m = max(0.22, upper_m - 0.02)
                forearm_m = max(0.20, forearm_m - 0.02)
            if feedback.reward < 0.25:
                forearm_m = min(0.34, forearm_m + 0.02)
            if feedback.breakdown.energy_penalty > 0.15:
                stiffness = max(0.7, stiffness - 0.2)

        side = problem.affected_side or "right"
        mount = f"torso_{side}" if side in {"left", "right"} else "torso_right"

        params = self._build_params(
            upper_m, forearm_m, grip_w, stiffness,
            elbow=(elbow_lo, elbow_hi), wrist=(wrist_lo, wrist_hi),
            mount_frame=mount,
        )
        errors = self.validate(params, self._task_reach_m(problem))
        if errors:
            raise ValueError(f"Proposed morphology failed validation: {errors}")

        control_hints: dict[str, float] = {"ik_weight": 1.0, "grip_force_target": 0.35}
        return params, control_hints

    def propose_candidates(
        self,
        problem: ProblemSpec,
        feedback: SimFeedback | None = None,
        n: int = 2,
    ) -> list[tuple[DesignParams, dict[str, float]]]:
        """Generate n candidates with systematic dimensional variation."""
        base, base_hints = self.propose(problem, feedback)
        candidates: list[tuple[DesignParams, dict[str, float]]] = [(base, base_hints)]
        hints: dict[str, float] = {"ik_weight": 1.0, "grip_force_target": 0.35}

        variations = [
            dict(upper_m=0.32, forearm_m=0.28, elbow=(0.0, 145.0), wrist=(-70.0, 70.0)),
            dict(upper_m=0.26, forearm_m=0.22, elbow=(0.0, 120.0), wrist=(-50.0, 50.0)),
            dict(upper_m=0.34, forearm_m=0.30, elbow=(0.0, 135.0), wrist=(-60.0, 60.0)),
        ]
        reach = self._task_reach_m(problem)
        for v in variations[: n - 1]:
            grip_w = max(0.06, min(0.12, problem.constraints.grip_capacity * 0.15 + 0.06))
            params = self._build_params(
                v["upper_m"], v["forearm_m"], grip_w, 1.0,
                elbow=v["elbow"], wrist=v["wrist"],
            )
            if not self.validate(params, reach):
                candidates.append((params, hints))
            if len(candidates) >= n:
                break

        return candidates

    def validate(self, params: DesignParams, task_reach_m: float = 0.0) -> list[str]:
        """Return validation errors (empty = valid).

        Checks: positive link geometry, valid joint ranges, total reach vs task,
        and that each joint name is unique with a non-zero axis.
        """
        errors: list[str] = []
        total_reach = 0.0
        seen_joints: set[str] = set()

        for link in params.links:
            if link.length <= 0:
                errors.append(f"Link '{link.name}' has non-positive length {link.length}")
            if link.radius <= 0:
                errors.append(f"Link '{link.name}' has non-positive radius {link.radius}")
            total_reach += link.length

            for joint in link.joints:
                lo, hi = joint.range_deg
                if lo >= hi:
                    errors.append(
                        f"Joint '{joint.name}' range [{lo:.1f}, {hi:.1f}] deg invalid (lower >= upper)"
                    )
                if joint.type not in {"hinge", "slide"}:
                    errors.append(f"Joint '{joint.name}' has unknown type '{joint.type}'")
                if all(abs(a) < 1e-9 for a in joint.axis):
                    errors.append(f"Joint '{joint.name}' has zero axis vector")
                if joint.name in seen_joints:
                    errors.append(f"Duplicate joint name '{joint.name}'")
                seen_joints.add(joint.name)

        if total_reach < task_reach_m:
            errors.append(
                f"Total arm reach {total_reach:.3f} m < required {task_reach_m:.3f} m"
            )

        return errors

    def compare(
        self,
        candidates: list[DesignParams],
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
    def _build_params(
        upper_m: float,
        forearm_m: float,
        grip_w: float,
        stiffness: float,
        *,
        elbow: tuple[float, float] = (0.0, 130.0),
        wrist: tuple[float, float] = (-60.0, 60.0),
        mount_frame: str = "torso_right",
    ) -> DesignParams:
        links = (
            LinkDef(
                name="upper_arm", length=upper_m, radius=0.025,
                joints=(
                    JointDef("shoulder_flex", (0, 1, 0), (-90.0, 120.0)),
                    JointDef("shoulder_abduct", (1, 0, 0), (-60.0, 90.0)),
                ),
            ),
            LinkDef(
                name="forearm", length=forearm_m, radius=0.022,
                joints=(JointDef("elbow", (0, 1, 0), elbow),),
            ),
            LinkDef(
                name="gripper", length=0.06, radius=max(0.015, grip_w / 2),
                joints=(JointDef("wrist", (1, 0, 0), wrist),),
                rgba=(0.85, 0.6, 0.2, 1.0),
            ),
        )
        return DesignParams(
            upper_arm_len=upper_m,
            forearm_len=forearm_m,
            joint_stiffness=stiffness,
            grip_width=grip_w,
            joint_limits={"elbow": elbow, "wrist": wrist},
            links=links,
            mount_frame=mount_frame,
        )
