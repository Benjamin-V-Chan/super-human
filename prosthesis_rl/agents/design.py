"""Morphology design agent — Benji's owned module.

Responsibilities: ProblemSpec + RequirementsAgent brief + sim feedback ->
DesignParams candidates, spatial validation, multi-seed evaluation, and
design rationale reporting.

DesignParams.links (LinkDef chain) IS the MorphologySpec — the coordinator
added JointDef/LinkDef/default_arm_chain to contracts so no local duplicates.
EvalResult now lives in contracts too; re-exported here for back-compat.
Extended fields (reward_variance, dist, ROM violations, fatigue) proposed for
next contracts iteration (see morphology-phase2-3.md handoff).
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any

from prosthesis_rl.contracts import (
    DesignParams,
    EvalResult,
    JointDef,
    LinkDef,
    ProblemSpec,
    SimFeedback,
)

__all__ = ["DesignAgent", "EvalResult"]


# ── Extended eval for internal Phase-2/3 use (superset of contracts.EvalResult) ──

@dataclass
class _FullEval:
    """EvalResult + fields pending next contracts iteration."""
    task_id: str = ""
    num_rollouts: int = 0
    success_rate: float = 0.0
    mean_reward: float = 0.0
    reward_variance: float = 0.0
    mean_energy: float = 0.0
    mean_final_dist_cm: float = 0.0
    collision_rate: float = 0.0
    rom_violation_mean: float = 0.0
    peak_stress_mpa: float = 0.0
    predicted_life_years: float = float("inf")
    video_path: str = ""

    def to_eval_result(self) -> EvalResult:
        return EvalResult(
            task_id=self.task_id,
            num_rollouts=self.num_rollouts,
            success_rate=self.success_rate,
            mean_reward=self.mean_reward,
            mean_energy=self.mean_energy,
            collision_rate=self.collision_rate,
            video_path=self.video_path,
        )

    def summary_line(self, label: str = "") -> str:
        tag = f"[{label}] " if label else ""
        life_s = f"{self.predicted_life_years:.1f}yr" if self.predicted_life_years < 100 else ">=100yr"
        return (
            f"{tag}reward={self.mean_reward:+.3f} "
            f"(var={self.reward_variance:.3f}) "
            f"success={self.success_rate:.0%} "
            f"dist={self.mean_final_dist_cm:.1f}cm "
            f"energy={self.mean_energy:.0f}J "
            f"coll={self.collision_rate:.0%} "
            f"life={life_s}"
        )


# ── Agent ─────────────────────────────────────────────────────────────────────

class DesignAgent:
    """ProblemSpec + RequirementsAgent brief + sim feedback -> DesignParams candidates.

    Returns DesignParams with an explicit links chain so the MJCF builder,
    verifier, and RL env all see the same kinematic tree.
    """

    # ── Phase-1: single proposal ──────────────────────────────────────────────

    def propose(
        self,
        problem: ProblemSpec,
        feedback: SimFeedback | None = None,
        brief: dict[str, Any] | None = None,
    ) -> tuple[DesignParams, dict[str, float]]:
        """Return a validated DesignParams from problem, optional brief, and feedback.

        If `brief` is provided (from RequirementsAgent.derive()), its action-specific
        ROM targets and dimensions override the generic defaults.
        """
        if brief:
            upper_m, forearm_m, grip_w, stiffness, elbow, wrist = self._params_from_brief(brief)
        else:
            upper_m, forearm_m = 0.30, 0.26
            grip_w = max(0.06, min(0.12, problem.constraints.grip_capacity * 0.15 + 0.06))
            elbow = (0.0, 130.0)
            wrist = (-60.0, 60.0)
            stiffness = 1.0

        if feedback is not None:
            if feedback.breakdown.rom_penalty > 0.1:
                elbow = (elbow[0], min(160.0, elbow[1] + 20.0))
                wrist = (max(-80.0, wrist[0] - 10.0), min(80.0, wrist[1] + 10.0))
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
            elbow=elbow, wrist=wrist, mount_frame=mount,
        )
        errors = self.validate(params, self._task_reach_m(problem))
        if errors:
            raise ValueError(f"Proposed morphology failed validation: {errors}")

        grip_force = brief["design_params"]["grip_force_target_n"] / 100.0 if brief else 0.35
        control_hints: dict[str, float] = {"ik_weight": 1.0, "grip_force_target": grip_force}
        return params, control_hints

    # ── Phase-2: multi-candidate generation ──────────────────────────────────

    def propose_candidates(
        self,
        problem: ProblemSpec,
        feedback: SimFeedback | None = None,
        brief: dict[str, Any] | None = None,
        n: int = 3,
    ) -> list[tuple[DesignParams, dict[str, float]]]:
        """Generate n candidates with systematic dimensional variation around the base.

        When `brief` is provided, the base design uses action-specific ROM and
        dimensions; variations explore the envelope around that base.
        """
        base, base_hints = self.propose(problem, feedback, brief)
        candidates: list[tuple[DesignParams, dict[str, float]]] = [(base, base_hints)]
        hints: dict[str, float] = {**base_hints}

        side = problem.affected_side or "right"
        mount = f"torso_{side}" if side in {"left", "right"} else "torso_right"
        grip_w = base.grip_width
        base_elbow = next(
            (j.range_deg for l in base.links for j in l.joints if j.name == "elbow"),
            (0.0, 130.0),
        )
        base_wrist = next(
            (j.range_deg for l in base.links for j in l.joints if j.name == "wrist"),
            (-60.0, 60.0),
        )

        # Variation A: longer reach + wider ROM (prioritises far-target success)
        # Variation B: shorter + stiffer (prioritises energy efficiency and precision)
        variations = [
            dict(
                upper_m=min(0.36, base.upper_arm_len + 0.04),
                forearm_m=min(0.32, base.forearm_len + 0.04),
                elbow=(base_elbow[0], min(160.0, base_elbow[1] + 15.0)),
                wrist=(max(-80.0, base_wrist[0] - 10.0), min(80.0, base_wrist[1] + 10.0)),
                stiffness=base.joint_stiffness,
            ),
            dict(
                upper_m=max(0.22, base.upper_arm_len - 0.04),
                forearm_m=max(0.20, base.forearm_len - 0.04),
                elbow=(base_elbow[0], max(100.0, base_elbow[1] - 10.0)),
                wrist=(min(-40.0, base_wrist[0] + 10.0), max(40.0, base_wrist[1] - 10.0)),
                stiffness=min(1.4, base.joint_stiffness + 0.3),
            ),
        ]
        reach = self._task_reach_m(problem)
        for v in variations[: n - 1]:
            params = self._build_params(
                v["upper_m"], v["forearm_m"], grip_w, v["stiffness"],
                elbow=v["elbow"], wrist=v["wrist"], mount_frame=mount,
            )
            if not self.validate(params, reach):
                candidates.append((params, hints))
            if len(candidates) >= n:
                break

        return candidates

    # ── Phase-2: validation gates ─────────────────────────────────────────────

    def validate(self, params: DesignParams, task_reach_m: float = 0.0) -> list[str]:
        """Return validation errors (empty = valid).

        Checks: positive link geometry, valid joint ranges, total reach vs task,
        unique joint names, and non-zero joint axes.
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

    # ── Phase-2/3: multi-seed evaluation ─────────────────────────────────────

    def evaluate_candidates(
        self,
        candidates: list[tuple[DesignParams, dict[str, float]]],
        problem: ProblemSpec,
        verifier,
        cad,
        *,
        n_seeds: int = 10,
        n_targets: int = 4,
        seconds: float = 3.0,
        task_id: str = "reach_v1",
    ) -> list[_FullEval]:
        """Run each candidate through the verifier with n_seeds fixed seeds.

        Aggregates SimFeedback scalars into _FullEval, satisfying the evaluation
        protocol: >=10 fixed-seed rollouts, reporting success rate, mean reward,
        variance, distance, energy, collision rate, and stability.
        """
        results: list[_FullEval] = []

        for params, control_hints in candidates:
            mesh_dir = cad.export_arm(params)
            rewards, successes, energies, dists, colls, roms = [], [], [], [], [], []
            stress, life = 0.0, float("inf")

            for seed in range(n_seeds):
                fb: SimFeedback = verifier.evaluate(
                    problem, params, control_hints,
                    mesh_dir=mesh_dir,
                    n_targets=n_targets,
                    seconds=seconds,
                    seed=seed,
                )
                rewards.append(fb.reward)
                successes.append(fb.metrics.get("reach_success", 0.0))
                energies.append(fb.metrics.get("energy", 0.0))
                dists.append(fb.metrics.get("final_distance_cm", 0.0))
                colls.append(fb.metrics.get("self_collision", 0.0))
                roms.append(fb.metrics.get("rom_violation", 0.0))
                stress = max(stress, fb.metrics.get("peak_stress_mpa", 0.0))
                fb_life = fb.metrics.get("predicted_life_years", float("inf"))
                life = min(life, fb_life)

            results.append(_FullEval(
                task_id=task_id,
                num_rollouts=n_seeds,
                success_rate=statistics.mean(successes),
                mean_reward=statistics.mean(rewards),
                reward_variance=statistics.variance(rewards) if len(rewards) > 1 else 0.0,
                mean_energy=statistics.mean(energies),
                mean_final_dist_cm=statistics.mean(dists),
                collision_rate=statistics.mean(colls),
                rom_violation_mean=statistics.mean(roms),
                peak_stress_mpa=stress,
                predicted_life_years=life,
            ))

        return results

    # ── Phase-3: comparison and rationale ────────────────────────────────────

    def compare(
        self,
        candidates: list[DesignParams],
        eval_results: list[EvalResult | _FullEval],
    ) -> tuple[int, str]:
        """Pick best candidate by (mean_reward, success_rate, -collision_rate).

        Returns (index_of_best, one-line rationale).
        """
        if not candidates or not eval_results:
            raise ValueError("Need at least one candidate and one eval result")
        if len(candidates) != len(eval_results):
            raise ValueError("candidates and eval_results must have the same length")

        best_i = 0
        best = eval_results[0]
        for i, r in enumerate(eval_results[1:], 1):
            if (r.mean_reward, r.success_rate, -r.collision_rate) > (
                best.mean_reward, best.success_rate, -best.collision_rate,
            ):
                best_i, best = i, r

        rationale = (
            f"Candidate {best_i} selected: mean_reward={best.mean_reward:+.3f}, "
            f"success_rate={best.success_rate:.0%}, "
            f"collision_rate={best.collision_rate:.0%}"
        )
        return best_i, rationale

    def rationale_report(
        self,
        candidates: list[DesignParams],
        eval_results: list,
        best_i: int,
        rationale: str,
        action: str = "",
    ) -> str:
        """Format a human-readable design rationale comparing all candidates.

        Satisfies TECHNICAL_PLAN.md evaluation protocol: reports success rate,
        mean reward, variance, distance-to-goal, energy, collision, ROM violations,
        and stability (life estimate) per candidate.
        """
        lines: list[str] = []
        lines.append("=" * 70)
        lines.append("  MORPHOLOGY CANDIDATE COMPARISON — DESIGN RATIONALE")
        if action:
            lines.append(f"  Action: {action}")
        lines.append("=" * 70)
        lines.append(
            f"  {'#':<3} {'upper_m':>8} {'fore_m':>7} {'elbow_hi':>9} "
            f"{'wrist':>10} {'reward':>8} {'var':>6} {'succ':>6} "
            f"{'dist':>7} {'energy':>8} {'coll':>6} {'life':>8}"
        )
        lines.append("  " + "-" * 68)

        for i, (params, er) in enumerate(zip(candidates, eval_results)):
            elbow_hi = next(
                (j.range_deg[1] for l in params.links for j in l.joints if j.name == "elbow"), 0.0
            )
            wrist_hi = next(
                (j.range_deg[1] for l in params.links for j in l.joints if j.name == "wrist"), 0.0
            )
            tag = " ◄ BEST" if i == best_i else ""
            life = getattr(er, "predicted_life_years", float("inf"))
            life_str = f"{life:.1f}yr" if life < 100 else ">=100yr"
            var = getattr(er, "reward_variance", 0.0)
            dist = getattr(er, "mean_final_dist_cm", 0.0)
            lines.append(
                f"  {i:<3} {params.upper_arm_len:>8.3f} {params.forearm_len:>7.3f} "
                f"{elbow_hi:>9.1f}° {wrist_hi:>9.1f}° "
                f"{er.mean_reward:>+8.3f} {var:>6.3f} "
                f"{er.success_rate:>5.0%} {dist:>6.1f}cm "
                f"{er.mean_energy:>7.0f}J {er.collision_rate:>5.0%} "
                f"{life_str:>8}{tag}"
            )

        lines.append("  " + "-" * 68)
        lines.append(f"  WINNER: {rationale}")
        lines.append("")

        best_er = eval_results[best_i]
        best_life = getattr(best_er, "predicted_life_years", float("inf"))
        best_rom = getattr(best_er, "rom_violation_mean", 0.0)
        lines.append("  FAILURE MODE ANALYSIS (best candidate):")
        if best_er.success_rate < 0.5:
            lines.append("    ✗ Low success rate — IK struggles to reach FK-sampled targets.")
            lines.append("      Consider longer links or wider elbow range.")
        if best_er.mean_energy > 500:
            lines.append("    ✗ High energy — actuators are working hard; consider stiffer joints.")
        if best_er.collision_rate > 0.1:
            lines.append("    ✗ Self-collision detected — shorten links or tighten elbow range.")
        if best_rom > 0.01:
            lines.append("    ✗ ROM violations — joint limits may be too tight for the IK solver.")
        if best_life < 1.0:
            lines.append("    ✗ Very short predicted service life — peak torques exceed material limits.")
        if (
            best_er.success_rate >= 0.5
            and best_er.mean_energy <= 500
            and best_er.collision_rate <= 0.1
        ):
            lines.append("    ✓ No critical failure modes — design passes baseline gates.")

        lines.append("=" * 70)
        return "\n".join(lines)

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _task_reach_m(problem: ProblemSpec) -> float:
        return float(problem.constraints.rom.get("reach_m", 0.5))

    @staticmethod
    def _params_from_brief(brief: dict[str, Any]) -> tuple:
        """Extract (upper_m, forearm_m, grip_w, stiffness, elbow, wrist) from brief."""
        dp = brief.get("design_params", {})
        upper_m = float(dp.get("upper_arm_len", 0.30))
        forearm_m = float(dp.get("forearm_len", 0.26))
        grip_w = float(dp.get("grip_width", 0.08))
        stiffness = max(0.5, min(2.0, float(dp.get("joint_stiffness", 10.0)) / 10.0))

        rom = brief.get("rom_targets_deg", {})
        el = rom.get("elbow_flexion", [0.0, 130.0])
        wr = rom.get("wrist_rotation", [-60.0, 60.0])
        elbow: tuple[float, float] = (float(el[0]), float(el[1]))
        wrist: tuple[float, float] = (float(wr[0]), float(wr[1]))
        return upper_m, forearm_m, grip_w, stiffness, elbow, wrist

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
