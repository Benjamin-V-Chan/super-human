"""Evaluation agent that converts deterministic verifier rollouts to EvalResult."""

from __future__ import annotations

from pathlib import Path
from statistics import mean

from prosthesis_rl.cad.bridge import CadBridge
from prosthesis_rl.contracts import DesignParams, EvalResult, ProblemSpec
from prosthesis_rl.sim.verifier import Verifier


class EvaluationAgent:
    """Run a design over fixed seeds and publish the stable evaluation contract."""

    def __init__(
        self,
        verifier: Verifier | None = None,
        cad: CadBridge | None = None,
    ) -> None:
        self.verifier = verifier or Verifier()
        self.cad = cad or CadBridge()

    def evaluate(
        self,
        problem: ProblemSpec,
        design: DesignParams,
        *,
        task_id: str,
        seeds: tuple[int, ...] = (0, 1, 2),
        n_targets: int = 2,
        seconds: float = 1.5,
        mesh_dir: str | Path | None = None,
    ) -> EvalResult:
        if not task_id.strip():
            raise ValueError("task_id is required")
        if not seeds:
            raise ValueError("at least one evaluation seed is required")
        if n_targets < 1:
            raise ValueError("n_targets must be at least 1")
        if seconds <= 0:
            raise ValueError("seconds must be greater than zero")

        artifact_dir = Path(mesh_dir) if mesh_dir is not None else self.cad.export_arm(
            design, name=f"eval_{_safe_name(task_id)}"
        )
        feedback = [
            self.verifier.evaluate(
                problem,
                design,
                {"ik_weight": 1.0, "grip_force_target": 0.35},
                mesh_dir=artifact_dir,
                n_targets=n_targets,
                seconds=seconds,
                seed=seed,
            )
            for seed in seeds
        ]
        result = EvalResult(
            task_id=task_id,
            num_rollouts=len(seeds) * n_targets,
            success_rate=mean(item.metrics.get("reach_success", 0.0) for item in feedback),
            mean_reward=mean(item.reward for item in feedback),
            mean_energy=mean(item.metrics.get("energy", 0.0) for item in feedback),
            collision_rate=mean(item.metrics.get("self_collision", 0.0) for item in feedback),
            video_path="",
        )
        errors = result.validate()
        if errors:
            raise ValueError(f"invalid EvalResult: {errors}")
        return result


def _safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value)
    return cleaned.strip("_")[:64] or "task"
