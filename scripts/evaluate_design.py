"""JSON stdin/stdout bridge used by the local viewer evaluation endpoint."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prosthesis_rl.agents import EvaluationAgent  # noqa: E402
from prosthesis_rl.contracts import Constraints, DesignParams, ProblemSpec  # noqa: E402


def main() -> None:
    payload = json.load(sys.stdin)
    problem_data = payload.get("problem") or {}
    design_data = payload.get("design") or {}
    constraints = Constraints(
        rom=problem_data.get("rom") or {},
        residual_strength=problem_data.get("residual_strength") or {},
        grip_capacity=float(problem_data.get("grip_capacity") or 0.0),
    )
    tasks = problem_data.get("tasks") or []
    problem = ProblemSpec(
        tasks=[item if isinstance(item, dict) else {"id": str(item)} for item in tasks],
        constraints=constraints,
        primary_action=str(problem_data.get("primary_action") or ""),
        affected_side=str(problem_data.get("affected_side") or ""),
        residual_side=str(problem_data.get("residual_side") or ""),
    )
    design = DesignParams(
        upper_arm_len=float(design_data.get("upper_arm_len", 0.30)),
        forearm_len=float(design_data.get("forearm_len", 0.26)),
        joint_stiffness=float(design_data.get("joint_stiffness", 1.0)),
        grip_width=float(design_data.get("grip_width", 0.08)),
        mount_frame=str(design_data.get("mount_frame") or "torso_right"),
    )
    result = EvaluationAgent().evaluate(
        problem,
        design,
        task_id=str(payload.get("task_id") or "adl_task_v1"),
        seeds=tuple(int(seed) for seed in payload.get("seeds", [0, 1, 2])),
        n_targets=int(payload.get("n_targets", 2)),
        seconds=float(payload.get("seconds", 1.5)),
    )
    sys.stdout.write(result.to_json())


if __name__ == "__main__":
    main()
