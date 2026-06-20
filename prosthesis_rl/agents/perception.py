from __future__ import annotations

from pathlib import Path

from prosthesis_rl.contracts import Constraints, ProblemSpec


class PerceptionAgent:
    """Clip -> VLM -> validated ProblemSpec.

    Replace the stubbed VLM call with Claude vision or a Modal-hosted backend.
    """

    def infer_problem(self, clip_path: str | Path) -> ProblemSpec:
        clip_path = Path(clip_path)
        return ProblemSpec(
            tasks=[
                {
                    "id": "reach_1_1",
                    "name": "Reach target",
                    "source_clip": str(clip_path),
                    "target": {"x": 0.35, "y": 0.10, "z": 0.20},
                }
            ],
            constraints=Constraints(
                rom={"elbow_flexion": 120.0},
                residual_strength={"shoulder": 0.7},
                grip_capacity=0.45,
            ),
        )

