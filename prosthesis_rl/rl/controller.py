from __future__ import annotations

from prosthesis_rl.contracts import DesignParams, ProblemSpec


class ScriptedIKController:
    """First-pass controller so reward mostly reflects design quality."""

    def controls_for(self, problem: ProblemSpec, design: DesignParams) -> dict[str, float]:
        del problem, design
        return {"ik_weight": 1.0, "grip_force_target": 0.35}

