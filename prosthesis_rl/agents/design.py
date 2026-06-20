from __future__ import annotations

from prosthesis_rl.contracts import DesignParams, ProblemSpec, SimFeedback


class DesignAgent:
    """ProblemSpec + sim feedback -> DesignParams + control hints."""

    def propose(
        self,
        problem: ProblemSpec,
        feedback: SimFeedback | None = None,
    ) -> tuple[DesignParams, dict[str, float]]:
        del problem
        stiffness = 1.0
        if feedback and feedback.reward < 0.25:
            stiffness = 1.2

        params = DesignParams(
            upper_arm_len=0.30,
            forearm_len=0.26,
            joint_stiffness=stiffness,
            grip_width=0.08,
            joint_limits={"elbow": (0.0, 120.0), "wrist": (-45.0, 45.0)},
        )
        control_hints = {"ik_weight": 1.0, "grip_force_target": 0.35}
        return params, control_hints

