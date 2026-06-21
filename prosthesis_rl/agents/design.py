from __future__ import annotations

from prosthesis_rl.contracts import (
    DesignParams,
    JointDef,
    LinkDef,
    ProblemSpec,
    SimFeedback,
)


class DesignAgent:
    """ProblemSpec + sim feedback -> DesignParams (explicit DoF) + control hints.

    The arm's degrees of freedom are stated here, as an ordered `LinkDef` chain
    with per-joint axes and ranges, rather than assumed downstream by the MJCF
    builder. That chain is what cad.bridge skins with per-link meshes and what
    sim.mjcf_builder turns into bodies+joints, so the agent owns the topology.
    """

    def propose(
        self,
        problem: ProblemSpec,
        feedback: SimFeedback | None = None,
    ) -> tuple[DesignParams, dict[str, float]]:
        # Stiffen the actuators if the last design struggled to reach.
        stiffness = 1.2 if (feedback and feedback.reward < 0.25) else 1.0

        upper_arm_len, forearm_len, grip_width = 0.30, 0.26, 0.08

        # Declare the kinematic chain explicitly: shoulder (flex+abduct) on the
        # upper arm, elbow on the forearm, wrist on the gripper — a 4-DoF arm.
        # An overhead-reach task adds a 5th DoF (wrist pronation) for orientation.
        wants_orientation = any(
            t.get("type") in {"overhead", "pour", "rotate"}
            for t in problem.tasks
        )

        gripper_joints: tuple[JointDef, ...] = (
            JointDef("wrist", (1, 0, 0), (-45.0, 45.0)),
        )
        if wants_orientation:
            gripper_joints += (JointDef("wrist_rot", (0, 0, 1), (-90.0, 90.0)),)

        links = (
            LinkDef(
                name="upper_arm", length=upper_arm_len, radius=0.025,
                joints=(
                    JointDef("shoulder_flex", (0, 1, 0), (-90.0, 120.0)),
                    JointDef("shoulder_abduct", (1, 0, 0), (-60.0, 90.0)),
                ),
            ),
            LinkDef(
                name="forearm", length=forearm_len, radius=0.022,
                joints=(JointDef("elbow", (0, 1, 0), (0.0, 120.0)),),
            ),
            LinkDef(
                name="gripper", length=0.06, radius=max(0.015, grip_width / 2),
                joints=gripper_joints, rgba=(0.85, 0.6, 0.2, 1.0),
            ),
        )

        params = DesignParams(
            upper_arm_len=upper_arm_len,
            forearm_len=forearm_len,
            joint_stiffness=stiffness,
            grip_width=grip_width,
            joint_limits={"elbow": (0.0, 120.0), "wrist": (-45.0, 45.0)},
            links=links,
        )
        control_hints = {"ik_weight": 1.0, "grip_force_target": 0.35}
        return params, control_hints
