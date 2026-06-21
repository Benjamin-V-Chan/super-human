"""Requirements agent: problem deliverables -> CadGPT brief (fallback path)."""

from __future__ import annotations

import math

from prosthesis_rl.agents.requirements import RequirementsAgent, problem_deliverables
from prosthesis_rl.contracts import Constraints, ProblemSpec


def _spec(action: str) -> ProblemSpec:
    return ProblemSpec(
        tasks=[{"id": "grasp_1_1", "name": "Grasp object", "pain_points": ["one-handed"]}],
        constraints=Constraints(rom={"elbow_flexion": 130.0}, grip_capacity=0.4),
        primary_action=action,
        affected_side="left",
        residual_side="right",
    )


def test_problem_deliverables_shape():
    d = problem_deliverables(_spec("unscrewing a bottle cap"))
    assert d["primary_action"] == "unscrewing a bottle cap"
    assert d["affected_side"] == "left"
    assert d["adl_tasks"] == ["grasp_1_1"]
    assert "rom_deg" in d["observed_constraints"]


def test_fallback_brief_is_complete_and_mounts_correct_side():
    # Force the fallback (deterministic) path by bypassing the live call.
    agent = RequirementsAgent()
    brief = agent._derive_fallback(problem_deliverables(_spec("opening a food lid")))
    assert brief["mount_side"] == "left"
    assert brief["task"]["adl_category"] == "grasp"
    assert set(brief["rom_targets_deg"]) == {"shoulder_flexion", "elbow_flexion", "wrist_rotation"}
    assert "joint_limits_rad" in brief
    # radian limits are derived from the degree targets
    lo, hi = brief["rom_targets_deg"]["elbow_flexion"]
    assert brief["joint_limits_rad"]["elbow_flexion"] == [round(math.radians(lo), 4), round(math.radians(hi), 4)]


def test_action_profile_tunes_grip():
    agent = RequirementsAgent()
    zip_brief = agent._derive_fallback(problem_deliverables(_spec("zipping up a bag")))
    twist_brief = agent._derive_fallback(problem_deliverables(_spec("twisting a bottle cap")))
    # zipping demands a finer pincer than twisting a cap
    assert zip_brief["design_params"]["grip_width"] < twist_brief["design_params"]["grip_width"]
