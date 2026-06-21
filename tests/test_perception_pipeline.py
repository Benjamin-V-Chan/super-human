"""Gemma perception pipeline produces a valid ProblemSpec on the fallback path."""

from __future__ import annotations

from prosthesis_rl.agents.perception import PerceptionAgent
from prosthesis_rl.cv.backend import PerceptionBackend
from prosthesis_rl.contracts import ProblemSpec


def test_perception_agent_returns_valid_problem_spec():
    spec = PerceptionAgent().infer_problem("examples/adl/reach_1_1.mp4")
    assert isinstance(spec, ProblemSpec)
    assert spec.tasks, "tasks must be non-empty (contract requirement)"
    for task in spec.tasks:
        assert task["id"] in {"reach_1_1", "grasp_1_1", "feeding_1_1"}
    assert spec.constraints.rom, "rom constraints must be present"
    assert spec.constraints.grip_capacity >= 0.0


def test_backend_maps_detection_to_spec():
    backend = PerceptionBackend()
    detection = {
        "tasks": ["grasp", "feeding", "bogus_task"],
        "rom": {"elbow_flexion": [0.0, 130.0], "unknown_joint": 10.0},
        "residual_strength": {"shoulder": 0.5},
        "grip_capacity": 0.3,
        "pain_points": ["weak grip"],
    }
    spec = backend._to_problem_spec("clip.mp4", [], detection)
    ids = {t["id"] for t in spec.tasks}
    assert ids == {"grasp_1_1", "feeding_1_1"}  # bogus task dropped
    assert spec.constraints.rom == {"elbow_flexion": 130.0}  # range upper bound, unknown joint dropped
    assert spec.constraints.grip_capacity == 0.3


def test_empty_tasks_fall_back_to_reach():
    backend = PerceptionBackend()
    spec = backend._to_problem_spec("clip.mp4", [], {"tasks": []})
    assert [t["id"] for t in spec.tasks] == ["reach_1_1"]
