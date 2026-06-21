"""Perception -> design handoff: spec sheet + action/side threading."""

from __future__ import annotations

from prosthesis_rl.agents.design import DesignAgent
from prosthesis_rl.agents.spec_sheet import format_spec_sheet
from prosthesis_rl.contracts import Constraints, ProblemSpec


def _spec() -> ProblemSpec:
    return ProblemSpec(
        tasks=[{"id": "grasp_1_1", "name": "Grasp object", "pain_points": ["weak grip"]}],
        constraints=Constraints(rom={"elbow_flexion": 130.0}, grip_capacity=0.4),
        primary_action="picking up a water bottle",
        affected_side="left",
        residual_side="right",
    )


def test_spec_sheet_contains_core_specs():
    sheet = format_spec_sheet(_spec())
    assert "picking up a water bottle" in sheet
    assert "Affected side" in sheet and "left" in sheet
    assert "torso_left" in sheet  # mount frame follows affected side
    assert "DESIGN DIRECTIVE" in sheet
    assert "Grasp object" in sheet


def test_design_mounts_on_affected_side():
    spec = _spec()
    spec.constraints.rom["reach_m"] = 0.4
    morph, _ = DesignAgent().propose(spec)
    assert morph.mount_frame == "torso_left"


def test_design_defaults_mount_when_side_absent():
    spec = ProblemSpec(
        tasks=[{"id": "reach_1_1", "name": "Reach target"}],
        constraints=Constraints(rom={"reach_m": 0.4}),
    )
    morph, _ = DesignAgent().propose(spec)
    assert morph.mount_frame == "torso_right"  # backward-compatible default
