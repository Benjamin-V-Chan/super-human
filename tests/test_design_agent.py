"""Tests for DesignAgent: validation gates, propose logic, candidate comparison, MJCF output."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from prosthesis_rl.agents.design import DesignAgent, EvalResult
from prosthesis_rl.cad.bridge import CadBridge
from prosthesis_rl.contracts import (
    Constraints,
    DesignParams,
    JointDef,
    LinkDef,
    ProblemSpec,
    RewardBreakdown,
    SimFeedback,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _good_params(upper_m: float = 0.30, forearm_m: float = 0.26) -> DesignParams:
    return DesignParams(
        upper_arm_len=upper_m,
        forearm_len=forearm_m,
        joint_limits={"elbow": (0.0, 130.0), "wrist": (-60.0, 60.0)},
    )


def _problem(reach_m: float = 0.5) -> ProblemSpec:
    return ProblemSpec(constraints=Constraints(rom={"reach_m": reach_m}))


def _low_reward_feedback() -> SimFeedback:
    return SimFeedback(
        reward=0.10,
        breakdown=RewardBreakdown(success=0.1, energy_penalty=0.0, rom_penalty=0.0, collision_penalty=0.0),
    )


def _rom_feedback() -> SimFeedback:
    return SimFeedback(
        reward=0.40,
        breakdown=RewardBreakdown(success=0.5, energy_penalty=0.0, rom_penalty=0.2, collision_penalty=0.0),
    )


# ── validate() ────────────────────────────────────────────────────────────────

def test_validate_rejects_zero_length():
    agent = DesignAgent()
    bad = DesignParams(
        upper_arm_len=0.30, forearm_len=0.26,
        links=(
            LinkDef("upper_arm", length=0.0, radius=0.025,
                    joints=(JointDef("shoulder_flex", (0, 1, 0), (-90.0, 120.0)),)),
            LinkDef("forearm", length=0.26, radius=0.022,
                    joints=(JointDef("elbow", (0, 1, 0), (0.0, 130.0)),)),
        ),
    )
    errors = agent.validate(bad)
    assert any("non-positive length" in e for e in errors)


def test_validate_rejects_zero_radius():
    agent = DesignAgent()
    bad = DesignParams(
        links=(
            LinkDef("upper_arm", length=0.30, radius=0.0,
                    joints=(JointDef("shoulder_flex", (0, 1, 0), (-90.0, 120.0)),)),
        ),
    )
    errors = agent.validate(bad)
    assert any("non-positive radius" in e for e in errors)


def test_validate_rejects_inverted_joint_limits():
    agent = DesignAgent()
    bad = DesignParams(
        links=(
            LinkDef("upper_arm", length=0.30, radius=0.025,
                    joints=(JointDef("shoulder_flex", (0, 1, 0), (120.0, -90.0)),)),
        ),
    )
    errors = agent.validate(bad)
    assert any("invalid (lower >= upper)" in e for e in errors)


def test_validate_rejects_equal_joint_limits():
    agent = DesignAgent()
    bad = DesignParams(
        links=(
            LinkDef("upper_arm", length=0.30, radius=0.025,
                    joints=(JointDef("shoulder_flex", (0, 1, 0), (90.0, 90.0)),)),
        ),
    )
    errors = agent.validate(bad)
    assert any("invalid (lower >= upper)" in e for e in errors)


def test_validate_rejects_invalid_joint_type():
    agent = DesignAgent()
    bad = DesignParams(
        links=(
            LinkDef("upper_arm", length=0.30, radius=0.025,
                    joints=(JointDef("shoulder_flex", (0, 1, 0), (-90.0, 120.0), type="ball"),)),
        ),
    )
    errors = agent.validate(bad)
    assert any("unknown type" in e for e in errors)


def test_validate_rejects_zero_axis():
    agent = DesignAgent()
    bad = DesignParams(
        links=(
            LinkDef("upper_arm", length=0.30, radius=0.025,
                    joints=(JointDef("shoulder_flex", (0, 0, 0), (-90.0, 120.0)),)),
        ),
    )
    errors = agent.validate(bad)
    assert any("zero axis" in e for e in errors)


def test_validate_rejects_duplicate_joint_name():
    agent = DesignAgent()
    bad = DesignParams(
        links=(
            LinkDef("upper_arm", length=0.30, radius=0.025,
                    joints=(
                        JointDef("flex", (0, 1, 0), (-90.0, 120.0)),
                        JointDef("flex", (1, 0, 0), (-60.0, 90.0)),
                    )),
        ),
    )
    errors = agent.validate(bad)
    assert any("Duplicate" in e for e in errors)


def test_validate_rejects_insufficient_reach():
    agent = DesignAgent()
    params = DesignParams(upper_arm_len=0.10, forearm_len=0.10)
    errors = agent.validate(params, task_reach_m=0.5)
    assert any("< required" in e for e in errors)


def test_validate_passes_default_params():
    agent = DesignAgent()
    errors = agent.validate(DesignParams(), task_reach_m=0.5)
    assert errors == []


# ── propose() ─────────────────────────────────────────────────────────────────

def test_propose_returns_valid_params():
    agent = DesignAgent()
    params, hints = agent.propose(_problem())
    assert agent.validate(params, task_reach_m=0.5) == []
    assert "ik_weight" in hints
    assert params.dof == 4  # shoulder_flex, shoulder_abduct, elbow, wrist


def test_propose_links_match_scalar_params():
    agent = DesignAgent()
    params, _ = agent.propose(_problem())
    ua = next(l for l in params.links if l.name == "upper_arm")
    fa = next(l for l in params.links if l.name == "forearm")
    assert ua.length == pytest.approx(params.upper_arm_len)
    assert fa.length == pytest.approx(params.forearm_len)


def test_propose_adjusts_forearm_on_low_reward():
    agent = DesignAgent()
    default_params, _ = agent.propose(_problem())
    low_params, _ = agent.propose(_problem(), feedback=_low_reward_feedback())
    assert low_params.forearm_len > default_params.forearm_len


def test_propose_widens_joints_on_rom_penalty():
    agent = DesignAgent()
    default_params, _ = agent.propose(_problem())
    rom_params, _ = agent.propose(_problem(), feedback=_rom_feedback())
    # elbow range should be wider after ROM penalty
    default_elbow = next(j for l in default_params.links for j in l.joints if j.name == "elbow")
    rom_elbow = next(j for l in rom_params.links for j in l.joints if j.name == "elbow")
    assert rom_elbow.range_deg[1] > default_elbow.range_deg[1]


def test_propose_raises_on_impossible_reach():
    agent = DesignAgent()
    with pytest.raises(ValueError, match="failed validation"):
        agent.propose(_problem(reach_m=1.5))


# ── propose_candidates() ─────────────────────────────────────────────────────

def test_propose_candidates_returns_n():
    agent = DesignAgent()
    candidates = agent.propose_candidates(_problem(), n=2)
    assert len(candidates) == 2


def test_propose_candidates_all_valid():
    agent = DesignAgent()
    for params, _ in agent.propose_candidates(_problem(), n=2):
        assert agent.validate(params, task_reach_m=0.5) == []


# ── compare() ─────────────────────────────────────────────────────────────────

def test_compare_picks_higher_mean_reward():
    agent = DesignAgent()
    candidates = [DesignParams(), DesignParams()]
    results = [
        EvalResult(mean_reward=0.3, success_rate=0.5, collision_rate=0.1),
        EvalResult(mean_reward=0.7, success_rate=0.8, collision_rate=0.05),
    ]
    idx, rationale = agent.compare(candidates, results)
    assert idx == 1
    assert "Candidate 1" in rationale


def test_compare_breaks_tie_on_collision_rate():
    agent = DesignAgent()
    candidates = [DesignParams(), DesignParams()]
    results = [
        EvalResult(mean_reward=0.5, success_rate=0.7, collision_rate=0.2),
        EvalResult(mean_reward=0.5, success_rate=0.7, collision_rate=0.05),
    ]
    idx, _ = agent.compare(candidates, results)
    assert idx == 1


def test_compare_raises_on_length_mismatch():
    agent = DesignAgent()
    with pytest.raises(ValueError):
        agent.compare([DesignParams()], [EvalResult(), EvalResult()])


# ── export_arm() ──────────────────────────────────────────────────────────────

def test_export_arm_creates_per_link_stls(tmp_path: Path):
    params = DesignParams()
    bridge = CadBridge(output_dir=tmp_path / "stl")
    mesh_dir = bridge.export_arm(params, name="test_arm")
    assert mesh_dir.is_dir()
    link_names = [l.name for l in params.links]
    for lname in link_names:
        assert (mesh_dir / f"{lname}.stl").exists(), f"missing {lname}.stl"


def test_export_arm_stls_are_nonempty(tmp_path: Path):
    params = DesignParams()
    bridge = CadBridge(output_dir=tmp_path / "stl")
    mesh_dir = bridge.export_arm(params, name="test_arm")
    for stl in mesh_dir.glob("*.stl"):
        assert stl.stat().st_size > 80, f"{stl.name} is too small to be a valid STL"


# ── export_mjcf() ─────────────────────────────────────────────────────────────

def test_export_mjcf_produces_valid_xml(tmp_path: Path):
    params = DesignParams()
    bridge = CadBridge(output_dir=tmp_path / "stl")
    mjcf_path = bridge.export_mjcf(params, name="test_arm")
    assert mjcf_path.exists()
    root = ET.parse(str(mjcf_path)).getroot()
    assert root.tag == "mujoco"


def test_export_mjcf_contains_joint_names(tmp_path: Path):
    params = DesignParams()
    bridge = CadBridge(output_dir=tmp_path / "stl")
    mjcf_path = bridge.export_mjcf(params, name="test_arm")
    content = mjcf_path.read_text()
    for jname in params.joint_names:
        assert jname in content, f"joint '{jname}' missing from MJCF"
