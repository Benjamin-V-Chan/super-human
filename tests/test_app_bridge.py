from __future__ import annotations

import json
from pathlib import Path

from prosthesis_rl.app_bridge import build_cad, build_policy, derive_design


PROBLEM = {
    "primary_action": "reach for a bottle",
    "affected_side": "left",
    "tasks": ["reach"],
    "rom": {"reach_m": 0.4},
    "residual_strength": {"shoulder": 0.6},
    "grip_capacity": 0.4,
}


def test_browser_bridge_runs_design_policy_and_cad_handoffs(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)

    design = derive_design(PROBLEM)
    assert design["mount_frame"] == "torso_left"
    assert design["dof"] == 4
    assert design["control_hints"]["ik_weight"] == 1.0

    policy = build_policy(PROBLEM, design, "reach policy")
    policy_path = Path(policy["path"])
    assert policy["kind"] == "scripted_ik"
    assert policy_path.is_file()
    payload = json.loads(policy_path.read_text())
    assert payload["dof"] == design["dof"]
    assert payload["joints"] == design["joint_names"]

    cad = build_cad(design, "browser_candidate")
    assert Path(cad["path"]).is_file()
    assert cad["file"] == "browser_candidate.stl"
    assert cad["triangles"] > 0
    assert cad["bytes"] > 84
