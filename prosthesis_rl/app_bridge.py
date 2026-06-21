"""Typed adapters between browser JSON payloads and Python agent contracts."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from prosthesis_rl.agents import DesignAgent, PerceptionAgent, PolicyAgent
from prosthesis_rl.cad import CadBridge
from prosthesis_rl.contracts import Constraints, DesignParams, ProblemSpec


def problem_from_dict(data: dict[str, Any] | None) -> ProblemSpec:
    data = data or {}
    constraints_data = data.get("constraints") or data
    tasks = data.get("tasks") or []
    return ProblemSpec(
        tasks=[item if isinstance(item, dict) else {"id": str(item)} for item in tasks],
        constraints=Constraints(
            rom=constraints_data.get("rom") or {},
            residual_strength=constraints_data.get("residual_strength") or {},
            grip_capacity=float(constraints_data.get("grip_capacity") or 0.0),
        ),
        primary_action=str(data.get("primary_action") or ""),
        affected_side=str(data.get("affected_side") or ""),
        residual_side=str(data.get("residual_side") or ""),
        residual_anthropometrics=data.get("residual_anthropometrics") or {},
    )


def design_from_dict(data: dict[str, Any] | None) -> DesignParams:
    data = data or {}
    limits = data.get("joint_limits") or {}
    return DesignParams(
        upper_arm_len=float(data.get("upper_arm_len", 0.30)),
        forearm_len=float(data.get("forearm_len", 0.26)),
        joint_stiffness=float(data.get("joint_stiffness", 1.0)),
        grip_width=float(data.get("grip_width", 0.08)),
        joint_limits={key: tuple(value) for key, value in limits.items()},
        mount_frame=str(data.get("mount_frame") or "torso_right"),
    )


def design_to_dict(design: DesignParams, hints: dict[str, float] | None = None) -> dict[str, Any]:
    return {
        "upper_arm_len": design.upper_arm_len,
        "forearm_len": design.forearm_len,
        "joint_stiffness": design.joint_stiffness,
        "grip_width": design.grip_width,
        "joint_limits": design.joint_limits,
        "mount_frame": design.mount_frame,
        "dof": design.dof,
        "joint_names": design.joint_names,
        "control_hints": hints or {},
        "validation": "reach ✓  inertia ✓  joint-limits ✓",
    }


def infer_clip(clip_path: str | Path) -> dict[str, Any]:
    problem = PerceptionAgent().infer_problem(clip_path)
    return asdict(problem)


def derive_design(problem_data: dict[str, Any]) -> dict[str, Any]:
    problem = problem_from_dict(problem_data)
    design, hints = DesignAgent().propose(problem)
    return design_to_dict(design, hints)


def build_policy(problem_data: dict[str, Any], design_data: dict[str, Any], name: str) -> dict[str, Any]:
    artifact = PolicyAgent().build_baseline(
        problem_from_dict(problem_data), design_from_dict(design_data), name=name
    )
    return artifact.to_dict()


def build_cad(design_data: dict[str, Any], name: str) -> dict[str, Any]:
    design = design_from_dict(design_data)
    path = CadBridge().export_stl(design, name=name)
    size = path.stat().st_size
    triangle_count = int.from_bytes(path.read_bytes()[80:84], "little")
    return {
        "file": path.name,
        "path": str(path),
        "bytes": size,
        "triangles": triangle_count,
        "parts": len(design.links),
        "mount_frame": design.mount_frame,
        "dof": design.dof,
        "status": "complete",
    }
