"""Requirements agent: perception problem -> engineering spec for CadGPT.

Bridges the perception stage and Benji's CAD model. Takes the specific problem
deliverables (action, affected side, observed constraints) and DETERMINES the
ROM targets, the ADL task, and the design specs that CadGPT should optimize the
geometry around.

LLM-backed (Gemini reasoning over the action) with a deterministic, per-action
fallback so the loop stays green with no key.
"""

from __future__ import annotations

import json
import math
import os
import re
from typing import Any

from prosthesis_rl.contracts import ProblemSpec

DEFAULT_MODEL = "gemini-2.5-flash"

# Per-action emphasis for the deterministic fallback. Each entry nudges the
# design envelope toward what that action physically demands.
_ACTION_PROFILES: dict[str, dict[str, Any]] = {
    "twist": {"wrist_rotation_deg": 90.0, "grip_width": 0.06, "grip_force_n": 18.0, "note": "cap/lid twisting needs strong pronation/supination + firm cylindrical grip"},
    "tear": {"wrist_rotation_deg": 70.0, "grip_width": 0.05, "grip_force_n": 12.0, "note": "tearing needs a firm pinch and a braced second contact"},
    "fold": {"wrist_rotation_deg": 60.0, "grip_width": 0.07, "grip_force_n": 8.0, "note": "folding needs light, precise pinch and flat pressing"},
    "zip": {"wrist_rotation_deg": 50.0, "grip_width": 0.03, "grip_force_n": 10.0, "note": "zipping needs a fine pincer grip on a small pull"},
    "plug": {"wrist_rotation_deg": 60.0, "grip_width": 0.03, "grip_force_n": 8.0, "note": "inserting a cable needs fine pincer precision and steady alignment"},
    "pour": {"wrist_rotation_deg": 110.0, "grip_width": 0.07, "grip_force_n": 14.0, "note": "pouring needs wide wrist rotation under load"},
    "default": {"wrist_rotation_deg": 60.0, "grip_width": 0.08, "grip_force_n": 12.0, "note": "general-purpose grasp envelope"},
}


def problem_deliverables(spec: ProblemSpec) -> dict[str, Any]:
    """Serialize the perception findings as the problem deliverables JSON."""
    pain_points: list[str] = []
    for task in spec.tasks:
        for pp in task.get("pain_points", []):
            if pp not in pain_points:
                pain_points.append(pp)
    return {
        "primary_action": spec.primary_action,
        "affected_side": spec.affected_side,
        "residual_side": spec.residual_side,
        "adl_tasks": [t.get("id") for t in spec.tasks],
        "observed_constraints": {
            "rom_deg": dict(spec.constraints.rom),
            "residual_strength": dict(spec.constraints.residual_strength),
            "grip_capacity": spec.constraints.grip_capacity,
        },
        "pain_points": pain_points,
    }


def _api_key() -> str | None:
    return os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")


def _use_vertex() -> bool:
    return os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in {"1", "true", "yes"}


def _extract_json(text: str) -> dict[str, Any] | None:
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


class RequirementsAgent:
    """Problem deliverables -> CAD brief (ROM, task, design specs) for CadGPT."""

    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.environ.get("GEMMA_MODEL", DEFAULT_MODEL)

    @property
    def available(self) -> bool:
        return _api_key() is not None or _use_vertex()

    def derive(self, problem: ProblemSpec) -> dict[str, Any]:
        """Return the CAD brief: optimized inputs for Benji's CadGPT model."""
        deliverables = problem_deliverables(problem)
        if self.available:
            try:
                brief = self._derive_live(deliverables)
                brief["source"] = f"llm:{self.model}"
                return self._finalize(brief, deliverables)
            except Exception as exc:  # noqa: BLE001 - never break the loop
                fallback = self._derive_fallback(deliverables)
                fallback["source"] = f"fallback_after_error: {type(exc).__name__}"
                return fallback
        return self._derive_fallback(deliverables)

    # -- LLM path -----------------------------------------------------------

    def _derive_live(self, deliverables: dict[str, Any]) -> dict[str, Any]:
        from google import genai
        from google.genai import types
        from google.genai.types import HttpOptions

        prompt = (
            "You are a prosthetics requirements engineer. Given the perception "
            "deliverables for an upper-limb prosthesis candidate, determine the "
            "engineering spec a CAD model should optimize around so the "
            "prosthesis can perform the specific action.\n\n"
            f"DELIVERABLES:\n{json.dumps(deliverables, indent=2)}\n\n"
            "Return ONLY JSON with this shape (lengths in meters, angles in "
            "DEGREES, forces in newtons, torques in N·m):\n"
            "{\n"
            '  "task": {"primary_action": "...", "adl_category": "reach|grasp|feeding", "task_id": "..."},\n'
            '  "mount_side": "left|right",\n'
            '  "rom_targets_deg": {"shoulder_flexion": [lo,hi], "elbow_flexion": [lo,hi], "wrist_rotation": [lo,hi]},\n'
            '  "design_params": {"upper_arm_len": 0.30, "forearm_len": 0.26, "grip_width": 0.06, "grip_force_target_n": 15.0, "joint_stiffness": 10.0},\n'
            '  "actuator_torque_nm": {"shoulder_flexion": 20.0, "elbow_flexion": 15.0},\n'
            '  "rationale": "one sentence tying the specs to the action"\n'
            "}\n"
            "Tune grip_width, grip_force, and wrist_rotation to what the specific "
            "action physically demands."
        )
        if _use_vertex():
            client = genai.Client(http_options=HttpOptions(api_version="v1"))
        else:
            client = genai.Client(api_key=_api_key())
        resp = client.models.generate_content(
            model=self.model,
            contents=[prompt],
            config=types.GenerateContentConfig(temperature=0.0),
        )
        brief = _extract_json(resp.text or "")
        if brief is None:
            raise ValueError("Requirements LLM response was not parseable JSON")
        return brief

    # -- Deterministic fallback --------------------------------------------

    def _derive_fallback(self, deliverables: dict[str, Any]) -> dict[str, Any]:
        action = (deliverables.get("primary_action") or "").lower()
        profile = _ACTION_PROFILES["default"]
        for key, prof in _ACTION_PROFILES.items():
            if key != "default" and key in action:
                profile = prof
                break

        side = deliverables.get("affected_side") or "right"
        rom = deliverables.get("observed_constraints", {}).get("rom_deg", {})
        sh = float(rom.get("shoulder_flexion", 110.0))
        el = float(rom.get("elbow_flexion", 130.0))
        wr = float(profile["wrist_rotation_deg"])
        tasks = deliverables.get("adl_tasks") or ["grasp_1_1"]

        brief = {
            "task": {
                "primary_action": deliverables.get("primary_action", ""),
                "adl_category": tasks[0].split("_")[0] if tasks else "grasp",
                "task_id": tasks[0],
            },
            "mount_side": side,
            "rom_targets_deg": {
                "shoulder_flexion": [0.0, sh],
                "elbow_flexion": [0.0, el],
                "wrist_rotation": [-wr, wr],
            },
            "design_params": {
                "upper_arm_len": 0.30,
                "forearm_len": 0.26,
                "grip_width": profile["grip_width"],
                "grip_force_target_n": profile["grip_force_n"],
                "joint_stiffness": 10.0,
            },
            "actuator_torque_nm": {"shoulder_flexion": 20.0, "elbow_flexion": 15.0},
            "rationale": profile["note"],
            "source": "fallback",
        }
        return self._finalize(brief, deliverables)

    # -- shared post-processing --------------------------------------------

    @staticmethod
    def _finalize(brief: dict[str, Any], deliverables: dict[str, Any]) -> dict[str, Any]:
        """Add radian joint_limits (sim/CAD convention) derived from rom_targets."""
        rom = brief.get("rom_targets_deg", {})
        brief["joint_limits_rad"] = {
            joint: [round(math.radians(lo), 4), round(math.radians(hi), 4)]
            for joint, (lo, hi) in (
                (j, v) for j, v in rom.items() if isinstance(v, (list, tuple)) and len(v) == 2
            )
        }
        brief.setdefault("mount_side", deliverables.get("affected_side", ""))
        return brief
