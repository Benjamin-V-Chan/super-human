"""CV/perception backend: clip -> frames -> Gemma -> ProblemSpec.

This is the real pipeline behind the perception stage. It samples frames from an
ADL clip, runs Gemma video analysis for problem detection, then maps the
detection onto the validated ``ProblemSpec`` contract.

Everything degrades gracefully: missing clip, no ffmpeg, or no Gemma key all
fall back to a deterministic detection so the end-to-end loop stays green.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from prosthesis_rl.contracts import Constraints, ProblemSpec
from prosthesis_rl.cv.frames import extract_frames
from prosthesis_rl.cv.gemma import GemmaVideoAnalyzer

_ALLOWED_TASKS = {"reach", "grasp", "feeding"}
_TASK_IDS = {"reach": "reach_1_1", "grasp": "grasp_1_1", "feeding": "feeding_1_1"}
_TASK_NAMES = {"reach": "Reach target", "grasp": "Grasp object", "feeding": "Feeding motion"}


class PerceptionBackend:
    """Frame extraction + Gemma analysis, producing a validated ProblemSpec."""

    def __init__(self, analyzer: GemmaVideoAnalyzer | None = None, n_frames: int = 6) -> None:
        self.analyzer = analyzer or GemmaVideoAnalyzer()
        self.n_frames = n_frames

    def extract_frames(self, clip_path: str | Path) -> list[Path]:
        return extract_frames(clip_path, n_frames=self.n_frames)

    def detect_pain_points(self, frame_paths: list[Path]) -> dict[str, Any]:
        return self.analyzer.analyze(frame_paths)

    def infer_problem(self, clip_path: str | Path) -> ProblemSpec:
        frames = self.extract_frames(clip_path)
        detection = self.detect_pain_points(frames)
        return self._to_problem_spec(clip_path, frames, detection)

    # -- mapping: Gemma detection -> ProblemSpec contract -------------------

    def _to_problem_spec(
        self, clip_path: str | Path, frames: list[Path], detection: dict[str, Any]
    ) -> ProblemSpec:
        clip_path = str(clip_path)
        pain_points = list(detection.get("pain_points", []))

        raw_tasks = [t for t in detection.get("tasks", []) if t in _ALLOWED_TASKS]
        if not raw_tasks:  # contract requires a non-empty task list
            raw_tasks = ["reach"]

        tasks = [
            {
                "id": _TASK_IDS[t],
                "name": _TASK_NAMES[t],
                "source_clip": clip_path,
                "pain_points": pain_points,
            }
            for t in raw_tasks
        ]

        constraints = Constraints(
            rom=self._clean_rom(detection.get("rom", {})),
            residual_strength=self._clean_strength(detection.get("residual_strength", {})),
            grip_capacity=float(detection.get("grip_capacity", 0.45)),
        )
        return ProblemSpec(tasks=tasks, constraints=constraints)

    @staticmethod
    def _clean_rom(rom: Any) -> dict[str, float]:
        allowed = {"shoulder_flexion", "elbow_flexion", "wrist_rotation"}
        out: dict[str, float] = {}
        if isinstance(rom, dict):
            for joint, value in rom.items():
                if joint in allowed:
                    try:
                        out[joint] = float(value[1] if isinstance(value, (list, tuple)) else value)
                    except (TypeError, ValueError):
                        continue
        return out or {"elbow_flexion": 120.0}

    @staticmethod
    def _clean_strength(strength: Any) -> dict[str, float]:
        out: dict[str, float] = {}
        if isinstance(strength, dict):
            for region, value in strength.items():
                try:
                    out[str(region)] = float(value)
                except (TypeError, ValueError):
                    continue
        return out or {"shoulder": 0.7}
