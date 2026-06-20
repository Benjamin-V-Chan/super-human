from __future__ import annotations

from prosthesis_rl.agents import ProsthesisLoop


def evaluate_clip(clip_path: str) -> float:
    feedback = ProsthesisLoop().run_once(clip_path)
    return feedback.reward

