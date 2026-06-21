"""Gemma video analysis for ADL problem detection.

Feeds sampled frames to a Gemma multimodal model (via the Google GenAI API) and
asks it to detect the patient's functional difficulties — which ADL tasks are
hard, and the rough physical constraints (range of motion, residual strength,
grip) implied by the footage.

Design rules:
- **Never break the loop.** If there is no API key, the SDK is missing, or the
  call fails, ``analyze`` returns a deterministic stub detection so the
  perception stage still produces a valid ProblemSpec.
- Output is a plain dict (the "detection") that ``cv.backend`` maps onto the
  ProblemSpec contract. Gemma is not asked to emit the contract directly — it is
  asked for observations, and we own the schema mapping.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

# Gemma 3 multimodal, served through the Gemini API. Override with GEMMA_MODEL.
DEFAULT_MODEL = "gemma-3-27b-it"

_PROMPT = """You are a rehabilitation-robotics assistant analyzing frames from a
short clip of a person attempting activities of daily living (ADL) with an
upper-limb impairment.

From the frames, identify:
1. Which ADL tasks the person struggles with, chosen ONLY from:
   ["reach", "grasp", "feeding"].
2. The physical limitations you can infer, as rough numbers:
   - rom: per-joint range of motion in DEGREES for any of
     {shoulder_flexion, elbow_flexion, wrist_rotation}
   - residual_strength: per-region 0..1 (e.g. {"shoulder": 0.6})
   - grip_capacity: a single 0..1 estimate of achievable grip
3. A short pain_points list of observed difficulties.

Respond with ONLY a JSON object, no prose, of this exact shape:
{
  "tasks": ["reach"],
  "rom": {"elbow_flexion": 120.0},
  "residual_strength": {"shoulder": 0.6},
  "grip_capacity": 0.4,
  "pain_points": ["limited elbow extension when reaching"]
}
"""

# Deterministic detection used whenever the live model is unavailable. Keeps the
# end-to-end loop reproducible and green without a key.
_STUB_DETECTION: dict[str, Any] = {
    "tasks": ["reach", "grasp"],
    "rom": {"shoulder_flexion": 95.0, "elbow_flexion": 120.0, "wrist_rotation": 45.0},
    "residual_strength": {"shoulder": 0.7, "elbow": 0.6},
    "grip_capacity": 0.45,
    "pain_points": ["limited elbow extension when reaching", "weak grip on grasp"],
    "source": "stub",
}


def _api_key() -> str | None:
    return os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")


def _extract_json(text: str) -> dict[str, Any] | None:
    """Pull the first JSON object out of a model response."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


class GemmaVideoAnalyzer:
    """Analyze sampled ADL frames with Gemma, with a deterministic fallback."""

    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.environ.get("GEMMA_MODEL", DEFAULT_MODEL)

    @property
    def available(self) -> bool:
        return _api_key() is not None

    def analyze(self, frame_paths: list[Path]) -> dict[str, Any]:
        if not frame_paths or not self.available:
            return dict(_STUB_DETECTION)
        try:
            return self._analyze_live(frame_paths)
        except Exception as exc:  # noqa: BLE001 - never break the loop on a CV error
            fallback = dict(_STUB_DETECTION)
            fallback["source"] = f"stub_after_error: {type(exc).__name__}"
            return fallback

    def _analyze_live(self, frame_paths: list[Path]) -> dict[str, Any]:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=_api_key())
        parts: list[Any] = [_PROMPT]
        for path in frame_paths:
            data = Path(path).read_bytes()
            parts.append(types.Part.from_bytes(data=data, mime_type="image/jpeg"))

        response = client.models.generate_content(model=self.model, contents=parts)
        detection = _extract_json(response.text or "")
        if detection is None:
            raise ValueError("Gemma response was not parseable JSON")
        detection.setdefault("source", f"gemma:{self.model}")
        return detection
