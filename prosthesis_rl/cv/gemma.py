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

# Multimodal model for problem detection. Override with GEMMA_MODEL.
# Gemini Flash works on both the API-key and Vertex paths and is fast/cheap;
# swap to a Gemma 3 id (e.g. "gemma-3-27b-it") on the API-key path if preferred.
DEFAULT_MODEL = "gemini-2.5-flash"

_PROMPT = """You are the perception module of a custom upper-limb prosthetic
design system. Every subject you analyze is a candidate for an upper-limb
prosthesis: one of their arms/hands is absent, amputated, or non-functional, and
the other arm is compensating for everyday tasks. Your job is to read their
functional situation from the video frames and infer what the prosthesis must
restore.

Reason from the footage:
- Watch which hand/arm actually does the work (reaching, holding, manipulating).
  That is the COMPENSATING (residual functioning) side.
- The opposite side is the one that is missing/non-functional and NEEDS the
  prosthesis. If only one hand is ever visible performing actions, the other
  side is the side to prosthetize.

Infer and report:
1. affected_side: "left" or "right" — the side that needs the prosthesis.
2. residual_side: "left" or "right" — the functioning side doing the work.
3. tasks: ADL tasks the prosthesis must enable, chosen ONLY from
   ["reach", "grasp", "feeding"].
4. rom: per-joint range of motion in DEGREES the prosthesis should provide for
   {shoulder_flexion, elbow_flexion, wrist_rotation}.
5. residual_strength: per-region 0..1 of the residual functioning limb.
6. grip_capacity: a single 0..1 grip the prosthesis must deliver.
7. pain_points: short observed difficulties the prosthesis should solve.

Decide affected_side and residual_side ONLY from what the footage shows — they
are opposite sides. Do not default to a side; determine it from which hand is
visibly doing the work.

Respond with ONLY a JSON object, no prose, of this exact shape (the
"<left|right>" placeholders mean: choose based on the video):
{
  "affected_side": "<left|right>",
  "residual_side": "<left|right>",
  "tasks": ["reach", "grasp"],
  "rom": {"shoulder_flexion": 110.0, "elbow_flexion": 130.0, "wrist_rotation": 60.0},
  "residual_strength": {"shoulder": 0.7, "elbow": 0.6},
  "grip_capacity": 0.4,
  "pain_points": ["one-handed compensation for bimanual tasks"]
}
"""

# Deterministic detection used whenever the live model is unavailable. Keeps the
# end-to-end loop reproducible and green without a key.
_STUB_DETECTION: dict[str, Any] = {
    "affected_side": "left",
    "residual_side": "right",
    "tasks": ["reach", "grasp"],
    "rom": {"shoulder_flexion": 95.0, "elbow_flexion": 120.0, "wrist_rotation": 45.0},
    "residual_strength": {"shoulder": 0.7, "elbow": 0.6},
    "grip_capacity": 0.45,
    "pain_points": ["one-handed compensation for bimanual daily tasks"],
    "source": "stub",
}


def _api_key() -> str | None:
    return os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")


def _use_vertex() -> bool:
    """True when configured to auth via Google Cloud ADC instead of an API key."""
    return os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in {"1", "true", "yes"}


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
        # Live either via an API key, or via Vertex AI / Google Cloud ADC.
        return _api_key() is not None or _use_vertex()

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
        from google.genai.types import HttpOptions

        if _use_vertex():
            # Vertex AI path: auth via Google Cloud ADC + project/location env.
            client = genai.Client(http_options=HttpOptions(api_version="v1"))
        else:
            client = genai.Client(api_key=_api_key())
        parts: list[Any] = [_PROMPT]
        for path in frame_paths:
            data = Path(path).read_bytes()
            parts.append(types.Part.from_bytes(data=data, mime_type="image/jpeg"))

        # temperature=0 for reproducible reads (PRD determinism requirement).
        config = types.GenerateContentConfig(temperature=0.0)
        response = client.models.generate_content(
            model=self.model, contents=parts, config=config
        )
        detection = _extract_json(response.text or "")
        if detection is None:
            raise ValueError("Gemma response was not parseable JSON")
        detection.setdefault("source", f"gemma:{self.model}")
        return detection
