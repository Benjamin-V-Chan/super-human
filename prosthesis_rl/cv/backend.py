from __future__ import annotations

from pathlib import Path


class PerceptionBackend:
    """Frame extraction and VLM backend placeholder."""

    def extract_frames(self, clip_path: str | Path) -> list[Path]:
        return [Path(clip_path)]

    def detect_pain_points(self, frame_paths: list[Path]) -> list[dict[str, object]]:
        return [{"frame": str(path), "label": "reach difficulty"} for path in frame_paths]

