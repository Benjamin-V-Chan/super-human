"""Sample frames from an ADL clip for downstream vision analysis.

Uses ffmpeg (available on most machines) to grab N evenly-spaced frames. Falls
back gracefully: if the clip is missing or ffmpeg is unavailable, returns an
empty list so callers can degrade to a stub rather than crash the loop.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def extract_frames(clip_path: str | Path, n_frames: int = 6, out_dir: str | Path | None = None) -> list[Path]:
    """Return up to ``n_frames`` evenly-spaced frame images from ``clip_path``.

    If ``clip_path`` is itself an image, it is returned as a single frame. If the
    clip cannot be read (missing file, no ffmpeg, decode error), returns ``[]``.
    """

    clip_path = Path(clip_path)
    if not clip_path.exists():
        return []

    if clip_path.suffix.lower() in _IMAGE_SUFFIXES:
        return [clip_path]

    if not _has_ffmpeg():
        return []

    out_dir = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="adl_frames_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir / "frame_%03d.jpg"

    # thumbnail filter picks representative frames across the whole clip,
    # then we cap the count to n_frames.
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(clip_path),
        "-vf",
        f"thumbnail,fps=1/2",
        "-frames:v",
        str(n_frames),
        "-y",
        str(pattern),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return []

    frames = sorted(p for p in out_dir.glob("frame_*.jpg"))
    return frames[:n_frames]
