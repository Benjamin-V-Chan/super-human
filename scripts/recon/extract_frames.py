"""Stage [A]: MP4 -> a clean set of frames for photogrammetry.

Runs locally on macOS (uses OpenCV, already installed). Samples the video at a
target rate, drops motion-blurred frames (low Laplacian variance), optionally
downscales, and writes zero-padded JPEGs that the Modal COLMAP app consumes.

    python3 scripts/recon/extract_frames.py <video.mp4> -o frames/ --fps 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def sharpness(gray: np.ndarray) -> float:
    """Variance of the Laplacian — higher = sharper (less motion blur)."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def extract(video: str, outdir: str, target_fps: float = 3.0,
            max_frames: int = 300, max_width: int = 1600,
            sharp_keep: float = 0.6) -> int:
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"could not open {video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stride = max(1, round(src_fps / target_fps))

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.jpg"):
        old.unlink()

    # Pass 1: collect candidate frames + sharpness at the target stride.
    cands: list[tuple[float, np.ndarray]] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            if frame.shape[1] > max_width:
                s = max_width / frame.shape[1]
                frame = cv2.resize(frame, None, fx=s, fy=s,
                                   interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cands.append((sharpness(gray), frame))
        idx += 1
    cap.release()

    if not cands:
        raise SystemExit("no frames extracted")

    # Drop the blurriest fraction, then cap the count by uniform subsampling so
    # camera coverage stays even around the room.
    sharps = np.array([c[0] for c in cands])
    cutoff = np.quantile(sharps, 1.0 - sharp_keep)
    kept = [f for s, f in cands if s >= cutoff]
    if len(kept) > max_frames:
        sel = np.linspace(0, len(kept) - 1, max_frames).round().astype(int)
        kept = [kept[i] for i in sel]

    for i, frame in enumerate(kept):
        cv2.imwrite(str(out / f"frame_{i:05d}.jpg"), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])

    print(f"[extract_frames] {video}: {total} frames @ {src_fps:.1f}fps "
          f"-> sampled {len(cands)} -> kept {len(kept)} sharp frames in {out}/")
    return len(kept)


def main() -> None:
    ap = argparse.ArgumentParser(description="MP4 -> photogrammetry frames")
    ap.add_argument("video")
    ap.add_argument("-o", "--outdir", default="frames")
    ap.add_argument("--fps", type=float, default=3.0,
                    help="target sampling rate (3 is a good room-scan default)")
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--max-width", type=int, default=1600)
    ap.add_argument("--sharp-keep", type=float, default=0.6,
                    help="fraction of sharpest frames to keep (0-1)")
    args = ap.parse_args()
    extract(args.video, args.outdir, args.fps, args.max_frames,
            args.max_width, args.sharp_keep)


if __name__ == "__main__":
    main()
