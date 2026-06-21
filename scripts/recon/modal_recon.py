"""Stage [B]/[C]: frames -> room mesh, on a Modal cloud GPU (COLMAP).

RGB-only photogrammetry: COLMAP runs Structure-from-Motion (camera poses +
intrinsics), then CUDA dense MVS, then Poisson surface reconstruction. No depth
and no RTG-SLAM needed — a plain phone video is enough. Output is a single
coloured mesh (.ply) you then feed to mesh_to_mjcf.py locally.

Prereqs (already done in this repo's setup):
    pip install modal
    modal token set --token-id ... --token-secret ...   # auth

Run:
    # 1) make frames locally first
    python3 scripts/recon/extract_frames.py myroom.mp4 -o frames/ --fps 3
    # 2) reconstruct on the cloud GPU
    modal run scripts/recon/modal_recon.py --frames-dir frames --out room_mesh.ply
    # 3) convert to a MuJoCo scene locally
    python3 scripts/recon/mesh_to_mjcf.py room_mesh.ply -o assets/scenes/room --scale 1.0

Cost note: ~A10G for a few minutes per scan. Set --quality low while iterating.
"""

from __future__ import annotations

import io
import subprocess
import tarfile
from pathlib import Path

import modal

# Official CUDA-enabled COLMAP image; add a Modal-managed Python for the entrypoint.
# The remote fn only shells out to the `colmap` CLI + uses stdlib, so no pip deps.
image = modal.Image.from_registry("colmap/colmap:latest", add_python="3.11")

app = modal.App("room-recon", image=image)


def _tar_frames(frames_dir: str) -> bytes:
    buf = io.BytesIO()
    paths = sorted(Path(frames_dir).glob("*.jpg")) + sorted(Path(frames_dir).glob("*.png"))
    if not paths:
        raise SystemExit(f"no .jpg/.png frames in {frames_dir} "
                         "(run extract_frames.py first)")
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for p in paths:
            tar.add(p, arcname=p.name)
    print(f"[modal_recon] packed {len(paths)} frames")
    return buf.getvalue()


@app.function(gpu="A10G", timeout=3600)
def reconstruct(frames_tar: bytes, quality: str = "medium",
                data_type: str = "video", mesher: str = "poisson") -> bytes:
    """Run COLMAP automatic reconstruction on the GPU; return the mesh .ply bytes."""
    work = Path("/work")
    img_dir = work / "images"
    ws = work / "ws"
    img_dir.mkdir(parents=True, exist_ok=True)
    ws.mkdir(parents=True, exist_ok=True)

    with tarfile.open(fileobj=io.BytesIO(frames_tar), mode="r:gz") as tar:
        tar.extractall(img_dir)
    n = len(list(img_dir.iterdir()))
    print(f"[reconstruct] {n} frames; quality={quality} data_type={data_type}")

    cmd = [
        "colmap", "automatic_reconstructor",
        "--workspace_path", str(ws),
        "--image_path", str(img_dir),
        "--data_type", data_type,      # 'video' -> sequential matching
        "--quality", quality,           # low|medium|high|extreme
        "--dense", "1",
        "--mesher", mesher,             # poisson|delaunay
        "--use_gpu", "1",
    ]
    print("[reconstruct] $", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    print(proc.stdout[-4000:])
    if proc.returncode != 0:
        print(proc.stderr[-4000:])
        raise RuntimeError(f"COLMAP failed (exit {proc.returncode})")

    # automatic_reconstructor writes dense/<i>/meshed-{poisson,delaunay}.ply
    candidates = sorted(ws.glob(f"dense/*/meshed-{mesher}.ply")) \
        or sorted(ws.glob("dense/*/meshed-*.ply"))
    if not candidates:
        raise RuntimeError("no mesh produced — likely too few matched frames "
                           "(more overlap / slower pan / more light)")
    mesh = candidates[0].read_bytes()
    print(f"[reconstruct] mesh {candidates[0]} ({len(mesh)/1e6:.1f} MB)")
    return mesh


@app.local_entrypoint()
def main(frames_dir: str = "frames", out: str = "room_mesh.ply",
         quality: str = "medium", data_type: str = "video",
         mesher: str = "poisson") -> None:
    frames_tar = _tar_frames(frames_dir)
    mesh = reconstruct.remote(frames_tar, quality=quality, data_type=data_type,
                              mesher=mesher)
    Path(out).write_bytes(mesh)
    print(f"\n[modal_recon] wrote {out} ({len(mesh)/1e6:.1f} MB)")
    print(f"next: python3 scripts/recon/mesh_to_mjcf.py {out} "
          f"-o assets/scenes/room --scale 1.0")
