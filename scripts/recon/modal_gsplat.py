"""Train a real 3D Gaussian Splat of the room on a Modal GPU.

Integrates the official Inria reference implementation
(https://github.com/graphdeco-inria/gaussian-splatting): we COLMAP the room frames
(their convert.py), build the CUDA rasterizer + simple-knn submodules, and run
their train.py. Output is `point_cloud.ply` — the trained gaussians — which the
web viewer renders behind the prosthesis arm.

This is the splat counterpart to modal_recon.py (which makes a *collision mesh*);
both start from the same `extract_frames.py` output.

    # 1) frames already exist (frames/), or:
    python3 scripts/recon/extract_frames.py assets/clips/room.mp4 -o frames/ --fps 3
    # 2) train the splat on the cloud GPU
    modal run scripts/recon/modal_gsplat.py --frames-dir frames --out room.ply --iterations 7000

Cost: an A10G for ~15-40 min (COLMAP + 3DGS training). Lower --iterations to iterate.
"""

from __future__ import annotations

import io
import subprocess
import tarfile
from pathlib import Path

import modal

# CUDA *devel* base (has nvcc) so the diff-gaussian-rasterization / simple-knn
# CUDA extensions compile. Torch must be installed before the submodules build.
CUDA_ARCH = "8.6"  # A10G = Ampere sm_86
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("git", "colmap", "imagemagick", "ffmpeg", "libgl1", "libglib2.0-0",
                 "wget", "build-essential", "g++", "gcc")
    .env({"CC": "gcc", "CXX": "g++"})  # steer torch cpp_extension to g++, not clang++
    .pip_install(
        "torch==2.1.2", "torchvision==0.16.2",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install("plyfile", "tqdm", "numpy", "opencv-python-headless")
    .pip_install("setuptools", "wheel", "ninja")  # build deps for --no-build-isolation
    .run_commands(
        "git clone https://github.com/graphdeco-inria/gaussian-splatting /gs --recursive",
        # Build the CUDA submodules against the installed torch. --no-build-isolation
        # is required: their setup.py imports torch at build time, which pip's
        # isolated build env would otherwise hide.
        f"cd /gs && TORCH_CUDA_ARCH_LIST={CUDA_ARCH} pip install --no-build-isolation "
        "./submodules/diff-gaussian-rasterization ./submodules/simple-knn",
        # fused-ssim is required by newer train.py revisions; install if present.
        f"cd /gs && if [ -d submodules/fused-ssim ]; then "
        f"TORCH_CUDA_ARCH_LIST={CUDA_ARCH} pip install --no-build-isolation "
        f"./submodules/fused-ssim; fi",
    )
    # APPENDED LAST (after the cached CUDA build) so this cheap layer doesn't
    # invalidate it: torch 2.1.2 needs numpy<2; the base pulled numpy 2.x
    # ("RuntimeError: Numpy is not available" at train).
    .pip_install("numpy<2")
)

app = modal.App("room-gsplat", image=image)


def _tar_frames(frames_dir: str, max_frames: int = 60) -> bytes:
    buf = io.BytesIO()
    paths = sorted(Path(frames_dir).glob("*.jpg")) + sorted(Path(frames_dir).glob("*.png"))
    if not paths:
        raise SystemExit(f"no .jpg/.png frames in {frames_dir} (run extract_frames.py first)")
    # Subsample uniformly: exhaustive COLMAP matching is O(n^2) and the CPU-SIFT
    # pass OOM-kills the container on ~100+ frames. ~60 keeps good coverage, fast.
    if len(paths) > max_frames:
        n = len(paths)
        idx = sorted({round(i * (n - 1) / (max_frames - 1)) for i in range(max_frames)})
        paths = [paths[i] for i in idx]
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for p in paths:
            tar.add(p, arcname=p.name)
    print(f"[modal_gsplat] packed {len(paths)} frames")
    return buf.getvalue()


@app.function(gpu="A10G", timeout=3600, memory=32768)  # 32GB: CPU-SIFT COLMAP is RAM-hungry
def train_splat(frames_tar: bytes, iterations: int = 7000) -> bytes:
    """COLMAP the frames, then train Inria 3DGS; return point_cloud.ply bytes."""
    work = Path("/work")
    data = work / "data"
    inp = data / "input"
    out = work / "out"
    inp.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    with tarfile.open(fileobj=io.BytesIO(frames_tar), mode="r:gz") as tar:
        tar.extractall(inp)
    n = len(list(inp.iterdir()))
    print(f"[gsplat] {n} frames -> COLMAP (convert.py, CPU SIFT)", flush=True)

    def run(cmd: list[str], cwd: str = "/gs") -> None:
        print("[gsplat] $", " ".join(cmd), flush=True)
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        print(p.stdout[-3000:])
        if p.returncode != 0:
            print(p.stderr[-4000:])
            raise RuntimeError(f"command failed (exit {p.returncode}): {' '.join(cmd)}")

    # 1) COLMAP -> sparse model in the layout train.py expects (data/sparse/0 + images/).
    run(["python", "convert.py", "-s", str(data), "--no_gpu"])

    # 2) Train the gaussians.
    run([
        "python", "train.py", "-s", str(data), "-m", str(out),
        "--iterations", str(iterations),
        "--save_iterations", str(iterations),
        "--test_iterations", "-1",
        "--data_device", "cpu",
        "--disable_viewer",
    ])

    ply = out / "point_cloud" / f"iteration_{iterations}" / "point_cloud.ply"
    if not ply.exists():
        cands = sorted(out.glob("point_cloud/iteration_*/point_cloud.ply"))
        if not cands:
            raise RuntimeError("training produced no point_cloud.ply")
        ply = cands[-1]
    blob = ply.read_bytes()
    print(f"[gsplat] {ply} ({len(blob)/1e6:.1f} MB gaussians)", flush=True)
    return blob


@app.local_entrypoint()
def main(frames_dir: str = "frames", out: str = "room.ply", iterations: int = 7000) -> None:
    blob = train_splat.remote(_tar_frames(frames_dir), iterations=iterations)
    Path(out).write_bytes(blob)
    print(f"\n[modal_gsplat] wrote {out} ({len(blob)/1e6:.1f} MB)")
    print(f"next: copy to webdemo and render — python3 scripts/demo/export_web_splat.py {out}")
