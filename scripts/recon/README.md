# Room Recon → MuJoCo Scene

Turn a **plain RGB phone video of a room** into a **collision mesh the prosthesis
arm can physically interact with** in MuJoCo. This is how we source real "house
environments" for the Phase-1 robustness scenes in
[../../STRESS_TEST_PLAN.md](../../STRESS_TEST_PLAN.md) instead of hand-built primitives.

## Why not RTG-SLAM?

The original idea was [RTG-SLAM](https://github.com/MisEty/RTG-SLAM). It doesn't fit:

| RTG-SLAM needs                                    | We have                                     | Verdict       |
| ------------------------------------------------- | ------------------------------------------- | ------------- |
| **RGB-D** (per-frame depth + intrinsics)          | RGB-only MP4                                | ✗             |
| **CUDA** GPU                                      | M3 Max (Metal) locally; cloud GPU via Modal | only on cloud |
| outputs **Gaussian splats** (render, not physics) | need a **collision mesh**                   | ✗             |

So we do RGB-only **photogrammetry** instead: COLMAP (SfM + dense MVS + Poisson)
reconstructs a mesh directly from the video — no depth, no splats — and we convert
that mesh into convex MuJoCo collision geometry. Reconstruction is **offline and
one-time**: bake the room once, reuse the asset in every RL rollout (no real-time
SLAM, which would be pure risk in the loop).

## Pipeline

```text
room.mp4 ──[A]──> frames/ ──[B/C]──> room_mesh.ply ──[D/E]──> assets/scenes/<name>/
          extract_frames    modal_recon (CLOUD GPU)   mesh_to_mjcf      scene.xml + room.xml
          (Mac, cv2)         COLMAP photogrammetry     (Mac: clean,      (load in MuJoCo;
                                                        align, CoACD)     inject via room_asset)
```

| Stage                         | Script                                  | Runs on                    |
| ----------------------------- | --------------------------------------- | -------------------------- |
| **[A]** sample sharp frames   | `extract_frames.py`                     | this Mac                   |
| **[B/C]** SfM + dense + mesh  | `modal_recon.py`                        | **Modal cloud GPU** (A10G) |
| **[D/E]** mesh → MuJoCo scene | `mesh_to_mjcf.py`                       | this Mac                   |
| integration helper            | `../../prosthesis_rl/sim/room_asset.py` | this Mac                   |
| local end-to-end test         | `_selftest.py` (no GPU/cloud needed)    | this Mac                   |

## Run it

```bash
# 0) one-time: Modal is already authed (~/.modal.toml). Sim deps installed:
#    mujoco, trimesh, coacd, fast-simplification, opencv, modal.

# [A] video -> sharp frames  (tune --fps for how fast you panned)
python3 scripts/recon/extract_frames.py assets/clips/myroom.mp4 -o frames/ --fps 3

# [B/C] reconstruct on the cloud GPU  (start with --quality low to iterate fast)
modal run scripts/recon/modal_recon.py --frames-dir frames --out room_mesh.ply --quality low

# [D/E] mesh -> MuJoCo scene  (see "Metric scale" below)
python3 scripts/recon/mesh_to_mjcf.py room_mesh.ply -o assets/scenes/myroom --scale 1.0

# verify it loads + collides
python3 prosthesis_rl/sim/room_asset.py assets/scenes/myroom
```

Then from `sim/scenes.py`:

```python
from prosthesis_rl.sim import room_asset
env_xml = room_asset.inject_into(robot_xml, "assets/scenes/myroom")  # arm + room
```

## Two things to get right

- **Metric scale.** Monocular photogrammetry is _up-to-scale_ — the mesh has the
  right shape but arbitrary units. The arm is in meters, so pass `--scale` so one
  known real length matches. E.g. if a 0.80 m table comes out 1.6 units long,
  `--scale 0.5`. (Measure once in any mesh viewer, or eyeball a doorway ≈ 2.0 m.)
- **Floor alignment** is automatic (RANSAC finds the dominant plane, rotates its
  normal to +Z, drops the floor to z=0). If your scan is weird, pass `--no-align`
  and orient by hand.

## Capture tips for a good reconstruction

Slow, steady pan; lots of overlap between frames; even lighting; avoid blank walls
and mirrors/glass (no texture → SfM fails). 20–40 s of video is plenty. If COLMAP
reports "no mesh produced," there weren't enough matched frames — reshoot slower.

## Verified locally

`_selftest.py` builds a synthetic tilted room, runs the full **[D/E]** half, loads
it in MuJoCo, and drops a probe sphere that collides and settles — proving the
clean → align → CoACD → MJCF path without needing the GPU. Run: `python3 scripts/recon/_selftest.py`.

The only stage that _requires_ the cloud (and your specific video) is **[B/C]**.

```

```
