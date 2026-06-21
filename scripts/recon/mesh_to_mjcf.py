"""Turn a raw reconstructed room mesh into a MuJoCo-ready scene.

Pipeline stage [D]/[E] of scripts/recon (see README.md). Runs locally on macOS
(no GPU): clean -> floor-align -> metric-scale -> decimate (visual) ->
convex-decompose (collision, via CoACD) -> emit MJCF.

The reconstruction from COLMAP/photogrammetry is a single triangle soup in an
arbitrary, up-to-scale world frame. MuJoCo needs (a) gravity-aligned geometry,
(b) metric units, and (c) *convex* collision geoms. This module produces all of
that plus a standalone loadable scene for verification.

Usable both as a CLI and as an importable library (see _selftest.py / room_asset.py).
"""

from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

# ----------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------


def load_clean(path: str | Path, keep_fraction: float = 0.02) -> trimesh.Trimesh:
    """Load a mesh and drop tiny disconnected junk components.

    keep_fraction: drop connected components whose face count is below this
    fraction of the largest component (photogrammetry leaves floating specks).
    """
    mesh = trimesh.load(str(path), force="mesh", process=True)
    if not isinstance(mesh, trimesh.Trimesh) or mesh.faces.shape[0] == 0:
        raise ValueError(f"{path} did not load as a non-empty triangle mesh")

    components = mesh.split(only_watertight=False)
    if len(components) > 1:
        sizes = np.array([c.faces.shape[0] for c in components])
        cutoff = sizes.max() * keep_fraction
        kept = [c for c, s in zip(components, sizes) if s >= cutoff]
        mesh = trimesh.util.concatenate(kept) if kept else mesh

    mesh.merge_vertices()
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    return mesh


def ransac_floor(vertices: np.ndarray, iters: int = 2000, seed: int = 0
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Find the dominant planar surface (assumed: the floor) via RANSAC.

    Returns (unit_normal, point_on_plane). Deterministic for a fixed seed so the
    whole pipeline stays reproducible (the PRD/STRESS_TEST_PLAN require it).
    """
    rng = np.random.default_rng(seed)
    pts = vertices
    if len(pts) > 20000:  # subsample for speed; floor is large, survives sampling
        pts = pts[rng.choice(len(pts), 20000, replace=False)]

    diag = np.linalg.norm(pts.max(0) - pts.min(0))
    thresh = max(diag * 0.01, 1e-6)

    best_inliers, best_plane = -1, None
    for _ in range(iters):
        tri = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n = n / norm
        d = np.abs((pts - tri[0]) @ n)
        inliers = int((d < thresh).sum())
        if inliers > best_inliers:
            best_inliers, best_plane = inliers, (n, tri[0])

    n, p = best_plane
    # Orient the normal so the bulk of the geometry sits on the +normal side
    # (the room is *above* its floor).
    if ((vertices - p) @ n).mean() < 0:
        n = -n
    return n, p


def align_to_floor(mesh: trimesh.Trimesh, normal: np.ndarray, point: np.ndarray
                   ) -> trimesh.Trimesh:
    """Rotate so the floor normal -> +Z and translate so the floor sits at z=0."""
    z = np.array([0.0, 0.0, 1.0])
    n = normal / np.linalg.norm(normal)
    v = np.cross(n, z)
    s = np.linalg.norm(v)
    if s < 1e-9:
        R = np.eye(3)
    else:
        c = float(np.dot(n, z))
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))

    mesh = mesh.copy()
    T = np.eye(4)
    T[:3, :3] = R
    mesh.apply_transform(T)
    mesh.apply_translation([0, 0, -mesh.bounds[0][2]])  # min-z -> 0
    return mesh


def decimate(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    """Reduce face count for the *visual* geom (collision uses convex hulls)."""
    if mesh.faces.shape[0] <= target_faces:
        return mesh
    try:
        return mesh.simplify_quadric_decimation(face_count=target_faces)
    except Exception as exc:  # backend missing or failed -> keep full res, warn
        print(f"[mesh_to_mjcf] decimation skipped ({exc}); using full-res visual")
        return mesh


def convex_decompose(mesh: trimesh.Trimesh, threshold: float = 0.06,
                     max_hulls: int = 32, seed: int = 0) -> list[trimesh.Trimesh]:
    """CoACD convex decomposition -> list of convex collision pieces.

    threshold: concavity tolerance (lower = more, tighter hulls). 0.05-0.08 is a
    good room-scale default.
    """
    import coacd

    try:
        coacd.set_log_level("error")
    except Exception:
        pass
    cmesh = coacd.Mesh(np.asarray(mesh.vertices, dtype=np.float64),
                       np.asarray(mesh.faces, dtype=np.int32))
    parts = coacd.run_coacd(cmesh, threshold=threshold, max_convex_hull=max_hulls,
                            seed=seed)
    out = []
    for verts, faces in parts:
        part = trimesh.Trimesh(vertices=np.asarray(verts), faces=np.asarray(faces),
                               process=True)
        if part.faces.shape[0] > 0 and part.volume > 1e-9:
            out.append(part.convex_hull)  # guarantee convex for MuJoCo
    return out


# ----------------------------------------------------------------------------
# MJCF emission
# ----------------------------------------------------------------------------


@dataclass
class SceneSpec:
    name: str
    visual_obj: str
    collision_stls: list[str]
    friction: tuple[float, float, float] = (1.0, 0.05, 0.001)


def _asset_xml(spec: SceneSpec, base: Path, abspaths: bool) -> str:
    def ref(p: str) -> str:
        return str((base / p).resolve()) if abspaths else os.path.basename(p)

    lines = [f'    <mesh name="{spec.name}_visual" file="{ref(spec.visual_obj)}"/>']
    for i, stl in enumerate(spec.collision_stls):
        lines.append(f'    <mesh name="{spec.name}_col_{i:03d}" file="{ref(stl)}"/>')
    return "\n".join(lines)


def _body_xml(spec: SceneSpec, pos=(0, 0, 0)) -> str:
    fr = " ".join(str(x) for x in spec.friction)
    lines = [f'    <body name="{spec.name}" pos="{pos[0]} {pos[1]} {pos[2]}">',
             f'      <geom type="mesh" mesh="{spec.name}_visual" contype="0" '
             f'conaffinity="0" group="2" rgba="0.82 0.82 0.85 1"/>']
    for i in range(len(spec.collision_stls)):
        lines.append(
            f'      <geom name="{spec.name}_col_{i:03d}" type="mesh" '
            f'mesh="{spec.name}_col_{i:03d}" group="3" rgba="0.9 0.5 0.2 0.35" '
            f'friction="{fr}" condim="3"/>')
    lines.append("    </body>")
    return "\n".join(lines)


def assets_fragment(spec: SceneSpec, base: Path) -> str:
    """`<asset>` mesh lines with ABSOLUTE paths (for <include>-free injection)."""
    return _asset_xml(spec, base, abspaths=True)


def body_fragment(spec: SceneSpec, pos=(0, 0, 0)) -> str:
    """`<body>` geom block for the room (visual + convex collision)."""
    return _body_xml(spec, pos)


def write_scene(outdir: str | Path, visual: trimesh.Trimesh,
                convex_parts: list[trimesh.Trimesh], name: str = "room",
                add_probe: bool = True) -> Path:
    """Write meshes + a standalone loadable scene.xml + an injectable room.xml."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    visual.export(out / "visual.obj")
    col_files = []
    for i, part in enumerate(convex_parts):
        fn = f"collision_{i:03d}.stl"
        part.export(out / fn)
        col_files.append(fn)

    spec = SceneSpec(name=name, visual_obj="visual.obj", collision_stls=col_files)

    # Standalone, self-contained scene (relative meshdir) for `mjpython`/viewer/test.
    probe = ""
    if add_probe:
        top = float(visual.bounds[1][2]) + 0.3
        cx, cy = visual.centroid[:2]
        probe = (f'\n    <body name="probe" pos="{cx:.3f} {cy:.3f} {top:.3f}">'
                 '\n      <freejoint/>'
                 '\n      <geom type="sphere" size="0.05" rgba="1 0.1 0.1 1" '
                 'mass="0.2"/>\n    </body>')
    scene = f"""<mujoco model="{name}_scene">
  <compiler meshdir="." angle="radian" autolimits="true"/>
  <option gravity="0 0 -9.81" integrator="implicitfast"/>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.2 0.3 0.4"
             rgb2="0.1 0.15 0.2" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.1"/>
{_asset_xml(spec, out, abspaths=False)}
  </asset>
  <worldbody>
    <light pos="0 0 4" dir="0 0 -1" diffuse="0.9 0.9 0.9"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="grid"/>
{_body_xml(spec)}{probe}
  </worldbody>
</mujoco>
"""
    (out / "scene.xml").write_text(scene)

    # Injectable fragment (absolute mesh paths -> no meshdir needed in parent).
    room = (f'<mujocoinclude>\n  <asset>\n{assets_fragment(spec, out)}\n  </asset>\n'
            f'  <worldbody>\n{body_fragment(spec)}\n  </worldbody>\n</mujocoinclude>\n')
    (out / "room.xml").write_text(room)

    print(f"[mesh_to_mjcf] wrote {out/'scene.xml'} "
          f"({len(col_files)} convex collision pieces)")
    return out / "scene.xml"


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------


def convert(mesh_path: str | Path, outdir: str | Path, *, scale: float = 1.0,
            align: bool = True, target_faces: int = 40000,
            coacd_threshold: float = 0.06, max_hulls: int = 32,
            name: str = "room", add_probe: bool = True, seed: int = 0) -> Path:
    mesh = load_clean(mesh_path)
    if scale != 1.0:
        mesh.apply_scale(scale)
    if align:
        n, p = ransac_floor(mesh.vertices, seed=seed)
        mesh = align_to_floor(mesh, n, p)
    visual = decimate(mesh, target_faces)
    parts = convex_decompose(mesh, threshold=coacd_threshold, max_hulls=max_hulls,
                             seed=seed)
    if not parts:
        raise RuntimeError("CoACD produced no collision pieces; check the mesh")
    return write_scene(outdir, visual, parts, name=name, add_probe=add_probe)


def main() -> None:
    ap = argparse.ArgumentParser(description="Room mesh -> MuJoCo scene")
    ap.add_argument("mesh", help="input mesh (.ply/.obj from recon)")
    ap.add_argument("-o", "--outdir", default="assets/scenes/room")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="metric scale factor (monocular recon is up-to-scale; "
                         "set so one known length matches reality in meters)")
    ap.add_argument("--no-align", action="store_true",
                    help="skip RANSAC floor alignment (already gravity-aligned)")
    ap.add_argument("--target-faces", type=int, default=40000)
    ap.add_argument("--coacd-threshold", type=float, default=0.06)
    ap.add_argument("--max-hulls", type=int, default=32)
    ap.add_argument("--name", default="room")
    ap.add_argument("--no-probe", action="store_true")
    args = ap.parse_args()

    convert(args.mesh, args.outdir, scale=args.scale, align=not args.no_align,
            target_faces=args.target_faces, coacd_threshold=args.coacd_threshold,
            max_hulls=args.max_hulls, name=args.name, add_probe=not args.no_probe)


if __name__ == "__main__":
    main()
