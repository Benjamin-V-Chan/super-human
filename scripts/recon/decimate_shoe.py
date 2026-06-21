"""Decimate the baked Gizmo shoe into a light, browser-friendly STL.

The raw Gizmo export (assets/objects/shoe/a_single_sneaker_running_shoe.xml) is a
~1M-face inline mesh — fine for one hero render, fatal for a 30-agent browser
grid. This extracts that mesh, quadric-decimates it to a few thousand faces,
recenters it so its sole sits at z=0 and it's centred in x/y, scales it to a real
shoe length, and writes webdemo/assets/scenes/arm_links/shoe.stl. The scene
builders define one `shoe_mesh` asset from it and instance it per agent (MuJoCo
shares the mesh data; only the draw calls multiply).

    python3 scripts/recon/decimate_shoe.py            # default ~2500 faces
    python3 scripts/recon/decimate_shoe.py --faces 4000 --length 0.26
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
GIZMO_XML = ROOT / "assets" / "objects" / "shoe" / "a_single_sneaker_running_shoe.xml"
OUT = ROOT / "webdemo" / "assets" / "scenes" / "arm_links" / "shoe.stl"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--faces", type=int, default=2500, help="target face count")
    ap.add_argument("--length", type=float, default=0.25, help="target shoe length (m)")
    ap.add_argument("--src", default=str(GIZMO_XML))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    import mujoco
    import trimesh

    src = Path(args.src)
    if not src.exists():
        print(f"[shoe] no Gizmo bake at {src} — rebake via gizmo_assets.py", file=sys.stderr)
        return 1

    m = mujoco.MjModel.from_xml_path(str(src))
    va, nv = int(m.mesh_vertadr[0]), int(m.mesh_vertnum[0])
    fa, nf = int(m.mesh_faceadr[0]), int(m.mesh_facenum[0])
    verts = np.array(m.mesh_vert[3 * va:3 * (va + nv)], dtype=np.float64).reshape(-1, 3)
    faces = np.array(m.mesh_face[3 * fa:3 * (fa + nf)], dtype=np.int64).reshape(-1, 3)
    print(f"[shoe] source mesh: {nv} verts, {nf} faces")

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    target = max(200, min(args.faces, len(mesh.faces)))
    try:
        mesh = mesh.simplify_quadric_decimation(face_count=target)
    except TypeError:                                  # older trimesh signature
        mesh = mesh.simplify_quadric_decimation(target)
    print(f"[shoe] decimated to {len(mesh.faces)} faces")

    # Reorient so the shoe lies flat: the longest axis -> forward (y), the
    # shortest -> up (z). The Gizmo export comes in standing on its length.
    mesh.apply_translation(-mesh.bounds.mean(axis=0))   # centre on origin
    order = np.argsort(mesh.extents)                    # [short, mid, long] axis ids
    perm = [int(order[1]), int(order[2]), int(order[0])]  # x<-mid, y<-long, z<-short
    mesh.vertices = mesh.vertices[:, perm]
    mesh.fix_normals()                                  # keep winding outward after permute
    # Scale to the target length (now along y), sole on the floor.
    mesh.apply_scale(args.length / float(mesh.extents[1]))
    mesh.apply_translation([0, 0, -mesh.bounds[0][2]])  # min z -> 0
    bnds = mesh.bounds
    print(f"[shoe] final extents (m): {np.round(mesh.extents, 3).tolist()}  "
          f"z range [{bnds[0][2]:.3f}, {bnds[1][2]:.3f}]")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(out)
    print(f"[shoe] wrote {out.relative_to(ROOT)}  ({out.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
