"""DWG -> mesh, so an agent-produced .dwg CAD part can enter MuJoCo.

MuJoCo cannot read DWG (only STL/OBJ/MSH). This bridges it:

    DWG  --dwg2dxf (LibreDWG)-->  DXF  --ezdxf-->  triangles  -->  STL/Trimesh

The recovered mesh then flows into the SAME path as everything else
(mesh_to_mjcf convex-decomposition -> MJCF). Also ships `mesh_to_dwg()` so we can
synthesize a real DWG to test against (DXF round-trip via dxf2dwg).

CLI:
    python3 scripts/recon/dwg_to_mesh.py part.dwg -o part.stl
    python3 scripts/recon/dwg_to_mesh.py --mesh-to-dwg arm.stl -o arm.dwg   # make a DWG
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import trimesh


# ---------------------------------------------------------------------------
# DWG <-> DXF (LibreDWG command-line tools)
# ---------------------------------------------------------------------------

def _require(tool: str) -> str:
    p = shutil.which(tool)
    if not p:
        raise RuntimeError(
            f"'{tool}' not found. Install LibreDWG: `brew install libredwg` "
            "(provides dwg2dxf + dxf2dwg). Alternatives: ODA File Converter, FreeCAD.")
    return p


def dwg_to_dxf(dwg: str | Path, out_dxf: str | Path) -> Path:
    _require("dwg2dxf")
    out_dxf = Path(out_dxf)
    out_dxf.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["dwg2dxf", "-y", "-o", str(out_dxf), str(dwg)],
                       capture_output=True, text=True)
    if r.returncode != 0 or not out_dxf.exists():
        raise RuntimeError(f"dwg2dxf failed: {r.stderr.strip() or r.stdout.strip()}")
    return out_dxf


def dxf_to_dwg(dxf: str | Path, out_dwg: str | Path) -> Path:
    _require("dxf2dwg")
    out_dwg = Path(out_dwg)
    out_dwg.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["dxf2dwg", "-y", "-o", str(out_dwg), str(dxf)],
                       capture_output=True, text=True)
    if r.returncode != 0 or not out_dwg.exists():
        raise RuntimeError(f"dxf2dwg failed: {r.stderr.strip() or r.stdout.strip()}")
    return out_dwg


# ---------------------------------------------------------------------------
# DXF -> triangles (ezdxf): handles 3DFACE, MESH, POLYFACE/POLYMESH
# ---------------------------------------------------------------------------

def _add_poly(verts, faces, pts):
    if len(pts) < 3:
        return
    base = len(verts)
    verts.extend(tuple(p) for p in pts)
    for k in range(1, len(pts) - 1):              # fan-triangulate
        faces.append((base, base + k, base + k + 1))


def _parse_3dfaces_ascii(dxf: Path):
    """Read 3DFACE entities straight from DXF tag pairs (ignores handles).

    Robust to the non-unique/invalid handles LibreDWG emits, which crash ezdxf.
    3DFACE encodes its 4 corners as group codes 10-13 (x), 20-23 (y), 30-33 (z).
    """
    lines = dxf.read_text(errors="ignore").splitlines()
    verts: list[tuple] = []
    faces: list[tuple] = []
    in_face = False
    cur: dict[int, float] = {}

    def flush():
        pts = [(cur[10 + k], cur[20 + k], cur[30 + k])
               for k in range(4) if (10 + k) in cur and (20 + k) in cur and (30 + k) in cur]
        if len(pts) >= 3:
            uniq = [pts[0], pts[1], pts[2]]
            if len(pts) == 4 and pts[3] != pts[2]:
                uniq.append(pts[3])
            _add_poly(verts, faces, uniq)

    i, n = 0, len(lines) - 1
    while i < n:
        code, val = lines[i].strip(), lines[i + 1].strip()
        i += 2
        if code == "0":
            if in_face:
                flush()
            in_face = (val == "3DFACE")
            cur = {}
        elif in_face and code.isdigit():
            c = int(code)
            if c in (10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33):
                try:
                    cur[c] = float(val)
                except ValueError:
                    pass
    if in_face:
        flush()
    return verts, faces


def dxf_to_trimesh(dxf: str | Path) -> trimesh.Trimesh:
    dxf = Path(dxf)
    verts, faces = _parse_3dfaces_ascii(dxf)      # primary: handle-agnostic

    if not faces:                                  # fallback: ezdxf for MESH/POLYFACE
        try:
            import ezdxf
            from ezdxf import recover
            try:
                doc = ezdxf.readfile(str(dxf))
            except Exception:
                doc, _ = recover.readfile(str(dxf))
            for e in doc.modelspace():
                if e.dxftype() == "MESH":
                    md = e.get_data()
                    base = len(verts)
                    verts.extend(tuple(v) for v in md.vertices)
                    for f in md.faces:
                        f = list(f)
                        for k in range(1, len(f) - 1):
                            faces.append((base + f[0], base + f[k], base + f[k + 1]))
                elif e.dxftype() == "POLYLINE" and getattr(e, "is_poly_face_mesh", False):
                    for f in e.faces():
                        _add_poly(verts, faces, [tuple(v.dxf.location) for v in f])
        except Exception:
            pass

    if not faces:
        raise RuntimeError(
            "no 3D mesh geometry found (only 3DSOLID/ACIS bodies?). Those need a "
            "CAD kernel (FreeCAD/ODA) to tessellate — export the part as a "
            "mesh/3DFACE DWG, or convert via FreeCAD to STL first.")

    mesh = trimesh.Trimesh(vertices=np.array(verts), faces=np.array(faces),
                           process=True)
    mesh.merge_vertices()
    mesh.remove_unreferenced_vertices()
    return mesh


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def dwg_to_mesh(path: str | Path, out: str | Path | None = None) -> trimesh.Trimesh:
    """Load a .dwg or .dxf into a Trimesh; optionally export to STL/OBJ at `out`."""
    path = Path(path)
    if path.suffix.lower() == ".dwg":
        with tempfile.TemporaryDirectory() as td:
            dxf = dwg_to_dxf(path, Path(td) / "part.dxf")
            mesh = dxf_to_trimesh(dxf)
    elif path.suffix.lower() == ".dxf":
        mesh = dxf_to_trimesh(path)
    else:
        raise ValueError(f"expected .dwg/.dxf, got {path.suffix}")
    if out:
        mesh.export(str(out))
        print(f"[dwg_to_mesh] {path.name} -> {out}  "
              f"({len(mesh.vertices)} verts, {len(mesh.faces)} faces)")
    return mesh


def mesh_to_dwg(mesh_path: str | Path, out_dwg: str | Path) -> Path:
    """Test/utility: mesh (STL/OBJ/PLY) -> DXF 3DFACEs -> DWG (real .dwg)."""
    import ezdxf

    mesh = trimesh.load(str(mesh_path), force="mesh")
    doc = ezdxf.new("R2000")
    msp = doc.modelspace()
    for tri in mesh.faces:
        a, b, c = (mesh.vertices[i] for i in tri)
        msp.add_3dface([tuple(a), tuple(b), tuple(c), tuple(c)])
    with tempfile.TemporaryDirectory() as td:
        dxf = Path(td) / "part.dxf"
        doc.saveas(dxf)
        return dxf_to_dwg(dxf, out_dwg)


def main() -> None:
    ap = argparse.ArgumentParser(description="DWG/DXF <-> mesh bridge")
    ap.add_argument("input")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--mesh-to-dwg", action="store_true",
                    help="reverse: input is a mesh, write a .dwg to --out")
    args = ap.parse_args()
    if args.mesh_to_dwg:
        mesh_to_dwg(args.input, args.out)
        print(f"[dwg_to_mesh] wrote DWG {args.out}")
    else:
        dwg_to_mesh(args.input, args.out)


if __name__ == "__main__":
    main()
