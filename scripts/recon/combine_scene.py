"""Combine a reconstructed room scene + the prosthesis CAD into ONE runnable MJCF.

Pulls the two halves together:
  - room  : assets/scenes/<name>/room.xml   (from extract->modal_recon->mesh_to_mjcf)
  - arm   : the prosthesis CAD output (CadBridge -> binary STL)

Both are non-convex meshes, so both get CoACD convex-decomposed for collision and
kept full-res for visuals — identical treatment. Emits combined_scene.xml and can
load + step + render it to prove the assembled world simulates.

    python3 scripts/recon/combine_scene.py --scene assets/scenes/demo_room --render

NOTE on formats: MuJoCo reads STL/OBJ/MSH meshes only. CadBridge already emits STL.
If your CAD tool exports DWG/STEP instead, convert to STL first (e.g. FreeCAD:
  freecadcmd -c "import Mesh; Mesh.Mesh('part.dwg').write('part.stl')"), then pass
--arm-mesh part.stl. DWG cannot be loaded by MuJoCo directly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import trimesh

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "recon"))
sys.path.insert(0, str(ROOT))

import mesh_to_mjcf as m2m  # noqa: E402


def arm_mesh_from_cad(outdir: Path, name: str = "arm") -> Path:
    """Generate the prosthesis arm STL from DesignParams via CadBridge."""
    from prosthesis_rl.cad.bridge import CadBridge
    from prosthesis_rl.contracts import DesignParams

    outdir.mkdir(parents=True, exist_ok=True)
    bridge = CadBridge(output_dir=outdir)
    return bridge.export_stl(DesignParams(), name=name)


def _load_arm_mesh(arm_mesh: Path) -> trimesh.Trimesh:
    """Load the prosthesis mesh, auto-converting DWG/DXF -> mesh first."""
    arm_mesh = Path(arm_mesh)
    if arm_mesh.suffix.lower() in (".dwg", ".dxf"):
        import dwg_to_mesh as d2m
        print(f"[combine] CAD is {arm_mesh.suffix.upper()} -> converting "
              "(LibreDWG dwg2dxf -> 3DFACE parse -> mesh)")
        return d2m.dwg_to_mesh(arm_mesh)
    return trimesh.load(str(arm_mesh), force="mesh")


def prepare_arm(arm_mesh: Path, outdir: Path, name: str = "arm",
                coacd_threshold: float = 0.05) -> tuple[Path, list[Path]]:
    """Copy the visual mesh + write CoACD convex collision pieces for the arm."""
    mesh = _load_arm_mesh(arm_mesh)
    visual = outdir / f"{name}_visual.stl"
    mesh.export(visual)
    parts = m2m.convex_decompose(mesh, threshold=coacd_threshold, max_hulls=24)
    col_paths = []
    for i, part in enumerate(parts):
        p = outdir / f"{name}_col_{i:03d}.stl"
        part.export(p)
        col_paths.append(p)
    print(f"[combine] arm: 1 visual + {len(col_paths)} convex collision pieces")
    return visual, col_paths


def _arm_xml(visual: Path, cols: list[Path], pos, euler, name="prosthesis"):
    assets = [f'    <mesh name="{name}_visual" file="{visual.resolve()}"/>']
    for i, c in enumerate(cols):
        assets.append(f'    <mesh name="{name}_col_{i:03d}" file="{c.resolve()}"/>')
    geoms = [f'      <geom type="mesh" mesh="{name}_visual" contype="0" '
             f'conaffinity="0" group="2" rgba="0.2 0.6 0.95 1"/>']
    for i in range(len(cols)):
        geoms.append(f'      <geom name="{name}_col_{i:03d}" type="mesh" '
                     f'mesh="{name}_col_{i:03d}" group="3" '
                     f'rgba="0.2 0.6 0.95 0.4"/>')
    p = " ".join(f"{x:.4f}" for x in pos)
    e = " ".join(f"{x:.4f}" for x in euler)
    body = (f'    <body name="{name}" pos="{p}" euler="{e}">\n'
            + "\n".join(geoms) + f"\n    </body>")
    return "\n".join(assets), body


def build_combined(scene_dir: str | Path, outdir: str | Path,
                   arm_mesh: Path | None = None,
                   mount_pos=(0.0, 0.0, 1.0), mount_euler=(1.5708, 0, 0),
                   add_target=True) -> Path:
    """Assemble room + arm into one MJCF and write combined_scene.xml."""
    from prosthesis_rl.sim import room_asset

    scene_dir = Path(scene_dir)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if arm_mesh is None:
        arm_mesh = arm_mesh_from_cad(outdir)
    visual, cols = prepare_arm(Path(arm_mesh), outdir)
    arm_assets, arm_body = _arm_xml(visual, cols, mount_pos, mount_euler)

    target = ""
    if add_target:
        # a free-falling object so you can SEE the combined world stepping physics
        target = ('    <body name="target" pos="0.18 -0.20 1.30">\n'
                  '      <freejoint/>\n'
                  '      <geom type="cylinder" size="0.035 0.06" '
                  'rgba="0.9 0.3 0.2 1" mass="0.25"/>\n    </body>')

    base = f"""<mujoco model="combined">
  <compiler angle="radian" autolimits="true"/>
  <option gravity="0 0 -9.81" integrator="implicitfast"/>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.2 0.3 0.4"
             rgb2="0.1 0.15 0.2" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="6 6" reflectance="0.1"/>
{arm_assets}
  </asset>
  <worldbody>
    <light pos="0 0 4" dir="0 0 -1" diffuse="0.9 0.9 0.9"/>
    <light pos="2 2 3" dir="-1 -1 -1" diffuse="0.5 0.5 0.5"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="grid"/>
{arm_body}
{target}
  </worldbody>
</mujoco>
"""
    # Splice in the reconstructed room (asset meshes + room body, absolute paths).
    combined = room_asset.inject_into(base, scene_dir)
    out_xml = outdir / "combined_scene.xml"
    out_xml.write_text(combined)
    print(f"[combine] wrote {out_xml}")
    return out_xml


def main() -> None:
    ap = argparse.ArgumentParser(description="room scene + prosthesis -> one MJCF")
    ap.add_argument("--scene", default="assets/scenes/demo_room",
                    help="reconstructed room scene dir (has room.xml)")
    ap.add_argument("--arm-mesh", default=None,
                    help="prosthesis mesh (STL/OBJ). Default: generate via CadBridge")
    ap.add_argument("--out", default="assets/combined")
    ap.add_argument("--render", action="store_true",
                    help="load + step physics + write combined.mp4/.png")
    args = ap.parse_args()

    out_xml = build_combined(args.scene, args.out,
                             arm_mesh=Path(args.arm_mesh) if args.arm_mesh else None)

    # Always smoke-test that the assembled model actually loads + steps.
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(out_xml))
    data = mujoco.MjData(model)
    for _ in range(300):
        mujoco.mj_step(model, data)
    n_arm = sum(1 for i in range(model.ngeom)
                if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
                and mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i).startswith("prosthesis_col"))
    n_room = sum(1 for i in range(model.ngeom)
                 if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
                 and mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i).startswith("room_col"))
    print(f"[combine] LOADS + STEPS OK: {model.ngeom} geoms total "
          f"({n_arm} arm-collision + {n_room} room-collision), {model.nbody} bodies")

    if args.render:
        import render_preview as rp
        rp.render_mujoco_sim(out_xml, ROOT / "combined.mp4", ROOT / "combined.png",
                             seconds=4.0)


if __name__ == "__main__":
    main()
