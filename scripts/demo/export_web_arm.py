"""Generate the MuJoCo-WASM web scene from a REAL arm STL (not the procedural placeholder).

Renders a designed-arm STL as the arm body in webdemo/ — mounted at the shoulder with
base joints so it stays interactive (actuator sliders move it, drag-to-poke applies force).

A monolithic STL is ONE rigid body: it swings from the shoulder but cannot bend at an
elbow/wrist. For an articulated real arm, export one STL per link with
`CadBridge.export_arm(design)` (origin at each joint) and build with
`build_mjcf(design, mesh_dir=...)` — the placeholder slot here is the single-mesh path.

    python3 scripts/demo/export_web_arm.py                          # default: assets/combined/arm_visual.stl
    python3 scripts/demo/export_web_arm.py --arm-stl path/to/arm.stl
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Decorative seated wearer (capsule primitives, visual-only) — same body the
# Python demo uses, so the browser scene matches. Needs ROOT on sys.path first.
from prosthesis_rl.sim.mjcf_builder import _human  # noqa: E402

SCENES = ROOT / "webdemo" / "assets" / "scenes"
DEFAULT_STL = ROOT / "assets" / "combined" / "arm_visual.stl"
MOUNT = (0.0, -0.40, 1.00)
TARGET = (0.0, 0.10, 0.63)  # within the rigid arm's reach sphere (~0.64 m from shoulder)


def build_xml(mesh_file: str) -> str:
    mx, my, mz = MOUNT
    tx, ty, tz = TARGET
    return f"""<mujoco model="prosthesis_env">
  <compiler angle="radian" autolimits="true" meshdir="."/>
  <option gravity="0 0 -9.81" integrator="implicitfast" timestep="0.002"/>
  <visual>
    <global offwidth="1280" offheight="720"/>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.6 0.6 0.6"/>
    <rgba haze="0.85 0.88 0.92 1"/>
  </visual>
  <asset>
    <texture name="skybox" type="skybox" builtin="gradient"
             rgb1="0.55 0.62 0.72" rgb2="0.10 0.12 0.16" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.30 0.32 0.36"
             rgb2="0.22 0.24 0.28" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="14 14" reflectance="0.05"/>
    <mesh name="arm_mesh" file="{mesh_file}"/>
  </asset>
  <worldbody>
    <light pos="0 0 4" dir="0 0 -1" diffuse="0.9 0.9 0.9"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="grid"/>
    <body name="mount" pos="{mx} {my} {mz}">
      <geom name="mount_geom" type="box" size="0.045 0.045 0.045" rgba="0.30 0.30 0.35 1"/>
      <body name="arm" pos="0 0 0">
        <joint name="shoulder_flex" type="hinge" axis="0 1 0" range="-1.571 2.094"
               damping="0.8" armature="0.02"/>
        <joint name="shoulder_abduct" type="hinge" axis="1 0 0" range="-1.047 1.571"
               damping="0.8" armature="0.02"/>
        <geom name="arm_geom" type="mesh" mesh="arm_mesh" rgba="0.20 0.60 0.95 1" density="1010"/>
        <site name="ee" pos="0 0 -0.64" size="0.015" rgba="0.95 0.2 0.2 1" group="1"/>
      </body>
    </body>{_human(MOUNT, "left")}
    <body name="target_marker" pos="{tx} {ty} {tz}">
      <geom type="sphere" size="0.03" rgba="0.1 0.9 0.2 0.85" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
  <contact>
    <exclude body1="mount" body2="arm"/>
  </contact>
  <actuator>
    <position name="act_shoulder_flex" joint="shoulder_flex" kp="220" kv="18"
              ctrlrange="-1.571 2.094" forcerange="-150 150"/>
    <position name="act_shoulder_abduct" joint="shoulder_abduct" kp="220" kv="18"
              ctrlrange="-1.047 1.571" forcerange="-150 150"/>
  </actuator>
</mujoco>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Render a real arm STL into the MuJoCo-WASM web demo")
    ap.add_argument("--arm-stl", default=str(DEFAULT_STL),
                    help="path to the designed arm STL (default: assets/combined/arm_visual.stl)")
    args = ap.parse_args()

    stl = Path(args.arm_stl)
    if not stl.exists():
        raise SystemExit(f"STL not found: {stl}")

    SCENES.mkdir(parents=True, exist_ok=True)
    mesh_file = stl.name
    shutil.copy(stl, SCENES / mesh_file)

    xml = build_xml(mesh_file)
    (SCENES / "arm.xml").write_text(xml)

    # Validate it compiles (meshdir="." resolves relative to the written arm.xml).
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(SCENES / "arm.xml"))
    info = trimesh.load(stl, force="mesh")
    print(f"arm STL: {stl.relative_to(ROOT)}  ({len(info.faces)} tris, "
          f"watertight={info.is_watertight})")
    print(f"wrote {(SCENES / 'arm.xml').relative_to(ROOT)} + copied {mesh_file}  "
          f"(nbody={m.nbody} ngeom={m.ngeom} nu={m.nu})")
    print(f"loader must include '{mesh_file}' in downloadExampleScenesFolder allFiles")


if __name__ == "__main__":
    main()
