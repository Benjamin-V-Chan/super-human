"""Export the ARTICULATED arm + a trained-policy trajectory for in-browser playback.

Produces everything the standalone MuJoCo-WASM playback page (webdemo/playback.html)
needs to show the *learned* reach moving the *real per-link CAD arm* in a browser:

  webdemo/assets/scenes/arm_links/<link>.stl   one STL per articulated body (relative)
  webdemo/assets/scenes/arm_articulated.xml    build_mjcf(design, mesh_dir=...) scene
  webdemo/assets/scenes/arm_trajectory.json    policy qpos per frame + target

Unlike export_web_arm.py (single fused STL, hand-driven), this is the multi-link
articulated path: the arm bends at every joint and plays back the PPO policy.

`write_web_scene` is reused by scripts/demo/train_live.py (the live dashboard).

    python3 scripts/demo/export_web_playback.py --policy assets/policies/reach_ppo
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SCENES = ROOT / "webdemo" / "assets" / "scenes"
LINKS_DIR = SCENES / "arm_links"
MOUNT = (0.0, -0.40, 1.00)
TARGET = (0.0, 0.22, 0.95)  # fixed, comfortably reachable forward reach


def write_web_scene(design, mesh_dir, target=TARGET) -> Path:
    """Copy per-link STLs + write the articulated web scene XML; return its path.

    Rewrites build_mjcf's absolute mesh paths to relative (arm_links/) with
    meshdir=".", and adds a visible green target marker (the WASM loader renders
    geoms, not <site>s). Validates the scene compiles before returning.
    """
    import mujoco

    from prosthesis_rl.sim.mjcf_builder import build_mjcf

    LINKS_DIR.mkdir(parents=True, exist_ok=True)
    for link in design.links:
        shutil.copy(mesh_dir / f"{link.name}.stl", LINKS_DIR / f"{link.name}.stl")

    xml = build_mjcf(design, mount_pos=MOUNT, target_pos=tuple(target), mesh_dir=mesh_dir)
    xml = xml.replace('<compiler angle="radian" autolimits="true"/>',
                      '<compiler angle="radian" autolimits="true" meshdir="."/>')
    xml = xml.replace(f'file="{Path(mesh_dir).resolve()}/', 'file="arm_links/')
    tx, ty, tz = target
    marker = (f'\n    <body name="target_marker" mocap="true" pos="{tx:.5g} {ty:.5g} {tz:.5g}">'
              '\n      <geom type="sphere" size="0.03" rgba="0.1 0.9 0.2 0.85" '
              'contype="0" conaffinity="0"/>\n    </body>')
    xml = xml.replace("  </worldbody>", marker + "\n  </worldbody>")

    out = SCENES / "arm_articulated.xml"
    out.write_text(xml)
    mujoco.MjModel.from_xml_path(str(out))  # validate the relative mesh paths compile
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Export articulated arm + policy trajectory for the web")
    ap.add_argument("--policy", default="assets/policies/reach_ppo",
                    help="trained PPO policy dir/zip")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    import mujoco

    from prosthesis_rl.contracts import DesignParams
    from prosthesis_rl.cad.bridge import CadBridge
    from prosthesis_rl.sim.mjcf_builder import build_mjcf
    from prosthesis_rl.rl.rollout import load_policy, run_policy_reach

    design = DesignParams()
    mesh_dir = CadBridge().export_arm(design, name="candidate")

    web_scene = write_web_scene(design, mesh_dir, TARGET)
    web_model = mujoco.MjModel.from_xml_path(str(web_scene))

    # Simulate the trained policy reaching the fixed target; record qpos per frame.
    sim_model = mujoco.MjModel.from_xml_string(
        build_mjcf(design, mount_pos=MOUNT, mesh_dir=mesh_dir), {})
    target = np.array(TARGET)
    policy = load_policy(str((ROOT / args.policy) if not Path(args.policy).is_absolute()
                             else Path(args.policy)))
    data = mujoco.MjData(sim_model)
    frames: list[list[float]] = []

    def cb(d):
        frames.append([float(x) for x in d.qpos[: sim_model.nq]])

    metrics, _ = run_policy_reach(sim_model, data, design, target, policy,
                                  seconds=args.seconds, fps=args.fps, frame_cb=cb)

    traj = {
        "dt": 1.0 / args.fps,
        "fps": args.fps,
        "nq": int(sim_model.nq),
        "joints": design.joint_names,
        "links": [link.name for link in design.links],
        "target": [float(x) for x in target],
        "mount": list(MOUNT),
        "success": bool(metrics.reach_success),
        "final_cm": float(metrics.final_distance) * 100.0,
        "frames": frames,
    }
    (SCENES / "arm_trajectory.json").write_text(json.dumps(traj))

    mesh_geoms = sum(1 for g in range(web_model.ngeom)
                     if web_model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH)
    print(f"[web] arm_articulated.xml  nbody={web_model.nbody} ngeom={web_model.ngeom} "
          f"mesh_geoms={mesh_geoms} nu={web_model.nu}")
    print(f"[web] arm_links/: {[p.name for p in sorted(LINKS_DIR.glob('*.stl'))]}")
    print(f"[web] arm_trajectory.json: {len(frames)} frames, reach="
          f"{'HIT' if metrics.reach_success else 'miss'} ({traj['final_cm']:.1f} cm)")
    print("[web] open the playback page:  cd webdemo && python3 -m http.server 8011 "
          " ->  http://localhost:8011/playback.html")


if __name__ == "__main__":
    main()
