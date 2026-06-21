"""Offscreen-render the arm inside the Gizmo showcase scene (no browser/WebGL needed).

Mirrors scripts/demo/scene_server.py's showcase config + alignment, builds the
merged MJCF (arm + Gizmo scene + waypoint markers), scripts the hand through the
task waypoints, and writes a video + keyframe PNGs under assets/renders/.

    python3 scripts/demo/render_scene.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import imageio.v2 as imageio
import mujoco

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from prosthesis_rl.agents.scenario import ScenarioAgent
from prosthesis_rl.contracts import DesignParams
from prosthesis_rl.sim.control import nearest_reachable
from prosthesis_rl.sim.gizmo_asset import waypoint_markers
from prosthesis_rl.sim.gizmo_scene_merge import inject_gizmo_scene
from prosthesis_rl.sim.mjcf_builder import build_mjcf

ARM_MESH = ROOT / "webdemo" / "assets" / "scenes" / "arm_links"
OUT = ROOT / "assets" / "renders"


def _load_scene_server():
    spec = importlib.util.spec_from_file_location("scene_server", ROOT / "scripts" / "demo" / "scene_server.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ss = _load_scene_server()
    scene_mjcf = ss.SHOWCASE_SCENE
    if not scene_mjcf.is_file():
        print(f"[render] showcase scene missing: {scene_mjcf}")
        return 1
    print(f"[render] scene: {scene_mjcf.name}  | task: {ss.SHOWCASE_ACTION!r}")

    scenario = ScenarioAgent(use_llm=False).for_action(ss.SHOWCASE_ACTION)
    primary = scenario.primary_waypoint()
    design = DesignParams()

    xml = build_mjcf(design, mount_pos=tuple(scenario.mount_pos),
                     target_pos=tuple(primary.pos), mesh_dir=str(ARM_MESH), add_human=True)
    offset = ss._align_offset(scene_mjcf, scenario)
    print(f"[render] scene offset: {tuple(round(v, 2) for v in offset)}")
    merged = inject_gizmo_scene(xml, scene_mjcf, offset=offset)
    merged = merged.replace("</worldbody>", waypoint_markers(scenario.waypoints) + "\n</worldbody>", 1)

    model = mujoco.MjModel.from_xml_string(merged, {})
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    print(f"[render] merged model: {model.nbody} bodies, {model.ngeom} geoms")

    qadr = [int(model.joint(n).qposadr[0]) for n in design.joint_names]
    configs = []
    for wp in scenario.waypoints:
        _, q, res = nearest_reachable(model, design, np.asarray(wp.pos, dtype=float))
        configs.append(np.asarray(q, dtype=float))
        print(f"[render]   waypoint {wp.name} {tuple(round(v,2) for v in wp.pos)}: reach within {res*100:.1f} cm")
    seq = [np.zeros(len(qadr))] + configs
    fps, n_leg, n_hold = 30, int(2.2 * 30), int(0.7 * 30)
    smooth = lambda t: t * t * (3.0 - 2.0 * t)
    qframes = []
    for a, b in zip(seq, seq[1:]):
        for k in range(1, n_leg + 1):
            qframes.append(a + (b - a) * smooth(k / n_leg))
        qframes.extend([b] * n_hold)

    # camera: side-front view framing the arm + the task object
    renderer = mujoco.Renderer(model, height=720, width=1280)
    cam = mujoco.MjvCamera()
    tgt = np.asarray(primary.pos)
    cam.lookat[:] = 0.5 * np.asarray(scenario.mount_pos) + 0.5 * tgt + np.array([0, 0, 0.0])
    cam.distance, cam.elevation, cam.azimuth = 2.2, -10, 120

    imgs = []
    for qf in qframes:
        for a, v in zip(qadr, qf):
            data.qpos[a] = float(v)
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam)
        imgs.append(renderer.render())

    imageio.mimsave(OUT / "showcase.mp4", imgs, fps=fps)
    keys = {"01_start": 0, "02_pointA": n_leg + n_hold - 1, "03_pointB": len(imgs) - 1}
    for name, idx in keys.items():
        imageio.imwrite(OUT / f"showcase_{name}.png", imgs[idx])
    print(f"[render] wrote {len(imgs)} frames -> {OUT}/showcase.mp4 + 3 keyframes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
