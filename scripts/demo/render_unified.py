"""Unified demo: a prosthesis arm performs an ADL reach, scored for robustness +
estimated lifespan, rendered to ONE mp4.

Default backdrop is a clean studio stage (no synthetic room). Point `--scene` at
any `assets/scenes/<name>/` produced by scripts/recon to drop the SAME demo into
a real reconstructed room — that is the only change needed to swap environments.

    # clean stage (works today, no room needed)
    python3 scripts/demo/render_unified.py

    # real reconstructed room (after: scripts/recon/build_scene.py room.mp4 -o assets/scenes/myroom)
    python3 scripts/demo/render_unified.py --scene assets/scenes/myroom --out myroom_demo

The articulated arm is a functional placeholder for the CAD/DWG arm coming from
the design pipeline; `--arm-visual <mesh>` is reserved to skin the real geometry
onto the moving skeleton.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def autoplace_target(design, mount_pos, scene_dir: str | None, seed: int = 0):
    """Pick a kinematically reachable target (and, with a room, a clear one).

    Forward-samples real joint configs (reachable by construction), runs FK, and
    keeps the best forward, table-height end-effector pose. When a room is present
    it also requires gripper clearance from the room collision geoms.
    """
    import mujoco

    from prosthesis_rl.sim.mjcf_builder import build_mjcf, EE_SITE, ARM, joint_ranges
    from prosthesis_rl.sim import room_asset

    xml = build_mjcf(design, mount_pos=mount_pos)
    if scene_dir:
        xml = room_asset.inject_into(xml, scene_dir)
    m = mujoco.MjModel.from_xml_string(xml, {})
    d = mujoco.MjData(m)
    ee_id = m.site(EE_SITE).id
    grip = m.geom("gripper_geom").id
    cols = [g for g in range(m.ngeom) if m.geom_group[g] == 3]
    qadr = np.array([m.joint(n).qposadr[0] for n in ARM], dtype=int)
    ranges = joint_ranges(design)
    lo = np.array([ranges[n][0] for n in ARM])
    hi = np.array([ranges[n][1] for n in ARM])
    rng = np.random.default_rng(seed)
    fy, fz = float(mount_pos[1]), float(mount_pos[2])

    best = None
    for _ in range(5000):
        d.qpos[qadr] = rng.uniform(lo, hi)
        mujoco.mj_forward(m, d)
        ee = np.asarray(d.site_xpos[ee_id], dtype=float)
        if not (fz - 0.40 <= ee[2] <= fz + 0.05):     # natural reach height band
            continue
        clr = 1.0
        if cols:
            clr = min(mujoco.mj_geomDistance(m, d, grip, g, 1.0, None) for g in cols)
            if clr < 0.06:
                continue
        score = (ee[1] - fy) + 0.5 * clr               # forward, and clear if room
        if best is None or score > best[0]:
            best = (score, ee.copy(), clr)
    if best is None:
        return np.asarray([mount_pos[0], fy + 0.42, fz - 0.12]), 0.0
    return best[1], best[2]


def task_reward(metrics) -> float:
    """Robustness-style scalar (STRESS_TEST_PLAN reward shape, single scene)."""
    energy_penalty = min(0.30, metrics.energy / 300.0)
    return float(metrics.reach_success - energy_penalty
                 - 0.20 * metrics.self_collision - metrics.rom_violation)


def overlay(frame, lines, flags):
    import cv2

    img = np.ascontiguousarray(frame)
    pad, lh = 14, 26
    box_h = pad * 2 + lh * len(lines)
    sub = img[0:box_h, 0:440].astype(np.float32)
    img[0:box_h, 0:440] = (sub * 0.32).astype(np.uint8)
    palette = {"ok": (90, 230, 110), "bad": (240, 110, 90),
               "head": (255, 255, 255), "dim": (190, 200, 210)}
    for i, (text, flag) in enumerate(zip(lines, flags)):
        cv2.putText(img, text, (pad, pad + lh * (i + 1) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, palette[flag], 1, cv2.LINE_AA)
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description="Unified prosthesis stress-test demo")
    ap.add_argument("--scene", default=None,
                    help="assets/scenes/<name>/ to inject; omit for a clean studio stage")
    ap.add_argument("--arm-visual", default=None,
                    help="override CAD mesh dir to skin onto the moving arm")
    ap.add_argument("--policy", default=None,
                    help="trained PPO policy dir/zip (e.g. assets/policies/reach_ppo); "
                         "omit for scripted DLS-IK")
    ap.add_argument("--out", default="unified_demo")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import mujoco

    from prosthesis_rl.contracts import DesignParams
    from prosthesis_rl.cad.bridge import CadBridge
    from prosthesis_rl.sim.mjcf_builder import build_mjcf, EE_SITE
    from prosthesis_rl.sim.control import ReachController, run_reach
    from prosthesis_rl.sim import room_asset
    from prosthesis_rl.fatigue import estimate_lifespan

    scene_dir = None
    scene_name = "studio stage"
    if args.scene:
        scene_dir = str((ROOT / args.scene) if not Path(args.scene).is_absolute()
                        else Path(args.scene))
        scene_name = Path(scene_dir).name + " (reconstructed room)"

    design = DesignParams()
    mount_pos = (0.0, -0.40, 1.00)

    # Per-link CAD meshes -> the simulated arm IS the design's geometry.
    mesh_dir = args.arm_visual or CadBridge().export_arm(design, name="candidate")

    control_mode = "scripted IK"
    policy = None
    if args.policy:
        from prosthesis_rl.rl.rollout import load_policy
        ppath = (ROOT / args.policy) if not Path(args.policy).is_absolute() else Path(args.policy)
        policy = load_policy(str(ppath))
        control_mode = "learned policy (PPO)"

    print(f"[demo] backdrop={scene_name}  control={control_mode}  DoF={design.dof}  placing target ...")
    target, clr = autoplace_target(design, mount_pos, scene_dir, seed=args.seed)
    print(f"[demo] target={np.round(target,3)}  clearance={clr:.2f} m")

    xml = build_mjcf(design, mount_pos=mount_pos, target_pos=tuple(target), mesh_dir=mesh_dir)
    if scene_dir:
        xml = room_asset.inject_into(xml, scene_dir)
    xml = xml.replace(
        "<worldbody>",
        '<worldbody>\n    <light pos="2 -1 3" dir="-1 0.5 -1" diffuse="0.5 0.5 0.5"/>'
        '\n    <light pos="-2 -1 3" dir="1 0.5 -1" diffuse="0.4 0.4 0.4"/>', 1)
    model = mujoco.MjModel.from_xml_string(xml, {})
    data = mujoco.MjData(model)
    ee_id = model.site(EE_SITE).id

    renderer = mujoco.Renderer(model, height=720, width=1280)
    cam = mujoco.MjvCamera()
    # Frame the whole seated wearer from the front-left (the prosthesis side) so
    # the body and the reaching arm are both visible, not the person's back.
    cam.lookat[:] = 0.5 * (np.asarray(mount_pos) + np.asarray(target))
    cam.distance = 2.5
    cam.elevation = -12

    raw, dists = [], []

    def cb(d):
        cam.azimuth = 250 - 35 * len(raw) / (args.seconds * args.fps)
        renderer.update_scene(d, camera=cam)
        raw.append(renderer.render().copy())
        dists.append(float(np.linalg.norm(target - d.site_xpos[ee_id])))

    if policy is not None:
        from prosthesis_rl.rl.rollout import run_policy_reach
        metrics, log = run_policy_reach(model, data, design, target, policy,
                                        seconds=args.seconds, fps=args.fps, frame_cb=cb)
    else:
        ctrl = ReachController(model, design, target)
        metrics, log = run_reach(model, data, ctrl, seconds=args.seconds,
                                 fps=args.fps, frame_cb=cb)

    reward = task_reward(metrics)
    # Elbow torque column drives the lifespan estimate; fall back to the mid joint
    # for designs whose chain has no joint literally named "elbow".
    joints = list(log.joints)
    elbow_idx = joints.index("elbow") if "elbow" in joints else len(joints) // 2
    life = estimate_lifespan(log.as_array()[:, elbow_idx], design)
    print(f"[demo] reach={'HIT' if metrics.reach_success else 'MISS'} "
          f"final={metrics.final_distance*100:.1f}cm energy={metrics.energy:.1f}J "
          f"reward={reward:+.2f} life={life.display_years} peak={life.peak_stress_mpa:.1f}MPa")

    import imageio.v2 as imageio

    frames = []
    for i, fr in enumerate(raw):
        d_cm = dists[i] * 100
        reached_now = d_cm <= 5.0
        lines = [
            "PROSTHESIS STRESS-TEST  -  unified demo",
            f"backdrop: {scene_name}",
            f"arm: {design.dof}-DoF CAD mesh   control: {control_mode}",
            f"ADL reach: {'REACHED' if reached_now else 'reaching'}  dist {d_cm:4.1f} cm",
            f"energy: {metrics.energy:5.1f} J    self-collision: {'yes' if metrics.self_collision else 'no'}",
            f"robustness reward: {reward:+.2f}",
            f"est. service life: {life.display_years}",
            "sim estimate - not a certified medical device",
        ]
        flags = ["head", "dim", "dim", "ok" if reached_now else "dim",
                 "bad" if metrics.self_collision else "dim",
                 "ok" if reward > 0 else "bad", "ok", "dim"]
        frames.append(overlay(fr, lines, flags))

    out_mp4 = ROOT / f"{args.out}.mp4"
    out_png = ROOT / f"{args.out}.png"
    imageio.mimsave(out_mp4, frames, fps=args.fps)
    imageio.imwrite(out_png, frames[int(len(frames) * 0.7)])
    print(f"[demo] wrote {out_mp4.name} ({len(frames)} frames) + {out_png.name}")


if __name__ == "__main__":
    main()
