"""Watch-it-learn: render the PPO policy improving across training into ONE mp4.

Trains the reach policy cumulatively, pausing at checkpoints (0 = untrained) to
roll out and record the SAME reach on the mesh arm. Early clips flail; later ones
snap to the target — so the video literally shows the joint-trajectory policy
learning in sim. This is the native counterpart to the in-browser playback.

    python3 scripts/demo/render_training.py --out training_progress
    python3 scripts/demo/render_training.py --checkpoints 0 20000 60000 150000

Headless offscreen rendering (same path as render_unified) — no browser needed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def banner(frame, lines, flags):
    import cv2

    img = np.ascontiguousarray(frame)
    pad, lh = 14, 28
    box_h = pad * 2 + lh * len(lines)
    sub = img[0:box_h, 0:520].astype(np.float32)
    img[0:box_h, 0:520] = (sub * 0.30).astype(np.uint8)
    palette = {"ok": (90, 230, 110), "bad": (240, 140, 90),
               "head": (255, 255, 255), "dim": (190, 200, 210)}
    for i, (text, flag) in enumerate(zip(lines, flags)):
        cv2.putText(img, text, (pad, pad + lh * (i + 1) - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, palette[flag], 1, cv2.LINE_AA)
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description="Render PPO learning-progress video")
    ap.add_argument("--out", default="training_progress")
    ap.add_argument("--checkpoints", type=int, nargs="+",
                    default=[0, 20000, 60000, 150000, 300000],
                    help="cumulative training steps to snapshot (0 = untrained)")
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import mujoco
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    from prosthesis_rl.contracts import DesignParams
    from prosthesis_rl.cad.bridge import CadBridge
    from prosthesis_rl.sim.mjcf_builder import build_mjcf, EE_SITE
    from prosthesis_rl.rl.env import ReachEnv
    from prosthesis_rl.rl.rollout import run_policy_reach

    design = DesignParams()
    mount_pos = (0.0, -0.40, 1.00)
    mesh_dir = CadBridge().export_arm(design, name="candidate")

    # Vec env for training (same recipe as rl.train).
    venv = DummyVecEnv([
        (lambda i=i: ReachEnv(design, mesh_dir=mesh_dir, seed=args.seed + i))
        for i in range(args.n_envs)
    ])
    model = PPO("MlpPolicy", venv, seed=args.seed, verbose=0,
                n_steps=512, batch_size=512, gae_lambda=0.95, gamma=0.99,
                learning_rate=3e-4, ent_coef=0.0, n_epochs=10,
                policy_kwargs={"net_arch": [128, 128]})

    # One fixed render model + a fixed reachable target so improvement is visible.
    xml = build_mjcf(design, mount_pos=mount_pos, mesh_dir=mesh_dir)
    xml = xml.replace(
        "<worldbody>",
        '<worldbody>\n    <light pos="2 -1 3" dir="-1 0.5 -1" diffuse="0.5 0.5 0.5"/>', 1)
    rmodel = mujoco.MjModel.from_xml_string(xml, {})
    target = np.array([0.0, 0.22, 0.95])  # fixed, comfortably reachable forward reach
    ee_id = rmodel.site(EE_SITE).id

    renderer = mujoco.Renderer(rmodel, height=720, width=1280)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = 0.5 * (np.asarray(mount_pos) + np.asarray(target))
    cam.distance, cam.elevation, cam.azimuth = 2.0, -14, 120

    all_frames = []
    prev = 0
    for ck in args.checkpoints:
        if ck > prev:
            model.learn(ck - prev, reset_num_timesteps=False, progress_bar=False)
            prev = ck

        data = mujoco.MjData(rmodel)
        clip, dists = [], []

        def cb(d):
            renderer.update_scene(d, camera=cam)
            clip.append(renderer.render().copy())
            dists.append(float(np.linalg.norm(target - d.site_xpos[ee_id])))

        metrics, _ = run_policy_reach(rmodel, data, design, target, model,
                                      seconds=args.seconds, fps=args.fps, frame_cb=cb)
        verdict = "REACHED" if metrics.reach_success else "still learning"
        tag = "untrained (random policy)" if ck == 0 else f"PPO training: {ck/1000:.0f}k steps"
        for fr, dcm in ((f, dd * 100) for f, dd in zip(clip, dists)):
            hit = dcm <= 5.0
            all_frames.append(banner(fr, [
                "PROSTHESIS RL  -  watch it learn",
                tag,
                f"reach: {verdict}   dist {dcm:4.1f} cm",
            ], ["head", "dim", "ok" if hit else "bad"]))
        print(f"[learn] {tag:32}  final {metrics.final_distance*100:5.1f} cm  "
              f"{'HIT' if metrics.reach_success else 'miss'}")

    venv.close()
    import imageio.v2 as imageio

    out_mp4 = ROOT / f"{args.out}.mp4"
    imageio.mimsave(out_mp4, all_frames, fps=args.fps)
    print(f"[learn] wrote {out_mp4.name} ({len(all_frames)} frames, "
          f"{len(args.checkpoints)} checkpoints)")


if __name__ == "__main__":
    main()
