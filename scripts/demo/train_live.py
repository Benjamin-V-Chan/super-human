"""Live PPO training with an in-browser dashboard — watch the policy learn for real.

Runs *actual* PPO on the reach task and, after every rollout, streams two things
into webdemo/assets/live/ (written atomically so the browser never reads a partial
file):

  status.json      growing history of reward / success / losses (the dashboard)
  trajectory.json  the CURRENT policy's eval rollout (the arm in the viewer)

It also serves webdemo/ so you just open one URL. Nothing is pre-baked: the arm in
the browser is whatever the live policy does *right now*, and it visibly improves
as training proceeds.

    python3 scripts/demo/train_live.py --steps 400000 --port 8011
    # then open  http://localhost:8011/live.html

Stop with Ctrl-C (the page keeps the last state).
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

WEBDEMO = ROOT / "webdemo"
LIVE = WEBDEMO / "assets" / "live"
MOUNT = (0.0, -0.40, 1.00)
FIXED_TARGET = (0.0, 0.22, 0.95)  # the reach shown in the viewer (same every eval)


def atomic_write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def serve(directory: Path, port: int):
    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a):  # keep the console clean
            pass

    handler = functools.partial(QuietHandler, directory=str(directory))
    httpd = http.server.ThreadingHTTPServer(("", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def main() -> None:
    ap = argparse.ArgumentParser(description="Live PPO reach training + browser dashboard")
    ap.add_argument("--steps", type=int, default=400_000)
    ap.add_argument("--port", type=int, default=8011)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-seconds", type=float, default=5.0)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    import mujoco
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.utils import safe_mean
    from stable_baselines3.common.vec_env import DummyVecEnv

    from prosthesis_rl.contracts import DesignParams
    from prosthesis_rl.cad.bridge import CadBridge
    from prosthesis_rl.sim.mjcf_builder import build_mjcf, EE_SITE
    from prosthesis_rl.sim.control import sample_reachable_targets
    from prosthesis_rl.rl.env import ReachEnv
    from prosthesis_rl.rl.rollout import run_policy_reach

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "export_web_playback", ROOT / "scripts" / "demo" / "export_web_playback.py")
    ewp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ewp)

    design = DesignParams()
    mesh_dir = CadBridge().export_arm(design, name="candidate")
    ewp.write_web_scene(design, mesh_dir, FIXED_TARGET)  # geometry for the viewer
    LIVE.mkdir(parents=True, exist_ok=True)

    # Eval model (shared) + a fixed set of random targets for the success metric.
    eval_model = mujoco.MjModel.from_xml_string(
        build_mjcf(design, mount_pos=MOUNT, mesh_dir=mesh_dir), {})
    ee_id = eval_model.site(EE_SITE).id
    eval_targets = sample_reachable_targets(eval_model, design, n=4, seed=123)
    fixed_target = np.array(FIXED_TARGET)

    # Initial "starting up" status so the page has something to show immediately.
    atomic_write_json(LIVE / "status.json",
                      {"running": True, "step": 0, "total": args.steps, "history": []})

    history: list[dict] = []
    t0 = time.time()

    def evaluate(model):
        # Streamed trajectory: the fixed reach, current policy.
        data = mujoco.MjData(eval_model)
        frames: list[list[float]] = []
        m_fixed, _ = run_policy_reach(
            eval_model, data, design, fixed_target, model,
            seconds=args.eval_seconds, fps=args.fps,
            frame_cb=lambda d: frames.append([float(x) for x in d.qpos[: eval_model.nq]]))
        # Success metric over the fixed random eval set.
        succ, finals = 0, []
        for tgt in eval_targets:
            d2 = mujoco.MjData(eval_model)
            mm, _ = run_policy_reach(eval_model, d2, design, tgt, model,
                                     seconds=args.eval_seconds, fps=args.fps)
            succ += int(mm.reach_success)
            finals.append(mm.final_distance)
        return frames, m_fixed, succ / len(eval_targets), float(np.mean(finals)) * 100.0

    class LiveCallback(BaseCallback):
        def _on_rollout_end(self) -> None:
            frames, m_fixed, success_rate, mean_final_cm = evaluate(self.model)
            lv = self.logger.name_to_value

            def g(k):
                v = lv.get(k)
                return float(v) if v is not None else None

            buf = self.model.ep_info_buffer
            reward = float(safe_mean([e["r"] for e in buf])) if len(buf) else None

            history.append({
                "step": int(self.num_timesteps),
                "reward": reward,
                "success_rate": success_rate,
                "final_cm": mean_final_cm,
                "value_loss": g("train/value_loss"),
                "policy_loss": g("train/policy_gradient_loss"),
                "entropy": g("train/entropy_loss"),
                "approx_kl": g("train/approx_kl"),
                "explained_variance": g("train/explained_variance"),
            })
            atomic_write_json(LIVE / "status.json", {
                "running": True, "step": int(self.num_timesteps),
                "total": args.steps, "elapsed_s": round(time.time() - t0, 1),
                "history": history,
            })
            atomic_write_json(LIVE / "trajectory.json", {
                "dt": 1.0 / args.fps, "fps": args.fps, "nq": int(eval_model.nq),
                "joints": design.joint_names,
                "links": [link.name for link in design.links],
                "target": list(FIXED_TARGET), "mount": list(MOUNT),
                "success": bool(m_fixed.reach_success),
                "final_cm": float(m_fixed.final_distance) * 100.0,
                "step": int(self.num_timesteps), "frames": frames,
            })
            rw = history[-1]["reward"]
            print(f"[live] step {self.num_timesteps:>7}  "
                  f"reward {rw:.2f}  " if rw is not None else
                  f"[live] step {self.num_timesteps:>7}  reward --  ", end="")
            print(f"success {success_rate:.2f}  final {mean_final_cm:.1f}cm  "
                  f"fixed-reach {'HIT' if m_fixed.reach_success else 'miss'}", flush=True)
            return None

        def _on_step(self) -> bool:
            return True

    venv = DummyVecEnv([
        (lambda i=i: Monitor(ReachEnv(design, mesh_dir=mesh_dir, seed=args.seed + i)))
        for i in range(args.n_envs)
    ])
    model = PPO("MlpPolicy", venv, seed=args.seed, verbose=0,
                n_steps=512, batch_size=512, gae_lambda=0.95, gamma=0.99,
                learning_rate=3e-4, ent_coef=0.0, n_epochs=10,
                policy_kwargs={"net_arch": [128, 128]})

    serve(WEBDEMO, args.port)
    print(f"[live] serving  http://localhost:{args.port}/live.html   "
          f"(training {args.steps} steps; Ctrl-C to stop)")

    try:
        model.learn(total_timesteps=args.steps, progress_bar=False,
                    callback=LiveCallback())
    except KeyboardInterrupt:
        print("\n[live] interrupted")
    finally:
        done = dict(running=False, step=int(model.num_timesteps),
                    total=args.steps, elapsed_s=round(time.time() - t0, 1),
                    history=history)
        atomic_write_json(LIVE / "status.json", done)
        model.save(ROOT / "assets" / "policies" / "reach_live")
        print(f"[live] done. saved assets/policies/reach_live.zip. "
              f"server still up at http://localhost:{args.port}/live.html — Ctrl-C to exit.")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
