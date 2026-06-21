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


def serve(directory: Path, port: int, reset_event: threading.Event):
    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a):  # keep the console clean
            pass

        def do_GET(self):  # the "Reset training" button hits GET /reset
            if self.path.split("?")[0] == "/reset":
                reset_event.set()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(b"reset")
                return
            return super().do_GET()

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
    ap.add_argument("--scenario", default=None,
                    help="train an ADL scenario instead of the fixed reach, "
                         'e.g. --scenario "tie my shoe" (see scenario_orchestrator --list)')
    ap.add_argument("--fleet", type=int, default=0,
                    help="show a grid of N agents (one PPO policy, N randomized task "
                         'placements) in live.html, e.g. --fleet 36 --scenario "tie my shoe"')
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
    from prosthesis_rl.rl.scenario_env import ScenarioReachEnv
    from prosthesis_rl.rl.rollout import run_policy_reach

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "export_web_playback", ROOT / "scripts" / "demo" / "export_web_playback.py")
    ewp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ewp)

    design = DesignParams()
    mesh_dir = WEBDEMO / "assets" / "scenes" / "arm_links"

    # Scenario mode: an agent drops the arm into a real ADL scene (posture +
    # objects + task waypoints) instead of the fixed forward reach. mount/target/
    # env/web-scene all derive from it; with no --scenario, everything below keeps
    # the original fixed-reach behaviour exactly.
    scenario = None
    mount = MOUNT
    # Fleet: one PPO policy, N randomized task placements, shown as a grid in the
    # browser. The training envs randomize the target across FLEET_BAND (domain
    # randomization); each dashboard eval rolls the policy on the N fixed targets
    # and streams all N rollouts into fleet.json for the grid scene arm_fleet.xml.
    fleet = max(0, int(args.fleet))
    fleet_targets: list[np.ndarray] = []
    fleet_band = None
    fleet_nq = 0
    LIVE.mkdir(parents=True, exist_ok=True)
    (LIVE / "fleet.json").unlink(missing_ok=True)  # stale fleet stream from a prior run

    if args.scenario:
        from prosthesis_rl.agents.scenario import ScenarioAgent
        scenario = ScenarioAgent().for_action(args.scenario)
        probe = ScenarioReachEnv(scenario, design, mesh_dir=mesh_dir, snap_samples=4000)
        mount = tuple(float(x) for x in scenario.mount_pos)
        primary_t = probe.primary_target()
        target_pos = tuple(float(x) for x in primary_t)
        print(f"[live] scenario '{scenario.task_id}' ({scenario.source}) "
              f"posture={scenario.posture} target={tuple(round(x,2) for x in target_pos)}",
              flush=True)
        if fleet:
            from fleet_scene import build_fleet_scene
            # Randomized laces band on the prosthesis's open (-x) side — wide enough
            # that each agent's reach/trajectory visibly differs.
            lo = np.array([-0.24, 0.17, 0.11]); hi = np.array([-0.10, 0.33, 0.22])
            fleet_band = (lo, hi)
            _frng = np.random.default_rng(args.seed + 7)
            fleet_targets = [_frng.uniform(lo, hi) for _ in range(fleet)]
            cell_specs = [((t[0], t[1] + 0.02, 0.07), tuple(t)) for t in fleet_targets]
            cols = int(round(fleet ** 0.5)) or 1
            _, fleet_nq, _dofc = build_fleet_scene(design, mount, cell_specs, cols=cols)
            print(f"[live] FLEET: {fleet} agents on randomized tie-shoe placements "
                  f"({cols} cols, nq={fleet_nq}) -> arm_fleet.xml", flush=True)
        else:
            # Single agent: one task-target glow (no mid-air intermediate dots).
            ewp.write_web_scene(design, mesh_dir, target_pos, mount_pos=mount,
                                objects=scenario.objects)
    else:
        target_pos = FIXED_TARGET
        if not (WEBDEMO / "assets" / "scenes" / "arm_articulated.xml").exists():
            ewp.write_web_scene(design, mesh_dir, FIXED_TARGET)  # geometry for the viewer

    # Eval model (shared) + a fixed set of jittered targets for the success metric.
    eval_model = mujoco.MjModel.from_xml_string(
        build_mjcf(design, mount_pos=mount, mesh_dir=mesh_dir), {})
    ee_id = eval_model.site(EE_SITE).id
    fixed_target = np.array(target_pos)
    # Success/final-dist are measured around the SAME demo target the viewer shows
    # (small jitter), so "Success rate" agrees with the "fixed reach: HIT" badge
    # instead of using faraway random targets the policy never sees.
    _jrng = np.random.default_rng(123)
    eval_targets = [fixed_target + _jrng.uniform(-0.03, 0.03, size=3) for _ in range(5)]

    # Initial "starting up" status so the page has something to show immediately.
    atomic_write_json(LIVE / "status.json",
                      {"running": True, "step": 0, "total": args.steps, "history": []})

    history: list[dict] = []
    t0_holder = [time.time()]  # reset per training run (mutable for the callback)
    reset_event = threading.Event()

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

    def evaluate_fleet(model):
        """Roll the policy on each of the N randomized targets; stack into one
        cell-major qpos stream (agent 0's joints, then agent 1's, ...)."""
        per_agent = []
        for t in fleet_targets:
            d = mujoco.MjData(eval_model)
            fr: list[list[float]] = []
            mm, _ = run_policy_reach(
                eval_model, d, design, np.asarray(t, dtype=float), model,
                seconds=args.eval_seconds, fps=args.fps,
                frame_cb=lambda dd: fr.append([float(x) for x in dd.qpos[: eval_model.nq]]))
            per_agent.append((fr, mm.final_distance, mm.reach_success))
        n_frames = min(len(p[0]) for p in per_agent)
        fleet_frames = [[q for p in per_agent for q in p[0][f]] for f in range(n_frames)]
        succ = sum(int(p[2]) for p in per_agent) / len(per_agent)
        mean_cm = float(np.mean([p[1] for p in per_agent])) * 100.0
        return fleet_frames, succ, mean_cm

    class LiveCallback(BaseCallback):
        def _on_rollout_end(self) -> None:
            if fleet:
                frames, success_rate, mean_final_cm = evaluate_fleet(self.model)
                m_fixed = None
            else:
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
                "total": args.steps, "elapsed_s": round(time.time() - t0_holder[0], 1),
                "history": history,
            })
            if fleet:
                atomic_write_json(LIVE / "fleet.json", {
                    "dt": 1.0 / args.fps, "fps": args.fps, "nq": fleet_nq,
                    "dof": design.dof, "cols": int(round(fleet ** 0.5)) or 1,
                    "n_agents": fleet, "joints": design.joint_names,
                    "scenario": (scenario.task_id if scenario else None),
                    "success_rate": success_rate, "mean_cm": mean_final_cm,
                    "step": int(self.num_timesteps), "frames": frames,
                })
            else:
                atomic_write_json(LIVE / "trajectory.json", {
                    "dt": 1.0 / args.fps, "fps": args.fps, "nq": int(eval_model.nq),
                    "joints": design.joint_names,
                    "links": [link.name for link in design.links],
                    "target": list(target_pos), "mount": list(mount),
                    "scenario": (scenario.task_id if scenario else None),
                    "posture": (scenario.posture if scenario else None),
                    "success": bool(m_fixed.reach_success),
                    "final_cm": float(m_fixed.final_distance) * 100.0,
                    "step": int(self.num_timesteps), "frames": frames,
                })
            rw = history[-1]["reward"]
            tag = f"fleet({fleet})" if fleet else \
                f"fixed-reach {'HIT' if m_fixed.reach_success else 'miss'}"
            print(f"[live] step {self.num_timesteps:>7}  "
                  f"reward {rw:.2f}  " if rw is not None else
                  f"[live] step {self.num_timesteps:>7}  reward --  ", end="")
            print(f"success {success_rate:.2f}  final {mean_final_cm:.1f}cm  {tag}", flush=True)
            return None

        def _on_step(self) -> bool:
            return not reset_event.is_set()  # False stops learn() so we can restart

    def make_env(i: int):
        if scenario is not None:
            return Monitor(ScenarioReachEnv(scenario, design, mesh_dir=mesh_dir,
                                            seed=args.seed + i, target_band=fleet_band))
        return Monitor(ReachEnv(design, mesh_dir=mesh_dir, seed=args.seed + i))

    venv = DummyVecEnv([(lambda i=i: make_env(i)) for i in range(args.n_envs)])

    def fresh_model():
        return PPO("MlpPolicy", venv, seed=args.seed, verbose=0,
                   n_steps=512, batch_size=512, gae_lambda=0.95, gamma=0.99,
                   learning_rate=3e-4, ent_coef=0.0, n_epochs=10,
                   policy_kwargs={"net_arch": [128, 128]})

    def write_done(model):
        atomic_write_json(LIVE / "status.json", dict(
            running=False, step=int(model.num_timesteps), total=args.steps,
            elapsed_s=round(time.time() - t0_holder[0], 1), history=list(history)))

    serve(WEBDEMO, args.port, reset_event)
    print(f"[live] serving  http://localhost:{args.port}/live.html   "
          f"(training {args.steps} steps; click Reset or Ctrl-C)")

    model = None
    try:
        while True:  # one full training run per loop; Reset (GET /reset) restarts it
            history.clear()
            t0_holder[0] = time.time()
            reset_event.clear()
            model = fresh_model()
            atomic_write_json(LIVE / "status.json",
                              {"running": True, "step": 0, "total": args.steps, "history": []})
            print("[live] training from scratch…", flush=True)
            model.learn(total_timesteps=args.steps, progress_bar=False,
                        callback=LiveCallback())
            if reset_event.is_set():
                print("[live] reset requested — restarting from scratch", flush=True)
                continue
            write_done(model)
            # Scenario runs save under their own name so they never clobber the
            # deliberately-trained fixed-reach policy (assets/policies/reach_live).
            policy_name = f"reach_{scenario.task_id}" if scenario else "reach_live"
            model.save(ROOT / "assets" / "policies" / policy_name)
            print(f"[live] done. saved {policy_name}.zip — click Reset to retrain.", flush=True)
            while not reset_event.is_set():
                time.sleep(0.3)
    except KeyboardInterrupt:
        print("\n[live] interrupted")
        if model is not None:
            write_done(model)


if __name__ == "__main__":
    main()
