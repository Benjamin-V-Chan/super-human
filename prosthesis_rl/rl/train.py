"""Train a PPO policy that drives the arm's joints to reach ADL targets.

This replaces the old training stub with a real loop: PPO (stable-baselines3)
learns joint-space trajectories on `rl.env.ReachEnv` — the same MuJoCo arm the
demo renders, skinned with the per-link CAD meshes — and the learned policy is
saved to `assets/policies/<name>.zip`. The demo loads it via `--policy`.

    python3 -m prosthesis_rl.rl.train --timesteps 150000 --out reach_ppo

The trained arm is design-specific: the observation/action sizes follow the
design's DoF, so re-train if the agent changes the kinematic chain.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from prosthesis_rl.cad.bridge import CadBridge
from prosthesis_rl.contracts import DesignParams

POLICY_DIR = Path("assets/policies")


def _make_env_fn(design: DesignParams, mesh_dir, seed: int, scenario=None):
    """Env factory. With a ScenarioSpec, the arm is dropped into that ADL scene
    (posture + objects + reach waypoints); otherwise it's the generic reach dot."""
    def _init():
        if scenario is not None:
            from prosthesis_rl.rl.scenario_env import ScenarioReachEnv
            return ScenarioReachEnv(scenario, design, mesh_dir=mesh_dir, seed=seed,
                                    add_markers=False)
        from prosthesis_rl.rl.env import ReachEnv
        return ReachEnv(design, mesh_dir=mesh_dir, seed=seed)

    return _init


def _make_progress_callback(progress_cb, total: int, interval: int = 500):
    """Return a proper SB3 BaseCallback that fires progress_cb every N steps."""
    from stable_baselines3.common.callbacks import BaseCallback

    class _Cb(BaseCallback):
        def _on_step(self) -> bool:
            if self.num_timesteps % interval < self.training_env.num_envs:
                try:
                    buf = self.model.ep_info_buffer
                    mean_rew = float(sum(e["r"] for e in buf) / len(buf)) if buf else 0.0
                except Exception:
                    mean_rew = 0.0
                progress_cb({
                    "timestep": self.num_timesteps,
                    "mean_reward": mean_rew,
                    "progress": self.num_timesteps / max(1, total),
                })
            return True

    return _Cb()


def train_reach_policy(
    timesteps: int = 150_000,
    *,
    name: str = "reach_ppo",
    design: DesignParams | None = None,
    n_envs: int = 4,
    seed: int = 0,
    eval_episodes: int = 20,
    verbose: int = 1,
    progress_cb=None,
    scenario=None,
) -> dict[str, object]:
    """Train and save a PPO reach policy; return a small training summary.

    Pass a `scenario` (ScenarioSpec) to train the arm on a real ADL task scene
    derived from the clip — posture + objects + reach waypoints — instead of the
    generic floating-dot reach. Eval then measures the task-completing reach.
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    design = design or DesignParams()
    # Skin the env with the same per-link meshes the demo uses, so the policy
    # transfers to the rendered arm without a domain gap.
    mesh_dir = CadBridge().export_arm(design, name="candidate")

    venv = DummyVecEnv([
        _make_env_fn(design, mesh_dir, seed + i, scenario=scenario) for i in range(n_envs)
    ])

    model = PPO(
        "MlpPolicy", venv, seed=seed, verbose=verbose,
        n_steps=512, batch_size=512, gae_lambda=0.95, gamma=0.99,
        learning_rate=3e-4, ent_coef=0.0, n_epochs=10,
        policy_kwargs={"net_arch": [128, 128]},
    )

    cb = _make_progress_callback(progress_cb, timesteps) if progress_cb else None
    model.learn(total_timesteps=timesteps, progress_bar=False, callback=cb)

    POLICY_DIR.mkdir(parents=True, exist_ok=True)
    out_path = POLICY_DIR / name
    model.save(out_path)

    summary = {
        "policy": str(out_path) + ".zip",
        "timesteps": timesteps,
        "dof": design.dof,
        "joints": design.joint_names,
        "mesh_dir": str(mesh_dir),
    }
    if scenario is not None:
        summary["scenario"] = scenario.task_id
    if eval_episodes > 0:
        summary["eval"] = evaluate_policy_success(out_path, design, mesh_dir,
                                                   episodes=eval_episodes, seed=seed + 999,
                                                   scenario=scenario)
    venv.close()
    return summary


def _resolve_scenarios(scenarios, *, reach: float):
    """Coerce a mixed list (ScenarioSpec | dict | action string) into specs.

    An empty/None list means "the whole built-in ADL battery".
    """
    from prosthesis_rl.agents.scenario import ScenarioAgent, library_scenarios
    from prosthesis_rl.contracts import ScenarioSpec

    if not scenarios:
        return library_scenarios(reach=reach)

    agent = ScenarioAgent(reach=reach)
    out: list[ScenarioSpec] = []
    for s in scenarios:
        if isinstance(s, ScenarioSpec):
            out.append(s)
        elif isinstance(s, dict):
            out.append(ScenarioSpec.from_dict(s))
        else:                                   # free-text action -> library/LLM
            out.append(agent.for_action(str(s)))
    return out


def train_scenario_policy(
    scenarios=None,
    timesteps: int = 300_000,
    *,
    name: str = "scenario_ppo",
    design: DesignParams | None = None,
    mesh_dir="auto",
    seed: int = 0,
    reach: float = 0.62,
    snap_samples: int = 4000,
    n_steps: int = 512,
    eval_episodes: int = 10,
    verbose: int = 1,
) -> dict[str, object]:
    """Train ONE generalist PPO across the ADL scenarios, then save it.

    Each scenario becomes a sub-env in the vectorised env, so PPO sees every task
    (crouch to the laces, lean to the bottle, pull the drawer) within one policy.
    The observation carries the goal, so the single policy is goal-conditioned and
    transfers across the scenes. `scenarios=None` trains on the full built-in
    battery; pass action strings or ScenarioSpecs to scope it.
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    from prosthesis_rl.rl.scenario_env import ScenarioReachEnv

    design = design or DesignParams()
    specs = _resolve_scenarios(scenarios, reach=reach)
    # "auto": skin the arm with the real CAD meshes (their inertia drives the
    # torques the fatigue model reads). None/path: caller controls the skin.
    if mesh_dir == "auto":
        mesh_dir = CadBridge().export_arm(design, name="candidate")

    def _make(spec, s):
        def _init():
            return ScenarioReachEnv(spec, design, mesh_dir=mesh_dir, seed=s,
                                    snap_samples=snap_samples, add_markers=False)
        return _init

    venv = DummyVecEnv([_make(spec, seed + i) for i, spec in enumerate(specs)])
    model = PPO(
        "MlpPolicy", venv, seed=seed, verbose=verbose,
        n_steps=n_steps, batch_size=n_steps, gae_lambda=0.95, gamma=0.99,
        learning_rate=3e-4, ent_coef=0.0, n_epochs=10,
        policy_kwargs={"net_arch": [128, 128]},
    )
    model.learn(total_timesteps=timesteps, progress_bar=False)

    POLICY_DIR.mkdir(parents=True, exist_ok=True)
    out_path = POLICY_DIR / name
    model.save(out_path)
    venv.close()

    summary: dict[str, object] = {
        "policy": str(out_path) + ".zip",
        "timesteps": timesteps,
        "dof": design.dof,
        "scenarios": [sp.task_id for sp in specs],
        "mesh_dir": str(mesh_dir),
    }
    if eval_episodes > 0:
        from prosthesis_rl.rl.stress_test import stress_test_battery

        report = stress_test_battery(specs, str(out_path), design=design,
                                     mesh_dir=mesh_dir, snap_samples=snap_samples)
        summary["stress_test"] = report.to_dict()
    return summary


def evaluate_policy_success(
    policy_path, design: DesignParams, mesh_dir, *, episodes: int = 20, seed: int = 0,
    scenario=None,
) -> dict[str, float]:
    """Roll out the saved policy; report success rate + mean final distance.

    With a `scenario`, every episode is pinned to the task-completing waypoint
    (the highest-weight one — laces, cap, handle), so success measures *doing the
    task* rather than reaching an arbitrary dot.
    """
    import numpy as np
    from stable_baselines3 import PPO

    model = PPO.load(str(policy_path))
    if scenario is not None:
        from prosthesis_rl.rl.scenario_env import ScenarioReachEnv
        primary = scenario.waypoints.index(scenario.primary_waypoint())
        env = ScenarioReachEnv(scenario, design, mesh_dir=mesh_dir, seed=seed,
                               eval_waypoint=primary, add_markers=False)
    else:
        from prosthesis_rl.rl.env import ReachEnv
        env = ReachEnv(design, mesh_dir=mesh_dir, seed=seed)
    successes, finals = 0, []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        info = {"distance": 1.0, "success": 0.0}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
        successes += int(info["success"])
        finals.append(info["distance"])
    return {
        "episodes": episodes,
        "success_rate": successes / episodes,
        "mean_final_cm": float(np.mean(finals)) * 100.0,
    }


def run_training_stub(tasks_per_rollout: int = 10) -> dict[str, object]:
    """Deprecated: kept for back-compat. Use train_reach_policy()."""
    return {
        "status": "deprecated",
        "use_instead": "prosthesis_rl.rl.train.train_reach_policy",
        "rollouts_per_task": tasks_per_rollout,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a PPO reach policy for the arm")
    ap.add_argument("--timesteps", type=int, default=150_000)
    ap.add_argument("--out", default="reach_ppo", help="policy name under assets/policies/")
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-episodes", type=int, default=20)
    args = ap.parse_args()

    summary = train_reach_policy(
        args.timesteps, name=args.out, n_envs=args.n_envs,
        seed=args.seed, eval_episodes=args.eval_episodes,
    )
    print("\n[train] done:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
