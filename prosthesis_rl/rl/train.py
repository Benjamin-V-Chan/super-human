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


def _make_env_fn(design: DesignParams, mesh_dir, seed: int):
    from prosthesis_rl.rl.env import ReachEnv

    def _init():
        env = ReachEnv(design, mesh_dir=mesh_dir, seed=seed)
        return env

    return _init


class _ProgressCallback:
    """Minimal SB3-compatible callback that fires a progress_cb every N steps."""

    def __init__(self, progress_cb, total: int, interval: int = 500) -> None:
        self._cb = progress_cb
        self._total = total
        self._interval = interval
        self.num_timesteps = 0

    # SB3 callbacks duck-type: init / on_step / on_rollout_end
    def init_callback(self, model) -> None:
        self._model = model

    def on_training_start(self, *_, **__) -> None:
        pass

    def on_rollout_start(self) -> None:
        pass

    def on_rollout_end(self) -> None:
        pass

    def on_step(self) -> bool:
        self.num_timesteps = self._model.num_timesteps
        if self.num_timesteps % self._interval < self._model.n_envs:
            try:
                ep_rew = self._model.ep_info_buffer
                mean_rew = (
                    float(sum(e["r"] for e in ep_rew) / len(ep_rew))
                    if ep_rew else 0.0
                )
            except Exception:
                mean_rew = 0.0
            self._cb({
                "timestep": self.num_timesteps,
                "mean_reward": mean_rew,
                "progress": self.num_timesteps / max(1, self._total),
            })
        return True

    def on_training_end(self) -> None:
        pass


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
) -> dict[str, object]:
    """Train and save a PPO reach policy; return a small training summary."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    design = design or DesignParams()
    # Skin the env with the same per-link meshes the demo uses, so the policy
    # transfers to the rendered arm without a domain gap.
    mesh_dir = CadBridge().export_arm(design, name="candidate")

    venv = DummyVecEnv([
        _make_env_fn(design, mesh_dir, seed + i) for i in range(n_envs)
    ])

    model = PPO(
        "MlpPolicy", venv, seed=seed, verbose=verbose,
        n_steps=512, batch_size=512, gae_lambda=0.95, gamma=0.99,
        learning_rate=3e-4, ent_coef=0.0, n_epochs=10,
        policy_kwargs={"net_arch": [128, 128]},
    )

    cb = _ProgressCallback(progress_cb, timesteps) if progress_cb else None
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
    if eval_episodes > 0:
        summary["eval"] = evaluate_policy_success(out_path, design, mesh_dir,
                                                   episodes=eval_episodes, seed=seed + 999)
    venv.close()
    return summary


def evaluate_policy_success(
    policy_path, design: DesignParams, mesh_dir, *, episodes: int = 20, seed: int = 0,
) -> dict[str, float]:
    """Roll out the saved policy; report success rate + mean final distance."""
    import numpy as np
    from stable_baselines3 import PPO

    from prosthesis_rl.rl.env import ReachEnv

    model = PPO.load(str(policy_path))
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
