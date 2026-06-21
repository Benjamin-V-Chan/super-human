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
import math
from pathlib import Path

from prosthesis_rl.cad.bridge import CadBridge
from prosthesis_rl.contracts import DesignParams

POLICY_DIR = Path("assets/policies")


def _checkpoint_path(out_path: Path) -> Path:
    return Path(f"{out_path}.zip")


def _validate_training_args(timesteps: int, n_envs: int, eval_episodes: int, name: str) -> str:
    if timesteps < 1:
        raise ValueError("timesteps must be at least 1")
    if n_envs < 1:
        raise ValueError("n_envs must be at least 1")
    if eval_episodes < 0:
        raise ValueError("eval_episodes cannot be negative")
    clean_name = name.removesuffix(".zip")
    if not clean_name or Path(clean_name).name != clean_name:
        raise ValueError("name must be a filename, not a path")
    return clean_name


def _make_env_fn(design: DesignParams, mesh_dir, seed: int):
    from prosthesis_rl.rl.env import ReachEnv

    def _init():
        env = ReachEnv(design, mesh_dir=mesh_dir, seed=seed)
        return env

    return _init


def _make_progress_callback(progress_cb, total: int, start: int = 0, interval: int = 500):
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
                    "progress": min(1.0, (self.num_timesteps - start) / max(1, total)),
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
    mesh_dir: str | Path | None = None,
    output_dir: str | Path = POLICY_DIR,
    resume_from: str | Path | None = None,
) -> dict[str, object]:
    """Train and save a PPO reach policy; return a small training summary."""
    name = _validate_training_args(timesteps, n_envs, eval_episodes, name)
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    design = design or DesignParams()
    # Skin the env with the same per-link meshes the demo uses, so the policy
    # transfers to the rendered arm without a domain gap.
    mesh_dir = Path(mesh_dir) if mesh_dir is not None else CadBridge().export_arm(
        design, name=f"{name}_training"
    )

    venv = DummyVecEnv([
        _make_env_fn(design, mesh_dir, seed + i) for i in range(n_envs)
    ])

    try:
        if resume_from is not None:
            model = PPO.load(str(resume_from), env=venv)
            start_steps = int(model.num_timesteps)
        else:
            rollout_steps = max(8, min(512, math.ceil(timesteps / n_envs)))
            batch_size = min(512, rollout_steps * n_envs)
            model = PPO(
                "MlpPolicy", venv, seed=seed, verbose=verbose,
                n_steps=rollout_steps, batch_size=batch_size, gae_lambda=0.95, gamma=0.99,
                learning_rate=3e-4, ent_coef=0.0, n_epochs=10,
                policy_kwargs={"net_arch": [128, 128]},
            )
            start_steps = 0

        cb = (
            _make_progress_callback(progress_cb, timesteps, start=start_steps)
            if progress_cb else None
        )
        model.learn(
            total_timesteps=timesteps,
            progress_bar=False,
            callback=cb,
            reset_num_timesteps=resume_from is None,
        )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / name
        model.save(out_path)
        checkpoint_path = _checkpoint_path(out_path)

        summary = {
            "policy": str(checkpoint_path),
            "timesteps": int(model.num_timesteps),
            "training_steps": int(model.num_timesteps) - start_steps,
            "requested_training_steps": timesteps,
            "dof": design.dof,
            "joints": design.joint_names,
            "mesh_dir": str(mesh_dir),
        }
        if eval_episodes > 0:
            summary["eval"] = evaluate_policy_success(
                out_path, design, mesh_dir, episodes=eval_episodes, seed=seed + 999
            )
        return summary
    finally:
        venv.close()


def evaluate_policy_success(
    policy_path, design: DesignParams, mesh_dir, *, episodes: int = 20, seed: int = 0,
) -> dict[str, float]:
    """Roll out the saved policy; report success rate + mean final distance."""
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    import numpy as np
    from stable_baselines3 import PPO

    from prosthesis_rl.rl.env import ReachEnv

    model = PPO.load(str(policy_path))
    env = ReachEnv(design, mesh_dir=mesh_dir, seed=seed)
    successes, finals = 0, []
    try:
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
    finally:
        env.close()
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
