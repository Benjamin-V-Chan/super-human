"""Policy/RL agent that owns checkpoint training, validation, and loading."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Callable

from prosthesis_rl.contracts import DesignParams, PolicyArtifact, ProblemSpec
from prosthesis_rl.rl.controller import ScriptedIKController
from prosthesis_rl.rl.rollout import load_policy
from prosthesis_rl.rl.train import train_reach_policy


class PolicyAgent:
    """Stable public boundary around the PPO implementation."""

    def __init__(
        self,
        trainer: Callable[..., dict[str, Any]] = train_reach_policy,
        loader: Callable[[str | Path], Any] = load_policy,
    ) -> None:
        self._trainer = trainer
        self._loader = loader

    def build_baseline(
        self,
        problem: ProblemSpec,
        design: DesignParams,
        *,
        name: str,
        output_dir: str | Path = "assets/policies",
    ) -> PolicyArtifact:
        """Persist a runnable scripted-IK policy for immediate, reliable use."""

        safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)
        safe_name = safe_name.strip("_")[:64]
        if not safe_name:
            raise ValueError("policy name is required")
        path = Path(output_dir) / f"{safe_name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        controls = ScriptedIKController().controls_for(problem, design)
        payload = {
            "kind": "scripted_ik",
            "controller": "prosthesis_rl.sim.control.ReachController",
            "controls": controls,
            "dof": design.dof,
            "joints": design.joint_names,
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        artifact = PolicyArtifact(
            kind="scripted_ik",
            path=str(path),
            metadata={
                "dof": design.dof,
                "joints": design.joint_names,
                "controls": controls,
            },
        )
        errors = artifact.validate()
        if errors:
            raise ValueError(f"invalid policy artifact: {errors}")
        return artifact

    def train(
        self,
        design: DesignParams,
        *,
        timesteps: int,
        name: str,
        mesh_dir: str | Path | None = None,
        resume_from: str | Path | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        summary = self._trainer(
            timesteps,
            name=name,
            design=design,
            mesh_dir=mesh_dir,
            resume_from=resume_from,
            **kwargs,
        )
        artifact = PolicyArtifact(
            kind="rl_checkpoint",
            path=str(summary["policy"]),
            metadata={
                "algorithm": "PPO",
                "dof": design.dof,
                "joints": list(design.joint_names),
                "observation_size": 2 * design.dof + 6,
                "action_size": design.dof,
                "timesteps": int(summary.get("timesteps", timesteps)),
            },
        )
        errors = artifact.validate()
        if errors:
            raise ValueError(f"invalid policy artifact: {errors}")
        return {**summary, "artifact": artifact.to_dict()}

    def load(
        self,
        artifact: PolicyArtifact | str | Path,
        *,
        design: DesignParams | None = None,
    ) -> Any:
        if not isinstance(artifact, PolicyArtifact):
            artifact = PolicyArtifact(kind="rl_checkpoint", path=str(artifact))
        errors = artifact.validate()
        if errors:
            raise ValueError(f"invalid policy artifact: {errors}")
        if artifact.kind != "rl_checkpoint":
            raise ValueError("PolicyAgent can only load rl_checkpoint artifacts")

        checkpoint = Path(artifact.path)
        if not checkpoint.exists() and checkpoint.suffix != ".zip":
            checkpoint = Path(f"{checkpoint}.zip")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"policy checkpoint not found: {checkpoint}")

        policy = self._loader(checkpoint)
        if design is not None:
            expected_action = (design.dof,)
            expected_observation = (2 * design.dof + 6,)
            if tuple(policy.action_space.shape) != expected_action:
                raise ValueError(
                    f"checkpoint action shape {policy.action_space.shape} does not match "
                    f"design shape {expected_action}"
                )
            if tuple(policy.observation_space.shape) != expected_observation:
                raise ValueError(
                    f"checkpoint observation shape {policy.observation_space.shape} does not "
                    f"match design shape {expected_observation}"
                )
        return policy
