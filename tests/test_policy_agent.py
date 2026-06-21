from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from prosthesis_rl.agents import PolicyAgent
from prosthesis_rl.contracts import DesignParams, PolicyArtifact
from prosthesis_rl.rl.rollout import build_obs, map_action, run_policy_reach
from prosthesis_rl.rl.train import _checkpoint_path, _validate_training_args
from prosthesis_rl.sim.mjcf_builder import build_mjcf


def test_policy_agent_returns_inspectable_artifact(tmp_path: Path) -> None:
    checkpoint = tmp_path / "reach.zip"
    checkpoint.write_bytes(b"checkpoint")
    calls = {}

    def trainer(timesteps, **kwargs):
        calls.update({"timesteps": timesteps, **kwargs})
        return {"policy": str(checkpoint), "timesteps": 2500, "eval": {"success_rate": 0.6}}

    result = PolicyAgent(trainer=trainer).train(
        DesignParams(),
        timesteps=500,
        name="reach",
        mesh_dir="mesh",
        resume_from="prior.zip",
    )

    assert calls["resume_from"] == "prior.zip"
    assert result["artifact"]["kind"] == "rl_checkpoint"
    assert result["artifact"]["path"] == str(checkpoint)
    assert result["artifact"]["metadata"]["action_size"] == 4
    assert result["artifact"]["metadata"]["observation_size"] == 14
    assert result["artifact"]["metadata"]["timesteps"] == 2500


def test_policy_agent_validates_checkpoint_shape(tmp_path: Path) -> None:
    checkpoint = tmp_path / "reach.zip"
    checkpoint.write_bytes(b"checkpoint")
    policy = SimpleNamespace(
        action_space=SimpleNamespace(shape=(3,)),
        observation_space=SimpleNamespace(shape=(12,)),
    )
    agent = PolicyAgent(loader=lambda path: policy)

    with pytest.raises(ValueError, match="action shape"):
        agent.load(checkpoint, design=DesignParams())


def test_policy_agent_resolves_zip_suffix(tmp_path: Path) -> None:
    checkpoint = tmp_path / "reach.zip"
    checkpoint.write_bytes(b"checkpoint")
    policy = SimpleNamespace(
        action_space=SimpleNamespace(shape=(4,)),
        observation_space=SimpleNamespace(shape=(14,)),
    )
    loaded = []
    agent = PolicyAgent(loader=lambda path: loaded.append(path) or policy)

    assert agent.load(tmp_path / "reach", design=DesignParams()) is policy
    assert loaded == [checkpoint]


def test_policy_contract_rejects_unsupported_kind() -> None:
    artifact = PolicyArtifact(kind="unknown", path="policy.bin")
    assert artifact.validate() == ["unsupported policy kind: unknown"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"timesteps": 0, "n_envs": 1, "eval_episodes": 0, "name": "p"}, "timesteps"),
        ({"timesteps": 1, "n_envs": 0, "eval_episodes": 0, "name": "p"}, "n_envs"),
        ({"timesteps": 1, "n_envs": 1, "eval_episodes": -1, "name": "p"}, "eval_episodes"),
        ({"timesteps": 1, "n_envs": 1, "eval_episodes": 0, "name": "../p"}, "filename"),
    ],
)
def test_training_arguments_are_validated(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        _validate_training_args(**kwargs)


def test_checkpoint_path_preserves_dotted_names() -> None:
    assert _checkpoint_path(Path("policy.v2")) == Path("policy.v2.zip")


def test_observation_and_action_contracts() -> None:
    obs = build_obs(
        q=[0.0, 1.0],
        qd=[0.0, 20.0],
        ee=[1.0, 2.0, 3.0],
        target=[0.5, 1.5, 2.5],
        mid=[0.0, 0.0],
        half=[1.0, 2.0],
    )
    assert obs.dtype == np.float32
    assert obs.shape == (10,)
    assert obs[3] == 10.0
    assert np.allclose(
        map_action([2.0, -2.0], [0.0, 0.0], [1.0, 2.0], [-1.0, -2.0], [1.0, 2.0]),
        [1.0, -2.0],
    )
    with pytest.raises(ValueError, match="matching shapes"):
        map_action([0.0], [0.0, 0.0], [1.0, 1.0], [-1.0, -1.0], [1.0, 1.0])


def test_rollout_control_rate_is_independent_of_render_fps() -> None:
    mujoco = pytest.importorskip("mujoco")
    design = DesignParams()
    model = mujoco.MjModel.from_xml_string(build_mjcf(design), {})
    target = np.array([0.1, 0.0, 0.7])

    class CountingPolicy:
        def __init__(self):
            self.calls = 0

        def predict(self, obs, deterministic=True):
            self.calls += 1
            return np.zeros(design.dof), None

    counts = []
    for fps in (10, 40):
        policy = CountingPolicy()
        run_policy_reach(
            model,
            mujoco.MjData(model),
            design,
            target,
            policy,
            seconds=0.2,
            fps=fps,
            control_hz=25.0,
        )
        counts.append(policy.calls)

    assert counts == [5, 5]
