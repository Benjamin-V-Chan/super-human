from __future__ import annotations

from pathlib import Path

import pytest

from prosthesis_rl.agents import EvaluationAgent
from prosthesis_rl.contracts import (
    DesignParams,
    ProblemSpec,
    RewardBreakdown,
    SimFeedback,
)


class FakeCad:
    def __init__(self) -> None:
        self.names = []

    def export_arm(self, design, name="candidate") -> Path:
        self.names.append(name)
        return Path("/tmp") / name


class FakeVerifier:
    def __init__(self) -> None:
        self.calls = []

    def evaluate(self, problem, design, hints, **kwargs) -> SimFeedback:
        del problem, design, hints
        self.calls.append(kwargs)
        seed = kwargs["seed"]
        success = 0.5 + 0.1 * seed
        return SimFeedback(
            reward=success - 0.1,
            breakdown=RewardBreakdown(success=success, energy_penalty=0.1),
            metrics={
                "reach_success": success,
                "energy": 10.0 + seed,
                "self_collision": 0.1 * seed,
            },
        )


def test_evaluation_agent_aggregates_fixed_seed_rollouts() -> None:
    cad = FakeCad()
    verifier = FakeVerifier()
    result = EvaluationAgent(verifier=verifier, cad=cad).evaluate(
        ProblemSpec(),
        DesignParams(),
        task_id="reach bottle/v1",
        seeds=(0, 1, 2),
        n_targets=4,
        seconds=2.0,
    )

    assert result.task_id == "reach bottle/v1"
    assert result.num_rollouts == 12
    assert result.success_rate == pytest.approx(0.6)
    assert result.mean_reward == pytest.approx(0.5)
    assert result.mean_energy == pytest.approx(11.0)
    assert result.collision_rate == pytest.approx(0.1)
    assert cad.names == ["eval_reach_bottle_v1"]
    assert [call["seed"] for call in verifier.calls] == [0, 1, 2]
    assert all(call["n_targets"] == 4 for call in verifier.calls)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"task_id": ""}, "task_id"),
        ({"task_id": "task", "seeds": ()}, "seed"),
        ({"task_id": "task", "n_targets": 0}, "n_targets"),
        ({"task_id": "task", "seconds": 0.0}, "seconds"),
    ],
)
def test_evaluation_agent_rejects_invalid_runs(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        EvaluationAgent(verifier=FakeVerifier(), cad=FakeCad()).evaluate(
            ProblemSpec(), DesignParams(), **kwargs
        )
