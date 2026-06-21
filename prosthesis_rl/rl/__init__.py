"""Lazy public RL API; importing the package does not import training dependencies."""


def train_reach_policy(*args, **kwargs):
    from prosthesis_rl.rl.train import train_reach_policy as _train

    return _train(*args, **kwargs)


def evaluate_policy_success(*args, **kwargs):
    from prosthesis_rl.rl.train import evaluate_policy_success as _evaluate

    return _evaluate(*args, **kwargs)


def run_training_stub(*args, **kwargs):
    from prosthesis_rl.rl.train import run_training_stub as _stub

    return _stub(*args, **kwargs)

__all__ = [
    "evaluate_policy_success",
    "run_training_stub",
    "train_reach_policy",
]
