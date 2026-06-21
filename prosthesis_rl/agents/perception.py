from __future__ import annotations

from pathlib import Path

from prosthesis_rl.contracts import ProblemSpec
from prosthesis_rl.cv.backend import PerceptionBackend


class PerceptionAgent:
    """Clip -> Gemma video analysis -> validated ProblemSpec.

    Delegates the heavy lifting to ``PerceptionBackend`` (frame extraction +
    Gemma problem detection + contract mapping). With no Gemma key the backend
    falls back to a deterministic detection, so this always returns a valid
    ProblemSpec and the end-to-end loop stays green.
    """

    def __init__(self, backend: PerceptionBackend | None = None) -> None:
        self.backend = backend or PerceptionBackend()

    def infer_problem(self, clip_path: str | Path) -> ProblemSpec:
        return self.backend.infer_problem(clip_path)
