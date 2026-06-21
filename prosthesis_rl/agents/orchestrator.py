from __future__ import annotations

from pathlib import Path

from prosthesis_rl.agents.design import DesignAgent
from prosthesis_rl.agents.perception import PerceptionAgent
from prosthesis_rl.cad.bridge import CadBridge
from prosthesis_rl.contracts import SimFeedback
from prosthesis_rl.sim.verifier import Verifier


class ProsthesisLoop:
    """Owns the end-to-end loop across perception, design, CAD, and sim."""

    def __init__(
        self,
        perception: PerceptionAgent | None = None,
        design: DesignAgent | None = None,
        cad: CadBridge | None = None,
        verifier: Verifier | None = None,
    ) -> None:
        self.perception = perception or PerceptionAgent()
        self.design = design or DesignAgent()
        self.cad = cad or CadBridge()
        self.verifier = verifier or Verifier()

    def run_once(self, clip_path: str | Path) -> SimFeedback:
        problem = self.perception.infer_problem(clip_path)
        params, control_hints = self.design.propose(problem)
        # Per-link meshes (one STL per articulated body) + manifest; the verifier
        # skins the simulated arm with these instead of discarding the geometry.
        mesh_dir = self.cad.export_arm(params)
        return self.verifier.evaluate(problem, params, control_hints, mesh_dir=mesh_dir)

