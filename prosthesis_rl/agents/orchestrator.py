from __future__ import annotations

from pathlib import Path

from prosthesis_rl.agents.design import DesignAgent
from prosthesis_rl.agents.perception import PerceptionAgent
from prosthesis_rl.agents.spec_sheet import format_spec_sheet
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
        emit_spec_sheet: bool = True,
    ) -> None:
        self.perception = perception or PerceptionAgent()
        self.design = design or DesignAgent()
        self.cad = cad or CadBridge()
        self.verifier = verifier or Verifier()
        self.emit_spec_sheet = emit_spec_sheet

    def run_once(self, clip_path: str | Path) -> SimFeedback:
        problem = self.perception.infer_problem(clip_path)
        # Perception -> Design handoff: the formatted core-spec sheet the design
        # agent optimizes around.
        spec_sheet = format_spec_sheet(problem)
        if self.emit_spec_sheet:
            print(spec_sheet)
        params, control_hints = self.design.propose(problem)
        stl_path = self.cad.export_stl(params)
        return self.verifier.evaluate(problem, params, control_hints, stl_path=stl_path)

