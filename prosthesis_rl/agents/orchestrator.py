from __future__ import annotations

from pathlib import Path
from typing import Any

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
        spec_sheet = format_spec_sheet(problem)
        if self.emit_spec_sheet:
            print(spec_sheet)
        params, control_hints = self.design.propose(problem)
        mesh_dir = self.cad.export_arm(params)
        return self.verifier.evaluate(problem, params, control_hints, mesh_dir=mesh_dir)

    def run_optimized(
        self,
        clip_path: str | Path,
        emit=None,
        quick_mode: bool = False,
    ) -> dict[str, Any]:
        """Full iterative CAD↔sim RL feedback loop with SSE streaming.

        Args:
            clip_path: path to ADL video clip
            emit: Emitter instance for SSE streaming (created if None)
            quick_mode: reduce seeds/timesteps for fast demo

        Returns result dict with best_params, stats, trajectory, rl_result.
        """
        from prosthesis_rl.pipeline.events import Emitter
        from prosthesis_rl.pipeline.loop import DesignOptimizationLoop

        emitter = emit or Emitter()
        loop = DesignOptimizationLoop(quick_mode=quick_mode)
        return loop.run(clip_path, emitter)

