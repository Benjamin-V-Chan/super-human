from __future__ import annotations

from pathlib import Path

from prosthesis_rl.contracts import DesignParams


class CadBridge:
    """DesignParams -> CAD model -> STL.

    The current implementation writes a placeholder STL. Swap this with
    CadQuery/OpenSCAD inside Daytona once the sandbox is wired.
    """

    def __init__(self, output_dir: str | Path = "assets/stl") -> None:
        self.output_dir = Path(output_dir)

    def export_stl(self, params: DesignParams, name: str = "candidate") -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stl_path = self.output_dir / f"{name}.stl"
        stl_path.write_text(self._placeholder_stl(params), encoding="utf-8")
        return stl_path

    def _placeholder_stl(self, params: DesignParams) -> str:
        return "\n".join(
            [
                "solid prosthesis_candidate",
                f"  // upper_arm_len={params.upper_arm_len}",
                f"  // forearm_len={params.forearm_len}",
                f"  // joint_stiffness={params.joint_stiffness}",
                f"  // grip_width={params.grip_width}",
                "endsolid prosthesis_candidate",
                "",
            ]
        )

