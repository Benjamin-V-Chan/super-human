"""Format a ProblemSpec into a core-specs handoff sheet for the design agent.

The perception stage emits this so the design agent has a single, readable brief
of what to optimize the prosthesis around: the specific action observed, which
side needs the device, and the target ROM / strength / grip envelope.
"""

from __future__ import annotations

from prosthesis_rl.contracts import ProblemSpec


def format_spec_sheet(spec: ProblemSpec) -> str:
    """Render the core specs as a formatted, human-readable brief."""
    affected = spec.affected_side or "?"
    residual = spec.residual_side or "?"
    mount = f"torso_{affected}" if affected in {"left", "right"} else "torso_?"
    action = spec.primary_action or "(unspecified)"
    task_names = [t.get("name", t.get("id", "?")) for t in spec.tasks]

    lines: list[str] = []
    lines.append("=" * 56)
    lines.append("  PERCEPTION → DESIGN  ·  CORE SPEC SHEET")
    lines.append("=" * 56)
    lines.append(f"  Primary action      : {action}")
    lines.append(f"  Affected side       : {affected}  (needs prosthesis)")
    lines.append(f"  Residual side       : {residual}  (compensating)")
    lines.append(f"  Mount frame         : {mount}")
    lines.append(f"  ADL task categories : {', '.join(task_names) or '—'}")
    lines.append("-" * 56)
    lines.append("  Target ROM (deg):")
    if spec.constraints.rom:
        for joint, value in spec.constraints.rom.items():
            lines.append(f"    - {joint:<18}: {value:g}")
    else:
        lines.append("    - (none provided)")
    lines.append("  Residual strength (0..1):")
    if spec.constraints.residual_strength:
        for region, value in spec.constraints.residual_strength.items():
            lines.append(f"    - {region:<18}: {value:g}")
    else:
        lines.append("    - (none provided)")
    lines.append(f"  Grip capacity (0..1) : {spec.constraints.grip_capacity:g}")
    lines.append("-" * 56)

    pain_points: list[str] = []
    for task in spec.tasks:
        for pp in task.get("pain_points", []):
            if pp not in pain_points:
                pain_points.append(pp)
    lines.append("  Observed difficulties:")
    if pain_points:
        for pp in pain_points:
            lines.append(f"    • {pp}")
    else:
        lines.append("    • (none observed)")
    lines.append("-" * 56)
    lines.append(
        f"  DESIGN DIRECTIVE: optimize a {affected}-side prosthesis to perform "
        f'"{action}".'
    )
    lines.append("=" * 56)
    return "\n".join(lines)
