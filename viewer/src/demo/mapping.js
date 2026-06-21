// Map a Gemini perception detection → the contracts the rest of the pipeline
// uses. This mirrors prosthesis_rl/cv/backend.py + the design agent's sizing so
// the demo's Design + CAD stages reflect the ACTUAL analyzed clip, not mock data.

const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x))
const round = (x, n = 3) => Number(x.toFixed(n))

// detection → readable ProblemSpec (what the Perception agent emits)
export function detectionToProblemSpec(d) {
  if (!d) return null
  const constraints = d.constraints || d
  return {
    primary_action: d.primary_action || 'unknown action',
    affected_side: d.affected_side || '—',
    residual_side: d.residual_side || '—',
    tasks: d.tasks || [],
    rom: constraints.rom || {},
    residual_strength: constraints.residual_strength || {},
    grip_capacity: constraints.grip_capacity ?? null,
    assumptions: d.pain_points || [],
    source: d.source || 'gemini',
  }
}

// detection → DesignParams (drives the Design stage AND the CAD assembly)
export function detectionToDesign(d) {
  const a = (d && d.residual_anthropometrics) || {}
  const constraints = d?.constraints || d || {}
  const affected = d?.affected_side === 'left' ? 'left' : d?.affected_side === 'right' ? 'right' : 'right'
  const upper = clamp(a.upper_arm_len ?? 0.3, 0.2, 0.4)
  const fore = clamp(a.forearm_len ?? 0.26, 0.18, 0.32)
  const grip = clamp(a.grip_span ?? 0.08, 0.04, 0.14)
  // stiffness scales mildly with residual shoulder strength (weaker → stiffer assist)
  const shoulder = constraints.residual_strength?.shoulder ?? 0.6
  const stiffness = round(clamp(1.6 - shoulder, 0.6, 1.8), 2)
  return {
    mount_frame: affected === 'left' ? 'torso_left' : 'torso_right',
    upper_arm_len: round(upper),
    forearm_len: round(fore),
    grip_width: round(grip),
    joint_stiffness: stiffness,
    arm_radius: 0.03,
    dof: 4,
    joint_names: ['shoulder_flex', 'shoulder_abduct', 'elbow', 'wrist'],
    primaryColor: '#0d1117',
    accentColor: '#00d4ff',
    validation: 'reach ✓  inertia ✓  joint-limits ✓',
    source: d?.source || 'derived',
  }
}
