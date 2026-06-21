// Generate a MuJoCo MJCF (XML) for the prosthetic arm from DesignParams.
// A 4-DoF arm hangs from a torso mount and reaches; position actuators on each
// joint let the evaluator drive it toward a reach pose. Lengths/stiffness come
// straight from the analyzed design, so the sim reflects the actual model.
export function buildMjcf(design = {}) {
  const upper = clamp(design.upper_arm_len ?? 0.3, 0.18, 0.42)
  const fore = clamp(design.forearm_len ?? 0.26, 0.16, 0.34)
  const grip = clamp(design.grip_width ?? 0.08, 0.04, 0.16)
  const r = clamp(design.arm_radius ?? 0.03, 0.018, 0.05)
  const r2 = r * 0.9
  const halfGrip = grip / 2
  const k = clamp(design.joint_stiffness ?? 1.0, 0.5, 2.0)
  const kp = (30 * k).toFixed(1)
  const kp2 = (24 * k).toFixed(1)
  const kp3 = (14 * k).toFixed(1)
  const damp = (0.6 * Math.sqrt(k)).toFixed(3)

  return `<mujoco model="armasai_prosthesis">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.004" gravity="0 0 -9.81" integrator="implicitfast"/>
  <default>
    <joint damping="${damp}"/>
    <geom density="700" friction="1 0.1 0.1"/>
  </default>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="3 3 0.1" pos="0 0 0" rgba="0.15 0.15 0.2 1"/>
    <body name="mount" pos="0 0 1.1">
      <geom type="box" size="0.05 0.07 0.08" rgba="0.12 0.12 0.18 1"/>
      <body name="upper" pos="0 0 -0.02">
        <joint name="shoulder_flex" type="hinge" axis="0 1 0" range="-2.0 2.0"/>
        <joint name="shoulder_abduct" type="hinge" axis="1 0 0" range="-1.2 1.2"/>
        <geom type="capsule" fromto="0 0 0 0 0 ${-upper}" size="${r}" rgba="0.06 0.07 0.1 1"/>
        <body name="forearm" pos="0 0 ${-upper}">
          <joint name="elbow" type="hinge" axis="0 1 0" range="0 2.4"/>
          <geom type="capsule" fromto="0 0 0 0 0 ${-fore}" size="${r2}" rgba="0.09 0.1 0.14 1"/>
          <body name="hand" pos="0 0 ${-fore}">
            <joint name="wrist" type="hinge" axis="1 0 0" range="-1.2 1.2"/>
            <geom type="box" size="${r2} ${halfGrip} ${r2}" rgba="0.85 0.6 0.2 1"/>
            <site name="grip" pos="0 0 ${-r2 - 0.02}" size="0.012" rgba="1 0.6 0.2 1"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="a_sf" joint="shoulder_flex" kp="${kp}" ctrlrange="-2 2"/>
    <position name="a_sa" joint="shoulder_abduct" kp="${kp}" ctrlrange="-1.2 1.2"/>
    <position name="a_el" joint="elbow" kp="${kp2}" ctrlrange="0 2.4"/>
    <position name="a_wr" joint="wrist" kp="${kp3}" ctrlrange="-1.2 1.2"/>
  </actuator>
</mujoco>`
}

// Reference reach pose (qpos order: shoulder_flex, shoulder_abduct, elbow, wrist).
export const REACH_POSE = [0.9, 0.2, 1.2, 0.0]

function clamp(x, lo, hi) { return Math.min(hi, Math.max(lo, x)) }
