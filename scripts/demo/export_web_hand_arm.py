"""Build a ~23-DoF prosthesis: a positioning ARM + the dexterous Shadow Hand.

Grafts the vendored Shadow Hand (webdemo/assets/scenes/shadow_hand/right_hand.xml,
20 actuators / 24 joints) onto a 3-DoF capsule arm (shoulder flex+abduct, elbow), so
the hand can be positioned at a target AND its fingers can close to grasp. Output is a
single browser-loadable scene (meshes resolve from shadow_hand/assets, already vendored).

    python3 scripts/demo/export_web_hand_arm.py

Writes webdemo/assets/scenes/arm_hand.xml and validates it compiles.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SCENES = ROOT / "webdemo" / "assets" / "scenes"
HAND_XML = SCENES / "shadow_hand" / "right_hand.xml"
OUT = SCENES / "arm_hand.xml"

MOUNT = (0.0, -0.40, 1.00)
UPPER = 0.28
FORE = 0.24
# Hand attach pose at the forearm tip. quat reorients the Shadow Hand (its root uses
# quat "1 -1 1 -1") so the palm faces forward/down off the end of the arm.
HAND_POS = (0.0, 0.0, -FORE)
HAND_QUAT = (0.5, -0.5, 0.5, -0.5)


def _inner(root: ET.Element, tag: str) -> str:
    el = root.find(tag)
    if el is None:
        return ""
    return "\n".join(ET.tostring(c, encoding="unicode") for c in el)


def main() -> None:
    import mujoco

    hand = ET.parse(HAND_XML).getroot()
    hand_default = _inner(hand, "default")
    hand_asset = _inner(hand, "asset")
    hand_actuator = _inner(hand, "actuator")
    hand_contact = _inner(hand, "contact")
    hand_tendon = _inner(hand, "tendon")

    # The hand's root body (rh_forearm) — re-pose it onto the arm wrist.
    hand_body = hand.find("worldbody").find("body")
    hand_body.set("pos", " ".join(str(v) for v in HAND_POS))
    hand_body.set("quat", " ".join(str(v) for v in HAND_QUAT))
    hand_body_str = ET.tostring(hand_body, encoding="unicode")

    mx, my, mz = MOUNT
    xml = f"""<mujoco model="arm_hand">
  <compiler angle="radian" meshdir="shadow_hand/assets" autolimits="true"/>
  <option gravity="0 0 -9.81" integrator="implicitfast" timestep="0.002"/>
  <visual>
    <global offwidth="1280" offheight="720"/>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.6 0.6 0.6"/>
    <rgba haze="0.85 0.88 0.92 1"/>
  </visual>
  <asset>
    <texture name="skybox" type="skybox" builtin="gradient"
             rgb1="0.55 0.62 0.72" rgb2="0.10 0.12 0.16" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.30 0.32 0.36"
             rgb2="0.22 0.24 0.28" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="14 14" reflectance="0.05"/>
{hand_asset}
  </asset>
  <default>
{hand_default}
  </default>
  <worldbody>
    <light pos="0 0 4" dir="0 0 -1" diffuse="0.9 0.9 0.9"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="grid"/>
    <body name="mount" pos="{mx} {my} {mz}">
      <geom name="mount_geom" type="box" size="0.045 0.045 0.045" rgba="0.3 0.3 0.35 1"/>
      <body name="upper_arm" pos="0 0 0">
        <joint name="shoulder_flex" type="hinge" axis="0 1 0" range="-1.571 2.094"
               damping="1.0" armature="0.02"/>
        <joint name="shoulder_abduct" type="hinge" axis="1 0 0" range="-1.047 1.571"
               damping="1.0" armature="0.02"/>
        <geom name="upper_arm_geom" type="capsule" fromto="0 0 0 0 0 {-UPPER}"
              size="0.028" rgba="0.7 0.72 0.78 1"/>
        <body name="forearm" pos="0 0 {-UPPER}">
          <joint name="elbow" type="hinge" axis="0 1 0" range="0 2.269"
                 damping="1.0" armature="0.02"/>
          <geom name="forearm_geom" type="capsule" fromto="0 0 0 0 0 {-FORE}"
                size="0.024" rgba="0.7 0.72 0.78 1"/>
          {hand_body_str}
          <site name="ee" pos="0 0 {-(FORE + 0.10):.4g}" size="0.014"
                rgba="0.95 0.2 0.2 1" group="1"/>
        </body>
      </body>
    </body>
    <body name="target_marker" mocap="true" pos="-0.10 0.12 0.55">
      <geom type="sphere" size="0.025" rgba="0.1 0.9 0.2 0.5" contype="0" conaffinity="0"/>
    </body>
    <!-- graspable scene: a can on a pedestal, right where the reach pose puts the hand -->
    <body name="table" pos="-0.10 0.12 0.215">
      <geom name="table_geom" type="box" size="0.10 0.10 0.215" rgba="0.40 0.33 0.26 1"/>
    </body>
    <body name="can" pos="-0.10 0.12 0.50">
      <freejoint/>
      <geom name="can_geom" type="cylinder" size="0.03 0.05" rgba="0.85 0.2 0.2 1"
            mass="0.12" friction="1.6 0.05 0.001" condim="4"/>
    </body>
  </worldbody>
  <contact>
    <exclude body1="mount" body2="upper_arm"/>
    <exclude body1="upper_arm" body2="forearm"/>
    <exclude body1="forearm" body2="rh_forearm"/>
    <exclude body1="can" body2="mount"/>
    <exclude body1="can" body2="upper_arm"/>
    <exclude body1="can" body2="forearm"/>
    <exclude body1="can" body2="rh_forearm"/>
    <exclude body1="can" body2="rh_wrist"/>
{hand_contact}
  </contact>
  <tendon>
{hand_tendon}
  </tendon>
  <actuator>
    <position name="act_shoulder_flex" joint="shoulder_flex" kp="250" kv="22"
              ctrlrange="-1.571 2.094" forcerange="-200 200"/>
    <position name="act_shoulder_abduct" joint="shoulder_abduct" kp="250" kv="22"
              ctrlrange="-1.047 1.571" forcerange="-200 200"/>
    <position name="act_elbow" joint="elbow" kp="250" kv="22"
              ctrlrange="0 2.269" forcerange="-200 200"/>
{hand_actuator}
  </actuator>
</mujoco>
"""
    OUT.write_text(xml)
    m = mujoco.MjModel.from_xml_path(str(OUT))
    print(f"arm_hand.xml: nu={m.nu} njnt={m.njnt} nbody={m.nbody} "
          f"(arm 3 + hand {m.nu - 3})  ee+target sites ok")


if __name__ == "__main__":
    main()
