// Scripted grasp for the 23-DoF hand+arm (arm_hand.xml). When a grasp task is active
// it runs a reach -> descend -> close -> hold -> release loop, driving data.ctrl, so the
// dexterous hand visibly practices grabbing the can. (Reliable lift needs finer hand
// positioning than the 3-DoF arm gives; this is the scripted first pass.)

import { nameToId } from "./mujocoUtils.js";

const ABOVE = [0.04, 0.98, 1.14]; // arm pose: hand above the can
const AT = [0.17, 0.89, 0.95]; // arm pose: hand at the can
const CLOSE = {
  rh_A_THJ5: 1.0,
  rh_A_THJ4: 1.0,
  rh_A_THJ2: 0.5,
  rh_A_THJ1: 1.4,
  rh_A_FFJ3: 1.4,
  rh_A_FFJ0: 2.6,
  rh_A_MFJ3: 1.4,
  rh_A_MFJ0: 2.6,
  rh_A_RFJ3: 1.4,
  rh_A_RFJ0: 2.6,
  rh_A_LFJ3: 1.4,
  rh_A_LFJ0: 2.6,
};
// [label, frames, armPose, fingersClosed]
const PHASES = [
  ["reach", 80, ABOVE, 0],
  ["descend", 55, AT, 0],
  ["close", 55, AT, 1],
  ["hold", 60, AT, 1],
  ["release", 45, ABOVE, 0],
];

export class GraspController {
  constructor(parent) {
    this.parent = parent;
    this.active = false;
    this.t = 0;
    this.phase = 0;
    this._model = null;
  }

  enable(on) {
    this.active = !!on;
    this.t = 0;
    this.phase = 0;
  }

  get phaseName() {
    return this.active ? PHASES[this.phase][0] : "idle";
  }

  _bind() {
    const m = this.parent.model;
    this.armAct = ["act_shoulder_flex", "act_shoulder_abduct", "act_elbow"].map(
      (n) => nameToId(m, m.name_actuatoradr, m.nu, n),
    );
    this.fingerAct = {};
    for (const k in CLOSE) {
      const id = nameToId(m, m.name_actuatoradr, m.nu, k);
      if (id >= 0) this.fingerAct[id] = CLOSE[k];
    }
    this._model = m;
  }

  update() {
    if (!this.active) return;
    const p = this.parent;
    if (!p.model || !p.data) return;
    if (p.model !== this._model) this._bind();
    if (!this.armAct || this.armAct[0] < 0) return; // not the hand scene
    const d = p.data;
    const [, dur, arm, closed] = PHASES[this.phase];

    for (let i = 0; i < 3; i++) {
      const a = this.armAct[i];
      if (a >= 0) d.ctrl[a] += (arm[i] - d.ctrl[a]) * 0.06; // smooth toward pose
    }
    for (const id in this.fingerAct) {
      const tgt = closed ? this.fingerAct[id] : 0.0;
      d.ctrl[id] += (tgt - d.ctrl[id]) * 0.08;
    }

    if (++this.t >= dur) {
      this.t = 0;
      this.phase = (this.phase + 1) % PHASES.length;
    }
  }
}
