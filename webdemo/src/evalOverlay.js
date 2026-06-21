// Live multi-objective GRADER overlay for the MuJoCo-WASM prosthesis demo.
//
// The reward shaping lives in Python (prosthesis_rl RewardBreakdown). This mirrors it
// in-browser: each frame it reads live MuJoCo state, computes a 4-component grade driven
// by the ACTIVE TASK's weights (from tasks.json), and draws bars + a rolling chart.
// Pure DOM + canvas, no chart dependency.
//
//   total = w.success*success - w.effort*effort - w.rom*rom - w.stability*stability
//     success   = clamp(1 - dist(ee,target)/REACH_RANGE)   (HIT when dist <= threshold)
//     effort    = min(1, sum|actuator_force*qvel| / NORM_EFFORT)
//     rom       = total joint travel beyond limits (Python rom_penalty; ~0 within range)
//     stability = max(contact, smoothness)  (smoothness penalizes jerk/tremor)
//
// The per-frame computation produces a `grade` object {components, weights, total, source}.
// A future source:"hud" swaps local compute for a grade fetched from the Python HUD env;
// the renderer only consumes the object, so the swap is one branch in update().

import { nameToId } from "./mujocoUtils.js";

const HIST = 260; // chart samples kept
const NORM_EFFORT = 200; // W, normalizes the effort penalty into [0,1]
const REACH_RANGE = 0.6; // m, distance over which `success` decays 1 -> 0
const QVEL_DELTA_NORM = 1.5; // per-frame |Δqvel| that maps to smoothness=1
const ROM_NORM = 0.5; // rad of cumulative over-limit travel that maps rom to 1
const ARM_BODIES = ["mount", "arm", "upper_arm", "forearm", "gripper"];
const DEFAULT_WEIGHTS = { success: 1.0, effort: 0.3, rom: 1.0, stability: 0.2 };
const DEFAULT_THR_CM = 8;
const BEST_KEY = "armasai_best_v1";

const COMPS = [
  ["success", "#5be36e"],
  ["effort", "#ffb74d"],
  ["rom", "#f0c64b"],
  ["stability", "#e36e6e"],
];

export class EvalOverlay {
  constructor(parent) {
    this.parent = parent; // MuJoCoDemo
    this._model = null;
    this.hist = { total: [], success: [], penalty: [] };
    this.task = null;
    this.weights = { ...DEFAULT_WEIGHTS };
    this.thrCm = DEFAULT_THR_CM;
    this.qvelPrev = null;
    this.bestByTask = this._loadBests();
    this._buildDOM();
  }

  setTask(task) {
    this.task = task;
    this.weights = (task && task.grader && task.grader.weights) || {
      ...DEFAULT_WEIGHTS,
    };
    this.thrCm =
      (task && task.grader && task.grader.success_threshold_cm) ||
      DEFAULT_THR_CM;
    if (this.el) {
      this.el.task.textContent = task
        ? `${task.name}  ·  ${task.arm} arm`
        : "—";
    }
  }

  // ---- DOM ----------------------------------------------------------------
  _buildDOM() {
    const wrap = document.createElement("div");
    wrap.style.cssText = `position:absolute;top:12px;left:12px;width:340px;
      background:rgba(12,16,22,0.74);color:#e6edf3;border-radius:10px;padding:12px 14px;
      font:12px/1.4 -apple-system,Segoe UI,Roboto,sans-serif;z-index:50;pointer-events:none;
      box-shadow:0 6px 24px rgba(0,0,0,.35);backdrop-filter:blur(6px);`;
    const btn = `pointer-events:auto;cursor:pointer;background:rgba(255,255,255,.08);
      border:1px solid rgba(255,255,255,.18);color:#cfe;border-radius:5px;width:22px;height:20px;
      font:13px/1 sans-serif;`;
    wrap.innerHTML = `
      <div style="display:flex;align-items:center;gap:6px;">
        <div style="flex:1;font-weight:700;letter-spacing:.04em;">
          PROSTHESIS EVAL <span style="opacity:.5;font-weight:400">&middot; grader breakdown</span></div>
        <button id="ev-min" title="minimize" style="${btn}">–</button>
        <button id="ev-hide" title="hide" style="${btn}">×</button>
      </div>
      <div id="ev-body">
        <div id="ev-task" style="font-size:11px;opacity:.7;margin:2px 0 8px;">—</div>
        <div style="display:flex;gap:10px;margin-bottom:8px;">
          <div style="flex:1"><div id="ev-reward" style="font-size:26px;font-weight:700;color:#5be36e">+0.00</div>
            <div style="opacity:.55">reward</div></div>
          <div style="flex:1"><div id="ev-dist" style="font-size:26px;font-weight:700">--
            <span id="ev-hit" style="font-size:11px;color:#5be36e"></span></div>
            <div style="opacity:.55">reach dist (cm)</div></div>
        </div>
        <div id="ev-bars"></div>
        <canvas id="ev-chart" width="312" height="86"
          style="width:312px;height:86px;background:rgba(0,0,0,.28);border-radius:6px;display:block;margin-top:8px"></canvas>
        <div style="display:flex;gap:12px;margin-top:5px;font-size:10px;opacity:.85">
          <span style="color:#5be36e">&#9632; reward</span>
          <span style="color:#5aa9ff">&#9632; success</span>
          <span style="color:#ffb74d">&#9632; total penalty</span>
        </div>
        <div id="ev-board" style="margin-top:7px;font-size:10px;opacity:.8;
          border-top:1px solid rgba(255,255,255,.1);padding-top:6px"></div>
      </div>`;
    document.body.appendChild(wrap);
    this.wrap = wrap;
    this.body = wrap.querySelector("#ev-body");
    this.hidden = false;
    this.minimized = false;

    // Floating restore tab, shown only when fully hidden.
    const tab = document.createElement("button");
    tab.textContent = "▣ EVAL";
    tab.style.cssText = `position:absolute;top:12px;left:12px;z-index:50;display:none;
      pointer-events:auto;cursor:pointer;background:rgba(12,16,22,0.8);color:#cfe;
      border:1px solid rgba(255,255,255,.18);border-radius:8px;padding:6px 10px;font:12px sans-serif;`;
    document.body.appendChild(tab);
    this.tab = tab;

    wrap.querySelector("#ev-min").onclick = () => {
      this.minimized = !this.minimized;
      this.body.style.display = this.minimized ? "none" : "block";
      wrap.querySelector("#ev-min").textContent = this.minimized ? "+" : "–";
    };
    wrap.querySelector("#ev-hide").onclick = () => {
      this.hidden = true;
      wrap.style.display = "none";
      tab.style.display = "block";
    };
    tab.onclick = () => {
      this.hidden = false;
      tab.style.display = "none";
      wrap.style.display = "block";
    };

    // Build the 4 component bars programmatically.
    const bars = wrap.querySelector("#ev-bars");
    this.bars = {};
    for (const [key, color] of COMPS) {
      const row = document.createElement("div");
      row.style.cssText =
        "display:flex;align-items:center;gap:6px;margin:3px 0;font-size:11px;";
      const lab = document.createElement("div");
      lab.style.cssText = "width:118px;opacity:.85";
      const track = document.createElement("div");
      track.style.cssText =
        "flex:1;height:8px;background:rgba(255,255,255,.08);border-radius:4px;overflow:hidden;";
      const fill = document.createElement("div");
      fill.style.cssText = `height:100%;width:0%;background:${color};`;
      track.appendChild(fill);
      const val = document.createElement("div");
      val.style.cssText =
        "width:34px;text-align:right;opacity:.9;font-variant-numeric:tabular-nums";
      row.appendChild(lab);
      row.appendChild(track);
      row.appendChild(val);
      bars.appendChild(row);
      this.bars[key] = { lab, fill, val };
    }

    this.el = {
      reward: wrap.querySelector("#ev-reward"),
      dist: wrap.querySelector("#ev-dist"),
      hit: wrap.querySelector("#ev-hit"),
      task: wrap.querySelector("#ev-task"),
      board: wrap.querySelector("#ev-board"),
    };
    this.ctx = wrap.querySelector("#ev-chart").getContext("2d");
    this.W = 312;
    this.H = 86;
  }

  // ---- model binding (re-runs when the scene changes) ---------------------
  _idByName(adr, count, name) {
    return nameToId(this.parent.model, adr, count, name);
  }
  _bind() {
    const m = this.parent.model;
    this.eeId = this._idByName(m.name_siteadr, m.nsite, "ee");
    this.targetBody = this._idByName(m.name_bodyadr, m.nbody, "target_marker");
    // All geoms belonging to arm bodies (for the contact / stability check).
    const armBodyIds = new Set();
    for (const nm of ARM_BODIES) {
      const id = this._idByName(m.name_bodyadr, m.nbody, nm);
      if (id >= 0) armBodyIds.add(id);
    }
    this.armGeoms = new Set();
    for (let g = 0; g < m.ngeom; g++)
      if (armBodyIds.has(m.geom_bodyid[g])) this.armGeoms.add(g);

    this._model = m;
    this.hist = { total: [], success: [], penalty: [] };
    this.qvelPrev = new Float64Array(m.nv);
    // Weights/task survive an arm reload (taskPanel set them via setTask before reload).
    if (this.parent.activeTask) this.setTask(this.parent.activeTask);
    const hasArm = this.eeId >= 0;
    this.wrap.style.display = hasArm && !this.hidden ? "block" : "none";
    if (this.tab)
      this.tab.style.display = hasArm && this.hidden ? "block" : "none";
  }

  // ---- per-frame update ---------------------------------------------------
  update() {
    const p = this.parent;
    if (!p.model || !p.data) return;
    if (p.model !== this._model) this._bind();
    if (this.eeId < 0) return;
    const d = p.data,
      m = p.model;
    const w = this.weights;

    // success
    const ee = [
      d.site_xpos[this.eeId * 3],
      d.site_xpos[this.eeId * 3 + 1],
      d.site_xpos[this.eeId * 3 + 2],
    ];
    let tg = ee;
    if (this.targetBody >= 0)
      tg = [
        d.xpos[this.targetBody * 3],
        d.xpos[this.targetBody * 3 + 1],
        d.xpos[this.targetBody * 3 + 2],
      ];
    const dist = Math.hypot(ee[0] - tg[0], ee[1] - tg[1], ee[2] - tg[2]);
    const success = Math.max(0, Math.min(1, 1 - dist / REACH_RANGE));
    const hit = dist <= this.thrCm / 100;

    // effort
    let effortRaw = 0,
      peak = 0;
    for (let i = 0; i < m.nu; i++) {
      const f = d.actuator_force[i] || 0;
      effortRaw += Math.abs(f * (d.qvel[i] || 0));
      peak = Math.max(peak, Math.abs(f));
    }
    const effort = Math.min(1, effortRaw / NORM_EFFORT);

    // rom: total joint motion BEYOND limits (Python rom_penalty; ~0 within range,
    // so the natural rest pose isn't penalized — it spikes only on a real violation)
    let romRaw = 0;
    for (let j = 0; j < m.njnt; j++) {
      if (m.jnt_limited && !m.jnt_limited[j]) continue;
      const lo = m.jnt_range[2 * j],
        hi = m.jnt_range[2 * j + 1];
      if (!(hi > lo)) continue;
      const q = d.qpos[m.jnt_qposadr[j]];
      romRaw += Math.max(lo - q, 0) + Math.max(q - hi, 0);
    }
    const rom = Math.min(1, romRaw / ROM_NORM);

    // stability = max(contact, smoothness)
    let contact = 0;
    try {
      for (let c = 0; c < d.ncon; c++) {
        const ct = d.contact.get(c);
        if (
          ct &&
          (this.armGeoms.has(ct.geom1) || this.armGeoms.has(ct.geom2))
        ) {
          contact = 1;
          break;
        }
      }
    } catch (e) {
      /* contact API unavailable -> skip */
    }
    let dv = 0;
    for (let i = 0; i < m.nv; i++) {
      const x = (d.qvel[i] || 0) - this.qvelPrev[i];
      dv += x * x;
      this.qvelPrev[i] = d.qvel[i] || 0;
    }
    const smoothness = Math.min(1, Math.sqrt(dv) / QVEL_DELTA_NORM);
    const stability = Math.max(contact, smoothness);

    // grade object — the drop-in seam for a future source:"hud"
    const grade = {
      components: { success, effort, rom, stability },
      weights: w,
      total:
        (w.success ?? 1) * success -
        (w.effort ?? 0) * effort -
        (w.rom ?? 0) * rom -
        (w.stability ?? 0) * stability,
      source: "local",
    };

    this._track(grade);
    this._render(grade, dist, hit, peak, contact);
  }

  _track(grade) {
    const id = this.task ? this.task.id : "_";
    if (!(id in this.bestByTask) || grade.total > this.bestByTask[id]) {
      this.bestByTask[id] = grade.total;
      this._saveBests();
    }
    this._push("total", grade.total);
    this._push("success", grade.components.success);
    const c = grade.components,
      w = grade.weights;
    this._push(
      "penalty",
      (w.effort ?? 0) * c.effort +
        (w.rom ?? 0) * c.rom +
        (w.stability ?? 0) * c.stability,
    );
  }

  _render(grade, dist, hit, peak, contact) {
    const c = grade.components,
      w = grade.weights,
      t = grade.total;
    this.el.reward.textContent = (t >= 0 ? "+" : "") + t.toFixed(2);
    this.el.reward.style.color =
      t >= 0.2 ? "#5be36e" : t >= 0 ? "#d6e36e" : "#e36e6e";
    this.el.dist.firstChild.textContent = (dist * 100).toFixed(1) + " ";
    this.el.hit.textContent = hit ? "HIT" : "";

    const wl = {
      success: w.success ?? 1,
      effort: w.effort ?? 0,
      rom: w.rom ?? 0,
      stability: w.stability ?? 0,
    };
    const labels = {
      success: `success ×${wl.success}`,
      effort: `effort ×${wl.effort}`,
      rom: `ROM viol. ×${wl.rom}`,
      stability: `stability ×${wl.stability}`,
    };
    for (const [key] of COMPS) {
      const v = c[key];
      this.bars[key].lab.textContent = labels[key];
      this.bars[key].fill.style.width = (v * 100).toFixed(0) + "%";
      this.bars[key].val.textContent = v.toFixed(2);
    }

    // per-task best leaderboard (top entries)
    const id = this.task ? this.task.id : "_";
    const rows = Object.entries(this.bestByTask)
      .filter(([k]) => k !== "_")
      .sort((a, b) => b[1] - a[1])
      .slice(0, 4)
      .map(
        ([k, v]) =>
          `<div style="display:flex;justify-content:space-between;${k === id ? "color:#9fe6b0" : ""}">
            <span>${k}</span><span>${v >= 0 ? "+" : ""}${v.toFixed(2)}</span></div>`,
      )
      .join("");
    this.el.board.innerHTML =
      `<div style="opacity:.6;margin-bottom:2px">best reward / task` +
      (contact ? ` · <span style="color:#e36e6e">contact</span>` : "") +
      `</div>${rows}`;

    this._draw();
  }

  _push(k, v) {
    const a = this.hist[k];
    a.push(v);
    if (a.length > HIST) a.shift();
  }

  _draw() {
    const ctx = this.ctx,
      W = this.W,
      H = this.H;
    ctx.clearRect(0, 0, W, H);
    const lo = -0.6,
      hi = 1.0;
    const y = (v) => H - ((Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo)) * H;
    ctx.strokeStyle = "rgba(255,255,255,.16)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, y(0));
    ctx.lineTo(W, y(0));
    ctx.stroke();
    for (const [k, color] of [
      ["penalty", "#ffb74d"],
      ["success", "#5aa9ff"],
      ["total", "#5be36e"],
    ]) {
      const a = this.hist[k];
      if (a.length < 2) continue;
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      for (let i = 0; i < a.length; i++) {
        const x = (i / (HIST - 1)) * W;
        const yy = y(k === "penalty" ? -a[i] : a[i]);
        i ? ctx.lineTo(x, yy) : ctx.moveTo(x, yy);
      }
      ctx.stroke();
    }
  }

  _loadBests() {
    try {
      return JSON.parse(localStorage.getItem(BEST_KEY) || "{}");
    } catch (e) {
      return {};
    }
  }
  _saveBests() {
    try {
      localStorage.setItem(BEST_KEY, JSON.stringify(this.bestByTask));
    } catch (e) {
      /* ignore */
    }
  }
}
