// Live eval / reward HUD for the MuJoCo-WASM prosthesis demo.
//
// The reward shaping lives in Python (scripts/demo + prosthesis_rl). This mirrors
// it in-browser: each animation frame it reads the live MuJoCo state, computes the
// same shaped reward, and draws rolling charts in an overlay. Pure DOM + canvas, no
// chart dependency.
//
//   reward = reach - 0.3*effort_penalty - 0.2*contact
//     reach  = max(0, 1 - dist(ee, target)/0.6)
//     effort = min(1, sum|actuator_force * qvel| / NORM_EFFORT)
//     contact = 1 if the arm geom touches anything, else 0

const HIST = 260; // samples kept on the chart
const NORM_EFFORT = 200; // W, normalizes the effort penalty into [0,1]
const REACH_RANGE = 0.6; // m, distance over which `reach` decays 1 -> 0

export class EvalOverlay {
  constructor(parent) {
    this.parent = parent; // MuJoCoDemo
    this._model = null;
    this.hist = { reward: [], reach: [], effort: [] };
    this.best = -Infinity;
    this._buildDOM();
  }

  // ---- DOM ----------------------------------------------------------------
  _buildDOM() {
    const wrap = document.createElement("div");
    wrap.style.cssText = `position:absolute;top:12px;left:12px;width:330px;
      background:rgba(12,16,22,0.72);color:#e6edf3;border-radius:10px;padding:12px 14px;
      font:12px/1.4 -apple-system,Segoe UI,Roboto,sans-serif;z-index:50;pointer-events:none;
      box-shadow:0 6px 24px rgba(0,0,0,.35);backdrop-filter:blur(6px);`;
    wrap.innerHTML = `
      <div style="font-weight:700;letter-spacing:.04em;margin-bottom:8px;">
        PROSTHESIS EVAL <span style="opacity:.5;font-weight:400">&middot; live reward</span></div>
      <div style="display:flex;gap:10px;margin-bottom:8px;">
        <div style="flex:1"><div id="ev-reward" style="font-size:26px;font-weight:700;color:#5be36e">+0.00</div>
          <div style="opacity:.55">reward &nbsp;<span id="ev-best" style="opacity:.7"></span></div></div>
        <div style="flex:1"><div id="ev-dist" style="font-size:26px;font-weight:700">--</div>
          <div style="opacity:.55">reach dist (cm)</div></div>
      </div>
      <div style="display:flex;gap:10px;margin-bottom:8px;font-size:11px;opacity:.9">
        <div style="flex:1">effort <b id="ev-eff">0</b> W</div>
        <div style="flex:1">peak &tau; <b id="ev-tau">0</b> Nm</div>
        <div style="flex:1">contact <b id="ev-col">no</b></div>
      </div>
      <canvas id="ev-chart" width="306" height="96"
        style="width:306px;height:96px;background:rgba(0,0,0,.28);border-radius:6px;display:block"></canvas>
      <div style="display:flex;gap:12px;margin-top:6px;font-size:10px;opacity:.85">
        <span style="color:#5be36e">&#9632; reward</span>
        <span style="color:#5aa9ff">&#9632; reach</span>
        <span style="color:#ffb74d">&#9632; effort penalty</span>
      </div>
      <div style="margin-top:5px;font-size:10px;opacity:.5">
        reward = reach &minus; 0.3&middot;effort &minus; 0.2&middot;contact</div>`;
    document.body.appendChild(wrap);
    this.wrap = wrap;
    this.el = {
      reward: wrap.querySelector("#ev-reward"),
      best: wrap.querySelector("#ev-best"),
      dist: wrap.querySelector("#ev-dist"),
      eff: wrap.querySelector("#ev-eff"),
      tau: wrap.querySelector("#ev-tau"),
      col: wrap.querySelector("#ev-col"),
    };
    this.ctx = wrap.querySelector("#ev-chart").getContext("2d");
    this.W = 306;
    this.H = 96;
  }

  // ---- model binding (re-runs when the scene changes) ---------------------
  _names() {
    return new Uint8Array(this.parent.model.names);
  }
  _idByName(adr, count, name) {
    const names = this._names();
    const td = new TextDecoder();
    for (let i = 0; i < count; i++) {
      let s = adr[i],
        e = s;
      while (e < names.length && names[e] !== 0) e++;
      if (td.decode(names.subarray(s, e)) === name) return i;
    }
    return -1;
  }
  _bind() {
    const m = this.parent.model;
    this.eeId = this._idByName(m.name_siteadr, m.nsite, "ee");
    this.targetBody = this._idByName(m.name_bodyadr, m.nbody, "target_marker");
    this.armGeom = this._idByName(m.name_geomadr, m.ngeom, "arm_geom");
    this._model = m;
    this.hist = { reward: [], reach: [], effort: [] };
    this.best = -Infinity;
    // Hide the panel for scenes that aren't our arm (no ee site).
    this.wrap.style.display = this.eeId >= 0 ? "block" : "none";
  }

  // ---- per-frame update ---------------------------------------------------
  update() {
    const p = this.parent;
    if (!p.model || !p.data) return;
    if (p.model !== this._model) this._bind();
    if (this.eeId < 0) return;
    const d = p.data,
      m = p.model;

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
    const reach = Math.max(0, 1 - dist / REACH_RANGE);

    let effort = 0,
      peak = 0;
    for (let i = 0; i < m.nu; i++) {
      const f = d.actuator_force[i] || 0;
      effort += Math.abs(f * (d.qvel[i] || 0));
      peak = Math.max(peak, Math.abs(f));
    }
    const effN = Math.min(1, effort / NORM_EFFORT);

    let coll = 0;
    try {
      for (let c = 0; c < d.ncon; c++) {
        const ct = d.contact.get(c);
        if (ct && (ct.geom1 === this.armGeom || ct.geom2 === this.armGeom)) {
          coll = 1;
          break;
        }
      }
    } catch (e) {
      /* contact API unavailable -> skip */
    }

    const reward = reach - 0.3 * effN - 0.2 * coll;
    if (reward > this.best) this.best = reward;

    this._push("reward", reward);
    this._push("reach", reach);
    this._push("effort", effN);

    this.el.reward.textContent = (reward >= 0 ? "+" : "") + reward.toFixed(2);
    this.el.reward.style.color =
      reward >= 0.2 ? "#5be36e" : reward >= 0 ? "#d6e36e" : "#e36e6e";
    this.el.best.textContent = "best " + this.best.toFixed(2);
    this.el.dist.textContent = (dist * 100).toFixed(1);
    this.el.eff.textContent = effort.toFixed(0);
    this.el.tau.textContent = peak.toFixed(0);
    this.el.col.textContent = coll ? "YES" : "no";
    this.el.col.style.color = coll ? "#e36e6e" : "#9fb8a8";

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
    const lo = -0.5,
      hi = 1.0;
    const y = (v) => H - ((v - lo) / (hi - lo)) * H;
    // zero baseline
    ctx.strokeStyle = "rgba(255,255,255,.16)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, y(0));
    ctx.lineTo(W, y(0));
    ctx.stroke();
    for (const [k, color] of [
      ["effort", "#ffb74d"],
      ["reach", "#5aa9ff"],
      ["reward", "#5be36e"],
    ]) {
      const a = this.hist[k];
      if (a.length < 2) continue;
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      for (let i = 0; i < a.length; i++) {
        const x = (i / (HIST - 1)) * W;
        const yy = y(a[i]);
        i ? ctx.lineTo(x, yy) : ctx.moveTo(x, yy);
      }
      ctx.stroke();
    }
  }
}
