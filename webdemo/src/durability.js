// Durability & Design Advisor — a panel embedded in the live dashboard (live.html).
// Call initDurabilityPanel(rootEl): it builds its own compact UI into rootEl, loads
// the stress-test report written by scripts/demo/durability_dashboard.py, and
// recomputes service life + design recommendations LIVE as you change material /
// usage / load — the fatigue model is cheap closed-form, so no re-rollout is needed.
//
// The fatigue + recommender math below is a faithful port of
// prosthesis_rl/fatigue/{estimate,recommend}.py; the per-task stress amplitudes
// (the expensive, physics-derived part) come from the report, captured at load 1.0.

const REPORT_URL = "./assets/durability/report.json";

// ── fatigue model (port of fatigue.estimate) ────────────────────────────────
function effectiveSigmaF(mat) {
  return mat.sigma_f_prime_mpa * mat.process_knockdown; // MPa
}
function yearsFromAmplitude(ampMPa, mat, usagePerDay) {
  if (ampMPa <= mat.endurance_limit_mpa || ampMPa <= 0) return Infinity;
  const N = 0.5 * Math.pow(ampMPa / effectiveSigmaF(mat), 1 / mat.basquin_b);
  return N / (usagePerDay * 365);
}
function displayYears(y) {
  if (!isFinite(y)) return "∞";
  if (y >= 100) return "≥100 yr";
  if (y >= 10) return y.toFixed(0) + " yr";
  if (y >= 1) return y.toFixed(1) + " yr";
  if (y * 12 >= 1) return (y * 12).toFixed(1) + " mo";
  return (y * 365).toFixed(0) + " days";
}
function multiplierDisplay(base, improved) {
  if (!isFinite(improved)) return isFinite(base) ? "∞" : "—";
  if (base <= 0 || !isFinite(base)) return "—";
  const m = improved / base;
  if (m >= 1000) return ">1000×";
  if (m >= 10) return m.toFixed(0) + "×";
  if (m >= 1.05) return m.toFixed(1) + "×";
  return "≈1×";
}

// ── recommender (port of fatigue.recommend) ─────────────────────────────────
// state: { taskId, joint, ampMPa, kt, radiusM, matKey, usage, baseYears }
function yearsUnder(
  state,
  mats,
  { kt, radiusM, matKey, torqueScale = 1 } = {},
) {
  kt = kt == null ? state.kt : kt;
  const r = radiusM == null ? state.radiusM : radiusM;
  const mat = mats[matKey || state.matKey];
  const amp =
    state.ampMPa *
    (kt / state.kt) *
    Math.pow(state.radiusM / r, 3) *
    torqueScale;
  return yearsFromAmplitude(amp, mat, state.usage);
}
function rec(state, o) {
  return {
    ...o,
    baseYears: state.baseYears,
    multiplier_display: multiplierDisplay(state.baseYears, o.improvedYears),
    inf: !isFinite(o.improvedYears),
  };
}
function ratioOf(r) {
  return isFinite(r.improvedYears)
    ? r.improvedYears / (r.baseYears || 1e-9)
    : 1e30;
}
function bestAltMaterial(state, mats, processes) {
  const base = state.baseYears;
  let best = null;
  for (const m of Object.values(mats)) {
    if (m.key === state.matKey || !processes.includes(m.process)) continue;
    const y = yearsUnder(state, mats, { matKey: m.key });
    const better =
      !isFinite(y) ||
      (isFinite(base) && y > base * 1.05) ||
      (base <= 0 && y > 0);
    if (!better) continue;
    const cmp = isFinite(y) ? y : Infinity;
    if (best === null || cmp > best.cmp) best = { cmp, m, y };
  }
  return best;
}
function cheapestSafeMaterial(state, mats, mat) {
  let best = null;
  for (const m of Object.values(mats)) {
    if (m.key === state.matKey || m.rel_cost >= mat.rel_cost) continue;
    const y = yearsUnder(state, mats, { matKey: m.key });
    if (isFinite(y)) continue; // must stay infinite-life
    if (best === null || m.rel_cost < best.m.rel_cost) best = { m, y };
  }
  return best;
}

function recommend(state, mats, targetYears = 10) {
  const mat = mats[state.matKey];
  const amp = state.ampMPa;
  const overBuilt = amp <= mat.endurance_limit_mpa;
  const margin = amp <= 0 ? Infinity : mat.endurance_limit_mpa / amp;
  const recs = overBuilt
    ? lighten(state, mats, mat)
    : improve(state, mats, mat, targetYears);
  const path = overBuilt
    ? recs[0] || null
    : cheapestFix(state, mats, targetYears);
  const headline = overBuilt
    ? `Over-built: the ${state.joint} runs at ${((1 / margin) * 100).toFixed(0)}% of its fatigue limit — ${margin.toFixed(1)}× spare load, life is effectively unlimited.`
    : `Fatigue-limited: the ${state.joint} sets a ${displayYears(state.baseYears)} life on ${state.taskId}.`;
  return {
    recommendations: recs,
    recommended_path: path,
    overBuilt,
    margin,
    headline,
  };
}

function improve(state, mats, mat, targetYears) {
  const base = state.baseYears;
  const recs = [];
  const r0 = state.radiusM;
  let rInf =
    state.ampMPa > mat.endurance_limit_mpa
      ? r0 * Math.cbrt(state.ampMPa / mat.endurance_limit_mpa)
      : r0;
  let r1;
  if (rInf <= 1.6 * r0) r1 = rInf;
  else {
    const ratio = base > 0 ? Math.max(targetYears / base, 1) : 100;
    r1 = r0 * Math.pow(ratio, -mat.basquin_b / 3);
  }
  recs.push(
    rec(state, {
      id: "thicken_joint",
      category: "geometry",
      title: `Thicken the ${state.joint} section`,
      action: `Increase joint section radius ${(r0 * 1000).toFixed(1)} → ${(r1 * 1000).toFixed(1)} mm (+${((r1 / r0 - 1) * 100).toFixed(0)}%)`,
      rationale:
        "N ∝ r³⁰ — a few mm on the loaded section is the single biggest lever.",
      improvedYears: yearsUnder(state, mats, { radiusM: r1 }),
      tradeoff: `+${(((r1 / r0) ** 2 - 1) * 100).toFixed(0)}% mass at that joint; trivial to print.`,
      effort: "low",
    }),
  );
  recs.push(
    rec(state, {
      id: "fillet",
      category: "manufacturing",
      title: "Add a generous fillet at the joint root",
      action: `Round the fillet to drop the stress-concentration Kt ${state.kt.toFixed(1)} → 1.4`,
      rationale:
        "N ∝ Kt⁻¹⁰ — a sharp internal corner throws away most of the life.",
      improvedYears: yearsUnder(state, mats, { kt: 1.4 }),
      tradeoff: "Essentially free — a CAD change, no mass/cost penalty.",
      effort: "low",
    }),
  );
  const plastic = bestAltMaterial(state, mats, ["FDM", "SLS/MJF"]);
  if (plastic) {
    recs.push(
      rec(state, {
        id: "material_plastic",
        category: "manufacturing",
        title: `Print the joint in ${plastic.m.name}`,
        action: `Switch the loaded part to ${plastic.m.name} (effective σ_f' ${effectiveSigmaF(mat).toFixed(0)} → ${effectiveSigmaF(plastic.m).toFixed(0)} MPa)`,
        rationale:
          "A higher effective fatigue strength lifts the whole S-N curve (N ∝ σ_f'¹⁰).",
        improvedYears: plastic.y,
        tradeoff: `~${(plastic.m.rel_cost / mat.rel_cost).toFixed(1)}× part cost.`,
        effort: "medium",
      }),
    );
  }
  const metal = bestAltMaterial(state, mats, ["CNC", "wrought"]);
  if (metal) {
    recs.push(
      rec(state, {
        id: "material_metal",
        category: "material",
        title: `Machine the limiting joint from ${metal.m.name}`,
        action: `Swap the ${state.joint} part to ${metal.m.name} (effective σ_f' ${effectiveSigmaF(mat).toFixed(0)} → ${effectiveSigmaF(metal.m).toFixed(0)} MPa)`,
        rationale:
          "N ∝ σ_f'¹⁰; the metal's endurance limit sits above this load, so it never fatigues.",
        improvedYears: metal.y,
        tradeoff: `${(metal.m.density_kg_m3 / mat.density_kg_m3).toFixed(1)}× denser part + CNC cost (Ti is the lighter, pricier alt).`,
        effort: "high",
      }),
    );
  }
  recs.push(
    rec(state, {
      id: "control_torque",
      category: "control",
      title: "Retrain for a gentler trajectory",
      action: "Raise the energy/torque penalty to cut peak joint torque ~15%",
      rationale:
        "N ∝ τ⁻¹⁰ — even a small reduction in peak load multiplies life, no hardware change.",
      improvedYears: yearsUnder(state, mats, { torqueScale: 0.85 }),
      tradeoff:
        "May slightly slow the motion or lower success; needs a retrain.",
      effort: "medium",
    }),
  );
  recs.sort((a, b) => b.inf - a.inf || ratioOf(b) - ratioOf(a));
  return recs;
}

function lighten(state, mats, mat) {
  const recs = [];
  const end = mat.endurance_limit_mpa;
  const a = state.ampMPa;
  const r0 = state.radiusM;
  if (a > 0) {
    let rMin = Math.max(r0 * Math.cbrt(a / (0.8 * end)), 0.5 * r0);
    if (rMin < 0.97 * r0) {
      recs.push(
        rec(state, {
          id: "slim_joint",
          category: "geometry",
          title: `Slim down the over-built ${state.joint} section`,
          action: `Reduce joint section radius ${(r0 * 1000).toFixed(1)} → ${(rMin * 1000).toFixed(1)} mm (${((rMin / r0 - 1) * 100).toFixed(0)}%)`,
          rationale:
            "The load is well under the endurance limit, so the section carries more material than fatigue needs.",
          improvedYears: yearsUnder(state, mats, { radiusM: rMin }),
          tradeoff: `−${((1 - (rMin / r0) ** 2) * 100).toFixed(0)}% mass; keep a margin for impact/static loads not in this model.`,
          effort: "low",
        }),
      );
    }
  }
  const cheaper = cheapestSafeMaterial(state, mats, mat);
  if (cheaper) {
    recs.push(
      rec(state, {
        id: "cheaper_material",
        category: "material",
        title: `Switch to cheaper ${cheaper.m.name}`,
        action: `Print the part in ${cheaper.m.name} (~${(cheaper.m.rel_cost / mat.rel_cost).toFixed(2)}× the cost) — still below its fatigue limit`,
        rationale:
          "With this much margin a lower-grade material stays in infinite-life territory.",
        improvedYears: cheaper.y,
        tradeoff: "Lower toughness/stiffness; re-check non-fatigue loads.",
        effort: "low",
      }),
    );
  }
  recs.push(
    rec(state, {
      id: "keep_design",
      category: "plan",
      title: "Or leave it — it already lasts",
      action:
        "No change needed for fatigue; spend the budget elsewhere (reach, grip, mass).",
      rationale:
        "Fatigue is not the limiting factor for this joint at this load.",
      improvedYears: state.baseYears,
      tradeoff: "—",
      effort: "low",
    }),
  );
  return recs;
}

const COMBOS = [
  ["round the fillet (Kt→1.4)", { kt: 1.4 }],
  [
    "round the fillet + retrain for ~15% lower torque",
    { kt: 1.4, torqueScale: 0.85 },
  ],
  ["round the fillet + thicken the joint ~20%", { kt: 1.4, radiusMult: 1.2 }],
  [
    "fillet + 20% thicker joint + ~15% lower torque",
    { kt: 1.4, radiusMult: 1.2, torqueScale: 0.85 },
  ],
  ["machine the joint from metal", { material: "__metal__" }],
];
function cheapestFix(state, mats, targetYears) {
  const metal = bestAltMaterial(state, mats, ["CNC", "wrought"]);
  const metalKey = metal ? metal.m.key : null;
  let best = null;
  for (const [desc, mods] of COMBOS) {
    let matKey = mods.material;
    if (matKey === "__metal__") {
      if (!metalKey) continue;
      matKey = metalKey;
    }
    const y = yearsUnder(state, mats, {
      kt: mods.kt,
      matKey,
      radiusM: state.radiusM * (mods.radiusMult || 1),
      torqueScale: mods.torqueScale || 1,
    });
    const cmp = isFinite(y) ? y : Infinity;
    if (best === null || cmp > best.cmp) best = { cmp, desc, y };
    if (!isFinite(y) || y >= targetYears) {
      best = { cmp, desc, y };
      break;
    }
  }
  if (!best) return null;
  return rec(state, {
    id: "path",
    category: "plan",
    title: "Recommended path",
    action: `Cheapest change that gets ${state.taskId} to a long service life: ${best.desc}.`,
    rationale:
      "Stacks the lowest-cost, lowest-risk levers until the load drops below the endurance limit (or clears the target).",
    improvedYears: best.y,
    tradeoff: "Balanced for cost/effort over a single heavy change.",
    effort: "low",
  });
}

// ── view ────────────────────────────────────────────────────────────────────
const esc = (s) =>
  String(s).replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const lifeColor = (y) =>
  !isFinite(y) || y >= 25
    ? "var(--accent)"
    : y >= 3
      ? "var(--warn)"
      : "var(--bad)";

let REPORT = null;
let MATS = {};
let els = {};

function injectStyle() {
  if (document.getElementById("dur-style")) return;
  const s = document.createElement("style");
  s.id = "dur-style";
  s.textContent = `
    #tab-durability .dur-controls { display:flex; flex-direction:column; gap:10px; margin-bottom:14px; }
    #tab-durability label { font-size:10.5px; color:var(--dim); text-transform:uppercase; letter-spacing:0.4px; display:block; margin-bottom:3px; }
    #tab-durability select, #tab-durability input[type=number] { width:100%; background:#1b2630; color:var(--fg); border:1px solid var(--line); border-radius:7px; padding:6px 8px; font-size:12px; }
    #tab-durability input[type=range] { width:100%; accent-color:var(--accent); }
    .dur-grid { display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-bottom:12px; }
    .dur-stat { background:#161f27; border:1px solid var(--line); border-radius:9px; padding:9px 11px; }
    .dur-stat .k { font-size:10px; color:var(--dim); text-transform:uppercase; letter-spacing:0.3px; }
    .dur-stat .v { font-size:20px; font-variant-numeric:tabular-nums; margin-top:2px; }
    .dur-stat .s { font-size:10.5px; color:var(--dim); margin-top:3px; }
    .dur-headline { background:#1b2630; border:1px solid var(--line); border-left:3px solid var(--accent); border-radius:8px; padding:9px 11px; margin-bottom:14px; font-size:12px; }
    #tab-durability h3 { font-size:11px; text-transform:uppercase; letter-spacing:0.5px; color:var(--dim); margin:16px 0 8px; }
    .dur-row { padding:7px 0; border-bottom:1px solid #1d2a33; }
    .dur-row .l1 { display:flex; justify-content:space-between; align-items:baseline; }
    .dur-row .t { font-weight:600; font-size:12.5px; }
    .dur-row .life { font-variant-numeric:tabular-nums; font-size:12.5px; }
    .dur-row .l2 { display:flex; justify-content:space-between; align-items:center; color:var(--dim); font-size:10.5px; margin-top:2px; }
    .dur-row.worst { background:rgba(239,111,111,0.08); border-radius:6px; padding-left:6px; padding-right:6px; }
    .dur-dot { display:inline-block; width:7px; height:7px; border-radius:50%; margin-left:5px; vertical-align:middle; }
    .mbar { display:inline-block; width:56px; height:6px; border-radius:4px; background:#20303b; overflow:hidden; vertical-align:middle; margin-left:5px; }
    .mbar > i { display:block; height:100%; }
    #tab-durability canvas { width:100%; height:140px; display:block; margin-top:4px; }
    .dur-sub { text-transform:none; letter-spacing:0; color:var(--dim); font-weight:400; font-size:10px; }
    .dur-chart { background:#161f27; border:1px solid var(--line); border-radius:9px; padding:8px 10px; margin-bottom:9px; }
    .dur-chart .ct { font-size:10.5px; color:var(--dim); }
    .dur-rec { background:#161f27; border:1px solid var(--line); border-radius:9px; padding:10px 11px; margin-bottom:8px; }
    .dur-rec.path { border-color:var(--accent); border-left:3px solid var(--accent); background:#1b2630; }
    .dur-rec .top { display:flex; justify-content:space-between; gap:8px; align-items:baseline; }
    .dur-rec .tt { font-size:12.5px; font-weight:600; }
    .dur-rec .mult { font-size:15px; font-variant-numeric:tabular-nums; white-space:nowrap; }
    .dur-rec .mult.inf { color:var(--accent); }
    .dur-rec .ac { font-size:11.5px; margin:4px 0 5px; }
    .dur-rec .ba { font-size:11px; font-variant-numeric:tabular-nums; color:var(--dim); display:flex; align-items:center; gap:6px; flex-wrap:wrap; }
    .dur-rec .ba b { color:var(--fg); }
    .dur-rec .why { font-size:10.5px; color:var(--dim); margin-top:5px; }
    .dur-rec .tags { margin-top:6px; display:flex; gap:5px; flex-wrap:wrap; }
    .dur-tag { font-size:9.5px; text-transform:uppercase; color:var(--dim); border:1px solid var(--line); border-radius:5px; padding:1px 6px; }
    .dur-tag.low { color:var(--accent); border-color:rgba(111,230,160,0.4); }
    .dur-tag.high { color:var(--warn); border-color:rgba(240,183,96,0.4); }
    #tab-durability .dur-err { color:var(--warn); padding:24px 2px; font-size:12.5px; }
    #tab-durability .dur-legend { font-size:10px; color:var(--dim); margin-top:6px; }
  `;
  document.head.appendChild(s);
}

function buildDom(root) {
  root.innerHTML = `
    <div class="dur-err" id="dur-err" style="display:none"></div>
    <div id="dur-body" style="display:none">
      <div class="dur-controls">
        <div><label>Joint material / process</label><select id="dur-material"></select></div>
        <div><label>Uses per day</label><input id="dur-usage" type="number" min="1" max="5000" step="10" value="300" /></div>
      </div>
      <div class="dur-chart"><div class="ct">Stress amplitude vs. endurance limit (MPa)</div><canvas id="dur-chart-stress" height="140"></canvas></div>
      <div class="dur-chart"><div class="ct" id="dur-torque-title">Joint torque over the motion (N·m)</div><canvas id="dur-chart-torque" height="150"></canvas><div class="dur-legend" id="dur-torque-legend"></div></div>
      <h3 id="dur-recs-title">Recommended actions</h3>
      <div id="dur-recs"></div>
    </div>`;
  els = {};
  for (const id of [
    "err",
    "body",
    "material",
    "usage",
    "chart-stress",
    "chart-torque",
    "torque-title",
    "torque-legend",
    "recs",
    "recs-title",
  ]) {
    els[id] = root.querySelector("#dur-" + id);
  }
}

function currentState() {
  const matKey = els.material.value;
  const usage = Math.max(1, +els.usage.value || 300);
  const rows = REPORT.results.map((r) => {
    const ampMPa = r.amplitude_mpa; // captured at load_factor 1.0
    const mat = MATS[matKey];
    return {
      ...r,
      ampMPa,
      years: yearsFromAmplitude(ampMPa, mat, usage),
      margin: ampMPa <= 0 ? Infinity : mat.endurance_limit_mpa / ampMPa,
    };
  });
  return { matKey, usage, rows };
}
function worstRow(rows) {
  return rows.reduce((w, r) =>
    (isFinite(r.years) ? r.years : Infinity) <
    (isFinite(w.years) ? w.years : Infinity)
      ? r
      : w,
  );
}

function render() {
  const { matKey, usage, rows } = currentState();
  const mat = MATS[matKey];
  // Defend against a stale/blank material selection or an empty report.
  if (!mat || !rows.length) {
    els.recs.innerHTML =
      '<div class="dur-rec"><div class="tt">No data</div><div class="ac">The report has no tasks or no material selected.</div></div>';
    return;
  }
  const worst = worstRow(rows);
  const state = {
    taskId: worst.task_id,
    joint: worst.critical_joint,
    ampMPa: worst.ampMPa,
    kt: REPORT.kt,
    radiusM: REPORT.radius_m,
    matKey,
    usage,
    baseYears: worst.years,
  };
  const plan = recommend(state, MATS);
  renderRecs(plan);
  drawStress(rows, mat);
  drawTorque(worst);
}

function renderRecs(plan) {
  const card = (r, isPath) => `
    <div class="dur-rec ${isPath ? "path" : ""}">
      <div class="top">
        <span class="tt">${isPath ? "★ " : ""}${esc(r.title)}</span>
        <span class="mult ${r.inf ? "inf" : ""}">${r.multiplier_display}</span>
      </div>
      <div class="ac">${esc(r.action)}</div>
      <div class="ba"><span><b>${displayYears(r.baseYears)}</b> → <b style="color:${lifeColor(r.improvedYears)}">${displayYears(r.improvedYears)}</b></span>
        <span class="dur-tag ${esc(r.effort)}">${esc(r.effort)}</span><span class="dur-tag">${esc(r.category)}</span></div>
    </div>`;
  const pathId = plan.recommended_path ? plan.recommended_path.id : null;
  let html = plan.recommended_path ? card(plan.recommended_path, true) : "";
  // Skip the list copy of whatever is already shown as the ★ recommended path
  // (in over-built mode the path IS the top lighten rec).
  html += plan.recommendations
    .filter((r) => r.id !== pathId)
    .map((r) => card(r, false))
    .join("");
  els.recs.innerHTML = html;
}

// ── canvas bar charts ───────────────────────────────────────────────────────
function fitCanvas(cv) {
  // Back the canvas at devicePixelRatio so charts are crisp on HiDPI/Retina, but
  // draw in CSS pixels (W,H) so the layout math is resolution-independent.
  const dpr = window.devicePixelRatio || 1;
  const W = cv.clientWidth || 320;
  const H = cv.clientHeight || 150;
  const bw = Math.round(W * dpr),
    bh = Math.round(H * dpr);
  if (cv.width !== bw) cv.width = bw;
  if (cv.height !== bh) cv.height = bh;
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, W, H };
}
function barChart(
  cv,
  items,
  { valueOf, label, color, refLine, refLabel, logCap },
) {
  const { ctx, W, H } = fitCanvas(cv);
  ctx.clearRect(0, 0, W, H);
  const padL = 32,
    padR = 8,
    padT = 8,
    padB = 40;
  const n = Math.max(1, items.length);
  const bw = (W - padL - padR) / n;
  const vals = items.map(valueOf);
  let vmax = 0;
  for (const v of vals) vmax = Math.max(vmax, logCap ? Math.min(v, logCap) : v);
  if (refLine != null) vmax = Math.max(vmax, refLine);
  vmax = vmax || 1;
  const sy = (v) =>
    H - padB - (Math.min(v, logCap || Infinity) / vmax) * (H - padT - padB);
  ctx.font = "8px -apple-system, Arial";
  ctx.strokeStyle = "#2a3742";
  ctx.beginPath();
  ctx.moveTo(padL, padT);
  ctx.lineTo(padL, H - padB);
  ctx.lineTo(W - padR, H - padB);
  ctx.stroke();
  ctx.fillStyle = "#7d93a3";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  ctx.fillText(vmax.toFixed(vmax >= 10 ? 0 : 1), padL - 3, sy(vmax));
  ctx.fillText("0", padL - 3, sy(0));
  if (refLine != null) {
    ctx.strokeStyle = "#ef6f6f";
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(padL, sy(refLine));
    ctx.lineTo(W - padR, sy(refLine));
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#ef6f6f";
    ctx.textAlign = "left";
    ctx.fillText(refLabel || "", padL + 2, sy(refLine) - 5);
  }
  items.forEach((it, i) => {
    const v = vals[i];
    const x = padL + i * bw + bw * 0.15,
      w = bw * 0.7,
      y = sy(v);
    ctx.fillStyle = typeof color === "function" ? color(it, v) : color;
    ctx.fillRect(x, y, w, H - padB - y);
    ctx.save();
    ctx.translate(x + w / 2, H - padB + 3);
    ctx.rotate(-Math.PI / 4);
    ctx.fillStyle = "#8aa0b0";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.fillText(String(label(it)).slice(0, 12), 0, 0);
    ctx.restore();
  });
}
function drawStress(rows, mat) {
  barChart(els["chart-stress"], rows, {
    valueOf: (r) => r.ampMPa,
    label: (r) => r.task_id,
    color: (r) => (r.ampMPa <= mat.endurance_limit_mpa ? "#6fe6a0" : "#ef6f6f"),
    refLine: mat.endurance_limit_mpa,
    refLabel: `endurance ${mat.endurance_limit_mpa} MPa`,
  });
}

// Torque-over-time line chart for the limiting task's most-loaded joint.
function drawTorque(row) {
  const cv = els["chart-torque"];
  const trace = (row && row.torque_trace) || [];
  els["torque-title"].textContent = row
    ? `Joint torque over the motion — ${row.task_id} · ${row.critical_joint} (N·m)`
    : "Joint torque over the motion (N·m)";
  const { ctx, W, H } = fitCanvas(cv);
  ctx.clearRect(0, 0, W, H);
  if (trace.length < 2) {
    ctx.fillStyle = "#7d93a3";
    ctx.font = "10px -apple-system, Arial";
    ctx.fillText("No torque trace in report.", 10, H / 2);
    els["torque-legend"].textContent = "";
    return;
  }
  const padL = 36,
    padR = 8,
    padT = 10,
    padB = 18;
  let vmin = Infinity,
    vmax = -Infinity;
  for (const v of trace) {
    vmin = Math.min(vmin, v);
    vmax = Math.max(vmax, v);
  }
  // Pad the range a touch and always include zero so sign is readable.
  vmin = Math.min(vmin, 0);
  vmax = Math.max(vmax, 0);
  const span = vmax - vmin || 1;
  vmax += span * 0.08;
  vmin -= span * 0.08;
  const n = trace.length;
  const sx = (i) => padL + (i / (n - 1)) * (W - padL - padR);
  const sy = (v) => padT + (1 - (v - vmin) / (vmax - vmin)) * (H - padT - padB);
  // axes
  ctx.strokeStyle = "#2a3742";
  ctx.beginPath();
  ctx.moveTo(padL, padT);
  ctx.lineTo(padL, H - padB);
  ctx.lineTo(W - padR, H - padB);
  ctx.stroke();
  // zero line
  ctx.strokeStyle = "#3a4854";
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.moveTo(padL, sy(0));
  ctx.lineTo(W - padR, sy(0));
  ctx.stroke();
  ctx.setLineDash([]);
  // y labels
  ctx.fillStyle = "#7d93a3";
  ctx.font = "8px -apple-system, Arial";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  ctx.fillText(vmax.toFixed(0), padL - 3, sy(vmax));
  ctx.fillText("0", padL - 3, sy(0));
  ctx.fillText(vmin.toFixed(0), padL - 3, sy(vmin));
  // filled area under the curve
  ctx.beginPath();
  ctx.moveTo(sx(0), sy(0));
  for (let i = 0; i < n; i++) ctx.lineTo(sx(i), sy(trace[i]));
  ctx.lineTo(sx(n - 1), sy(0));
  ctx.closePath();
  ctx.fillStyle = "rgba(111,230,160,0.12)";
  ctx.fill();
  // the torque line
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const x = sx(i),
      y = sy(trace[i]);
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  }
  ctx.strokeStyle = "#6fe6a0";
  ctx.lineWidth = 1.5;
  ctx.stroke();
  ctx.lineWidth = 1;
  // mark the peak
  let pi = 0;
  for (let i = 1; i < n; i++)
    if (Math.abs(trace[i]) > Math.abs(trace[pi])) pi = i;
  ctx.fillStyle = "#ef6f6f";
  ctx.beginPath();
  ctx.arc(sx(pi), sy(trace[pi]), 2.5, 0, Math.PI * 2);
  ctx.fill();
  els["torque-legend"].textContent = `peak |τ| ${Math.abs(trace[pi]).toFixed(
    1,
  )} N·m · span ${(vmax - vmin).toFixed(0)} N·m · ${
    row.steps || n
  } steps (${n} sampled) — x-axis = time through the motion`;
}

// ── public entry ─────────────────────────────────────────────────────────────
let inited = false;
let loaded = false;

export async function initDurabilityPanel(root) {
  if (inited) {
    if (loaded) redraw(); // re-fit canvases when the tab becomes visible again
    return;
  }
  inited = true;
  injectStyle();
  buildDom(root);
  let data;
  try {
    data = await (await fetch(REPORT_URL + "?_=" + Date.now())).json();
  } catch (e) {
    els.err.style.display = "block";
    els.err.innerHTML =
      "No durability report yet. Run<br><code>python3 scripts/demo/durability_dashboard.py --policy scenario_ppo</code><br>(add <code>--train</code> first) to generate it.";
    return;
  }
  REPORT = data;
  MATS = {};
  for (const m of data.materials || []) MATS[m.key] = m;
  if (!Object.keys(MATS).length || !(data.results && data.results.length)) {
    els.err.style.display = "block";
    els.err.innerHTML =
      "The durability report is missing its material database or task results — re-run <code>durability_dashboard.py</code> to regenerate it.";
    return;
  }
  els.material.innerHTML = (data.materials || [])
    .map(
      (m) =>
        `<option value="${esc(m.key)}" ${m.key === data.material_key ? "selected" : ""}>${esc(m.name)} — σ_f' ${effectiveSigmaF(m).toFixed(0)} MPa</option>`,
    )
    .join("");
  els.usage.value = data.usage_cycles_per_day || 300;
  els.body.style.display = "block";
  loaded = true;
  for (const id of ["material", "usage"])
    els[id].addEventListener("input", render);
  render();
}

function redraw() {
  if (!loaded) return;
  const { rows } = currentState();
  drawStress(rows, MATS[els.material.value]);
}
