/**
 * Super Human Pipeline Dashboard — main module
 *
 * Connects to pipeline_server.py via SSE, drives the 6-stage timeline,
 * runs a MuJoCo WASM viewer for live sim replay and final manual control,
 * and populates the stats bar.
 */

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import load_mujoco from "../node_modules/mujoco-js/dist/mujoco_wasm.js";

// ── Constants ──────────────────────────────────────────────────────────────────
const STAGES = [
  { id: "perception", label: "① Perception", desc: "Analyse video clip" },
  {
    id: "requirements",
    label: "② Requirements",
    desc: "Derive engineering spec",
  },
  { id: "design", label: "③ CAD Design", desc: "Generate arm candidates" },
  { id: "cad", label: "④ CAD Generation", desc: "Build STL + MJCF" },
  {
    id: "sim_eval",
    label: "⑤ Physics Simulation",
    desc: "MuJoCo stress testing",
  },
  { id: "rl_loop", label: "⑥ RL Optimization", desc: "PPO policy training" },
  { id: "final", label: "⑦ Final Product", desc: "Stats + arm presentation" },
];

// ── State ──────────────────────────────────────────────────────────────────────
let selectedClip = null;
let eventSource = null;
let startTime = null;
let elapsedTimer = null;

// MuJoCo WASM state
let mujoco = null;
let mjModel = null;
let mjData = null;
let bodies = {};
let currentQpos = null;
let trajectory = [];
let trajIdx = 0;
let playTraj = false;
let interacting = false;

// Three.js
let renderer, scene, camera, controls, animId;

// ── Init ───────────────────────────────────────────────────────────────────────
buildTimeline();
loadClips();
initViewer();
wireButtons();

// ── Timeline ───────────────────────────────────────────────────────────────────
function buildTimeline() {
  const tl = document.getElementById("timeline");
  tl.innerHTML = "";
  for (const s of STAGES) {
    const item = document.createElement("div");
    item.className = "stage-item";
    item.id = `stage-${s.id}`;
    item.innerHTML = `
      <div class="stage-header" onclick="toggleStage('${s.id}')">
        <div class="stage-dot"></div>
        <div class="stage-label">${s.label}</div>
        <div class="stage-elapsed" id="sel-${s.id}"></div>
        <div class="stage-chevron" id="chv-${s.id}">▸</div>
      </div>
      <div class="stage-body" id="sbody-${s.id}">
        <div style="color:var(--dim);font-size:12px;padding:8px 0">${s.desc}</div>
      </div>`;
    tl.appendChild(item);
  }
}

window.toggleStage = (id) => {
  const body = document.getElementById(`sbody-${id}`);
  const chv = document.getElementById(`chv-${id}`);
  if (body.classList.toggle("open")) chv.textContent = "▾";
  else chv.textContent = "▸";
};

function markStageActive(id) {
  const el = document.getElementById(`stage-${id}`);
  if (el) {
    el.className = "stage-item active";
    // Auto-open active stage
    const body = document.getElementById(`sbody-${id}`);
    if (body && !body.classList.contains("open")) {
      body.classList.add("open");
      document.getElementById(`chv-${id}`).textContent = "▾";
    }
  }
  setStatus(`Running: ${STAGES.find((s) => s.id === id)?.label || id}`);
  setProgress(STAGES.findIndex((s) => s.id === id) / STAGES.length);
}

function markStageDone(id, elapsed) {
  const el = document.getElementById(`stage-${id}`);
  if (el) el.className = "stage-item done";
  const sel = document.getElementById(`sel-${id}`);
  if (sel && elapsed)
    sel.textContent =
      elapsed < 1
        ? `${(elapsed * 1000).toFixed(0)}ms`
        : `${elapsed.toFixed(1)}s`;
  setProgress((STAGES.findIndex((s) => s.id === id) + 1) / STAGES.length);
}

function renderStageBody(id, html) {
  const body = document.getElementById(`sbody-${id}`);
  if (body) body.innerHTML = html;
}

// ── Event dispatch ─────────────────────────────────────────────────────────────
function dispatch(ev) {
  const { type, stage, data } = ev;

  if (type === "ping") return;

  if (type === "error") {
    markStageError(stage, data.message);
    return;
  }

  if (type === "stage_start") {
    markStageActive(stage);
    return;
  }

  if (type === "stage_done") {
    markStageDone(stage, data.elapsed_s);
    renderStageContent(stage, data);
    return;
  }

  if (type === "sim_frame") {
    if (data.qpos) updateViewerQpos(data.qpos);
    return;
  }

  if (type === "rl_step") {
    updateRLChart(data);
    return;
  }

  if (type === "done") {
    onPipelineDone(data);
    return;
  }
}

function renderStageContent(id, data) {
  switch (id) {
    case "perception":
      return renderPerception(data);
    case "requirements":
      return renderRequirements(data);
    case "design":
      return renderDesign(data);
    case "cad":
      return renderCad(data);
    case "sim_eval":
      return renderSimEval(data);
    case "rl_loop":
      return renderRLLoop(data);
    case "final":
      return renderFinal(data);
  }
}

// ── Stage renderers ────────────────────────────────────────────────────────────

function renderPerception(d) {
  const spec = d.spec || {};
  const rom = spec.rom_deg || {};
  let romRows = Object.entries(rom)
    .map(
      ([j, v]) =>
        `<div class="info-row"><span class="key">${j}</span><span class="val">${v}°</span></div>`,
    )
    .join("");
  renderStageBody(
    "perception",
    `
    <div class="info-row"><span class="key">Action</span><span class="val">${d.action || "—"}</span></div>
    <div class="info-row"><span class="key">Side</span>
      <span class="val"><span class="badge">${d.side || "—"}</span></span></div>
    ${romRows}
    <div class="info-row"><span class="key">Grip capacity</span>
      <span class="val">${spec.grip_capacity !== undefined ? (spec.grip_capacity * 100).toFixed(0) + "%" : "—"}</span></div>
  `,
  );
}

function renderRequirements(d) {
  const brief = d.brief || {};
  const rom = brief.rom_targets_deg || {};
  let romRows = Object.entries(rom)
    .map(
      ([j, [lo, hi]]) =>
        `<div class="bar-row">
      <div class="bar-label">${j}: ${lo.toFixed(0)}° – ${hi.toFixed(0)}°</div>
      <div class="bar-track"><div class="bar-fill" style="width:${Math.min(100, (hi - lo) / 1.8).toFixed(0)}%"></div></div>
    </div>`,
    )
    .join("");
  const dp = brief.design_params || {};
  renderStageBody(
    "requirements",
    `
    ${romRows}
    <div class="info-row" style="margin-top:6px"><span class="key">Upper arm</span>
      <span class="val">${dp.upper_arm_len !== undefined ? (dp.upper_arm_len * 1000).toFixed(0) + "mm" : "—"}</span></div>
    <div class="info-row"><span class="key">Forearm</span>
      <span class="val">${dp.forearm_len !== undefined ? (dp.forearm_len * 1000).toFixed(0) + "mm" : "—"}</span></div>
    <div class="info-row"><span class="key">Grip width</span>
      <span class="val">${dp.grip_width !== undefined ? (dp.grip_width * 1000).toFixed(0) + "mm" : "—"}</span></div>
    <div class="info-row"><span class="key">Grip force</span>
      <span class="val">${dp.grip_force_target_n !== undefined ? dp.grip_force_target_n.toFixed(0) + "N" : "—"}</span></div>
    <div class="info-row"><span class="key">Source</span>
      <span class="val"><span class="badge">${brief.source || "fallback"}</span></span></div>
  `,
  );
}

function renderDesign(d) {
  const cands = d.candidates || [];
  let rows = cands
    .map(
      (c, i) => `
    <tr>
      <td>#${i + 1}</td>
      <td>${(c.upper_m * 1000).toFixed(0)}mm</td>
      <td>${(c.fore_m * 1000).toFixed(0)}mm</td>
      <td>${c.dof}</td>
    </tr>`,
    )
    .join("");
  renderStageBody(
    "design",
    `
    <div style="color:var(--dim);font-size:11px;margin-bottom:6px">${d.n_candidates} candidates generated</div>
    <table class="candidate-table">
      <thead><tr><th>#</th><th>Upper arm</th><th>Forearm</th><th>DoF</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
  `,
  );
}

function renderCad(d) {
  const bs = d.build_sheet || {};
  // Update viewer with new MJCF when available
  if (d.name) loadViewerScene(d.name).catch(() => {});

  const links = bs.links || [];
  let linkRows = links
    .map(
      (lk) => `
    <tr>
      <td>${lk.name || "—"}</td>
      <td>${lk.length_mm !== undefined ? lk.length_mm.toFixed(0) + "mm" : "—"}</td>
      <td>${lk.wall_thickness_mm !== undefined ? lk.wall_thickness_mm.toFixed(1) + "mm" : "—"}</td>
      <td>${lk.mass_g !== undefined ? lk.mass_g.toFixed(0) + "g" : "—"}</td>
    </tr>`,
    )
    .join("");

  renderStageBody(
    "cad",
    `
    <div class="info-row"><span class="key">Material</span>
      <span class="val"><span class="badge">${d.material || bs.material || "—"}</span></span></div>
    <div class="info-row"><span class="key">Total mass</span>
      <span class="val">${bs.total_mass_g !== undefined ? bs.total_mass_g.toFixed(0) + "g" : "—"}</span></div>
    <div class="info-row"><span class="key">Reach envelope</span>
      <span class="val">${bs.reach_envelope_mm !== undefined ? bs.reach_envelope_mm.toFixed(0) + "mm" : "—"}</span></div>
    <div class="info-row"><span class="key">Printability</span>
      <span class="val">${bs.printability_ok ? "✓ OK" : "⚠ Issues"}</span></div>
    <table class="candidate-table" style="margin-top:6px">
      <thead><tr><th>Link</th><th>Length</th><th>Wall</th><th>Mass</th></tr></thead>
      <tbody>${linkRows}</tbody>
    </table>
  `,
  );
}

const _simBest = {};
function renderSimEval(d) {
  const results = d.eval_results || [];
  const bestIdx = d.best_index ?? -1;
  let rows = results
    .map(
      (r, i) => `
    <tr class="${i === bestIdx ? "best" : ""}">
      <td>#${r.candidate + 1}</td>
      <td>${(r.success_rate * 100).toFixed(0)}%</td>
      <td>${r.mean_reward !== undefined ? r.mean_reward.toFixed(3) : "—"}</td>
      <td>${r.predicted_life_years < 900 ? r.predicted_life_years.toFixed(1) + "yr" : "≥100yr"}</td>
    </tr>`,
    )
    .join("");

  _simBest.success = d.best_success;

  renderStageBody(
    "sim_eval",
    `
    <div class="info-row"><span class="key">Iteration</span>
      <span class="val">${d.iteration !== undefined ? d.iteration + 1 : "—"}</span></div>
    <div class="info-row"><span class="key">Best success rate</span>
      <span class="val ${d.best_success >= 0.4 ? "good" : "warn"}">${d.best_success !== undefined ? (d.best_success * 100).toFixed(0) + "%" : "—"}</span></div>
    <div class="info-row"><span class="key">Best reward</span>
      <span class="val">${d.best_reward !== undefined ? d.best_reward.toFixed(3) : "—"}</span></div>
    <table class="candidate-table" style="margin-top:6px">
      <thead><tr><th>#</th><th>Success</th><th>Reward</th><th>Lifespan</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
    <div style="color:var(--dim);font-size:11px;margin-top:6px">${d.rationale || ""}</div>
  `,
  );
}

// RL chart state
const _rlSteps = [],
  _rlRewards = [];
let _rlCanvas = null;

function renderRLLoop(d) {
  renderStageBody(
    "rl_loop",
    `
    <div class="info-row"><span class="key">IK baseline</span>
      <span class="val">${d.ik_success_rate !== undefined ? (d.ik_success_rate * 100).toFixed(0) + "%" : "—"}</span></div>
    <canvas id="rl-chart-canvas" width="600" height="120"></canvas>
    <div class="info-row" style="margin-top:4px"><span class="key">Timestep</span>
      <span class="val" id="rl-ts">0</span></div>
    <div class="info-row"><span class="key">Mean reward</span>
      <span class="val" id="rl-rew">—</span></div>
  `,
  );
  _rlCanvas = document.getElementById("rl-chart-canvas");
  _rlSteps.length = 0;
  _rlRewards.length = 0;
}

function updateRLChart(d) {
  _rlSteps.push(d.timestep || 0);
  _rlRewards.push(d.mean_reward || 0);
  document.getElementById("rl-ts") &&
    (document.getElementById("rl-ts").textContent =
      d.timestep >= 1000
        ? Math.round(d.timestep / 1000) + "k"
        : String(d.timestep));
  document.getElementById("rl-rew") &&
    (document.getElementById("rl-rew").textContent = (
      d.mean_reward || 0
    ).toFixed(3));
  if (_rlCanvas) drawRLChart(_rlCanvas);
}

function drawRLChart(canvas) {
  if (!_rlSteps.length) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.clientWidth,
    H = canvas.clientHeight;
  canvas.width = W;
  canvas.height = H;
  ctx.clearRect(0, 0, W, H);
  const pad = [8, 30, 20, 36];
  const xmin = _rlSteps[0],
    xmax = _rlSteps[_rlSteps.length - 1] || 1;
  let ymin = Math.min(..._rlRewards),
    ymax = Math.max(..._rlRewards);
  if (ymin === ymax) {
    ymin -= 0.1;
    ymax += 0.1;
  }
  const xs = (s) =>
    pad[3] + ((s - xmin) / (xmax - xmin)) * (W - pad[3] - pad[1]);
  const ys = (v) =>
    pad[0] + (1 - (v - ymin) / (ymax - ymin)) * (H - pad[0] - pad[2]);
  ctx.strokeStyle = "#2a3742";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad[0] + (i / 4) * (H - pad[0] - pad[2]);
    ctx.beginPath();
    ctx.moveTo(pad[3], y);
    ctx.lineTo(W - pad[1], y);
    ctx.stroke();
    const v = ymax - (i / 4) * (ymax - ymin);
    ctx.fillStyle = "#8aa0b0";
    ctx.font = "9px Arial";
    ctx.fillText(v.toFixed(2), 2, y + 3);
  }
  ctx.strokeStyle = "#6fe6a0";
  ctx.lineWidth = 2;
  ctx.beginPath();
  _rlSteps.forEach((s, i) => {
    if (i === 0) ctx.moveTo(xs(s), ys(_rlRewards[i]));
    else ctx.lineTo(xs(s), ys(_rlRewards[i]));
  });
  ctx.stroke();
  // axis x labels
  ctx.fillStyle = "#8aa0b0";
  ctx.font = "9px Arial";
  const xstep = Math.ceil(_rlSteps.length / 4);
  for (let i = 0; i < _rlSteps.length; i += xstep) {
    const ts = _rlSteps[i];
    ctx.fillText(
      ts >= 1000 ? Math.round(ts / 1000) + "k" : String(ts),
      xs(ts) - 8,
      H - 4,
    );
  }
}

function renderFinal(d) {
  const stats = d.stats || {};
  const joints = stats.joint_names || [];
  const jpos = stats.joint_positions_mm || {};
  let jointRows = joints
    .map((j) => {
      const p = jpos[j] || [0, 0, 0];
      return `<div class="info-row"><span class="key">${j}</span>
      <span class="val">[${p.map((v) => Math.round(v)).join(", ")}]mm</span></div>`;
    })
    .join("");

  const rlSuccess = stats.rl_success_rate || 0;
  const lifeOK = (stats.predicted_life_years || 0) >= 1;
  const passGate = rlSuccess >= 0.65 && lifeOK;

  renderStageBody(
    "final",
    `
    <div style="margin-bottom:8px">
      <span class="badge ${passGate ? "" : "warn"}" style="font-size:13px;padding:3px 10px">
        ${passGate ? "✓ PASS" : "⚠ PARTIAL"}
      </span>
    </div>
    <div class="info-row"><span class="key">Activity</span>
      <span class="val">${stats.primary_action || "—"}</span></div>
    <div class="info-row"><span class="key">Affected side</span>
      <span class="val"><span class="badge">${stats.affected_side || "—"}</span></span></div>
    <div class="info-row"><span class="key">IK success</span>
      <span class="val">${stats.ik_success_rate !== undefined ? (stats.ik_success_rate * 100).toFixed(0) + "%" : "—"}</span></div>
    <div class="info-row"><span class="key">RL success</span>
      <span class="val ${rlSuccess >= 0.65 ? "" : "warn"}">${(rlSuccess * 100).toFixed(0)}%</span></div>
    <div class="info-row"><span class="key">Predicted lifespan</span>
      <span class="val ${lifeOK ? "" : "warn"}">${stats.predicted_life_years < 900 ? (stats.predicted_life_years || 0).toFixed(1) + "yr" : "≥100yr"}</span></div>
    <div class="info-row"><span class="key">Peak stress</span>
      <span class="val">${stats.peak_stress_mpa !== undefined ? stats.peak_stress_mpa.toFixed(1) + " MPa" : "—"}</span></div>
    <div class="info-row"><span class="key">Design iterations</span>
      <span class="val">${(stats.iteration_count || 0) + 1}</span></div>
    <div style="margin-top:6px;font-size:11px;color:var(--dim)">Joint locations (mm from shoulder):</div>
    ${jointRows}
    <div style="color:var(--dim);font-size:11px;margin-top:8px">${d.rationale ? d.rationale.slice(0, 300) : ""}</div>
  `,
  );

  // Load trajectory if available
  if (d.trajectory && d.trajectory.length) {
    trajectory = d.trajectory;
    trajIdx = 0;
    playTraj = true;
  }
  updateStatsBar(stats);
  enableJointPanel(d.stats);
}

function updateStatsBar(stats) {
  const set = (id, val) => {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
  };
  set("s-material", stats.material || "PA12-CF");
  set(
    "s-mass",
    stats.total_mass_g !== undefined
      ? stats.total_mass_g.toFixed(0) + "g"
      : "—",
  );
  set(
    "s-reach",
    stats.reach_envelope_mm !== undefined
      ? stats.reach_envelope_mm + "mm"
      : "—",
  );
  set("s-dof", stats.dof !== undefined ? String(stats.dof) : "—");
  set(
    "s-ik",
    stats.ik_success_rate !== undefined
      ? (stats.ik_success_rate * 100).toFixed(0) + "%"
      : "—",
  );
  set(
    "s-rl",
    stats.rl_success_rate !== undefined
      ? (stats.rl_success_rate * 100).toFixed(0) + "%"
      : "—",
  );
  const life = stats.predicted_life_years;
  set(
    "s-life",
    life !== undefined && life < 900
      ? life.toFixed(1) + "yr"
      : life !== undefined
        ? "≥100yr"
        : "—",
  );
  set(
    "s-energy",
    stats.mean_energy_j !== undefined
      ? stats.mean_energy_j.toFixed(0) + "J"
      : "—",
  );
  set("s-joints", (stats.joint_names || []).join(" · ") || "—");
}

// ── Clip picker ────────────────────────────────────────────────────────────────
async function loadClips() {
  try {
    const clips = await fetch("/api/clips").then((r) => r.json());
    const grid = document.getElementById("clip-grid");
    grid.innerHTML = "";
    if (!clips.length) {
      grid.innerHTML =
        '<div style="color:var(--dim);font-size:12px;padding:8px">No clips found in test_vids/</div>';
      return;
    }
    for (const clip of clips) {
      const card = document.createElement("div");
      card.className = "clip-card";
      card.dataset.path = clip.path;
      const thumbUrl = `/api/thumbnail?clip=${encodeURIComponent(clip.path)}`;
      card.innerHTML = `
        <img src="${thumbUrl}" alt="${clip.name}" loading="lazy" />
        <div class="clip-name">${clip.name}</div>`;
      card.addEventListener("click", () => selectClip(clip, card));
      grid.appendChild(card);
    }
    // Auto-select first clip
    const first = grid.querySelector(".clip-card");
    if (first) first.click();
  } catch (err) {
    document.getElementById("clip-grid").innerHTML =
      `<div style="color:var(--danger);font-size:12px;padding:8px">Could not load clips: ${err}</div>`;
  }
}

function selectClip(clip, cardEl) {
  selectedClip = clip;
  document
    .querySelectorAll(".clip-card")
    .forEach((c) => c.classList.remove("selected"));
  cardEl.classList.add("selected");
  document.getElementById("start-btn").disabled = false;
  setStatus(`Selected: ${clip.name}`);
}

// ── Start / Stop ───────────────────────────────────────────────────────────────
function wireButtons() {
  document.getElementById("start-btn").addEventListener("click", startPipeline);
  document.getElementById("stop-btn").addEventListener("click", stopPipeline);
}

function startPipeline() {
  if (!selectedClip) return;
  if (eventSource) {
    eventSource.close();
    eventSource = null;
  }

  // Reset timeline
  buildTimeline();
  _rlSteps.length = 0;
  _rlRewards.length = 0;
  trajectory = [];
  playTraj = false;
  document.getElementById("joint-panel").style.display = "none";

  document.getElementById("start-btn").disabled = true;
  document.getElementById("stop-btn").disabled = false;
  setStatus("Connecting…");
  startTime = Date.now();
  clearInterval(elapsedTimer);
  elapsedTimer = setInterval(() => {
    const s = ((Date.now() - startTime) / 1000).toFixed(0);
    document.getElementById("elapsed-label").textContent = `${s}s`;
  }, 1000);

  const quick = document.getElementById("quick-mode").checked;
  const body = JSON.stringify({ clip: selectedClip.path, quick });

  fetch("/api/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body,
  })
    .then((resp) => {
      if (!resp.ok) throw new Error(`Server returned ${resp.status}`);
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      function pump() {
        return reader.read().then(({ done, value }) => {
          if (done) {
            onStreamDone();
            return;
          }
          buffer += decoder.decode(value, { stream: true });
          const parts = buffer.split("\n\n");
          buffer = parts.pop();
          for (const part of parts) {
            if (part.startsWith("data: ")) {
              try {
                dispatch(JSON.parse(part.slice(6)));
              } catch {}
            }
          }
          return pump();
        });
      }
      pump().catch((err) => setStatus("Stream error: " + err));
    })
    .catch((err) => {
      setStatus("Connection error: " + err);
      document.getElementById("start-btn").disabled = false;
      document.getElementById("stop-btn").disabled = true;
    });
}

function stopPipeline() {
  if (eventSource) {
    eventSource.close();
    eventSource = null;
  }
  clearInterval(elapsedTimer);
  document.getElementById("start-btn").disabled = false;
  document.getElementById("stop-btn").disabled = true;
  setStatus("Stopped");
}

function onStreamDone() {
  clearInterval(elapsedTimer);
  document.getElementById("start-btn").disabled = false;
  document.getElementById("stop-btn").disabled = true;
  setStatus("Pipeline complete");
  setProgress(1.0);
}

function onPipelineDone(data) {
  setStatus("Pipeline complete ✓");
  setProgress(1.0);
}

// ── Helpers ────────────────────────────────────────────────────────────────────
function setStatus(msg) {
  document.getElementById("status-label").textContent = msg;
}
function setProgress(frac) {
  document.getElementById("progress-fill").style.width =
    (frac * 100).toFixed(1) + "%";
}
function markStageError(id, msg) {
  const el = document.getElementById(`stage-${id}`);
  if (el) el.className = "stage-item error";
  renderStageBody(
    id,
    `<div style="color:var(--danger);font-size:12px;padding:4px 0">${msg}</div>`,
  );
}

// ── MuJoCo WASM Viewer ────────────────────────────────────────────────────────

async function initViewer() {
  try {
    mujoco = await load_mujoco();
    mujoco.FS.mkdir("/working");
    mujoco.FS.mount(mujoco.MEMFS, { root: "." }, "/working");

    // Init Three.js scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0.07, 0.13, 0.18);
    camera = new THREE.PerspectiveCamera(45, 1, 0.01, 20);
    camera.position.set(1.4, 1.2, 1.4);
    camera.lookAt(0, 0.5, 0);

    scene.add(new THREE.AmbientLight(0xffffff, 0.7));
    const key = new THREE.DirectionalLight(0xffffff, 2.0);
    key.position.set(2, 4, 3);
    key.castShadow = true;
    scene.add(key);
    scene.add(
      Object.assign(new THREE.DirectionalLight(0x99bbff, 0.6), {
        position: new THREE.Vector3(-2, 2, -1),
      }),
    );

    const wrap = document.getElementById("viewer-wrap");
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.domElement.id = "viewer-canvas";
    wrap.appendChild(renderer.domElement);

    controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0.4, 0);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;

    const resizeObserver = new ResizeObserver(() => resizeViewer());
    resizeObserver.observe(wrap);
    resizeViewer();

    document.getElementById("viewer-loading").style.display = "none";

    // Load default arm scene
    await loadViewerScene("default");

    // Start render loop
    function renderLoop() {
      animId = requestAnimationFrame(renderLoop);
      if (playTraj && trajectory.length) {
        const frame = trajectory[Math.floor(trajIdx) % trajectory.length];
        if (frame && frame.qpos) updateViewerQpos(frame.qpos);
        trajIdx += 0.5;
      } else if (currentQpos && mjModel && mjData) {
        // Hold last received qpos
      }
      if (mjModel && mjData) {
        updateBodies();
      }
      controls.update();
      renderer.render(scene, camera);
    }
    renderLoop();
  } catch (err) {
    document.getElementById("viewer-loading").textContent =
      "Viewer unavailable: " + err;
  }
}

function resizeViewer() {
  const wrap = document.getElementById("viewer-wrap");
  if (!renderer || !wrap) return;
  const w = wrap.clientWidth,
    h = wrap.clientHeight;
  renderer.setSize(w, h);
  if (camera) {
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
}

async function loadViewerScene(name) {
  try {
    const xmlText = await fetch(
      `/api/scene?name=${encodeURIComponent(name)}`,
    ).then((r) => {
      if (!r.ok) throw new Error(r.status);
      return r.text();
    });

    // Clean up old model
    if (mjData) {
      mjData.free?.();
      mjData = null;
    }
    if (mjModel) {
      mjModel.free?.();
      mjModel = null;
    }
    for (const b of Object.values(bodies)) scene.remove(b);
    bodies = {};

    // Preload any real STL meshes the scene references into the WASM FS, so the
    // model compiles with the actual CAD arm geometry instead of capsules. Each
    // <mesh file="upper_arm.stl"> is fetched from the server's /api/mesh route
    // (assets/stl/<scene>/<link>.stl) and written flat next to scene.xml.
    const meshFiles = [
      ...xmlText.matchAll(/<mesh\b[^>]*\bfile="([^"]+\.stl)"/g),
    ].map((m) => m[1]);
    for (const f of meshFiles) {
      const link = f.replace(/^.*\//, "").replace(/\.stl$/, "");
      try {
        const buf = await fetch(
          `/api/mesh?name=${encodeURIComponent(name)}&link=${encodeURIComponent(link)}`,
        ).then((r) => (r.ok ? r.arrayBuffer() : Promise.reject(r.status)));
        mujoco.FS.writeFile(`/working/${f}`, new Uint8Array(buf));
      } catch (e) {
        console.warn(
          `[pipeline] mesh ${f} unavailable (${e}); link will be invisible`,
        );
      }
    }

    // Write MJCF to WASM FS
    mujoco.FS.writeFile("/working/scene.xml", xmlText);

    mjModel = mujoco.MjModel.loadFromXML("/working/scene.xml");
    mjData = new mujoco.MjData(mjModel);
    mujoco.mj_forward(mjModel, mjData);

    bodies = buildViewerBodies(mjModel);
  } catch {}
}

function buildViewerBodies(model) {
  const bods = {};
  const meshes = {};
  for (let g = 0; g < model.ngeom; g++) {
    const b = model.geom_bodyid[g];
    const type = model.geom_type[g];
    if (type === model.mjtGeom?.mjGEOM_PLANE?.value) continue;
    const size = [
      model.geom_size[g * 3],
      model.geom_size[g * 3 + 1],
      model.geom_size[g * 3 + 2],
    ];
    if (!(b in bods)) {
      bods[b] = new THREE.Group();
      bods[b].bodyID = b;
      scene.add(bods[b]);
    }
    let geometry;
    const G = model.mjtGeom || {};
    if (type === (G.mjGEOM_SPHERE?.value ?? 2)) {
      geometry = new THREE.SphereGeometry(size[0], 16, 16);
    } else if (type === (G.mjGEOM_CAPSULE?.value ?? 5)) {
      geometry = new THREE.CapsuleGeometry(size[0], size[1] * 2, 8, 16);
    } else if (type === (G.mjGEOM_CYLINDER?.value ?? 4)) {
      geometry = new THREE.CylinderGeometry(size[0], size[0], size[1] * 2, 16);
    } else if (type === (G.mjGEOM_BOX?.value ?? 6)) {
      geometry = new THREE.BoxGeometry(size[0] * 2, size[2] * 2, size[1] * 2);
    } else if (type === (G.mjGEOM_MESH?.value ?? 7)) {
      const meshID = model.geom_dataid[g];
      if (!(meshID in meshes)) {
        geometry = new THREE.BufferGeometry();
        const vert = model.mesh_vert
          .subarray(
            model.mesh_vertadr[meshID] * 3,
            (model.mesh_vertadr[meshID] + model.mesh_vertnum[meshID]) * 3,
          )
          .slice();
        for (let v = 0; v < vert.length; v += 3) {
          const t = vert[v + 1];
          vert[v + 1] = vert[v + 2];
          vert[v + 2] = -t;
        }
        const faces = model.mesh_face.subarray(
          model.mesh_faceadr[meshID] * 3,
          (model.mesh_faceadr[meshID] + model.mesh_facenum[meshID]) * 3,
        );
        geometry.setAttribute("position", new THREE.BufferAttribute(vert, 3));
        geometry.setIndex(Array.from(faces));
        geometry.computeVertexNormals();
        meshes[meshID] = geometry;
      } else {
        geometry = meshes[meshID];
      }
    } else {
      geometry = new THREE.SphereGeometry(0.01);
    }

    const rgba = [
      model.geom_rgba[g * 4],
      model.geom_rgba[g * 4 + 1],
      model.geom_rgba[g * 4 + 2],
      model.geom_rgba[g * 4 + 3],
    ];
    const mat = new THREE.MeshPhysicalMaterial({
      color: new THREE.Color(rgba[0], rgba[1], rgba[2]),
      transparent: rgba[3] < 1,
      opacity: rgba[3],
      roughness: 0.6,
      metalness: 0.15,
    });
    const mesh = new THREE.Mesh(geometry, mat);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    bods[b].add(mesh);

    // Position / orientation from model (z-up → y-up swizzle)
    const gp = model.geom_pos;
    const gq = model.geom_quat;
    mesh.position.set(gp[g * 3], gp[g * 3 + 2], -gp[g * 3 + 1]);
    mesh.quaternion.set(
      -gq[g * 4 + 1],
      -gq[g * 4 + 3],
      gq[g * 4 + 2],
      gq[g * 4],
    );
  }
  return bods;
}

function updateViewerQpos(qpos) {
  if (!mjModel || !mjData) return;
  currentQpos = qpos;
  for (let i = 0; i < Math.min(qpos.length, mjModel.nq); i++) {
    mjData.qpos[i] = qpos[i];
  }
  mujoco.mj_forward(mjModel, mjData);
}

function updateBodies() {
  for (let b = 0; b < mjModel.nbody; b++) {
    if (!bodies[b]) continue;
    const xp = mjData.xpos,
      xq = mjData.xquat;
    bodies[b].position.set(xp[b * 3], xp[b * 3 + 2], -xp[b * 3 + 1]);
    bodies[b].quaternion.set(
      -xq[b * 4 + 1],
      -xq[b * 4 + 3],
      xq[b * 4 + 2],
      xq[b * 4],
    );
    bodies[b].updateWorldMatrix();
  }
}

// ── Joint panel (post-pipeline manual control) ─────────────────────────────────
function enableJointPanel(stats) {
  if (!stats || !mjModel) return;
  playTraj = false;

  const panel = document.getElementById("joint-panel");
  panel.style.display = "flex";
  panel.innerHTML = "<h3>Joint Controls</h3>";

  const joints = stats.joint_names || [];
  joints.forEach((name, i) => {
    const row = document.createElement("div");
    row.className = "joint-row";
    const slider = document.createElement("input");
    slider.type = "range";
    slider.min = -180;
    slider.max = 180;
    slider.value = 0;
    slider.step = 1;
    const valSpan = document.createElement("span");
    valSpan.className = "joint-val";
    valSpan.textContent = "0°";
    slider.addEventListener("input", () => {
      valSpan.textContent = slider.value + "°";
      if (!mjModel || !mjData) return;
      mjData.qpos[i] = (parseFloat(slider.value) * Math.PI) / 180;
      mujoco.mj_forward(mjModel, mjData);
    });
    const label = document.createElement("label");
    label.textContent = name;
    row.appendChild(label);
    row.appendChild(slider);
    row.appendChild(valSpan);
    panel.appendChild(row);
  });
}
