// Live MuJoCo-WASM viewer for PPO training in progress. Polls the files the live
// backend (scripts/demo/train_live.py) streams into assets/live/ and (a) plays the
// CURRENT policy's eval rollout on the articulated per-link CAD arm, (b) draws a
// toggleable wandb-style metrics dashboard on the right. Nothing is pre-baked —
// the arm and the curves update as training proceeds.
//
//   python3 scripts/demo/train_live.py --port 8011
//   open http://localhost:8011/live.html

import * as THREE from "three";
import { OrbitControls } from "../node_modules/three/examples/jsm/controls/OrbitControls.js";
import load_mujoco from "../node_modules/mujoco-js/dist/mujoco_wasm.js";
import { attachSplatBackdrop } from "./splatBackdrop.js";

const SCENE = "arm_articulated.xml";
const LINKS = ["upper_arm", "forearm", "gripper"];
const STATUS_URL = "./assets/live/status.json";
const TRAJ_URL = "./assets/live/trajectory.json";

const mujoco = await load_mujoco();

// --- virtual FS: static scene + per-link meshes -----------------------------
mujoco.FS.mkdir("/working");
mujoco.FS.mount(mujoco.MEMFS, { root: "." }, "/working");
mujoco.FS.writeFile(
  "/working/" + SCENE,
  await (await fetch("./assets/scenes/" + SCENE)).text(),
);
mujoco.FS.mkdir("/working/arm_links");
for (const link of LINKS) {
  mujoco.FS.writeFile(
    "/working/arm_links/" + link + ".stl",
    new Uint8Array(
      await (
        await fetch("./assets/scenes/arm_links/" + link + ".stl")
      ).arrayBuffer(),
    ),
  );
}

// --- swizzle helpers (MuJoCo z-up -> three.js y-up) -------------------------
function getPosition(buffer, index, target) {
  return target.set(
    buffer[index * 3 + 0],
    buffer[index * 3 + 2],
    -buffer[index * 3 + 1],
  );
}
function getQuaternion(buffer, index, target) {
  return target.set(
    -buffer[index * 4 + 1],
    -buffer[index * 4 + 3],
    buffer[index * 4 + 2],
    -buffer[index * 4 + 0],
  );
}

function buildBodies(model, scene) {
  const bodies = {};
  const meshes = {};
  for (let g = 0; g < model.ngeom; g++) {
    if (!(model.geom_group[g] < 3)) continue;
    const b = model.geom_bodyid[g];
    const type = model.geom_type[g];
    if (type == mujoco.mjtGeom.mjGEOM_PLANE.value) continue; // splat is the floor
    const size = [
      model.geom_size[g * 3 + 0],
      model.geom_size[g * 3 + 1],
      model.geom_size[g * 3 + 2],
    ];
    if (!(b in bodies)) {
      bodies[b] = new THREE.Group();
      bodies[b].bodyID = b;
      scene.add(bodies[b]);
    }

    let geometry = new THREE.SphereGeometry(size[0] * 0.5);
    if (type == mujoco.mjtGeom.mjGEOM_SPHERE.value) {
      geometry = new THREE.SphereGeometry(size[0]);
    } else if (type == mujoco.mjtGeom.mjGEOM_CAPSULE.value) {
      geometry = new THREE.CapsuleGeometry(size[0], size[1] * 2.0, 12, 20);
    } else if (type == mujoco.mjtGeom.mjGEOM_CYLINDER.value) {
      geometry = new THREE.CylinderGeometry(size[0], size[0], size[1] * 2.0);
    } else if (type == mujoco.mjtGeom.mjGEOM_BOX.value) {
      geometry = new THREE.BoxGeometry(size[0] * 2, size[2] * 2, size[1] * 2);
    } else if (type == mujoco.mjtGeom.mjGEOM_MESH.value) {
      const meshID = model.geom_dataid[g];
      if (!(meshID in meshes)) {
        geometry = new THREE.BufferGeometry();
        const vert = model.mesh_vert.subarray(
          model.mesh_vertadr[meshID] * 3,
          (model.mesh_vertadr[meshID] + model.mesh_vertnum[meshID]) * 3,
        );
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
    }

    const color = [
      model.geom_rgba[g * 4 + 0],
      model.geom_rgba[g * 4 + 1],
      model.geom_rgba[g * 4 + 2],
      model.geom_rgba[g * 4 + 3],
    ];
    const material = new THREE.MeshPhysicalMaterial({
      color: new THREE.Color(color[0], color[1], color[2]),
      transparent: color[3] < 1.0,
      opacity: color[3],
      roughness: 0.7,
      metalness: 0.1,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.castShadow = g != 0;
    mesh.receiveShadow = true;
    bodies[b].add(mesh);
    getPosition(model.geom_pos, g, mesh.position);
    getQuaternion(model.geom_quat, g, mesh.quaternion);
  }
  return bodies;
}

// --- scene / renderer -------------------------------------------------------
const model = mujoco.MjModel.loadFromXML("/working/" + SCENE);
const data = new mujoco.MjData(model);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0.07, 0.13, 0.18);
const camera = new THREE.PerspectiveCamera(
  45,
  window.innerWidth / window.innerHeight,
  0.01,
  100,
);
camera.position.set(1.6, 1.5, 1.6);
scene.add(camera);
scene.add(new THREE.AmbientLight(0xffffff, 0.7));
const key = new THREE.DirectionalLight(0xffffff, 2.0);
key.position.set(2, 4, 3);
key.castShadow = true;
scene.add(key);
const fill = new THREE.DirectionalLight(0x99bbff, 0.6);
fill.position.set(-2, 2, -1);
scene.add(fill);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(1.0);
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
document.body.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0.7, 0);
controls.enableDamping = true;
controls.update();
window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

const bodies = buildBodies(model, scene);

// Room Gaussian splat as the backdrop + floor (shared module + WASD/QE/ZX/P align
// tool). Fire-and-forget: the arm + dashboard render immediately while the (large)
// splat loads, then it pops in. Render-only — MuJoCo still owns all physics.
attachSplatBackdrop(scene, { align: true });

// --- trajectory playback (current policy) -----------------------------------
let traj = null;
let trajStep = -1;
let startMS = null;

function applyFrame(frame) {
  for (let i = 0; i < frame.length && i < model.nq; i++)
    data.qpos[i] = frame[i];
  mujoco.mj_forward(model, data);
}

function render(timeMS) {
  if (traj && traj.frames.length) {
    if (startMS === null) startMS = timeMS;
    const n = traj.frames.length;
    const total = n * traj.dt + 1.0; // hold final pose 1s then loop
    const tt = ((timeMS - startMS) / 1000.0) % total;
    applyFrame(traj.frames[Math.min(n - 1, Math.floor(tt / traj.dt))]);
  }
  for (let b = 0; b < model.nbody; b++) {
    if (bodies[b]) {
      getPosition(data.xpos, b, bodies[b].position);
      getQuaternion(data.xquat, b, bodies[b].quaternion);
      bodies[b].updateWorldMatrix();
    }
  }
  controls.update();
  renderer.render(scene, camera);
}
renderer.setAnimationLoop(render);

// --- dashboard --------------------------------------------------------------
// Only the metrics that tell you whether the arm is learning the task.
const METRICS = [
  {
    key: "reward",
    label: "Episode reward",
    fmt: (v) => v.toFixed(2),
    yfmt: (v) => v.toFixed(1),
  },
  {
    key: "success_rate",
    label: "Success rate",
    fmt: (v) => (v * 100).toFixed(0) + "%",
    yfmt: (v) => (v * 100).toFixed(0) + "%",
  },
  {
    key: "final_cm",
    label: "Mean final dist (cm)",
    fmt: (v) => v.toFixed(1) + " cm",
    yfmt: (v) => v.toFixed(0),
  },
];

const cardsEl = document.getElementById("dash-cards");
const cards = {};
for (const m of METRICS) {
  const card = document.createElement("div");
  card.className = "card";
  card.innerHTML =
    `<div class="row"><span class="title">${m.label}</span>` +
    `<span class="val" id="val-${m.key}">—</span></div>` +
    `<canvas id="cv-${m.key}" width="330" height="96"></canvas>`;
  cardsEl.appendChild(card);
  cards[m.key] = card.querySelector("canvas");
}

function fmtStep(s) {
  return s >= 1000 ? Math.round(s / 1000) + "k" : String(Math.round(s));
}

function drawChart(canvas, steps, vals, color, yfmt) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  const padL = 40,
    padR = 10,
    padT = 8,
    padB = 18; // room for y labels (left) + x labels (bottom)
  ctx.font = "9px -apple-system, Arial";

  const pts = [];
  for (let i = 0; i < vals.length; i++) {
    if (vals[i] === null || vals[i] === undefined || Number.isNaN(vals[i]))
      continue;
    pts.push([steps[i], vals[i]]);
  }
  if (pts.length < 2) {
    ctx.fillStyle = "#54697a";
    ctx.fillText("collecting…", padL, H / 2);
    return;
  }
  let xmin = pts[0][0],
    xmax = pts[pts.length - 1][0];
  let ymin = Infinity,
    ymax = -Infinity;
  for (const [, y] of pts) {
    ymin = Math.min(ymin, y);
    ymax = Math.max(ymax, y);
  }
  if (ymax - ymin < 1e-9) {
    ymax += 1;
    ymin -= 1;
  }
  const sx = (x) =>
    padL + ((x - xmin) / (xmax - xmin || 1)) * (W - padL - padR);
  const sy = (y) => H - padB - ((y - ymin) / (ymax - ymin)) * (H - padT - padB);

  // axes (left = value, bottom = step)
  ctx.strokeStyle = "#2a3742";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padL, padT);
  ctx.lineTo(padL, H - padB);
  ctx.lineTo(W - padR, H - padB);
  ctx.stroke();
  // zero gridline if the range crosses 0
  if (ymin < 0 && ymax > 0) {
    ctx.strokeStyle = "#1d2a33";
    ctx.beginPath();
    ctx.moveTo(padL, sy(0));
    ctx.lineTo(W - padR, sy(0));
    ctx.stroke();
  }

  // y-axis labels (max top, min bottom)
  ctx.fillStyle = "#7d93a3";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  ctx.fillText(yfmt(ymax), padL - 4, sy(ymax) + 4);
  ctx.fillText(yfmt(ymin), padL - 4, sy(ymin) - 4);
  // x-axis labels (first/last step + axis title)
  ctx.textBaseline = "top";
  ctx.textAlign = "left";
  ctx.fillText(fmtStep(xmin), padL, H - padB + 4);
  ctx.textAlign = "right";
  ctx.fillText(fmtStep(xmax), W - padR, H - padB + 4);
  ctx.textAlign = "center";
  ctx.fillStyle = "#54697a";
  ctx.fillText("step", (padL + (W - padR)) / 2, H - padB + 4);

  // series
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  pts.forEach(([x, y], i) =>
    i ? ctx.lineTo(sx(x), sy(y)) : ctx.moveTo(sx(x), sy(y)),
  );
  ctx.stroke();
  const [lx, ly] = pts[pts.length - 1];
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(sx(lx), sy(ly), 2.5, 0, 2 * Math.PI);
  ctx.fill();
}

const dotEl = document.getElementById("status-dot");
const metaEl = document.getElementById("dash-meta");
const barEl = document.getElementById("bar-fill");
const reachEl = document.getElementById("reach-val");
const subEl = document.getElementById("sub");

function updateDashboard(status) {
  const hist = status.history || [];
  const steps = hist.map((h) => h.step);
  const last = hist[hist.length - 1] || {};
  const pct = status.total
    ? Math.min(100, (100 * status.step) / status.total)
    : 0;
  barEl.style.width = pct.toFixed(1) + "%";
  dotEl.style.background = status.running ? "#6fe6a0" : "#888";
  metaEl.textContent =
    `step ${status.step.toLocaleString()} / ${status.total.toLocaleString()}` +
    (status.elapsed_s ? `  ·  ${status.elapsed_s.toFixed(0)}s` : "") +
    (status.running ? "  ·  training" : "  ·  done");
  for (const m of METRICS) {
    const v = last[m.key];
    document.getElementById("val-" + m.key).textContent =
      v === null || v === undefined ? "—" : m.fmt(v);
    drawChart(
      cards[m.key],
      steps,
      hist.map((h) => h[m.key]),
      m.key === "reward" || m.key === "success_rate" ? "#6fe6a0" : "#5aa9e6",
      m.yfmt || ((y) => String(Math.round(y))),
    );
  }
  if (subEl)
    subEl.textContent = status.running
      ? `training live · step ${status.step.toLocaleString()}`
      : `training complete · ${status.step.toLocaleString()} steps`;
}

// --- reset button -----------------------------------------------------------
const resetBtn = document.getElementById("reset-btn");
if (resetBtn) {
  resetBtn.addEventListener("click", async () => {
    resetBtn.disabled = true;
    resetBtn.textContent = "resetting…";
    try {
      await fetch("/reset?_=" + Date.now());
    } catch (e) {
      /* backend not the trainer (static server) — nothing to reset */
    }
    // Restart the viewer playback so it picks up the fresh policy's first rollout.
    traj = null;
    trajStep = -1;
    startMS = null;
    setTimeout(() => {
      resetBtn.disabled = false;
      resetBtn.textContent = "↻ Reset training";
    }, 1500);
  });
}

// --- poll loop --------------------------------------------------------------
async function poll() {
  try {
    const status = await (await fetch(STATUS_URL + "?_=" + Date.now())).json();
    updateDashboard(status);
  } catch (e) {
    if (subEl)
      subEl.textContent = "waiting for training backend (run train_live.py)…";
  }
  try {
    const t = await (await fetch(TRAJ_URL + "?_=" + Date.now())).json();
    if (t.step !== trajStep) {
      trajStep = t.step;
      traj = t;
      startMS = null; // restart playback on the new policy
      reachEl.innerHTML =
        `${t.success ? "<span style='color:#6fe6a0'>HIT</span>" : "miss"} ` +
        `(${t.final_cm.toFixed(1)} cm) · step ${t.step.toLocaleString()}`;
    }
  } catch (e) {
    /* trajectory not written yet */
  }
  setTimeout(poll, 1000);
}
poll();
