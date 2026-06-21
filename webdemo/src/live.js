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
    let isPlane = false;
    if (type == mujoco.mjtGeom.mjGEOM_PLANE.value) {
      isPlane = true;
      geometry = new THREE.PlaneGeometry(8, 8);
    } else if (type == mujoco.mjtGeom.mjGEOM_SPHERE.value) {
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
    if (!isPlane) getQuaternion(model.geom_quat, g, mesh.quaternion);
    else mesh.rotateX(-Math.PI / 2);
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
const METRICS = [
  { key: "reward", label: "Episode reward", fmt: (v) => v.toFixed(2) },
  {
    key: "success_rate",
    label: "Success rate",
    fmt: (v) => (v * 100).toFixed(0) + "%",
  },
  {
    key: "final_cm",
    label: "Mean final dist",
    fmt: (v) => v.toFixed(1) + " cm",
  },
  { key: "value_loss", label: "Value loss", fmt: (v) => v.toFixed(3) },
  { key: "policy_loss", label: "Policy loss", fmt: (v) => v.toFixed(4) },
  { key: "entropy", label: "Entropy", fmt: (v) => v.toFixed(3) },
  { key: "approx_kl", label: "Approx KL", fmt: (v) => v.toFixed(4) },
  {
    key: "explained_variance",
    label: "Explained variance",
    fmt: (v) => v.toFixed(2),
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
    `<canvas id="cv-${m.key}" width="340" height="64"></canvas>`;
  cardsEl.appendChild(card);
  cards[m.key] = card.querySelector("canvas");
}

function drawChart(canvas, steps, vals, color) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  const pts = [];
  for (let i = 0; i < vals.length; i++) {
    if (vals[i] === null || vals[i] === undefined || Number.isNaN(vals[i]))
      continue;
    pts.push([steps[i], vals[i]]);
  }
  if (pts.length < 2) return;
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
  const pad = 6;
  const sx = (x) => pad + ((x - xmin) / (xmax - xmin || 1)) * (W - 2 * pad);
  const sy = (y) => H - pad - ((y - ymin) / (ymax - ymin)) * (H - 2 * pad);
  // baseline grid
  ctx.strokeStyle = "#22303a";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad, H - pad);
  ctx.lineTo(W - pad, H - pad);
  ctx.stroke();
  // line
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  pts.forEach(([x, y], i) =>
    i ? ctx.lineTo(sx(x), sy(y)) : ctx.moveTo(sx(x), sy(y)),
  );
  ctx.stroke();
  // last point
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
    );
  }
  subEl.textContent = status.running
    ? `training live · step ${status.step.toLocaleString()}`
    : `training complete · ${status.step.toLocaleString()} steps`;
}

// --- toggle -----------------------------------------------------------------
const dash = document.getElementById("dash");
const toggle = document.getElementById("dash-toggle");
toggle.addEventListener("click", () => {
  dash.classList.toggle("hidden");
  toggle.textContent = dash.classList.contains("hidden")
    ? "Show metrics ◀"
    : "Hide metrics ▶";
});

// --- poll loop --------------------------------------------------------------
async function poll() {
  try {
    const status = await (await fetch(STATUS_URL + "?_=" + Date.now())).json();
    updateDashboard(status);
  } catch (e) {
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
