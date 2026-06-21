// Render the prosthesis arm INSIDE the real room as a 3D Gaussian Splat.
//
// The splat (assets/scenes/room.ply) is the actual Inria 3DGS training output from
// scripts/recon/modal_gsplat.py. We composite it behind the articulated MuJoCo arm
// (same WASM viewer as playback.js) using @mkkellogg/gaussian-splats-3d's DropInViewer.
//
// Monocular COLMAP/3DGS is up-to-scale in an arbitrary frame, so the splat needs to
// be aligned to the arm's metric world ONCE by hand: use the keyboard to move/rotate/
// scale it, press P to print the transform, then paste it into SPLAT_XFORM below.
//
//   python3 scripts/recon/modal_gsplat.py --frames-dir frames --out room.ply
//   cp room.ply webdemo/assets/scenes/room.ply
//   cd webdemo && python3 -m http.server 8011  ->  http://localhost:8011/splat.html

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import * as GaussianSplats3D from "@mkkellogg/gaussian-splats-3d";
import load_mujoco from "../node_modules/mujoco-js/dist/mujoco_wasm.js";

const SCENE = "arm_articulated.xml";
const LINKS = ["upper_arm", "forearm", "gripper"];
const SPLAT_URL = "./assets/scenes/room.ply";
const TRAJ_URL = "./assets/scenes/arm_trajectory.json";

// Hand-tuned alignment of the splat into the arm's world frame (edit after P-print).
const SPLAT_XFORM = {
  position: [0, 0, 0],
  rotationYdeg: 0,
  scale: 1.0,
};

const sub = document.getElementById("sub");

// ---------------------------------------------------------------------------
const mujoco = await load_mujoco();
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

// Build three.js meshes for the arm. Skip planes so the splat floor shows through.
function buildBodies(model, scene) {
  const bodies = {};
  const meshes = {};
  for (let g = 0; g < model.ngeom; g++) {
    if (!(model.geom_group[g] < 3)) continue;
    const type = model.geom_type[g];
    if (type == mujoco.mjtGeom.mjGEOM_PLANE.value) continue; // let the splat be the floor
    const b = model.geom_bodyid[g];
    const size = [
      model.geom_size[g * 3],
      model.geom_size[g * 3 + 1],
      model.geom_size[g * 3 + 2],
    ];
    if (!(b in bodies)) {
      bodies[b] = new THREE.Group();
      bodies[b].bodyID = b;
      scene.add(bodies[b]);
    }
    let geometry = new THREE.SphereGeometry(size[0] * 0.5);
    if (type == mujoco.mjtGeom.mjGEOM_SPHERE.value)
      geometry = new THREE.SphereGeometry(size[0]);
    else if (type == mujoco.mjtGeom.mjGEOM_CAPSULE.value)
      geometry = new THREE.CapsuleGeometry(size[0], size[1] * 2.0, 12, 20);
    else if (type == mujoco.mjtGeom.mjGEOM_CYLINDER.value)
      geometry = new THREE.CylinderGeometry(size[0], size[0], size[1] * 2.0);
    else if (type == mujoco.mjtGeom.mjGEOM_BOX.value)
      geometry = new THREE.BoxGeometry(size[0] * 2, size[2] * 2, size[1] * 2);
    else if (type == mujoco.mjtGeom.mjGEOM_MESH.value) {
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
      } else geometry = meshes[meshID];
    }
    const color = [
      model.geom_rgba[g * 4],
      model.geom_rgba[g * 4 + 1],
      model.geom_rgba[g * 4 + 2],
      model.geom_rgba[g * 4 + 3],
    ];
    const mesh = new THREE.Mesh(
      geometry,
      new THREE.MeshPhysicalMaterial({
        color: new THREE.Color(color[0], color[1], color[2]),
        transparent: color[3] < 1.0,
        opacity: color[3],
        roughness: 0.7,
        metalness: 0.1,
      }),
    );
    mesh.bodyID = b;
    bodies[b].add(mesh);
    getPosition(model.geom_pos, g, mesh.position);
    getQuaternion(model.geom_quat, g, mesh.quaternion);
  }
  return bodies;
}

// --- three.js scene ---------------------------------------------------------
const model = mujoco.MjModel.loadFromXML("/working/" + SCENE);
const data = new mujoco.MjData(model);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0.04, 0.06, 0.09);
const camera = new THREE.PerspectiveCamera(
  45,
  window.innerWidth / window.innerHeight,
  0.01,
  100,
);
camera.position.set(1.8, 1.5, 1.8);
scene.add(camera);
scene.add(new THREE.AmbientLight(0xffffff, 0.8));
const key = new THREE.DirectionalLight(0xffffff, 1.8);
key.position.set(2, 4, 3);
scene.add(key);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(1.0);
renderer.setSize(window.innerWidth, window.innerHeight);
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

// --- the Gaussian splat room ------------------------------------------------
let splatViewer = null;
try {
  splatViewer = new GaussianSplats3D.DropInViewer({
    gpuAcceleratedSort: true,
    sharedMemoryForWorkers: false, // python http.server sends no COOP/COEP headers
  });
  await splatViewer.addSplatScene(SPLAT_URL, {
    splatAlphaRemovalThreshold: 5,
    showLoadingUI: false,
    position: SPLAT_XFORM.position,
    rotation: [0, 0, 0, 1],
    scale: [SPLAT_XFORM.scale, SPLAT_XFORM.scale, SPLAT_XFORM.scale],
  });
  splatViewer.rotation.y = (SPLAT_XFORM.rotationYdeg * Math.PI) / 180;
  scene.add(splatViewer);
  sub.textContent =
    "room splat loaded — align with keyboard, P to print transform";
} catch (e) {
  sub.textContent =
    "no room.ply yet — train it: scripts/recon/modal_gsplat.py, then cp room.ply assets/scenes/";
  console.warn("splat load failed:", e);
}

// --- keyboard alignment of the splat ---------------------------------------
const STEP = 0.05;
window.addEventListener("keydown", (ev) => {
  if (!splatViewer) return;
  const p = splatViewer.position;
  switch (ev.key.toLowerCase()) {
    case "w":
      p.z -= STEP;
      break;
    case "s":
      p.z += STEP;
      break;
    case "a":
      p.x -= STEP;
      break;
    case "d":
      p.x += STEP;
      break;
    case "r":
      p.y += STEP;
      break;
    case "f":
      p.y -= STEP;
      break;
    case "q":
      splatViewer.rotation.y += 0.05;
      break;
    case "e":
      splatViewer.rotation.y -= 0.05;
      break;
    case "z":
      splatViewer.scale.multiplyScalar(1.03);
      break;
    case "x":
      splatViewer.scale.multiplyScalar(1 / 1.03);
      break;
    case "p":
      console.log(
        "SPLAT_XFORM = {",
        `position: [${p.x.toFixed(3)}, ${p.y.toFixed(3)}, ${p.z.toFixed(3)}],`,
        `rotationYdeg: ${((splatViewer.rotation.y * 180) / Math.PI).toFixed(1)},`,
        `scale: ${splatViewer.scale.x.toFixed(3)} };`,
      );
      break;
  }
});

// --- arm trajectory playback ------------------------------------------------
let traj = null;
let startMS = null;
try {
  traj = await (await fetch(TRAJ_URL + "?_=" + Date.now())).json();
} catch (e) {
  /* no trajectory yet — arm holds its rest pose */
}

function applyFrame(frame) {
  for (let i = 0; i < frame.length && i < model.nq; i++)
    data.qpos[i] = frame[i];
  mujoco.mj_forward(model, data);
}

function render(timeMS) {
  if (traj && traj.frames.length) {
    if (startMS === null) startMS = timeMS;
    const n = traj.frames.length;
    const total = n * traj.dt + 1.0;
    const tt = ((timeMS - startMS) / 1000.0) % total;
    applyFrame(traj.frames[Math.min(n - 1, Math.floor(tt / traj.dt))]);
  } else {
    mujoco.mj_forward(model, data);
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
