// Standalone MuJoCo-WASM playback of a trained-policy reach on the articulated,
// per-link CAD arm. Self-contained on purpose: it does NOT import main.js /
// mujocoUtils.js (those boot the interactive demo on import). Generate its inputs
// with `python3 scripts/demo/export_web_playback.py`:
//   assets/scenes/arm_articulated.xml  +  arm_links/*.stl  +  arm_trajectory.json
//
// Playback is kinematic: each frame we set the arm joint qpos from the recorded
// trajectory and mj_forward — so the browser shows exactly the motion the policy
// produced in sim, with no physics drift.

import * as THREE from "three";
import { OrbitControls } from "../node_modules/three/examples/jsm/controls/OrbitControls.js";
import load_mujoco from "../node_modules/mujoco-js/dist/mujoco_wasm.js";

const SCENE = "arm_articulated.xml";
const TRAJ = "./assets/scenes/arm_trajectory.json";

const mujoco = await load_mujoco();

// --- Emscripten virtual FS: scene XML + per-link STL meshes ------------------
mujoco.FS.mkdir("/working");
mujoco.FS.mount(mujoco.MEMFS, { root: "." }, "/working");
mujoco.FS.writeFile(
  "/working/" + SCENE,
  await (await fetch("./assets/scenes/" + SCENE)).text(),
);
mujoco.FS.mkdir("/working/arm_links");
const traj = await (await fetch(TRAJ)).json();
// The articulated scene references one STL per link under arm_links/.
for (const link of traj.links || []) {
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

// --- build three.js meshes from the compiled MuJoCo model -------------------
// Ported from mujocoUtils.loadSceneFromURL (geom loop), trimmed to the geom
// types this scene uses (plane/sphere/capsule/cylinder/box/mesh).
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
    if (!isPlane) {
      getQuaternion(model.geom_quat, g, mesh.quaternion);
    } else {
      mesh.rotateX(-Math.PI / 2);
    }
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

const sub = document.getElementById("sub");
sub.innerHTML =
  `${traj.joints.length}-DoF arm · PPO policy · ` +
  `reach ${traj.success ? "<b>HIT</b>" : "miss"} (${traj.final_cm.toFixed(1)} cm)` +
  `<br/>frames: ${traj.frames.length} · drag to orbit · loops`;

// --- playback loop: drive arm qpos from the recorded trajectory -------------
const tmpVec = new THREE.Vector3();
const tmpQuat = new THREE.Quaternion();
let startMS = null;

function applyFrame(idx) {
  const frame = traj.frames[idx];
  for (let i = 0; i < frame.length && i < model.nq; i++) {
    data.qpos[i] = frame[i];
  }
  mujoco.mj_forward(model, data);
}

function render(timeMS) {
  if (startMS === null) startMS = timeMS;
  const elapsed = (timeMS - startMS) / 1000.0;
  const n = traj.frames.length;
  // play through, then hold the final reached pose for 1s, then loop.
  const total = n * traj.dt + 1.0;
  const tt = elapsed % total;
  const idx = Math.min(n - 1, Math.floor(tt / traj.dt));
  applyFrame(idx);

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
