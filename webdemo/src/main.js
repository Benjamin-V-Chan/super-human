import * as THREE from "three";
import { GUI } from "../node_modules/three/examples/jsm/libs/lil-gui.module.min.js";
import { OrbitControls } from "../node_modules/three/examples/jsm/controls/OrbitControls.js";
import { DragStateManager } from "./utils/DragStateManager.js";
import {
  setupGUI,
  downloadExampleScenesFolder,
  loadSceneFromURL,
  reloadFunc,
  drawTendonsAndFlex,
  getPosition,
  getQuaternion,
  toMujocoPos,
  standardNormal,
  nameToId,
} from "./mujocoUtils.js";
import { EvalOverlay } from "./evalOverlay.js";
import { TaskPanel } from "./taskPanel.js";
import { GraspController } from "./graspController.js";
import load_mujoco from "../node_modules/mujoco-js/dist/mujoco_wasm.js";
import { attachSplatBackdrop } from "./splatBackdrop.js";

// Load the MuJoCo Module
const mujoco = await load_mujoco();

// Set up Emscripten's Virtual File System
var initialScene = "arm.xml";
mujoco.FS.mkdir("/working");
mujoco.FS.mount(mujoco.MEMFS, { root: "." }, "/working");
mujoco.FS.writeFile(
  "/working/" + initialScene,
  await (await fetch("./assets/scenes/" + initialScene)).text(),
);
// Preload the arm mesh so the model compiles in the constructor (before init()
// downloads the rest of the scene folder).
mujoco.FS.writeFile(
  "/working/arm_visual.stl",
  new Uint8Array(
    await (await fetch("./assets/scenes/arm_visual.stl")).arrayBuffer(),
  ),
);

export class MuJoCoDemo {
  constructor() {
    this.mujoco = mujoco;

    // Optional Gaussian-splat room backdrop (opt-in via ?splat). Render-only —
    // the splat never enters physics; MuJoCo still owns every collision.
    this.splatEnabled = new URLSearchParams(window.location.search).has(
      "splat",
    );

    // Load in the state from XML
    this.model = mujoco.MjModel.loadFromXML("/working/" + initialScene);
    this.data = new mujoco.MjData(this.model);

    // Define Random State Variables
    this.params = {
      scene: initialScene,
      paused: false,
      help: false,
      ctrlnoiserate: 0.0,
      ctrlnoisestd: 0.0,
      keyframeNumber: 0,
    };
    this.mujoco_time = 0.0;
    ((this.bodies = {}), (this.lights = {}));
    this.tmpVec = new THREE.Vector3();
    this.tmpQuat = new THREE.Quaternion();
    this.updateGUICallbacks = [];

    this.container = document.createElement("div");
    document.body.appendChild(this.container);

    this.scene = new THREE.Scene();
    this.scene.name = "scene";

    this.camera = new THREE.PerspectiveCamera(
      45,
      window.innerWidth / window.innerHeight,
      0.001,
      100,
    );
    this.camera.name = "PerspectiveCamera";
    this.camera.position.set(2.0, 1.7, 1.7);
    this.scene.add(this.camera);

    this.scene.background = new THREE.Color(0.15, 0.25, 0.35);
    this.scene.fog = new THREE.Fog(this.scene.background, 15, 25.5);

    this.ambientLight = new THREE.AmbientLight(0xffffff, 0.1 * 3.14);
    this.ambientLight.name = "AmbientLight";
    this.scene.add(this.ambientLight);

    this.spotlight = new THREE.SpotLight();
    this.spotlight.angle = 1.11;
    this.spotlight.distance = 10000;
    this.spotlight.penumbra = 0.5;
    this.spotlight.castShadow = true; // default false
    this.spotlight.intensity = this.spotlight.intensity * 3.14 * 10.0;
    this.spotlight.shadow.mapSize.width = 1024; // default
    this.spotlight.shadow.mapSize.height = 1024; // default
    this.spotlight.shadow.camera.near = 0.1; // default
    this.spotlight.shadow.camera.far = 100; // default
    this.spotlight.position.set(0, 3, 3);
    const targetObject = new THREE.Object3D();
    this.scene.add(targetObject);
    this.spotlight.target = targetObject;
    targetObject.position.set(0, 1, 0);
    this.scene.add(this.spotlight);

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(1.0); ////window.devicePixelRatio );
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap; // default THREE.PCFShadowMap
    THREE.ColorManagement.enabled = false;
    this.renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
    //this.renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
    //this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    //this.renderer.toneMappingExposure = 2.0;
    this.renderer.useLegacyLights = true;

    this.renderer.setAnimationLoop(this.render.bind(this));

    this.container.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0, 0.7, 0);
    this.controls.panSpeed = 2;
    this.controls.zoomSpeed = 1;
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.1;
    this.controls.screenSpacePanning = true;
    this.controls.update();

    window.addEventListener("resize", this.onWindowResize.bind(this));

    // Initialize the Drag State Manager.
    this.dragStateManager = new DragStateManager(
      this.scene,
      this.renderer,
      this.camera,
      this.container.parentElement,
      this.controls,
    );
  }

  async init() {
    // Download the the examples to MuJoCo's virtual file system
    await downloadExampleScenesFolder(mujoco);

    // Initialize the three.js Scene using the .xml Model in initialScene
    [this.model, this.data, this.bodies, this.lights] = await loadSceneFromURL(
      mujoco,
      initialScene,
      this,
    );

    // Generic mujoco_wasm "Controls" GUI removed — the Task panel + grader overlay
    // are the only UI. (Re-enable with `this.gui = new GUI(); setupGUI(this);`.)

    // Live eval/reward HUD (computes the reward shaping in-browser each frame).
    this.evalOverlay = new EvalOverlay(this);

    // Task/Dataset selector (drives setTask on toggle; reads assets/tasks.json).
    this.taskPanel = new TaskPanel(this);

    // Scripted grasp loop (runs only for grasp tasks on the hand scene).
    this.graspController = new GraspController(this);

    // Room Gaussian splat behind the live arm (opt-in via ?splat). Fire-and-
    // forget: the floor/fog are hidden synchronously, but the (large) splat
    // loads in the background so the arm + physics render immediately instead of
    // blocking init() on a multi-hundred-MB download. The splat pops in when ready.
    if (this.splatEnabled) this.enableSplatBackdrop();
  }

  /** Load the arm + target + grader weights for a task from tasks.json. */
  async setTask(task) {
    const ARM_SCENE = {
      single: "arm.xml",
      articulated: "arm_articulated.xml",
      hand: "arm_hand.xml",
    };
    const wantArm = ARM_SCENE[task.arm] || "arm.xml";
    if (this.params.scene !== wantArm) {
      this.params.scene = wantArm;
      await reloadFunc.call(this); // rebuilds model/data; overlay rebinds lazily
      // reloadFunc rebuilds the reflective floor; re-hide it so the splat shows.
      if (this.splatEnabled) this.hideFloorForSplat();
    }
    this.activeTask = task;
    this._applyTarget(task.target);
    if (this.evalOverlay) this.evalOverlay.setTask(task);
    if (this.graspController) this.graspController.enable(!!task.grasp);
  }

  /** Move the (mocap) target marker live, no reload. */
  _applyTarget(t) {
    const m = this.model,
      d = this.data;
    const bid = nameToId(m, m.name_bodyadr, m.nbody, "target_marker");
    if (bid < 0) return;
    const mid = m.body_mocapid[bid];
    if (mid < 0) return;
    d.mocap_pos[mid * 3] = t[0];
    d.mocap_pos[mid * 3 + 1] = t[1];
    d.mocap_pos[mid * 3 + 2] = t[2];
    this.mujoco.mj_forward(m, d);
  }

  /** Attach the room splat behind the live physics arm (opt-in via ?splat).
   *  The splat is render-only — MuJoCo still owns every collision — so this
   *  just hides the reflective floor + fog and drops the splat into the scene. */
  async enableSplatBackdrop() {
    this.hideFloorForSplat();
    if (!this.splat) {
      this.splat = await attachSplatBackdrop(this.scene, {
        align: true,
        statusEl: document.getElementById("sub"),
      });
    }
  }

  /** Hide the MuJoCo reflective floor + fog so the splat reads as the room.
   *  Must re-run after every scene (re)load — reloadFunc rebuilds the floor. */
  hideFloorForSplat() {
    this.scene.fog = null;
    this.scene.traverse((o) => {
      if (o.type === "Reflector") o.visible = false;
    });
  }

  onWindowResize() {
    this.camera.aspect = window.innerWidth / window.innerHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(window.innerWidth, window.innerHeight);
  }

  render(timeMS) {
    this.controls.update();

    // Scripted grasp drives data.ctrl before the physics steps below.
    if (this.graspController) this.graspController.update();

    if (!this.params["paused"]) {
      let timestep = this.model.opt.timestep;
      if (timeMS - this.mujoco_time > 35.0) {
        this.mujoco_time = timeMS;
      }
      while (this.mujoco_time < timeMS) {
        // Jitter the control state with gaussian random noise
        if (this.params["ctrlnoisestd"] > 0.0) {
          let rate = Math.exp(
            -timestep / Math.max(1e-10, this.params["ctrlnoiserate"]),
          );
          let scale = this.params["ctrlnoisestd"] * Math.sqrt(1 - rate * rate);
          let currentCtrl = this.data.ctrl;
          for (let i = 0; i < currentCtrl.length; i++) {
            currentCtrl[i] = rate * currentCtrl[i] + scale * standardNormal();
            this.params["Actuator " + i] = currentCtrl[i];
          }
        }

        // Clear old perturbations, apply new ones.
        for (let i = 0; i < this.data.qfrc_applied.length; i++) {
          this.data.qfrc_applied[i] = 0.0;
        }
        let dragged = this.dragStateManager.physicsObject;
        if (dragged && dragged.bodyID) {
          for (let b = 0; b < this.model.nbody; b++) {
            if (this.bodies[b]) {
              getPosition(this.data.xpos, b, this.bodies[b].position);
              getQuaternion(this.data.xquat, b, this.bodies[b].quaternion);
              this.bodies[b].updateWorldMatrix();
            }
          }
          let bodyID = dragged.bodyID;
          this.dragStateManager.update(); // Update the world-space force origin
          let force = toMujocoPos(
            this.dragStateManager.currentWorld
              .clone()
              .sub(this.dragStateManager.worldHit)
              .multiplyScalar(this.model.body_mass[bodyID] * 250),
          );
          let point = toMujocoPos(this.dragStateManager.worldHit.clone());
          mujoco.mj_applyFT(
            this.model,
            this.data,
            [force.x, force.y, force.z],
            [0, 0, 0],
            [point.x, point.y, point.z],
            bodyID,
            this.data.qfrc_applied,
          );

          // TODO: Apply pose perturbations (mocap bodies only).
        }

        mujoco.mj_step(this.model, this.data);

        this.mujoco_time += timestep * 1000.0;
      }
    } else if (this.params["paused"]) {
      this.dragStateManager.update(); // Update the world-space force origin
      let dragged = this.dragStateManager.physicsObject;
      if (dragged && dragged.bodyID) {
        let b = dragged.bodyID;
        getPosition(this.data.xpos, b, this.tmpVec, false); // Get raw coordinate from MuJoCo
        getQuaternion(this.data.xquat, b, this.tmpQuat, false); // Get raw coordinate from MuJoCo

        let offset = toMujocoPos(
          this.dragStateManager.currentWorld
            .clone()
            .sub(this.dragStateManager.worldHit)
            .multiplyScalar(0.3),
        );
        if (this.model.body_mocapid[b] >= 0) {
          // Set the root body's mocap position...
          console.log("Trying to move mocap body", b);
          let addr = this.model.body_mocapid[b] * 3;
          let pos = this.data.mocap_pos;
          pos[addr + 0] += offset.x;
          pos[addr + 1] += offset.y;
          pos[addr + 2] += offset.z;
        } else {
          // Set the root body's position directly...
          let root = this.model.body_rootid[b];
          let addr = this.model.jnt_qposadr[this.model.body_jntadr[root]];
          let pos = this.data.qpos;
          pos[addr + 0] += offset.x;
          pos[addr + 1] += offset.y;
          pos[addr + 2] += offset.z;
        }
      }

      mujoco.mj_forward(this.model, this.data);
    }

    // Update body transforms.
    for (let b = 0; b < this.model.nbody; b++) {
      if (this.bodies[b]) {
        getPosition(this.data.xpos, b, this.bodies[b].position);
        getQuaternion(this.data.xquat, b, this.bodies[b].quaternion);
        this.bodies[b].updateWorldMatrix();
      }
    }

    // Update light transforms.
    for (let l = 0; l < this.model.nlight; l++) {
      if (this.lights[l]) {
        getPosition(this.data.light_xpos, l, this.lights[l].position);
        getPosition(this.data.light_xdir, l, this.tmpVec);
        this.lights[l].lookAt(this.tmpVec.add(this.lights[l].position));
      }
    }

    // Draw Tendons and Flex verts
    drawTendonsAndFlex(this.mujocoRoot, this.model, this.data);

    // Update the live eval/reward HUD.
    if (this.evalOverlay) this.evalOverlay.update();

    // Render!
    this.renderer.render(this.scene, this.camera);
  }
}

let demo = new MuJoCoDemo();
await demo.init();
// Expose for an external (future live-HUD WebSocket) driver to select tasks.
window.demo = demo;
window.setTask = (task) => demo.setTask(task);
