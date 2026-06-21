// Reusable Gaussian-splat backdrop: add the room splat to ANY three.js scene as
// render-only pixels behind the MuJoCo geometry. The splat is the *look* of the
// room; MuJoCo owns all collisions (the splat is never in the physics world), so
// the two layers can't interfere by construction.
//
// Extracted from splat.js so the lightweight playback viewer AND the full
// physics/task viewer (main.js) share one implementation + one align transform
// (splatConfig.js). Used by:
//   splat.js                       always-on, align tool enabled
//   main.js  (?splat URL flag)     opt-in behind the live physics arm

import { SPLAT_URL, SPLAT_XFORM } from "./splatConfig.js";

// @mkkellogg/gaussian-splats-3d is imported DYNAMICALLY inside attachSplatBackdrop
// (not at module top). That keeps the splat lib out of the static import graph of
// any viewer that merely imports this module — main.js loads fine even when its
// page's importmap has no gaussian-splats-3d entry; only ?splat actually pulls it.

/**
 * @param {THREE.Scene} scene  scene to add the splat to
 * @param {object} opts
 *   url      splat .ply/.splat/.ksplat (default SPLAT_URL)
 *   xform    {position, rotationXdeg, rotationYdeg, scale} (default SPLAT_XFORM)
 *   align    register the WASD/QE/ZX/T-G/Z-X nudge + P-print tool (default false)
 *   statusEl optional element whose textContent reports load state
 * @returns {Promise<GaussianSplats3D.DropInViewer|null>} null if the splat is missing
 */
export async function attachSplatBackdrop(
  scene,
  { url = SPLAT_URL, xform = SPLAT_XFORM, align = false, statusEl = null } = {},
) {
  let viewer = null;
  try {
    const GaussianSplats3D = await import("@mkkellogg/gaussian-splats-3d");
    viewer = new GaussianSplats3D.DropInViewer({
      gpuAcceleratedSort: false, // pair with sharedMemoryForWorkers:false (no COOP/COEP)
      sharedMemoryForWorkers: false, // python http.server sends no COOP/COEP headers
    });
    await viewer.addSplatScene(url, {
      splatAlphaRemovalThreshold: 5,
      showLoadingUI: false,
      position: xform.position,
      rotation: [0, 0, 0, 1],
      scale: [xform.scale, xform.scale, xform.scale],
    });
    viewer.rotation.set(
      (xform.rotationXdeg * Math.PI) / 180,
      (xform.rotationYdeg * Math.PI) / 180,
      0,
    );
    scene.add(viewer);
    if (statusEl) {
      statusEl.textContent = align
        ? "room splat loaded — align with keyboard, P to print transform"
        : "room splat loaded";
    }
  } catch (e) {
    if (statusEl) {
      statusEl.textContent =
        "no room.ply yet — train it: scripts/recon/modal_gsplat.py, then cp to assets/scenes/";
    }
    console.warn("[splatBackdrop] splat load failed:", e);
    return null;
  }
  if (align) registerAlignTool(viewer);
  return viewer;
}

// Keyboard align tool — nudge the splat into the metric frame, press P to print a
// SPLAT_XFORM you can paste into splatConfig.js. Mouse stays with OrbitControls.
function registerAlignTool(viewer, step = 0.05) {
  window.addEventListener("keydown", (ev) => {
    const p = viewer.position;
    switch (ev.key.toLowerCase()) {
      case "w":
        p.z -= step;
        break;
      case "s":
        p.z += step;
        break;
      case "a":
        p.x -= step;
        break;
      case "d":
        p.x += step;
        break;
      case "r":
        p.y += step;
        break;
      case "f":
        p.y -= step;
        break;
      case "q":
        viewer.rotation.y += 0.05;
        break;
      case "e":
        viewer.rotation.y -= 0.05;
        break;
      case "t":
        viewer.rotation.x += 0.05;
        break;
      case "g":
        viewer.rotation.x -= 0.05;
        break;
      case "z":
        viewer.scale.multiplyScalar(1.03);
        break;
      case "x":
        viewer.scale.multiplyScalar(1 / 1.03);
        break;
      case "p":
        console.log(
          "SPLAT_XFORM = {",
          `position: [${p.x.toFixed(3)}, ${p.y.toFixed(3)}, ${p.z.toFixed(3)}],`,
          `rotationXdeg: ${((viewer.rotation.x * 180) / Math.PI).toFixed(1)},`,
          `rotationYdeg: ${((viewer.rotation.y * 180) / Math.PI).toFixed(1)},`,
          `scale: ${viewer.scale.x.toFixed(3)} };`,
        );
        break;
    }
  });
}
