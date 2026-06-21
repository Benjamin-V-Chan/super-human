// Single source of truth for the room Gaussian-splat backdrop and its alignment
// into the arm's metric (meters) world.
//
// The splat (assets/scenes/room.ply) is render-only pixels — it never has
// physics. Monocular COLMAP/3DGS is up-to-scale in an arbitrary frame, so it has
// to be aligned to the metric world ONCE by hand: open live.html (the splat align
// tool is on), nudge with WASD/QE/ZX, press P, and paste the printed SPLAT_XFORM
// below. The viewer then picks up the transform — align once, applies everywhere.

export const SPLAT_URL = "./assets/scenes/room.ply";

export const SPLAT_XFORM = {
  position: [0, 0, 0],
  rotationXdeg: 180, // flip the COLMAP/3DGS Y-down frame right-side up
  rotationYdeg: 0,
  scale: 1.0,
};
