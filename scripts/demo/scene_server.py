"""Live Gizmo scene generation for the web demo — the backend the browser talks to.

Serves webdemo/ and adds three endpoints so the viewer can drop the arm into a
*generated* environment instead of a Gaussian-splat backdrop:

    POST /generate-scene  {"action": "drink from a water bottle"}
        -> kick off (in a worker thread) the pipeline:
             action -> ScenarioAgent -> scene_prompt -> Gizmo bake (CACHED) ->
             slim self-contained MJCF -> splice the arm + authored waypoints in
           Returns immediately; the key never leaves the server.

    GET  /scene-status
        -> {status, stage, ready, scene_url, cached, error} for a progress UI.
           `stage` streams the Gizmo pipeline (submitting -> running -> exporting
           -> unpacking -> ready). First request for an action pays the bake
           (minutes); every repeat is an instant cache hit.

    GET  /scene-showcase
        -> publish a pre-baked scene from the cache instantly (no API call) so the
           demo always has something to show without waiting on a cold bake.

The merged scene is written to webdemo/assets/live/gizmo_scene.xml (self-contained
— inline meshes, no textures), which the browser fetches and loads into the WASM
viewer. Arm meshes are the existing arm_links/*.stl the viewer already serves.

    export GIZMO_API_KEY=sk-...            # or put it in .env (never committed)
    python3 scripts/demo/scene_server.py --port 8011
    # open http://localhost:8011/live.html  ->  "Generate scene"
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

WEBDEMO = ROOT / "webdemo"
LIVE = WEBDEMO / "assets" / "live"
ARM_XML = WEBDEMO / "assets" / "scenes" / "arm_articulated.xml"
ARM_MESH_DIR = WEBDEMO / "assets" / "scenes" / "arm_links"
PUBLISHED = LIVE / "gizmo_scene.xml"            # what the browser fetches
SCENE_URL = "./assets/live/gizmo_scene.xml"
# What the showcase demonstrates: a point-A -> point-B task in a chosen Gizmo scene.
SHOWCASE_ACTION = "open the drawer"
SHOWCASE_SCENE = (ROOT / "assets" / "scenes" / "gizmo" / "_drawer_cabinet"
                  / "js7dv7ry4ttbm1ccx8nvsbjms9892893.xml")

from prosthesis_rl.agents.scenario import ScenarioAgent, scene_prompt  # noqa: E402
from prosthesis_rl.sim import gizmo_scene as gs                        # noqa: E402
from prosthesis_rl.sim.gizmo_asset import inject_objects, waypoint_markers  # noqa: E402
from prosthesis_rl.sim.gizmo_scene_merge import (                      # noqa: E402
    inject_gizmo_scene, interactable_pos, spawn_site_pos)

# Single-job state (the viewer shows one scene at a time). Guarded by _lock.
_lock = threading.Lock()
_JOB = {"status": "idle", "stage": "", "ready": False, "cached": False,
        "action": "", "scene_url": "", "error": ""}


def _set(**kw) -> None:
    with _lock:
        _JOB.update(kw)


def _snapshot() -> dict:
    with _lock:
        return dict(_JOB)


def _align_offset(scene_mjcf: Path, scenario) -> tuple[float, float, float]:
    """Translate the scene so the task lines up with the arm (x/y only; floor kept).

    Prefer Gizmo's `*Robot_Spawn` site (line it up under the arm's shoulder); if the
    scene has none, line up the main interactable object with where the hand reaches
    (the primary waypoint) so the arm sits at the task. Z is never shifted so the
    scene stays on its floor."""
    mx, my, _ = scenario.mount_pos
    spawn = spawn_site_pos(scene_mjcf)
    if spawn:
        return (mx - spawn[0], my - spawn[1], 0.0)
    anchor = interactable_pos(scene_mjcf)
    if anchor:
        px, py, _ = scenario.primary_waypoint().pos
        return (px - anchor[0], py - anchor[1], 0.0)
    return (0.0, 0.0, 0.0)


def _publish(scene_mjcf: Path, scenario) -> None:
    """Splice the arm, the task objects, and the A/B waypoint markers into the scene
    and write the merged MJCF for the browser.

    The Gizmo geometry is the environment; the scenario's own objects (e.g. the
    phone + charger) are added as labelled props at the task points, and every
    waypoint becomes a green marker so the point-A -> point-B task reads clearly.
    """
    arm_xml = ARM_XML.read_text()
    offset = _align_offset(scene_mjcf, scenario)
    merged = inject_gizmo_scene(arm_xml, scene_mjcf, offset=offset)
    # The Gizmo scene already provides the task geometry (e.g. the cabinet/drawer);
    # we just mark the A/B reach points so the task reads clearly.
    merged = merged.replace(
        "</worldbody>", waypoint_markers(scenario.waypoints) + "\n</worldbody>", 1)
    LIVE.mkdir(parents=True, exist_ok=True)
    tmp = PUBLISHED.with_suffix(".xml.tmp")
    tmp.write_text(merged)
    tmp.replace(PUBLISHED)  # atomic — the browser never reads a half-written file
    _write_trajectory(scenario)


def _write_trajectory(scenario, *, fps: int = 30, seconds_per_leg: float = 2.2,
                      hold_s: float = 0.5) -> None:
    """Script the arm hand through the scenario's waypoints (rest -> A -> B -> …) and
    write trajectory.json, so the demo *shows the motion* without a trained policy.

    Each waypoint is snapped onto the arm's reachable manifold (`nearest_reachable`
    returns the joint config that gets closest, body-collision-free); the hand then
    interpolates joint-space between those configs with a smooth ease and dwells at
    each point — a clean "go to the charger, carry it to the phone, hold" motion.
    Frames are the arm joints only (the merged model lists them first, so the web
    viewer drives the arm and leaves the scene props at rest)."""
    import numpy as np
    import mujoco
    from prosthesis_rl.contracts import DesignParams
    from prosthesis_rl.sim.mjcf_builder import build_mjcf
    from prosthesis_rl.sim.control import nearest_reachable

    design = DesignParams()
    primary = scenario.primary_waypoint()
    xml = build_mjcf(design, mount_pos=tuple(scenario.mount_pos),
                     target_pos=tuple(primary.pos), mesh_dir=str(ARM_MESH_DIR),
                     add_human=True)
    model = mujoco.MjModel.from_xml_string(xml, {})
    n_dof = len(design.joint_names)

    # Reachable joint config for each waypoint (residual = how close it can get).
    configs, residual_last = [], 0.0
    for wp in scenario.waypoints:
        _, q, res = nearest_reachable(model, design, np.asarray(wp.pos, dtype=float))
        configs.append(np.asarray(q, dtype=float))
        residual_last = res

    seq = [np.zeros(n_dof)] + configs            # start from the neutral pose
    n_leg, n_hold = int(seconds_per_leg * fps), int(hold_s * fps)
    smooth = lambda t: t * t * (3.0 - 2.0 * t)   # ease in/out
    frames: list[list[float]] = []
    for a, b in zip(seq, seq[1:]):
        for k in range(1, n_leg + 1):
            frames.append([float(x) for x in (a + (b - a) * smooth(k / n_leg))])
        frames.extend([[float(x) for x in b]] * n_hold)  # dwell (grasp / insert)

    final_cm = residual_last * 100.0
    traj = {"dt": 1.0 / fps, "fps": fps, "nq": n_dof, "step": int(time.time()),
            "success": bool(final_cm < 4.0), "final_cm": final_cm, "frames": frames}
    LIVE.mkdir(parents=True, exist_ok=True)
    tmp = LIVE / "trajectory.json.tmp"
    tmp.write_text(json.dumps(traj))
    tmp.replace(LIVE / "trajectory.json")


def _worker(action: str) -> None:
    """Run the full action -> Gizmo scene -> merged MJCF pipeline (off the request thread)."""
    try:
        scenario = ScenarioAgent(use_llm=False).for_action(action)
        prompt = scene_prompt(scenario)
        _set(status="running", stage="checking cache", action=action,
             ready=False, error="", scene_url="")
        hit = gs.cached(prompt)
        bake = hit or gs.bake_scene(prompt, on_status=lambda s: _set(stage=s))
        _set(stage="splicing arm into scene", cached=bool(hit))
        _publish(bake.mjcf, scenario)
        _set(status="ready", stage="ready", ready=True, scene_url=SCENE_URL)
    except Exception as e:  # noqa: BLE001 — surface any failure to the UI, keep server up
        _set(status="error", stage="", ready=False, error=f"{type(e).__name__}: {e}")


def _start_job(action: str) -> dict:
    with _lock:
        if _JOB["status"] == "running":
            return {"ok": False, "busy": True, "stage": _JOB["stage"]}
        _JOB.update(status="running", stage="starting", action=action,
                    ready=False, error="", scene_url="")
    threading.Thread(target=_worker, args=(action,), daemon=True).start()
    return {"ok": True, "started": True, "action": action}


def _showcase() -> dict:
    """Publish the showcase scene instantly (no API call): the chosen Gizmo scene as
    the environment, with the SHOWCASE_ACTION point-A -> point-B task overlaid."""
    scene_mjcf = SHOWCASE_SCENE if SHOWCASE_SCENE.is_file() else None
    if scene_mjcf is None:  # fall back to the first valid cached scene
        if gs.CACHE_ROOT.is_dir():
            for d in sorted(gs.CACHE_ROOT.iterdir()):
                b = gs._read_bake(d)
                if b:
                    scene_mjcf = b.mjcf
                    break
    if scene_mjcf is None:
        return {"ok": False, "error": "no cached scene — run a /generate-scene first"}
    scenario = ScenarioAgent(use_llm=False).for_action(SHOWCASE_ACTION)
    _publish(scene_mjcf, scenario)
    _set(status="ready", stage="ready (showcase)", ready=True,
         cached=True, action=SHOWCASE_ACTION, scene_url=SCENE_URL, error="")
    return {"ok": True, "scene_url": SCENE_URL, "prompt": SHOWCASE_ACTION}


def serve(port: int) -> None:
    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a):  # quiet console
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            route = self.path.split("?")[0]
            if route == "/scene-status":
                return self._json(_snapshot())
            if route == "/scene-showcase":
                return self._json(_showcase())
            return super().do_GET()

        def do_POST(self):
            route = self.path.split("?")[0]
            if route == "/generate-scene":
                n = int(self.headers.get("Content-Length", 0))
                try:
                    payload = json.loads(self.rfile.read(n) or b"{}")
                except json.JSONDecodeError:
                    return self._json({"ok": False, "error": "bad JSON"}, 400)
                action = (payload.get("action") or "").strip()
                if not action:
                    return self._json({"ok": False, "error": "action is required"}, 400)
                return self._json(_start_job(action))
            self.send_error(404)

    handler = functools.partial(Handler, directory=str(WEBDEMO))
    httpd = http.server.ThreadingHTTPServer(("", port), handler)
    print(f"[scene-server] serving {WEBDEMO} on http://localhost:{port}/live.html", flush=True)
    print("[scene-server] POST /generate-scene {action}  |  GET /scene-status  |  GET /scene-showcase",
          flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[scene-server] bye")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8011)
    ap.add_argument("--check-key", action="store_true",
                    help="verify the Gizmo key (GET /v1/whoami) and exit")
    args = ap.parse_args()
    if args.check_key:
        print(json.dumps(gs.whoami(), indent=2))
        return 0
    if not ARM_XML.exists():
        print(f"[scene-server] WARNING: {ARM_XML} missing — run train_live.py once to "
              f"export the arm scene, or the merge will fail.", flush=True)
    serve(args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
