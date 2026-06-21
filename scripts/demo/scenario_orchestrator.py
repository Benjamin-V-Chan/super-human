"""Drop the arm into a real ADL scenario and prove it does the task.

This replaces the "reach a random floating point" demo. Pipeline:

    action / task  ->  ScenarioAgent  ->  ScenarioSpec (posture + objects + waypoints)
                                      ->  ScenarioReachEnv (scene + task goals)
                                      ->  scripted-IK rollout through the waypoints

It prints the chosen posture, the scene objects (with their Gizmo prompts), the
reach waypoints, and how close a scripted controller gets to each — so you can see
the arm crouch to the laces or lean to the bottle instead of poking a dot.

    # list the built-in ADL scenarios
    python3 scripts/demo/scenario_orchestrator.py --list

    # run one by free-text action (deterministic library; no key needed)
    python3 scripts/demo/scenario_orchestrator.py --action "tie my shoe"
    python3 scripts/demo/scenario_orchestrator.py --action "drink from a water bottle"

    # ask the LLM to invent a scene for a novel action (needs a valid OPENAI/GEMINI key)
    python3 scripts/demo/scenario_orchestrator.py --action "water a plant on a high shelf" --llm

    # bake the scene objects as real articulated Gizmo MJCF first (slow), then run
    python3 scripts/demo/scenario_orchestrator.py --action "tie my shoe" --bake-gizmo

Writes the resolved scenario JSON to assets/scenes/scenario_<task>.json for the
viewer / live trainer to pick up.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ARM_MESH_DIR = ROOT / "webdemo" / "assets" / "scenes" / "arm_links"
SCENES_OUT = ROOT / "assets" / "scenes"


def _load_dotenv() -> None:
    env = ROOT / ".env"
    if not env.is_file():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def bake_gizmo_objects(scenario) -> None:
    """Generate each scene object as a real articulated MJCF via the Gizmo API.

    Slow (Gizmo can sit on stage 1 for 10-18 min); off by default. On success the
    object's mjcf_dir is repointed at the bake so the env injects real geometry.
    """
    script = ROOT / "scripts" / "recon" / "gizmo_assets.py"
    for obj in scenario.objects:
        out_dir = ROOT / "assets" / "objects" / obj.name
        if list(out_dir.rglob("*.xml")) or list(out_dir.rglob("*.mjcf")):
            print(f"[bake] {obj.name}: cached -> {out_dir}")
            obj.mjcf_dir = str(out_dir)
            continue
        print(f"[bake] {obj.name}: generating via Gizmo — {obj.prompt!r}")
        rc = subprocess.call([sys.executable, str(script),
                              "--prompt", obj.prompt, "--name", obj.name])
        if rc == 0 and (list(out_dir.rglob("*.xml")) or list(out_dir.rglob("*.mjcf"))):
            obj.mjcf_dir = str(out_dir)
            print(f"[bake] {obj.name}: ready -> {out_dir}")
        else:
            print(f"[bake] {obj.name}: bake failed/timed out — using fallback box")


def scripted_rollout(scenario, *, seconds_per_wp: float = 2.0, fps: int = 30):
    """Drive the arm through the scenario waypoints and measure how close it gets.

    Each waypoint was pre-snapped to a reachable joint config; the position servo
    is commanded toward that config (so the arm visibly travels to the task point)
    and we report the residual at the snapped, reachable target.

    Returns a list of (waypoint_name, final_distance_m, hit) and the qpos frames.
    """
    import mujoco
    import numpy as np

    from prosthesis_rl.contracts import DesignParams
    from prosthesis_rl.rl.scenario_env import ScenarioReachEnv
    from prosthesis_rl.sim.control import REACH_SUCCESS_M
    from prosthesis_rl.sim.mjcf_builder import EE_SITE

    import copy

    design = DesignParams()
    mesh_dir = ARM_MESH_DIR if ARM_MESH_DIR.exists() else None
    # The reach proof uses non-colliding fallback boxes for the scene: the arm
    # must reach the task *point*, not bump a solid object (real Gizmo meshes are
    # collidable — that's for a later manipulation stage). Real objects are only
    # used for the hero render.
    box_scn = copy.deepcopy(scenario)
    for o in box_scn.objects:
        o.mjcf_dir = ""
    env = ScenarioReachEnv(box_scn, design, mesh_dir=mesh_dir, add_markers=True)
    model, data = env.model, env.data
    ee_id = model.site(EE_SITE).id

    names = [w.name for w in scenario.waypoints]
    targets = env.waypoint_targets
    configs = env.waypoint_configs
    ndof = design.dof
    dt = model.opt.timestep
    substeps = max(1, round((1.0 / fps) / dt))

    mujoco.mj_resetData(model, data)
    q_cmd = data.qpos[: ndof].copy()
    results, frames = [], []
    for name, tgt, q_goal in zip(names, targets, configs):
        for _ in range(int(seconds_per_wp * fps)):
            q_cmd += np.clip(q_goal - q_cmd, -0.05, 0.05)   # ease toward the config
            data.ctrl[: ndof] = q_cmd
            for _ in range(substeps):
                mujoco.mj_step(model, data)
            frames.append([float(x) for x in data.qpos[: model.nq]])
        d = float(np.linalg.norm(np.array(data.site_xpos[ee_id]) - tgt))
        results.append((name, d, d <= REACH_SUCCESS_M))
    return results, frames, env


def use_real_objects(scenario) -> int:
    """Point each object at its baked Gizmo MJCF when one exists. Returns the count."""
    n = 0
    for obj in scenario.objects:
        d = ROOT / "assets" / "objects" / obj.name
        if list(d.rglob("*.xml")) or list(d.rglob("*.mjcf")):
            obj.mjcf_dir = str(d)
            n += 1
    return n


def render_scene(scenario, out_png: Path) -> bool:
    """Render a hero frame of the arm at the task pose, with the real scene objects."""
    import imageio.v2 as imageio
    import mujoco
    import numpy as np

    from prosthesis_rl.contracts import DesignParams
    from prosthesis_rl.rl.scenario_env import ScenarioReachEnv
    from prosthesis_rl.sim.gizmo_asset import build_scenario_xml

    design = DesignParams()
    mesh_dir = ARM_MESH_DIR if ARM_MESH_DIR.exists() else None
    # Snapped joint config for the task-completing waypoint (fast box env).
    import copy
    box_scn = copy.deepcopy(scenario)
    for o in box_scn.objects:
        o.mjcf_dir = ""
    env = ScenarioReachEnv(box_scn, design, mesh_dir=mesh_dir, snap_samples=3000)
    idx = scenario.waypoints.index(scenario.primary_waypoint())
    q_goal, target = env.waypoint_configs[idx], env.waypoint_targets[idx]

    # Render model: real objects merged in (slow build, one-off).
    model = mujoco.MjModel.from_xml_string(
        build_scenario_xml(design, scenario, mesh_dir=mesh_dir), {})
    data = mujoco.MjData(model)
    data.qpos[: design.dof] = q_goal
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=720, width=1280)
    cam = mujoco.MjvCamera()
    # Side-on profile: reads as the body in a posture reaching toward the task
    # location (a crouch/lean), the whole point vs. poking a floating dot.
    mount = np.asarray(scenario.mount_pos)
    cam.lookat[:] = 0.45 * mount + 0.55 * np.asarray(target) + np.array([0.0, 0.0, 0.05])
    cam.distance, cam.elevation, cam.azimuth = 1.35, -8, 180
    renderer.update_scene(data, camera=cam)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(out_png, renderer.render())
    return True


def main() -> int:
    _load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--action", help="free-text action, e.g. 'tie my shoe'")
    ap.add_argument("--task-id", default="", help="optional ADL task id")
    ap.add_argument("--llm", action="store_true", help="prefer the LLM scene designer")
    ap.add_argument("--bake-gizmo", action="store_true",
                    help="generate scene objects as real Gizmo MJCF first (slow)")
    ap.add_argument("--real-objects", action="store_true",
                    help="use already-baked Gizmo MJCF for objects (assets/objects/<name>)")
    ap.add_argument("--render", action="store_true",
                    help="render a hero PNG of the arm at the task pose")
    ap.add_argument("--no-rollout", action="store_true", help="just resolve the scenario")
    ap.add_argument("--list", action="store_true", help="list built-in ADL scenarios")
    args = ap.parse_args()

    from prosthesis_rl.agents.scenario import ScenarioAgent, library_keys

    if args.list:
        print("Built-in ADL scenarios (match by keyword in --action):")
        for k in library_keys():
            print(f"  - {k}")
        return 0

    if not args.action:
        ap.error("--action is required (or use --list)")

    agent = ScenarioAgent()
    if args.llm and not agent.llm_available:
        print("[warn] --llm set but no OPENAI/GEMINI key found; using the library.")
    scenario = agent.for_action(args.action, task_id=args.task_id, prefer_llm=args.llm)

    problems = scenario.validate()
    if problems:
        print("[error] invalid scenario:", problems)
        return 1

    print("=" * 64)
    print(f"ACTION:   {args.action}")
    print(f"SCENARIO: {scenario.task_id}   (source: {scenario.source})")
    print(f"POSTURE:  {scenario.posture}   mount={tuple(round(x,2) for x in scenario.mount_pos)}")
    print(f"WHAT:     {scenario.description}")
    print(f"SUCCESS:  {scenario.success_condition}")
    print("OBJECTS:")
    for o in scenario.objects:
        print(f"  - {o.name:10s} @ {tuple(round(x,2) for x in o.pos)}  «{o.prompt}»")
    print("WAYPOINTS (the hand must reach these):")
    for w in scenario.waypoints:
        print(f"  - {w.name:10s} -> {tuple(round(x,2) for x in w.pos)}  "
              f"(w={w.weight}, tol={w.tolerance_m}m)")
    print("=" * 64)

    if args.bake_gizmo:
        bake_gizmo_objects(scenario)
    elif args.real_objects:
        n = use_real_objects(scenario)
        print(f"[real] using {n} baked Gizmo object(s)" if n else
              "[real] no baked objects found — using fallback boxes")

    SCENES_OUT.mkdir(parents=True, exist_ok=True)
    out_json = SCENES_OUT / f"scenario_{scenario.task_id or 'task'}.json"
    out_json.write_text(scenario.to_json(indent=2))
    print(f"[saved] {out_json.relative_to(ROOT)}")

    if args.no_rollout:
        return 0

    try:
        import mujoco  # noqa: F401
    except ImportError:
        print("[skip] mujoco not installed — scenario resolved but no rollout.")
        return 0

    print("\nRunning scripted-IK rollout through the waypoints…")
    results, frames, env = scripted_rollout(scenario)
    print(f"  arm reach budget: {env.reach*100:.0f} cm   frames: {len(frames)}")
    all_hit = True
    for name, dist, hit in results:
        print(f"  {name:10s}: final {dist*100:5.1f} cm   {'HIT ✅' if hit else 'miss ❌'}")
        all_hit = all_hit and hit
    print(f"\nTASK {'COMPLETE ✅ — arm reached every task point' if all_hit else 'PARTIAL — some points missed'}")

    if args.render:
        png = SCENES_OUT / f"scenario_{scenario.task_id or 'task'}.png"
        print(f"\nRendering hero frame -> {png.relative_to(ROOT)} …")
        try:
            render_scene(scenario, png)
            print(f"[render] saved {png.relative_to(ROOT)}")
        except Exception as exc:  # noqa: BLE001 - rendering is optional
            print(f"[render] skipped: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
