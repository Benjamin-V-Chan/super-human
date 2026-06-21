"""Durability dashboard backend: stress-test the arm, then serve live what-if + fixes.

This is the third dashboard. Where the live trainer shows the policy *learning*, this
one shows whether the resulting hardware *survives* — and what to change if it
doesn't. It runs the trained policy through the full ADL battery once (the expensive
part: a real MuJoCo rollout per task), captures the per-joint stress each task puts
on the arm, and writes one JSON the browser turns into:

  * a per-task table (success + predicted service life + fatigue margin),
  * lifespan / stress charts across the battery,
  * a ranked, *quantified* list of design fixes (thicken the joint, round the fillet,
    switch material, gentler control) — each with a proven before→after lifespan.

The fatigue model is cheap closed-form, so the page recomputes everything live as you
drag the material / usage / load-factor controls — no re-rollout needed. The battery
is captured at load_factor 1.0; the browser scales from there.

    # train on the battery (slow) then open the dashboard
    python3 scripts/demo/durability_dashboard.py --train --timesteps 300000 --port 8012
    # or stress-test an existing policy
    python3 scripts/demo/durability_dashboard.py --policy scenario_ppo --port 8012
    # then open  http://localhost:8012/live.html  and click the "Durability" tab

`--quick` does a tiny train + coarse snap to smoke-test the whole pipeline fast.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import os
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

WEBDEMO = ROOT / "webdemo"
OUT_DIR = WEBDEMO / "assets" / "durability"
POLICY_DIR = ROOT / "assets" / "policies"


def atomic_write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def serve(directory: Path, port: int) -> None:
    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass

    handler = functools.partial(QuietHandler, directory=str(directory))
    httpd = http.server.ThreadingHTTPServer(("", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()


def build_report(policy_path, *, design, mesh_dir, usage_cycles_per_day, snap_samples):
    """Run the battery at load_factor 1.0 and assemble the dashboard JSON."""
    from prosthesis_rl.fatigue.estimate import JOINT_RADIUS_M, KT_FILLET
    from prosthesis_rl.fatigue.materials import DEFAULT_MATERIAL_KEY, all_materials
    from prosthesis_rl.rl.stress_test import stress_test_battery

    report = stress_test_battery(
        None, policy_path, design=design, mesh_dir=mesh_dir,
        usage_cycles_per_day=usage_cycles_per_day, snap_samples=snap_samples,
        material=DEFAULT_MATERIAL_KEY, load_factor=1.0)

    out = report.to_dict()
    out["policy"] = Path(str(policy_path)).name
    out["material_key"] = DEFAULT_MATERIAL_KEY
    out["kt"] = KT_FILLET
    out["radius_m"] = JOINT_RADIUS_M
    # The whole material DB so the browser can recompute any swap live.
    out["materials"] = [m.to_dict() for m in all_materials()]
    return out, report


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="scenario_ppo",
                    help="policy name under assets/policies/ (or a full path)")
    ap.add_argument("--train", action="store_true",
                    help="train on the ADL battery first, then stress-test")
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--usage-cycles-per-day", type=int, default=300)
    ap.add_argument("--port", type=int, default=8012)
    ap.add_argument("--no-serve", action="store_true", help="write the JSON and exit")
    ap.add_argument("--quick", action="store_true",
                    help="tiny train + coarse snap for a fast end-to-end smoke run")
    args = ap.parse_args()

    try:
        import mujoco  # noqa: F401
        from stable_baselines3 import PPO  # noqa: F401
    except ImportError as exc:
        print(f"[error] needs mujoco + stable-baselines3: {exc}")
        return 1

    from prosthesis_rl.cad.bridge import CadBridge
    from prosthesis_rl.contracts import DesignParams

    design = DesignParams()
    snap_samples = 800 if args.quick else 4000
    mesh_dir = CadBridge().export_arm(design, name="candidate")

    policy_path = (Path(args.policy) if "/" in args.policy else POLICY_DIR / args.policy)

    if args.train:
        from prosthesis_rl.rl.train import train_scenario_policy

        timesteps = 4000 if args.quick else args.timesteps
        print(f"[train] PPO across the ADL battery — {timesteps} steps (slow)…")
        summary = train_scenario_policy(
            None, timesteps=timesteps, name=policy_path.name, design=design,
            mesh_dir=mesh_dir, snap_samples=snap_samples, eval_episodes=0)
        print(f"[train] saved {summary['policy']}")
    elif not (policy_path.with_suffix(".zip").exists() or policy_path.exists()):
        print(f"[error] no policy at {policy_path}.zip — pass --train to make one.")
        return 1

    print(f"[stress-test] running {policy_path.name} through the ADL battery…")
    out, report = build_report(
        policy_path, design=design, mesh_dir=mesh_dir,
        usage_cycles_per_day=args.usage_cycles_per_day, snap_samples=snap_samples)

    print()
    print(report.summary_table())

    atomic_write_json(OUT_DIR / "report.json", out)
    print(f"\n[saved] {(OUT_DIR / 'report.json').relative_to(ROOT)}")

    if args.no_serve:
        return 0
    serve(WEBDEMO, args.port)
    print(f"\n[durability] serving  http://localhost:{args.port}/live.html"
          f"   → click the \"Durability\" tab   (Ctrl-C to stop)")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        print("\n[durability] stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
