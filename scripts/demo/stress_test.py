"""Stress-test a trained arm policy across the ADL battery — lifespan + success.

Runs one trained PPO policy through every built-in daily task, captures the
actuator torque it generates, and reports two things per task: did the arm
*complete* it, and how many years the most-loaded joint would survive at that
load (via the fatigue model). The aggregate calls out the **limiting task** —
the daily action that decides how long the hardware lasts.

    # train one generalist policy on the whole ADL battery, then stress-test it
    python3 scripts/demo/stress_test.py --train --timesteps 300000

    # stress-test an already-trained policy
    python3 scripts/demo/stress_test.py --policy assets/policies/scenario_ppo

    # scope to specific actions and a heavier usage assumption
    python3 scripts/demo/stress_test.py --train --actions "tie my shoe" "drink water" \
        --usage-cycles-per-day 600

    # fast smoke run (tiny training) to check the pipeline end to end
    python3 scripts/demo/stress_test.py --train --quick

Writes the full report JSON to assets/scenes/stress_test_<policy>.json.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

POLICY_DIR = ROOT / "assets" / "policies"
OUT_DIR = ROOT / "assets" / "scenes"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="scenario_ppo",
                    help="policy name under assets/policies/ (or a full path)")
    ap.add_argument("--train", action="store_true",
                    help="train the policy on the battery first, then stress-test")
    ap.add_argument("--actions", nargs="*", default=None,
                    help="scope to these free-text actions (default: full ADL battery)")
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--usage-cycles-per-day", type=int, default=300,
                    help="how many times this ADL is performed per day")
    ap.add_argument("--material", default="pa12_cf_fdm",
                    help="joint material key (see prosthesis_rl.fatigue.materials)")
    ap.add_argument("--load-factor", type=float, default=1.0,
                    help="scale measured torque (payload/duty-margin factor)")
    ap.add_argument("--quick", action="store_true",
                    help="tiny training + coarse snap for a fast end-to-end smoke run")
    args = ap.parse_args()

    try:
        import mujoco  # noqa: F401
        from stable_baselines3 import PPO  # noqa: F401
    except ImportError as exc:
        print(f"[error] needs mujoco + stable-baselines3: {exc}")
        return 1

    from prosthesis_rl.cad.bridge import CadBridge
    from prosthesis_rl.contracts import DesignParams
    from prosthesis_rl.rl.stress_test import stress_test_battery

    design = DesignParams()
    scenarios = args.actions if args.actions else None
    snap_samples = 800 if args.quick else 4000
    # Skin the arm with the real CAD meshes once, and use the SAME skin to train
    # and to stress-test — link inertia sets the torques the fatigue model reads.
    mesh_dir = CadBridge().export_arm(design, name="candidate")

    policy_arg = args.policy
    policy_path = (Path(policy_arg) if "/" in policy_arg
                   else POLICY_DIR / policy_arg)

    if args.train:
        from prosthesis_rl.rl.train import train_scenario_policy

        timesteps = 4000 if args.quick else args.timesteps
        name = policy_path.name
        print(f"[train] PPO across the ADL battery — {timesteps} steps "
              f"(this is the slow part)…")
        summary = train_scenario_policy(
            scenarios, timesteps=timesteps, name=name, design=design,
            mesh_dir=mesh_dir, snap_samples=snap_samples, eval_episodes=0)
        print(f"[train] saved {summary['policy']}  scenarios={summary['scenarios']}")
    elif not (policy_path.with_suffix(".zip").exists() or policy_path.exists()):
        print(f"[error] no policy at {policy_path}.zip — pass --train to make one.")
        return 1

    print(f"\n[stress-test] running {policy_path.name} through the ADL battery…\n")
    report = stress_test_battery(
        scenarios, policy_path, design=design, mesh_dir=mesh_dir,
        usage_cycles_per_day=args.usage_cycles_per_day, snap_samples=snap_samples,
        material=args.material, load_factor=args.load_factor)

    print(report.summary_table())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / f"stress_test_{policy_path.name}.json"
    out_json.write_text(report.to_json())
    print(f"\n[saved] {out_json.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
