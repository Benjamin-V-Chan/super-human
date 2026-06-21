"""JSON-lines-free CLI bridge for viewer agent endpoints."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prosthesis_rl.app_bridge import (  # noqa: E402
    build_cad,
    build_policy,
    derive_design,
    infer_clip,
)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: app_bridge.py <perception|design|policy|cad>")
    operation = sys.argv[1]
    payload = json.load(sys.stdin)
    if operation == "perception":
        result = infer_clip(payload["clip_path"])
    elif operation == "design":
        result = derive_design(payload.get("problem") or {})
    elif operation == "policy":
        result = build_policy(
            payload.get("problem") or {}, payload.get("design") or {}, payload.get("name") or "policy"
        )
    elif operation == "cad":
        result = build_cad(payload.get("design") or {}, payload.get("name") or "candidate")
    else:
        raise ValueError(f"unsupported operation: {operation}")
    sys.stdout.write(json.dumps(result))


if __name__ == "__main__":
    main()
