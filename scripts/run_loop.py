from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prosthesis_rl.agents import ProsthesisLoop


def main() -> None:
    clips = sys.argv[1:] or ["examples/adl/reach_1_1.mp4"]
    if len(clips) == 1:
        result = ProsthesisLoop().run(clips[0])
    else:
        result = ProsthesisLoop().run_multi(clips)

    if hasattr(result, "to_json"):
        print(result.to_json(indent=2))
    elif isinstance(result, dict):
        # run_multi returns a dict — print key stats
        stats = result.get("stats", {})
        print(json.dumps(stats, indent=2))
    else:
        print(result)


if __name__ == "__main__":
    main()
