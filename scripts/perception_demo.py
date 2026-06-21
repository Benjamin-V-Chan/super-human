"""Demo: run the Gemma perception pipeline on a clip and print the ProblemSpec.

Usage:
    python scripts/perception_demo.py [path/to/clip.mp4]

With GOOGLE_API_KEY (or GEMINI_API_KEY) set, this runs real Gemma video
analysis. Without a key it falls back to a deterministic detection so the
pipeline still produces a valid ProblemSpec.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict

from prosthesis_rl.agents.perception import PerceptionAgent
from prosthesis_rl.cv.gemma import GemmaVideoAnalyzer


def main() -> None:
    clip = sys.argv[1] if len(sys.argv) > 1 else "examples/adl/reach_1_1.mp4"
    analyzer = GemmaVideoAnalyzer()
    print(f"clip: {clip}")
    print(f"gemma available: {analyzer.available} (model={analyzer.model})")

    spec = PerceptionAgent().infer_problem(clip)
    print(json.dumps(asdict(spec), indent=2))


if __name__ == "__main__":
    main()
