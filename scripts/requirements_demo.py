"""Perception -> problem deliverables JSON -> requirements agent -> CadGPT brief.

Usage:
    python scripts/requirements_demo.py [path/to/clip]
"""

from __future__ import annotations

import json
import sys

from prosthesis_rl.agents.perception import PerceptionAgent
from prosthesis_rl.agents.requirements import RequirementsAgent, problem_deliverables


def main() -> None:
    clip = sys.argv[1] if len(sys.argv) > 1 else "test_vids/IMG_9847 (1) (1).mov"

    spec = PerceptionAgent().infer_problem(clip)
    deliverables = problem_deliverables(spec)
    print("── PROBLEM DELIVERABLES (perception → requirements) ──")
    print(json.dumps(deliverables, indent=2))

    agent = RequirementsAgent()
    print(f"\nrequirements agent available: {agent.available} (model={agent.model})")
    brief = agent.derive(spec)
    print("\n── CADGPT BRIEF (requirements → Benji's CAD model) ──")
    print(json.dumps(brief, indent=2))


if __name__ == "__main__":
    main()
