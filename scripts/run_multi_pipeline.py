"""Multi-clip pipeline runner — spawned by viewer/server.js.

Reads JSON config from stdin, streams newline-delimited JSON events to stdout.
Each event is a PipelineEvent dict the SSE layer forwards to the browser.

Usage (from server.js):
    echo '{"clip_paths": ["test_vids/a.mp4", "test_vids/b.mp4"]}' | python3 scripts/run_multi_pipeline.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prosthesis_rl.pipeline.events import Emitter, PipelineEvent


class _StdoutEmitter(Emitter):
    """Writes each PipelineEvent as a newline-delimited JSON line to stdout."""

    def emit(self, event: PipelineEvent) -> None:
        try:
            line = json.dumps({"type": event.type, "stage": event.stage, "data": event.data})
            sys.stdout.write(line + "\n")
            sys.stdout.flush()
        except Exception:
            pass

    def close(self) -> None:
        sys.stdout.flush()


def main() -> None:
    raw = sys.stdin.read()
    try:
        config = json.loads(raw)
    except json.JSONDecodeError as e:
        sys.stdout.write(json.dumps({"type": "error", "stage": "startup", "message": f"Invalid JSON: {e}"}) + "\n")
        sys.exit(1)

    clip_paths = config.get("clip_paths") or []
    quick_mode = bool(config.get("quick_mode", False))

    if not clip_paths:
        sys.stdout.write(json.dumps({"type": "error", "stage": "startup", "message": "clip_paths is empty"}) + "\n")
        sys.exit(1)

    from prosthesis_rl.pipeline.loop import DesignOptimizationLoop

    emit = _StdoutEmitter()
    loop = DesignOptimizationLoop(quick_mode=quick_mode)

    try:
        loop.run_multi(clip_paths, emit)
    except Exception as exc:
        sys.stdout.write(json.dumps({"type": "error", "stage": "pipeline", "message": str(exc)}) + "\n")
        sys.stdout.flush()
        sys.exit(1)


if __name__ == "__main__":
    main()
