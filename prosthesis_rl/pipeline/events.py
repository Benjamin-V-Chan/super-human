"""Thread-safe SSE event emitter for the pipeline loop.

Usage in pipeline stages:
    em = Emitter()
    em.emit(PipelineEvent("stage_start", "perception", {}))
    em.emit(PipelineEvent("stage_done", "perception", {"action": "..."}))

Usage in SSE server:
    for chunk in em.get_stream():       # blocking iterator
        response.write(chunk)           # "data: {...}\\n\\n"
"""

from __future__ import annotations

import json
import queue
import threading
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator

_SENTINEL = object()


@dataclass
class PipelineEvent:
    type: str    # "stage_start"|"stage_done"|"sim_frame"|"rl_step"|"ping"|"error"|"done"
    stage: str   # "perception"|"requirements"|"design"|"cad"|"sim_eval"|"rl_loop"|"final"
    data: dict[str, Any] = field(default_factory=dict)


class Emitter:
    """Thread-safe sink: producer calls emit(), consumer iterates get_stream()."""

    def __init__(self) -> None:
        self._q: queue.Queue[object] = queue.Queue()
        self._closed = threading.Event()

    # ── Producer side ──────────────────────────────────────────────────────────

    def emit(self, event: PipelineEvent) -> None:
        if not self._closed.is_set():
            self._q.put(event)

    def close(self) -> None:
        """Signal the consumer that the stream is finished."""
        self._closed.set()
        self._q.put(_SENTINEL)

    # ── Consumer side ──────────────────────────────────────────────────────────

    def get_stream(self, timeout: float = 30.0) -> Iterator[str]:
        """Yield SSE-formatted strings.  Yields a ping every `timeout` seconds."""
        while True:
            try:
                item = self._q.get(timeout=timeout)
            except queue.Empty:
                yield "data: " + json.dumps({"type": "ping", "stage": "", "data": {}}) + "\n\n"
                continue
            if item is _SENTINEL:
                return
            ev: PipelineEvent = item  # type: ignore[assignment]
            payload = {"type": ev.type, "stage": ev.stage, "data": ev.data}
            yield "data: " + json.dumps(payload, default=str) + "\n\n"
