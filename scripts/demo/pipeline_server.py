"""HTTP + SSE server for the Super Human pipeline dashboard.

Serves webdemo/pipeline.html and exposes REST/SSE endpoints:
  GET  /                        → pipeline.html
  GET  /api/clips               → [{name, path, has_thumbnail}]
  GET  /api/thumbnail?clip=X    → JPEG first frame (ffmpeg) or placeholder PNG
  GET  /api/scene?name=X        → MJCF XML text
  GET  /api/mesh?name=X&link=Y  → STL binary
  POST /api/run                 → SSE stream (body: {"clip": "test_vids/..."})

Usage:
    PYTHONPATH=. python3 scripts/demo/pipeline_server.py [--port 8012] [--quick]
    # then open http://localhost:8012/pipeline.html
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import subprocess
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).parent.parent.parent
WEBDEMO = REPO / "webdemo"
TEST_VIDS = REPO / "test_vids"
ASSETS_MJCF = REPO / "assets" / "mjcf"
ASSETS_STL  = REPO / "assets" / "stl"

# Track ongoing run (one at a time for demo purposes)
_run_lock  = threading.Lock()
_active_loop: dict = {}

VIDEO_EXTS = {".mov", ".mp4", ".avi", ".webm", ".MOV", ".MP4"}


def _list_clips() -> list[dict]:
    clips = []
    if TEST_VIDS.exists():
        for p in sorted(TEST_VIDS.iterdir()):
            if p.suffix in VIDEO_EXTS:
                clips.append({
                    "name": p.name,
                    "path": str(p.relative_to(REPO)),
                    "has_thumbnail": False,  # generated on demand
                })
    return clips


def _thumbnail_bytes(clip_path: str) -> bytes | None:
    """Return first-frame JPEG bytes via ffmpeg, or None if unavailable."""
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(REPO / clip_path),
                "-vframes", "1", "-f", "image2", "-vcodec", "mjpeg",
                "-vf", "scale=320:-1", "pipe:1",
            ],
            capture_output=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout:
            return result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _placeholder_png() -> bytes:
    """Return a minimal 1×1 grey PNG as fallback thumbnail."""
    # Pre-baked 1×1 #555 PNG
    return bytes([
        137,80,78,71,13,10,26,10,0,0,0,13,73,72,68,82,0,0,0,1,0,0,0,1,
        8,2,0,0,0,144,119,83,222,0,0,0,12,73,68,65,84,8,215,99,88,85,85,
        0,0,0,36,0,8,189,199,195,84,0,0,0,0,73,69,78,68,174,66,96,130,
    ])


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # suppress default access log
        pass

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)

        path = parsed.path.rstrip("/") or "/"

        # ── Static serving ───────────────────────────────────────────────────
        if path == "/" or path == "/pipeline.html":
            self._serve_file(WEBDEMO / "pipeline.html", "text/html")
        elif path.startswith("/src/"):
            self._serve_file(WEBDEMO / path.lstrip("/"))
        elif path.startswith("/assets/"):
            self._serve_file(WEBDEMO / path.lstrip("/"))
        elif path.startswith("/node_modules/"):
            self._serve_file(WEBDEMO / path.lstrip("/"))

        # ── API ──────────────────────────────────────────────────────────────
        elif path == "/api/clips":
            self._json(_list_clips())
        elif path == "/api/thumbnail":
            clip = qs.get("clip", [""])[0]
            data = _thumbnail_bytes(clip)
            if data:
                self._respond(200, "image/jpeg", data)
            else:
                self._respond(200, "image/png", _placeholder_png())
        elif path == "/api/scene":
            name = qs.get("name", [""])[0]
            xml_path = ASSETS_MJCF / f"{name}.xml"
            if xml_path.exists():
                self._serve_file(xml_path, "application/xml")
            else:
                self._respond(404, "text/plain", b"not found")
        elif path == "/api/mesh":
            name = qs.get("name", [""])[0]
            link = qs.get("link", [""])[0]
            stl_path = ASSETS_STL / name / f"{link}.stl"
            if stl_path.exists():
                self._serve_file(stl_path, "application/octet-stream")
            else:
                # Fallback: check webdemo/assets/scenes/arm_links/
                fallback = WEBDEMO / "assets" / "scenes" / "arm_links" / f"{link}.stl"
                if fallback.exists():
                    self._serve_file(fallback, "application/octet-stream")
                else:
                    self._respond(404, "text/plain", b"not found")
        elif path == "/api/status":
            self._json({"active": bool(_active_loop), "name": _active_loop.get("name", "")})
        else:
            self._respond(404, "text/plain", b"not found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)
        path = parsed.path

        if path == "/api/run":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            clip = body.get("clip") or qs.get("clip", [""])[0]
            quick = body.get("quick", False)
            self._run_sse(clip, quick=quick)
        else:
            self._respond(404, "text/plain", b"not found")

    # ── SSE pipeline runner ───────────────────────────────────────────────────

    def _run_sse(self, clip: str, quick: bool = False) -> None:
        from prosthesis_rl.pipeline.events import Emitter, PipelineEvent
        from prosthesis_rl.pipeline.loop import DesignOptimizationLoop

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        emitter = Emitter()

        def _run():
            try:
                loop = DesignOptimizationLoop(quick_mode=quick)
                loop.run(clip, emitter)
            except Exception as exc:
                emitter.emit(PipelineEvent("error", "pipeline", {"message": str(exc)}))
                emitter.close()

        t = threading.Thread(target=_run, daemon=True)
        with _run_lock:
            _active_loop.clear()
            _active_loop.update({"name": clip, "thread": t})
        t.start()

        try:
            for chunk in emitter.get_stream(timeout=10.0):
                self.wfile.write(chunk.encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            _active_loop.clear()

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _respond(self, code: int, ctype: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj) -> None:
        body = json.dumps(obj, default=str).encode()
        self._respond(200, "application/json", body)

    def _serve_file(self, path: Path, ctype: str | None = None) -> None:
        if not path.exists() or not path.is_file():
            self._respond(404, "text/plain", b"not found")
            return
        ctype = ctype or mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        body = path.read_bytes()
        self._respond(200, ctype, body)


def main() -> None:
    ap = argparse.ArgumentParser(description="Super Human pipeline server")
    ap.add_argument("--port", type=int, default=8012)
    ap.add_argument("--quick", action="store_true", help="quick_mode (fewer seeds/timesteps)")
    args = ap.parse_args()

    sys.path.insert(0, str(REPO))
    os.chdir(REPO)

    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    url = f"http://localhost:{args.port}/pipeline.html"
    print(f"[pipeline-server] Listening on {url}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[pipeline-server] Stopped.")


if __name__ == "__main__":
    main()
