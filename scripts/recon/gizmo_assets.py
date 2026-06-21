"""Generate physics-ready, *interactable* task objects with the Gizmo API.

This is the third leg of our scene pipeline, and the one that gives the policy
something to actually *do*:

    modal_recon.py   RGB video  -> COLLISION MESH     (the static room you bump into)
    modal_gsplat.py  RGB video  -> GAUSSIAN SPLAT     (the pretty visual backdrop)
    gizmo_assets.py  TEXT prompt-> ARTICULATED MJCF   (the drawer/cup/doorknob you manipulate)

Gizmo (https://docs.gizmo.antimlabs.com) takes a natural-language prompt and
returns a single asset with real joints (revolute/prismatic), mass, inertia,
friction and collision geom, exported directly as **MJCF** — our sim format. We
bake an object once, drop it into `assets/objects/<name>/`, and reuse it at a
task target in every RL rollout (offline + one-time, exactly like recon: no live
API call sits in the loop).

    export GIZMO_API_KEY=sk-...                       # your key (never commit it)
    python3 scripts/recon/gizmo_assets.py --whoami     # smoke: confirm the key
    python3 scripts/recon/gizmo_assets.py \
        --prompt "a kitchen drawer that slides open on a prismatic joint" \
        --name drawer

Output: assets/objects/drawer/  (the unpacked MJCF + meshes). Placing it at an
ADL task target and splicing it into the arm env is the next stage —
prosthesis_rl/sim/gizmo_asset.py (mirrors room_asset.py).

Stdlib only (urllib + zipfile) — no new dependency. MuJoCo is imported lazily
just to verify the export loads.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path

def _load_dotenv() -> None:
    """Populate os.environ from the repo-root (or cwd) `.env`, without overriding
    anything already set. Stdlib-only — avoids a python-dotenv dependency."""
    candidates = [Path.cwd() / ".env", Path(__file__).resolve().parents[2] / ".env"]
    seen: set[Path] = set()
    for env_path in candidates:
        if env_path in seen or not env_path.is_file():
            continue
        seen.add(env_path)
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip().strip('"').strip("'")
            os.environ.setdefault(k, v)


_load_dotenv()
BASE_URL = os.environ.get("GIZMO_BASE_URL", "https://api.gizmo.antimlabs.com")
OUT_ROOT = Path("assets/objects")


# --------------------------------------------------------------------------- #
# Thin REST client (stdlib urllib).                                           #
# --------------------------------------------------------------------------- #
def _auth_headers(key: str, *, service_token: bool = False) -> dict[str, str]:
    """Gizmo accepts either a Bearer `authorization` header or a service token."""
    if service_token:
        return {"x-gizmo-service-token": key}
    return {"authorization": f"Bearer {key}"}


def _request(
    method: str,
    path: str,
    key: str,
    *,
    body: dict | None = None,
    query: dict | None = None,
    accept: str = "application/json",
    service_token: bool = False,
    timeout: int = 60,
) -> tuple[int, bytes, str]:
    """Return (status, raw_bytes, content_type). Raises only on transport error."""
    url = BASE_URL.rstrip("/") + path
    if query:
        url += "?" + urllib.parse.urlencode({k: v for k, v in query.items() if v is not None})
    headers = {"accept": accept, **_auth_headers(key, service_token=service_token)}
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["content-type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read(), r.headers.get("content-type", "")
    except urllib.error.HTTPError as e:
        return e.code, e.read(), e.headers.get("content-type", "") if e.headers else ""


def _json(method: str, path: str, key: str, **kw) -> dict:
    """_request that expects JSON and raises a readable error on >=400."""
    status, raw, _ = _request(method, path, key, **kw)
    try:
        payload = json.loads(raw or b"{}")
    except json.JSONDecodeError:
        payload = {"_raw": raw[:500].decode("utf-8", "replace")}
    if status >= 400:
        err = payload.get("error") or payload.get("detail") or payload
        raise SystemExit(f"Gizmo {method} {path} -> HTTP {status}: {err}")
    return payload


def _dig(obj, *names: str):
    """Depth-first search for the first value under any of `names` (schemas are loose)."""
    stack = [obj]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            for k, v in cur.items():
                if k in names and v not in (None, "", []):
                    return v
                stack.append(v)
        elif isinstance(cur, list):
            stack.extend(cur)
    return None


# --------------------------------------------------------------------------- #
# Pipeline: generate -> poll -> export -> unpack.                             #
# --------------------------------------------------------------------------- #
def generate_asset(key: str, prompt: str, *, pipeline: str | None, persist: bool, **kw) -> str:
    """POST /v1/assets -> job_id."""
    body = {"prompt": prompt, "persist": persist}
    if pipeline:
        body["asset_pipeline"] = pipeline
    payload = _json("POST", "/v1/assets", key, body=body, **kw)
    job_id = _dig(payload, "job_id", "jobId", "id")
    if not job_id:
        raise SystemExit(f"no job_id in /v1/assets response: {payload}")
    print(f"[gizmo] queued asset job {job_id}", flush=True)
    return job_id


def poll_job(key: str, job_id: str, *, timeout: int = 300, interval: float = 4.0, **kw) -> dict:
    """GET /v1/jobs/{id} until terminal. Returns the final job payload (with result)."""
    deadline = time.monotonic() + timeout
    last = ""
    while time.monotonic() < deadline:
        payload = _json("GET", f"/v1/jobs/{job_id}", key, query={"include_result": "true"}, **kw)
        status = (_dig(payload, "status") or "unknown").lower()
        if status != last:
            print(f"[gizmo] job {job_id}: {status}", flush=True)
            last = status
        if status in ("succeeded", "completed", "success"):
            return payload
        if status in ("failed", "error", "cancelled", "canceled"):
            raise SystemExit(f"job {job_id} {status}: {_dig(payload, 'error', 'message') or payload}")
        time.sleep(interval)
    raise SystemExit(f"job {job_id} did not finish within {timeout}s")


def export_asset(key: str, asset_id: str, *, fmt: str = "mjcf", **kw) -> bytes:
    """POST /v1/assets/{id}/export?format=mjcf -> zip bytes."""
    status, raw, ctype = _request(
        "POST", f"/v1/assets/{asset_id}/export", key,
        query={"format": fmt}, accept="application/zip", **kw,
    )
    if status >= 400:
        try:
            err = json.loads(raw).get("error", raw[:300].decode("utf-8", "replace"))
        except Exception:
            err = raw[:300].decode("utf-8", "replace")
        raise SystemExit(f"export {asset_id} ({fmt}) -> HTTP {status}: {err}")
    if "zip" not in ctype and not raw[:2] == b"PK":
        raise SystemExit(f"export returned {ctype!r}, expected a zip (got {len(raw)} bytes)")
    return raw


def unpack(zip_bytes: bytes, out_dir: Path) -> Path:
    """Extract the export zip; return the path to the primary .xml/.mjcf model."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(BytesIO(zip_bytes)) as zf:
        zf.extractall(out_dir)
        names = zf.namelist()
    print(f"[gizmo] unpacked {len(names)} files -> {out_dir}", flush=True)
    xmls = sorted(out_dir.rglob("*.xml")) + sorted(out_dir.rglob("*.mjcf"))
    if not xmls:
        raise SystemExit(f"no .xml/.mjcf in export ({names})")
    # Prefer a model-looking top-level file over an included fragment.
    xmls.sort(key=lambda p: (len(p.relative_to(out_dir).parts), p.name))
    return xmls[0]


def verify_loads(model_path: Path) -> None:
    """Lazy-import MuJoCo and confirm the export parses; print what makes it interactable."""
    try:
        import mujoco
    except ImportError:
        print("[gizmo] (mujoco not installed — skipping load check)")
        return
    try:
        model = mujoco.MjModel.from_xml_path(str(model_path))
    except Exception as e:  # noqa: BLE001 — surface the parse error, don't crash the bake
        print(f"[gizmo] WARNING: {model_path.name} did not load in MuJoCo: {e}")
        return
    print(f"[gizmo] loaded OK: {model.nbody} bodies, {model.njnt} joints, "
          f"{model.ngeom} geoms, {model.nmesh} meshes")
    if model.njnt == 0:
        print("[gizmo] note: 0 joints — this asset is rigid, not articulated.")


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def _key(args) -> str:
    key = args.api_key or os.environ.get("GIZMO_API_KEY")
    if not key:
        raise SystemExit("set GIZMO_API_KEY or pass --api-key")
    return key


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", help="natural-language description of the object to generate")
    ap.add_argument("--name", help="slug for assets/objects/<name>/ (default: derived from prompt)")
    ap.add_argument("--format", default="mjcf", choices=["mjcf", "usd", "sdf"], help="export format")
    ap.add_argument("--pipeline", default=None, help="asset_pipeline override (geometry pipeline)")
    ap.add_argument("--no-persist", action="store_true", help="don't persist the asset to Gizmo's S3")
    ap.add_argument("--timeout", type=int, default=300, help="seconds to wait for the job")
    ap.add_argument("--api-key", default=None, help="overrides $GIZMO_API_KEY")
    ap.add_argument("--service-token", action="store_true", help="auth via x-gizmo-service-token")
    ap.add_argument("--whoami", action="store_true", help="just verify the key (GET /v1/whoami) and exit")
    args = ap.parse_args(argv)

    key = _key(args)
    kw = {"service_token": args.service_token}

    if args.whoami:
        print(json.dumps(_json("GET", "/v1/whoami", key, **kw), indent=2))
        return 0

    if not args.prompt:
        ap.error("--prompt is required (or use --whoami)")

    name = args.name or "".join(c if c.isalnum() else "_" for c in args.prompt.lower())[:40].strip("_")
    out_dir = OUT_ROOT / name

    job_id = generate_asset(key, args.prompt, pipeline=args.pipeline, persist=not args.no_persist, **kw)
    job = poll_job(key, job_id, timeout=args.timeout, **kw)
    asset_id = _dig(job, "asset_id", "assetId")
    if not asset_id:
        raise SystemExit(f"job succeeded but no asset_id found: {job}")
    print(f"[gizmo] asset {asset_id} ready -> exporting {args.format}", flush=True)

    zip_bytes = export_asset(key, asset_id, fmt=args.format, **kw)
    model_path = unpack(zip_bytes, out_dir)
    if args.format == "mjcf":
        verify_loads(model_path)

    print(f"\n[gizmo] done: {model_path}")
    print(f"  next: position it at a task target and splice into the arm env via "
          f"prosthesis_rl/sim/gizmo_asset.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
