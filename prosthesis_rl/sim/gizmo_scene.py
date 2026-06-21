"""Whole-scene Gizmo baking with an on-disk cache.

Antim Labs / Gizmo (https://docs.gizmo.antimlabs.com) turns a natural-language
prompt into a *full, physics-ready scene* and exports it as MJCF — our sim
format. Unlike ``scripts/recon/gizmo_assets.py`` (which bakes ONE articulated
object and uses the scene only as a container, cancelling its job), this module
drives the **scene** endpoints:

    POST /v1/scenes {prompt}              -> job_id        (async, ~2-5 min)
    GET  /v1/jobs/{id}                    -> poll to done  -> scene_id
    POST /v1/scenes/{id}/export?format=.. -> a .zip of the MJCF + meshes

Because a bake costs minutes, **every result is cached on disk keyed by the
prompt**. A second request for the same problem/action returns instantly from the
cache; only a genuine cache miss pays the Gizmo latency. This is what makes the
"generate a scenario live in the demo" flow usable: the first person to ask for a
given ADL waits once; everyone after that gets it immediately.

    from prosthesis_rl.sim.gizmo_scene import bake_scene, cached
    res = bake_scene("a kitchen with a table and a water bottle on it")
    # res.mjcf  -> Path to the primary .xml (load this in MuJoCo / the WASM viewer)
    # res.files -> every file in the export (write all of these into the WASM FS)
    # res.cached -> True if served from disk (no API call, no wait)

Stdlib only (urllib + zipfile + hashlib) — no new dependency, mirroring the asset
baker. MuJoCo is imported lazily only to verify the export parses.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = ROOT / "assets" / "scenes" / "gizmo"
MANIFEST = "manifest.json"

# Status callback: bake_scene reports progress through this so a live UI can show
# the 2-5 min generation instead of a dead spinner. Stages: "cached", "submitting",
# "<gizmo job status>" (queued/running/...), "exporting", "unpacking", "ready".
StatusCb = Callable[[str], None]


def _load_dotenv() -> None:
    """Populate os.environ from repo-root/cwd `.env` without overriding live env.

    Stdlib-only — mirrors scripts/recon/gizmo_assets.py so the same GIZMO_API_KEY
    in `.env` works whether you run the CLI or the demo backend."""
    for env_path in (Path.cwd() / ".env", ROOT / ".env"):
        if not env_path.is_file():
            continue
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()
BASE_URL = os.environ.get("GIZMO_BASE_URL", "https://api.gizmo.antimlabs.com")


def get_api_key(explicit: str | None = None) -> str:
    key = explicit or os.environ.get("GIZMO_API_KEY")
    if not key:
        raise RuntimeError("set GIZMO_API_KEY (env or .env) or pass an explicit key")
    return key


# --------------------------------------------------------------------------- #
# Thin REST client (stdlib urllib) — raises RuntimeError so it's safe to run   #
# inside a backend worker thread (no SystemExit killing the server).           #
# --------------------------------------------------------------------------- #
def _auth_headers(key: str, *, service_token: bool = False) -> dict[str, str]:
    if service_token:
        return {"x-gizmo-service-token": key}
    return {"authorization": f"Bearer {key}"}


def _request(method, path, key, *, body=None, query=None, accept="application/json",
             service_token=False, timeout=120, retries=0):
    """Return (status, raw_bytes, content_type).

    HTTP error *responses* (>=400) come back normally so the caller can read the
    body. Transport-level failures (TLS handshake timeout, dropped connection,
    DNS) raise — but are retried up to `retries` times with backoff first, since
    those mean the request never reached the server. Only use retries>0 on
    idempotent calls (GET / export); a create POST must stay single-shot so a blip
    can't silently double-submit."""
    url = BASE_URL.rstrip("/") + path
    if query:
        url += "?" + urllib.parse.urlencode({k: v for k, v in query.items() if v is not None})
    headers = {"accept": accept, **_auth_headers(key, service_token=service_token)}
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["content-type"] = "application/json"
    attempt = 0
    while True:
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, r.read(), r.headers.get("content-type", "")
        except urllib.error.HTTPError as e:
            return e.code, e.read(), (e.headers.get("content-type", "") if e.headers else "")
        except (urllib.error.URLError, OSError):
            if attempt >= retries:
                raise
            attempt += 1
            time.sleep(min(2 ** attempt, 10))


def _json(method, path, key, **kw):
    status, raw, _ = _request(method, path, key, **kw)
    try:
        payload = json.loads(raw or b"{}")
    except json.JSONDecodeError:
        payload = {"_raw": raw[:500].decode("utf-8", "replace")}
    if status >= 400:
        err = payload.get("error") or payload.get("detail") or payload
        raise RuntimeError(f"Gizmo {method} {path} -> HTTP {status}: {err}")
    return payload


def _dig(obj, *names: str):
    """Depth-first search for the first non-empty value under any of `names`."""
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


def whoami(key: str | None = None, *, service_token: bool = False) -> dict:
    """GET /v1/whoami — confirm the key works before a long bake."""
    return _json("GET", "/v1/whoami", get_api_key(key), service_token=service_token)


# --------------------------------------------------------------------------- #
# Cache.                                                                       #
# --------------------------------------------------------------------------- #
def cache_key(prompt: str) -> str:
    """Stable directory name for a prompt: human-readable slug + content hash.

    Normalised (lower/whitespace-collapsed) so trivially-different spellings of
    the same request still hit, while the hash keeps distinct prompts distinct.
    """
    norm = re.sub(r"\s+", " ", (prompt or "").strip().lower())
    slug = re.sub(r"[^a-z0-9]+", "_", norm)[:40].strip("_") or "scene"
    digest = hashlib.sha1(norm.encode()).hexdigest()[:8]
    return f"{slug}__{digest}"


def cache_dir(prompt: str) -> Path:
    return CACHE_ROOT / cache_key(prompt)


@dataclass
class SceneBake:
    """A baked (or cached) Gizmo scene on disk."""
    dir: Path
    mjcf: Path                       # primary model .xml to load
    files: list[Path] = field(default_factory=list)  # every export file (meshes + xml)
    scene_id: str = ""
    prompt: str = ""
    key: str = ""
    cached: bool = False

    def to_manifest(self) -> dict:
        return {
            "prompt": self.prompt,
            "key": self.key,
            "scene_id": self.scene_id,
            "mjcf": self.mjcf.relative_to(self.dir).as_posix(),
            "files": [f.relative_to(self.dir).as_posix() for f in self.files],
        }


def _read_bake(d: Path, prompt: str = "") -> SceneBake | None:
    """Load a cached bake from dir `d` if its manifest + primary model exist."""
    man = d / MANIFEST
    if not man.is_file():
        return None
    try:
        m = json.loads(man.read_text())
    except json.JSONDecodeError:
        return None
    mjcf = d / m.get("mjcf", "")
    if not mjcf.is_file():
        return None
    files = [d / f for f in m.get("files", [])]
    return SceneBake(dir=d, mjcf=mjcf, files=[f for f in files if f.is_file()],
                     scene_id=m.get("scene_id", ""), prompt=prompt or m.get("prompt", ""),
                     key=m.get("key", ""), cached=True)


def cached(prompt: str) -> SceneBake | None:
    """Return the on-disk bake for `prompt`, or None if it hasn't been baked yet."""
    return _read_bake(cache_dir(prompt), prompt=prompt)


# --------------------------------------------------------------------------- #
# Scene pipeline: generate -> poll -> export -> unpack -> cache.               #
# --------------------------------------------------------------------------- #
def generate_scene(key, prompt, *, service_token=False) -> tuple[str, str]:
    """POST /v1/scenes {prompt} -> (job_id, scene_id?). Async — does NOT cancel.

    The scene endpoint is fire-and-poll: it returns a job_id immediately and the
    scene_id usually only materialises in the finished job's result (so it may be
    empty here)."""
    payload = _json("POST", "/v1/scenes", key, body={"prompt": prompt},
                    service_token=service_token)
    job_id = _dig(payload, "job_id", "jobId", "id")
    if not job_id:
        raise RuntimeError(f"no job_id in /v1/scenes response: {payload}")
    scene_id = _dig(payload, "scene_id", "sceneId") or ""
    return job_id, scene_id


def poll_job(key, job_id, *, timeout=1200, interval=4.0, on_status: StatusCb | None = None,
             service_token=False) -> dict:
    """GET /v1/jobs/{id} until terminal; return the final job payload.

    Reports each new status string through `on_status` so a live UI can follow the
    plan->generate->floorplan->validate stages. Raises RuntimeError on failure."""
    deadline = time.monotonic() + timeout
    last = ""
    while time.monotonic() < deadline:
        try:
            payload = _json("GET", f"/v1/jobs/{job_id}", key, query={"include_result": "true"},
                            service_token=service_token, retries=2)
        except (urllib.error.URLError, OSError) as e:
            # A single GET blipped during a long (10-20 min) poll — don't abandon
            # the whole job over one dropped connection; back off and try again.
            if on_status:
                on_status(f"poll retry (transient network error: {e})")
            time.sleep(interval)
            continue
        status = (_dig(payload, "status") or "unknown").lower()
        stage = _dig(payload, "stage", "phase", "message")
        label = f"{status}: {stage}" if stage else status
        if label != last:
            if on_status:
                on_status(label)
            last = label
        if status in ("succeeded", "completed", "success"):
            return payload
        if status in ("failed", "error", "cancelled", "canceled"):
            raise RuntimeError(f"job {job_id} {status}: {_dig(payload, 'error', 'message') or payload}")
        time.sleep(interval)
    raise RuntimeError(f"job {job_id} did not finish within {timeout}s")


def export_scene(key, scene_id, *, fmt="mjcf", service_token=False) -> bytes:
    """POST /v1/scenes/{id}/export {"format": fmt} -> zip bytes (MJCF + meshes).

    The scene export (unlike the single-asset export) takes the format in the JSON
    *body*, per ExportSceneBody; `robot_profile` is intentionally omitted — we
    splice our own prosthesis arm in rather than embedding a Gizmo robot."""
    status, raw, ctype = _request("POST", f"/v1/scenes/{scene_id}/export", key,
                                  body={"format": fmt}, accept="application/zip",
                                  service_token=service_token, timeout=180, retries=2)
    if status >= 400:
        try:
            err = json.loads(raw).get("error", raw[:300].decode("utf-8", "replace"))
        except Exception:
            err = raw[:300].decode("utf-8", "replace")
        raise RuntimeError(f"export scene {scene_id} ({fmt}) -> HTTP {status}: {err}")
    if "zip" not in ctype and raw[:2] != b"PK":
        raise RuntimeError(f"scene export returned {ctype!r}, expected a zip ({len(raw)} bytes)")
    return raw


def slim_scene_xml(xml_path: Path) -> list[str]:
    """Rewrite a Gizmo scene MJCF in place to be small + self-contained, deleting
    the texture PNGs it no longer needs. Returns the removed filenames.

    A scene export is ~10x bigger than it needs to be for us: tens of MB of PBR
    texture PNGs that nothing in our pipeline uses (the web viewer colours from
    geom rgba; RL ignores appearance; materials already carry a flat `rgba`). This
    strips every <texture>, drops the MuJoCo 3.x PBR <layer>/texture= refs from
    each <material> (so they keep just their rgba and still compile with no PNGs),
    and removes the PNG files — turning a ~55MB export into a ~5MB inline-mesh XML.
    Idempotent: re-running on an already-slim scene is a no-op."""
    import xml.etree.ElementTree as ET

    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    asset = root.find("asset")
    removed: list[str] = []
    if asset is not None:
        for tex in asset.findall("texture"):
            f = tex.get("file")
            if f:
                removed.append(f)
            asset.remove(tex)
        for mat in asset.findall("material"):
            mat.attrib.pop("texture", None)
            for layer in mat.findall("layer"):
                mat.remove(layer)
    tree.write(str(xml_path), encoding="unicode")
    # delete the now-unreferenced PNG files next to the model
    for name in removed:
        p = (xml_path.parent / name)
        if p.is_file():
            p.unlink()
    return removed


def _unpack(zip_bytes: bytes, out_dir: Path) -> tuple[Path, list[Path]]:
    """Extract the export zip into out_dir; return (primary .xml, all files)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(BytesIO(zip_bytes)) as zf:
        zf.extractall(out_dir)
    files = sorted(p for p in out_dir.rglob("*") if p.is_file())
    xmls = [p for p in files if p.suffix.lower() in (".xml", ".mjcf")]
    if not xmls:
        raise RuntimeError(f"no .xml/.mjcf in scene export ({[f.name for f in files]})")
    # Prefer a top-level, scene-looking model over an included fragment.
    xmls.sort(key=lambda p: (len(p.relative_to(out_dir).parts),
                             0 if "scene" in p.name.lower() else 1, p.name))
    return xmls[0], files


def verify_loads(model_path: Path, on_status: StatusCb | None = None) -> None:
    """Lazy-import MuJoCo and confirm the export parses (best-effort, never raises)."""
    try:
        import mujoco
    except ImportError:
        return
    try:
        model = mujoco.MjModel.from_xml_path(str(model_path))
    except Exception as e:  # noqa: BLE001 — a parse warning shouldn't fail the bake
        if on_status:
            on_status(f"warning: MuJoCo could not load export: {e}")
        return
    if on_status:
        on_status(f"loaded OK: {model.nbody} bodies, {model.njnt} joints, "
                  f"{model.ngeom} geoms, {model.nmesh} meshes")


def bake_scene(prompt: str, *, key: str | None = None, force: bool = False,
               timeout: int = 1200, on_status: StatusCb | None = None,
               service_token: bool = False, verify: bool = True,
               slim: bool = True) -> SceneBake:
    """Bake (or fetch from cache) a full Gizmo scene for `prompt` as MJCF.

    Cache-first: a previously-baked prompt returns instantly with ``cached=True``
    and no API call. On a miss it submits to Gizmo, polls the job (~2-5 min,
    reported via `on_status`), exports MJCF, unpacks into the cache dir, and
    writes a manifest. The unpack is atomic (tmp dir + rename) so a crash or a
    concurrent reader never sees a half-written scene.
    """
    def emit(s: str) -> None:
        if on_status:
            on_status(s)

    if not force:
        hit = cached(prompt)
        if hit is not None:
            emit("cached")
            return hit

    key = get_api_key(key)
    emit("submitting")
    job_id, scene_id = generate_scene(key, prompt, service_token=service_token)
    job = poll_job(key, job_id, timeout=timeout, on_status=on_status,
                   service_token=service_token)
    scene_id = scene_id or _dig(job, "scene_id", "sceneId") or ""
    if not scene_id:
        raise RuntimeError(f"scene job {job_id} finished but no scene_id found: {job}")

    emit("exporting")
    zip_bytes = export_scene(key, scene_id, fmt="mjcf", service_token=service_token)

    emit("unpacking")
    dest = cache_dir(prompt)
    tmp = dest.with_name(dest.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    primary, _ = _unpack(zip_bytes, tmp)
    if slim:
        slim_scene_xml(primary)  # strip unused texture PNGs -> ~10x smaller cache entry
    files = sorted(p for p in tmp.rglob("*") if p.is_file())  # recompute after slim
    bake = SceneBake(dir=tmp, mjcf=primary, files=files, scene_id=scene_id,
                     prompt=prompt, key=cache_key(prompt), cached=False)
    (tmp / MANIFEST).write_text(json.dumps(bake.to_manifest(), indent=2))
    if dest.exists():
        shutil.rmtree(dest)
    tmp.rename(dest)

    # Re-read so every path points at the final (renamed) location.
    final = _read_bake(dest, prompt=prompt)
    if final is None:
        raise RuntimeError(f"bake wrote {dest} but it failed manifest validation")
    final.cached = False
    if verify:
        verify_loads(final.mjcf, on_status)
    emit("ready")
    return final


# --------------------------------------------------------------------------- #
# CLI — offline bake / smoke test (mirrors gizmo_assets.py).                   #
# --------------------------------------------------------------------------- #
def _main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", help="natural-language scene description (the ADL problem)")
    ap.add_argument("--force", action="store_true", help="ignore the cache and re-bake")
    ap.add_argument("--timeout", type=int, default=1200,
                    help="seconds to wait for the job (scenes can exceed 10 min)")
    ap.add_argument("--api-key", default=None, help="overrides $GIZMO_API_KEY")
    ap.add_argument("--service-token", action="store_true", help="auth via x-gizmo-service-token")
    ap.add_argument("--whoami", action="store_true", help="just verify the key and exit")
    ap.add_argument("--list", action="store_true", help="list cached scenes and exit")
    args = ap.parse_args(argv)

    if args.list:
        if not CACHE_ROOT.is_dir():
            print("(no cached scenes yet)")
            return 0
        for d in sorted(CACHE_ROOT.iterdir()):
            b = _read_bake(d)
            if b:
                print(f"{d.name}\n   prompt: {b.prompt!r}\n   mjcf:   {b.mjcf}")
        return 0

    if args.whoami:
        print(json.dumps(whoami(args.api_key, service_token=args.service_token), indent=2))
        return 0

    if not args.prompt:
        ap.error("--prompt is required (or use --whoami / --list)")

    res = bake_scene(args.prompt, key=args.api_key, force=args.force, timeout=args.timeout,
                     on_status=lambda s: print(f"[gizmo-scene] {s}", flush=True),
                     service_token=args.service_token)
    tag = "cache hit" if res.cached else "baked"
    print(f"\n[gizmo-scene] {tag}: {res.mjcf}  ({len(res.files)} files)")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_main())
