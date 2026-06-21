"""Place a trained Gaussian splat into the web viewer.

Copies the Inria 3DGS output (point_cloud.ply / room.ply from modal_gsplat.py) to
webdemo/assets/scenes/room.ply, where live.html renders it behind the arm.

    modal run scripts/recon/modal_gsplat.py --frames-dir frames --out room.ply
    python3 scripts/demo/export_web_splat.py room.ply
    cd webdemo && python3 -m http.server 8011   ->  http://localhost:8011/live.html
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "webdemo" / "assets" / "scenes" / "room.ply"


def main() -> None:
    ap = argparse.ArgumentParser(description="Install a trained 3DGS splat into the web viewer")
    ap.add_argument("ply", nargs="?", default="room.ply",
                    help="path to the Inria point_cloud.ply (default: room.ply)")
    args = ap.parse_args()

    src = Path(args.ply)
    if not src.exists():
        sys.exit(f"splat not found: {src} — train it with scripts/recon/modal_gsplat.py")

    DEST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, DEST)
    mb = DEST.stat().st_size / 1e6
    print(f"[splat] copied {src} -> {DEST.relative_to(ROOT)} ({mb:.1f} MB)")
    print("[splat] open: cd webdemo && python3 -m http.server 8011 -> "
          "http://localhost:8011/live.html")
    if mb > 120:
        print("[splat] note: large .ply — browser load will be slow; consider a .ksplat "
              "conversion via the @mkkellogg gaussian-splats-3d CLI for production.")


if __name__ == "__main__":
    main()
