"""End-to-end self-test of the LOCAL half (no GPU / no Modal needed).

Builds a synthetic room (floor + walls + a table), tilts + translates it to
mimic a raw COLMAP frame, runs mesh_to_mjcf.convert(), loads the result in
MuJoCo, drops a probe sphere, and asserts it collides and settles above the
floor. This verifies clean -> floor-align -> CoACD -> MJCF without needing the
cloud reconstruction.

    python3 scripts/recon/_selftest.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import trimesh

import mesh_to_mjcf as m2m  # same dir


def synthetic_room() -> trimesh.Trimesh:
    parts = []
    parts.append(trimesh.creation.box(extents=[4, 4, 0.1],
                                       transform=trimesh.transformations.translation_matrix([0, 0, -0.05])))
    for tx, ty, sx, sy in [(2, 0, 0.1, 4), (-2, 0, 0.1, 4),
                           (0, 2, 4, 0.1), (0, -2, 4, 0.1)]:
        parts.append(trimesh.creation.box(
            extents=[sx, sy, 2.5],
            transform=trimesh.transformations.translation_matrix([tx, ty, 1.25])))
    # a table (raised platform) so the probe has something above the floor to hit
    parts.append(trimesh.creation.box(
        extents=[0.8, 0.5, 0.75],
        transform=trimesh.transformations.translation_matrix([0, 0, 0.375])))
    return trimesh.util.concatenate(parts)


def main() -> int:
    room = synthetic_room()

    # Tilt + translate to mimic an arbitrary, non-gravity-aligned recon frame.
    T = trimesh.transformations.rotation_matrix(np.radians(11), [1, 0, 0])
    T = trimesh.transformations.rotation_matrix(np.radians(7), [0, 1, 0]) @ T
    room.apply_transform(T)
    room.apply_translation([1.5, -2.0, 3.0])

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        room.export(td / "raw.ply")
        scene_xml = m2m.convert(td / "raw.ply", td / "scene", scale=1.0,
                                align=True, target_faces=20000,
                                coacd_threshold=0.05, add_probe=True)

        import mujoco
        model = mujoco.MjModel.from_xml_path(str(scene_xml))
        data = mujoco.MjData(model)

        n_col = sum(1 for i in range(model.ngeom)
                    if model.geom_group[i] == 3)
        assert n_col >= 2, f"expected multiple convex collision geoms, got {n_col}"

        # floor should be re-aligned near z=0
        mujoco.mj_forward(model, data)
        z0 = float(data.body("probe").xpos[2])

        max_contacts = 0
        for _ in range(1500):
            mujoco.mj_step(model, data)
            max_contacts = max(max_contacts, data.ncon)
        mujoco.mj_forward(model, data)
        zf = float(data.body("probe").xpos[2])
        speed = float(np.linalg.norm(data.body("probe").cvel[3:]))

    print(f"  collision geoms (convex pieces): {n_col}")
    print(f"  probe start z={z0:.3f} -> settle z={zf:.3f}")
    print(f"  max simultaneous contacts: {max_contacts}")
    print(f"  final probe speed: {speed:.4f}")

    ok = (np.isfinite(zf) and zf > 0.02 and max_contacts > 0 and speed < 0.05)
    print("\nRESULT:", "PASS ✅" if ok else "FAIL ❌")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
