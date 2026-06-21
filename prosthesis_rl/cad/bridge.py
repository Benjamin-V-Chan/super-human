from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh

from prosthesis_rl.contracts import DesignParams


class CadBridge:
    """DesignParams -> per-link 3D geometry -> one binary STL per moving link.

    Each link in the design's kinematic chain is meshed in its **own local frame**
    (origin at the proximal joint, body extending along -Z by `length`), matching
    sim.mjcf_builder's body convention. That lets MuJoCo attach the real geometry
    to each articulated body so it bends at the joints — a true robot arm rather
    than one fused, static blob. A `manifest.json` records the chain + mesh files
    for inspection / reload.

    Swap the capsule primitives here for CadQuery solids once that sandbox is up;
    the per-link contract (one STL per body, in joint frame) stays the same.
    """

    def __init__(self, output_dir: str | Path = "assets/stl") -> None:
        self.output_dir = Path(output_dir)

    # ── Primary: articulated per-link export ─────────────────────────────────

    def export_arm(self, params: DesignParams, name: str = "candidate") -> Path:
        """Write one `<link>.stl` per link + `manifest.json`; return the scene dir.

        Pass the returned dir to sim.mjcf_builder.build_mjcf(..., mesh_dir=dir)
        to skin the simulated arm with this geometry.
        """
        out = self.output_dir / name
        out.mkdir(parents=True, exist_ok=True)

        manifest_links = []
        for link in params.links:
            mesh = self._link_mesh(link.length, link.radius)
            mesh_file = f"{link.name}.stl"
            mesh.export(out / mesh_file)
            manifest_links.append({
                "name": link.name,
                "length": link.length,
                "radius": link.radius,
                "mesh": mesh_file,
                "rgba": list(link.rgba),
                "joints": [
                    {"name": j.name, "axis": list(j.axis),
                     "range_deg": list(j.range_deg), "type": j.type}
                    for j in link.joints
                ],
            })

        manifest = {
            "name": name,
            "dof": params.dof,
            "joint_order": params.joint_names,
            "links": manifest_links,
        }
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return out

    # ── Back-compat: one fused STL of the whole arm at zero pose ──────────────

    def export_stl(self, params: DesignParams, name: str = "candidate") -> Path:
        """Fuse all links (at the zero/extended pose) into a single STL.

        Kept for the recon scene-combine path; the articulated sim uses
        `export_arm`. Returns the path to the written `<name>.stl`.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        meshes, z = [], 0.0
        for link in params.links:
            m = self._link_mesh(link.length, link.radius)
            m.apply_translation([0.0, 0.0, z])
            meshes.append(m)
            z -= link.length
        fused = trimesh.util.concatenate(meshes)
        stl_path = self.output_dir / f"{name}.stl"
        fused.export(stl_path)
        return stl_path

    # ── Geometry ─────────────────────────────────────────────────────────────

    @staticmethod
    def _link_mesh(length: float, radius: float) -> trimesh.Trimesh:
        """A capsule from the local origin (z=0) down to (0, 0, -length).

        Built as cylinder + two hemispherical caps so the proximal end sits on
        the joint and the body hangs along -Z, matching the MJCF body tree.
        """
        h = max(1e-4, float(length))
        r = max(1e-4, float(radius))
        cyl = trimesh.creation.cylinder(radius=r, height=h)  # centred, along Z
        cyl.apply_translation([0.0, 0.0, -h / 2.0])          # -> spans 0 .. -h
        cap_top = trimesh.creation.icosphere(subdivisions=2, radius=r)
        cap_bot = trimesh.creation.icosphere(subdivisions=2, radius=r)
        cap_bot.apply_translation([0.0, 0.0, -h])
        mesh = trimesh.util.concatenate([cyl, cap_top, cap_bot])
        mesh.merge_vertices()
        return mesh
