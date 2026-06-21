from __future__ import annotations

import math
import struct
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from prosthesis_rl.contracts import DesignParams


class CadBridge:
    """DesignParams -> parametric 3D geometry -> binary STL.

    Generates a simplified prosthetic arm: shoulder socket + upper arm tube +
    elbow sphere + forearm tube + wrist disc. Good enough for MuJoCo import
    and viewer preview.  Swap per-link meshes with CadQuery once sandbox is up.
    """

    def __init__(self, output_dir: str | Path = "assets/stl") -> None:
        self.output_dir = Path(output_dir)

    def export_arm(self, params: DesignParams, name: str = "candidate") -> Path:
        """Export one STL per link into assets/stl/{name}/ for MuJoCo mesh skinning.

        The mjcf_builder skins each link body with {link.name}.stl from this dir
        when mesh_dir is passed. Returns the directory path.
        """
        mesh_dir = self.output_dir / name
        mesh_dir.mkdir(parents=True, exist_ok=True)
        for link in params.links:
            tris = self._link_triangles(link)
            stl_path = mesh_dir / f"{link.name}.stl"
            stl_path.write_bytes(self._pack_binary_stl(tris))
        return mesh_dir

    def export_mjcf(self, spec: Any, name: str = "candidate") -> Path:
        """Generate minimal valid MJCF from a DesignParams (or legacy dict).

        Writes to assets/mjcf/{name}.xml. Prefer sim.mjcf_builder.build_mjcf()
        for full env generation with room/human/target; this is a lightweight
        standalone export for inspection and handoff.
        """
        # Normalise to DesignParams for uniform handling
        params = self._coerce_to_design_params(spec)
        mjcf_dir = self.output_dir.parent / "mjcf"
        mjcf_dir.mkdir(parents=True, exist_ok=True)
        # Per-link STLs live next to the meshes export_arm() wrote: assets/stl/<name>/.
        stl_dir = self.output_dir / name

        root = ET.Element("mujoco", model=f"prosthesis_{name}")
        # meshdir="." so a flat <mesh file="upper_arm.stl"> resolves against the
        # STL files the web viewer preloads into the WASM FS (webdemo/src/pipeline.js).
        ET.SubElement(root, "compiler", angle="radian", meshdir=".")

        # Skin each link with its real CAD mesh when the STL exists; otherwise fall
        # back to a primitive capsule so the scene is always renderable.
        has_mesh: dict[str, bool] = {}
        asset_el = ET.SubElement(root, "asset")
        for link in params.links:
            has_mesh[link.name] = (stl_dir / f"{link.name}.stl").exists()
            if has_mesh[link.name]:
                ET.SubElement(asset_el, "mesh", name=f"{link.name}_mesh",
                              file=f"{link.name}.stl")
        if len(asset_el) == 0:
            root.remove(asset_el)

        worldbody = ET.SubElement(root, "worldbody")
        parent_body = ET.SubElement(worldbody, "body", name="mount", pos="0 0 0")

        prev_len = 0.0
        actuator_el = ET.SubElement(root, "actuator")

        for link in params.links:
            # Child offset is RELATIVE to the parent link (its length), so the chain
            # nests correctly — the gripper sits at the forearm tip, not below it.
            body = ET.SubElement(
                parent_body, "body",
                name=link.name,
                pos=f"0 0 {-prev_len:.5g}",
            )
            for joint in link.joints:
                lo_r = math.radians(joint.range_deg[0])
                hi_r = math.radians(joint.range_deg[1])
                ax = " ".join(f"{a:.5g}" for a in joint.axis)
                ET.SubElement(
                    body, "joint",
                    name=joint.name,
                    type=joint.type,
                    axis=ax,
                    range=f"{lo_r:.5g} {hi_r:.5g}",
                )
                ET.SubElement(
                    actuator_el, "motor",
                    joint=joint.name,
                    gear="20.0",
                )
            rgba = " ".join(f"{c:.5g}" for c in link.rgba)
            if has_mesh[link.name]:
                ET.SubElement(body, "geom", type="mesh",
                              mesh=f"{link.name}_mesh", rgba=rgba)
            else:
                ET.SubElement(
                    body, "geom",
                    type="capsule",
                    fromto=f"0 0 0 0 0 {-link.length:.5g}",
                    size=f"{link.radius:.5g}",
                    rgba=rgba,
                )
            prev_len = link.length
            parent_body = body

        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        mjcf_path = mjcf_dir / f"{name}.xml"
        tree.write(str(mjcf_path), encoding="unicode", xml_declaration=False)
        return mjcf_path

    def export_stl(self, params: Any, name: str = "candidate") -> Path:
        """Export STL from DesignParams or MorphologySpec (duck-typed)."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stl_path = self.output_dir / f"{name}.stl"
        design = self._coerce_to_design_params(params)
        triangles = self._build_arm_triangles(design)
        stl_path.write_bytes(self._pack_binary_stl(triangles))
        return stl_path

    # ── Geometry builders ────────────────────────────────────────────────────

    @staticmethod
    def _coerce_to_design_params(params: Any) -> DesignParams:
        """Return DesignParams from either DesignParams or MorphologySpec."""
        if isinstance(params, DesignParams):
            return params
        # MorphologySpec duck-typing: extract upper/forearm from links list
        links = getattr(params, "links", None) or params.get("links", [])
        upper_m = links[0].length_m if links else 0.30
        forearm_m = links[1].length_m if len(links) > 1 else 0.26
        joints = getattr(params, "joints", None) or params.get("joints", [])
        joint_limits: dict[str, tuple[float, float]] = {}
        for j in joints:
            jname = j.name if hasattr(j, "name") else j["name"]
            lo, hi = j.limits_rad if hasattr(j, "limits_rad") else j["limits_rad"]
            joint_limits[jname.split("_")[0]] = (math.degrees(lo), math.degrees(hi))
        return DesignParams(
            upper_arm_len=upper_m,
            forearm_len=forearm_m,
            joint_stiffness=1.0,
            grip_width=0.08,
            joint_limits=joint_limits,
        )

    def _link_triangles(self, link) -> list[tuple]:
        """Capsule geometry for a single LinkDef, extending along -Z from the origin."""
        radius = getattr(link, "radius", 0.025)
        length = getattr(link, "length", 0.30)
        tris = self._cylinder(
            center=(0.0, 0.0, -length / 2),
            axis=(0.0, 0.0, 1.0),
            radius=radius,
            height=length,
            segments=24,
        )
        tris += self._sphere(center=(0.0, 0.0, 0.0), radius=radius * 1.15, segments=12)
        tris += self._sphere(center=(0.0, 0.0, -length), radius=radius * 1.15, segments=12)
        return tris

    def _build_arm_triangles(self, params: DesignParams) -> list[tuple]:
        """Return list of (normal, v0, v1, v2) tuples for all arm components."""
        tris: list[tuple] = []
        r = 0.030  # tube radius (m)
        jr = r * 1.3  # joint radius

        # Shoulder socket (short wide cylinder at origin)
        tris += self._cylinder(
            center=(0, 0, 0), axis=(0, 1, 0),
            radius=r * 1.5, height=r * 2.5, segments=24,
        )

        # Upper arm
        ua_start = (0, -(r * 1.25), 0)
        tris += self._cylinder(
            center=(0, -(r * 1.25 + params.upper_arm_len / 2), 0),
            axis=(0, 1, 0), radius=r, height=params.upper_arm_len, segments=24,
        )

        # Elbow joint (sphere)
        elbow_y = -(r * 1.25 + params.upper_arm_len + jr)
        tris += self._sphere(center=(0, elbow_y, 0), radius=jr, segments=16)

        # Forearm (angled by elbow_deg, default 30°)
        elbow_deg = 30.0
        elbow_rad = math.radians(elbow_deg)
        dx = math.sin(elbow_rad) * (params.forearm_len / 2)
        dy = -math.cos(elbow_rad) * (params.forearm_len / 2)
        fa_center = (dx, elbow_y + dy, 0)
        fa_axis = (math.sin(elbow_rad), -math.cos(elbow_rad), 0)
        tris += self._cylinder(
            center=fa_center, axis=fa_axis,
            radius=r * 0.9, height=params.forearm_len, segments=24,
        )

        # Wrist disc at end of forearm
        wrist_x = math.sin(elbow_rad) * params.forearm_len
        wrist_y = elbow_y - math.cos(elbow_rad) * params.forearm_len
        tris += self._sphere(center=(wrist_x, wrist_y, 0), radius=r * 1.0, segments=12)

        return tris

    # ── Primitive generators ─────────────────────────────────────────────────

    def _cylinder(
        self, center: tuple, axis: tuple, radius: float, height: float, segments: int
    ) -> list[tuple]:
        """Tessellated cylinder (closed ends) oriented along axis."""
        tris: list[tuple] = []
        cx, cy, cz = center
        ax, ay, az = self._normalize(axis)

        # Build two perpendicular axes
        ux, uy, uz = self._perp(ax, ay, az)
        vx, vy, vz = self._cross(ax, ay, az, ux, uy, uz)

        top = [(cx + ax * height / 2, cy + ay * height / 2, cz + az * height / 2)]
        bot = [(cx - ax * height / 2, cy - ay * height / 2, cz - az * height / 2)]

        rings: list[list[tuple]] = [[], []]
        for i in range(segments):
            theta = 2 * math.pi * i / segments
            c, s = math.cos(theta) * radius, math.sin(theta) * radius
            for j, sign in enumerate([1, -1]):
                ox = cx + sign * ax * height / 2 + c * ux + s * vx
                oy = cy + sign * ay * height / 2 + c * uy + s * vy
                oz = cz + sign * az * height / 2 + c * uz + s * vz
                rings[j].append((ox, oy, oz))

        for i in range(segments):
            n = (i + 1) % segments
            # Side quads
            t0, t1 = rings[0][i], rings[0][n]
            b0, b1 = rings[1][i], rings[1][n]
            side_n = self._tri_normal(t0, t1, b0)
            tris.append((side_n, t0, t1, b0))
            tris.append((side_n, t1, b1, b0))
            # Top cap
            tris.append(((ax, ay, az), top[0], t0, t1))
            # Bottom cap
            tris.append(((-ax, -ay, -az), bot[0], rings[1][n], rings[1][i]))

        return tris

    def _sphere(self, center: tuple, radius: float, segments: int) -> list[tuple]:
        cx, cy, cz = center
        tris: list[tuple] = []
        rings = segments
        slices = segments * 2

        verts: list[list[tuple]] = []
        for i in range(rings + 1):
            phi = math.pi * i / rings
            row = []
            for j in range(slices):
                theta = 2 * math.pi * j / slices
                x = cx + radius * math.sin(phi) * math.cos(theta)
                y = cy + radius * math.cos(phi)
                z = cz + radius * math.sin(phi) * math.sin(theta)
                row.append((x, y, z))
            verts.append(row)

        for i in range(rings):
            for j in range(slices):
                n = (j + 1) % slices
                a, b = verts[i][j], verts[i][n]
                c, d = verts[i + 1][j], verts[i + 1][n]
                nm = self._tri_normal(a, b, c)
                tris.append((nm, a, b, c))
                tris.append((nm, b, d, c))

        return tris

    # ── Math helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _normalize(v: tuple) -> tuple:
        x, y, z = v
        mag = math.sqrt(x * x + y * y + z * z) or 1.0
        return (x / mag, y / mag, z / mag)

    @staticmethod
    def _perp(ax: float, ay: float, az: float) -> tuple:
        if abs(ax) <= abs(ay) and abs(ax) <= abs(az):
            return CadBridge._normalize((0, -az, ay))
        if abs(ay) <= abs(az):
            return CadBridge._normalize((-az, 0, ax))
        return CadBridge._normalize((-ay, ax, 0))

    @staticmethod
    def _cross(ax, ay, az, bx, by, bz) -> tuple:
        return (ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx)

    @staticmethod
    def _tri_normal(a: tuple, b: tuple, c: tuple) -> tuple:
        ax, ay, az = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        bx, by, bz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
        nx, ny, nz = ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx
        mag = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
        return (nx / mag, ny / mag, nz / mag)

    # ── Binary STL packer ────────────────────────────────────────────────────

    @staticmethod
    def _pack_binary_stl(triangles: list[tuple]) -> bytes:
        header = b"Prosthesis-RL candidate STL" + b"\x00" * (80 - 27)
        count = struct.pack("<I", len(triangles))
        body = bytearray()
        for tri in triangles:
            n = tri[0]
            vs = tri[1:4]
            body += struct.pack("<fff", *n)
            for v in vs:
                body += struct.pack("<fff", *v)
            body += struct.pack("<H", 0)
        return header + count + bytes(body)
