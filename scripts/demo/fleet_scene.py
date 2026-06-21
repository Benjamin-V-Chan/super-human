"""Build a grid MJCF of N independent arms for the live fleet view.

The live dashboard normally renders one arm playing the current PPO policy's
rollout. For the fleet, we instantiate the SAME arm N times in a grid — each cell
is its own randomized tie-shoe task (shoe + green target at a different spot) —
and stream one big qpos vector per frame (cell 0's joints, then cell 1's, ...).
The browser sets that qpos, runs FK once, and renders all N agents.

`build_fleet_scene` writes webdemo/assets/scenes/arm_fleet.xml with RELATIVE mesh
paths (arm_links/) so the WASM viewer can fetch them, and returns the per-cell
joint count + total nq so the trainer can lay out the streamed qpos to match.
"""

from __future__ import annotations

import math
from pathlib import Path

from prosthesis_rl.contracts import DesignParams
from prosthesis_rl.sim.mjcf_builder import _fmt, _human, _joint_range_rad

SCENES = Path(__file__).resolve().parents[2] / "webdemo" / "assets" / "scenes"
SHOE_STL = SCENES / "arm_links" / "shoe.stl"  # decimated Gizmo shoe (scripts/recon/decimate_shoe.py)


def _cell_arm(prefix: str, mount_world, design: DesignParams) -> str:
    """Nested arm body chain for one cell, names prefixed, mesh geoms shared."""
    links = design.links

    def joint_tag(j) -> str:
        lo, hi = _joint_range_rad(j)
        jtype = "slide" if j.type == "slide" else "hinge"
        return (f'<joint name="{prefix}{j.name}" type="{jtype}" axis="{_fmt(*j.axis)}" '
                f'range="{_fmt(lo, hi)}" damping="0.5" armature="0.01"/>')

    def link_geom(link) -> str:
        if link.mesh or True:  # all default links have a shared mesh by name
            return (f'<geom name="{prefix}{link.name}_geom" type="mesh" '
                    f'mesh="{link.name}_mesh" rgba="{_fmt(*link.rgba)}"/>')

    def body(k: int) -> str:
        link = links[k]
        joints = "\n        ".join(joint_tag(j) for j in link.joints)
        if k + 1 < len(links):
            child = body(k + 1)
        else:
            child = (f'<site name="{prefix}ee" pos="0 0 {_fmt(-link.length)}" '
                     'size="0.012" rgba="0.95 0.2 0.2 1" group="3"/>')
        pos = (0.0, 0.0, 0.0) if k == 0 else (0.0, 0.0, -links[k - 1].length)
        return (f'<body name="{prefix}{link.name}" pos="{_fmt(*pos)}">\n'
                f'        {joints}\n        {link_geom(link)}\n        {child}\n'
                f'      </body>')

    return (f'<body name="{prefix}mount" pos="{_fmt(*mount_world)}">\n'
            f'      <geom name="{prefix}mount_geom" type="box" size="0.04 0.04 0.04" '
            f'rgba="0.3 0.3 0.35 1"/>\n      {body(0)}\n    </body>')


def _cell(i: int, origin, mount_local, shoe_world, target_world, design: DesignParams,
          shoe_mesh: bool) -> str:
    """One full agent: arm + wearer + shoe + green target, offset to its grid cell."""
    p = f"c{i}_"
    dx, dy = origin
    mount_world = (mount_local[0] + dx, mount_local[1] + dy, mount_local[2])
    arm = _cell_arm(p, mount_world, design)
    # Wearer (visual). _human hardcodes name="human"; make it unique per cell.
    human = _human(mount_world, "left", collide=False).replace('name="human"', f'name="{p}human"')
    sx, sy = shoe_world[0] + dx, shoe_world[1] + dy
    tx, ty, tz = target_world[0] + dx, target_world[1] + dy, target_world[2]
    if shoe_mesh:  # the real (decimated) Gizmo shoe, a varied yaw per agent
        yaw = math.radians((i * 47) % 70 - 35)
        shoe = (f'<body name="{p}shoe" pos="{_fmt(sx, sy, 0.005)}" euler="0 0 {yaw:.4g}">'
                '<geom type="mesh" mesh="shoe_mesh" rgba="0.62 0.4 0.27 1" '
                'contype="0" conaffinity="0" group="2"/></body>')
    else:
        shoe = (f'<body name="{p}shoe" pos="{_fmt(sx, sy, 0.07)}">'
                '<geom type="box" size="0.05 0.12 0.045" rgba="0.55 0.35 0.22 1" '
                'contype="0" conaffinity="0" group="2"/></body>')
    target = (f'<body name="{p}target" pos="{_fmt(tx, ty, tz)}">'
              '<geom type="sphere" size="0.05" rgba="0.1 1 0.25 1" '
              'contype="0" conaffinity="0" group="0"/></body>')
    return f"    {arm}{human}\n    {shoe}\n    {target}"


def build_fleet_scene(
    design: DesignParams,
    mount_local,
    cell_specs,
    *,
    cols: int,
    spacing: float = 1.4,
) -> tuple[Path, int, int]:
    """Write arm_fleet.xml for the grid; return (path, nq, dof_per_cell).

    cell_specs[i] = (shoe_world_local, target_world_local) in the single-agent
    frame (the grid offset is applied here). qpos layout is cell-major: cell i's
    joints occupy [i*dof_per_cell : (i+1)*dof_per_cell].
    """
    n = len(cell_specs)
    dof = design.dof
    shoe_mesh = SHOE_STL.exists()
    mesh_assets = "".join(
        f'    <mesh name="{link.name}_mesh" file="arm_links/{link.name}.stl"/>\n'
        for link in design.links
    )
    if shoe_mesh:  # one shared mesh asset, instanced by every cell's shoe geom
        mesh_assets += '    <mesh name="shoe_mesh" file="arm_links/shoe.stl"/>\n'
    rows = math.ceil(n / cols)
    cells = []
    for i, (shoe_w, target_w) in enumerate(cell_specs):
        r, c = divmod(i, cols)
        # centre the grid on x; march rows back in -y so row 0 is nearest the camera
        dx = (c - (cols - 1) / 2.0) * spacing
        dy = (rows - 1 - r) * spacing
        cells.append(_cell(i, (dx, dy), mount_local, shoe_w, target_w, design, shoe_mesh))

    xml = f"""<mujoco model="fleet">
  <compiler angle="radian" autolimits="true" meshdir="."/>
  <option gravity="0 0 -9.81" integrator="implicitfast" timestep="0.002"/>
  <visual>
    <global offwidth="1280" offheight="720"/>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.6 0.6 0.6"/>
    <rgba haze="0.85 0.88 0.92 1"/>
  </visual>
  <asset>
    <texture name="skybox" type="skybox" builtin="gradient"
             rgb1="0.55 0.62 0.72" rgb2="0.10 0.12 0.16" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.30 0.32 0.36"
             rgb2="0.22 0.24 0.28" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="40 40" reflectance="0.05"/>
{mesh_assets}  </asset>
  <worldbody>
    <light pos="0 0 5" dir="0 0 -1" diffuse="0.9 0.9 0.9"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="grid"/>
{chr(10).join(cells)}
  </worldbody>
</mujoco>
"""
    SCENES.mkdir(parents=True, exist_ok=True)
    out = SCENES / "arm_fleet.xml"
    out.write_text(xml)
    return out, n * dof, dof
