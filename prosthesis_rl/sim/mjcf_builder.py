"""DesignParams -> MuJoCo MJCF for a prosthesis arm, from the agent's own DoF.

The body tree, joints, and per-link geometry all come from `design.links` (the
explicit kinematic chain the design agent emits) — the arm is no longer a fixed
4-DoF assumption baked in here. Pass `mesh_dir` (a cad.bridge `export_arm`
output) to skin each moving body with its real per-link STL; without it, each
link falls back to a primitive capsule sized by the chain.

    from prosthesis_rl.cad.bridge import CadBridge
    from prosthesis_rl.sim.mjcf_builder import build_mjcf
    mesh_dir = CadBridge().export_arm(design, name="cand")
    xml = build_mjcf(design, mount_pos=(0, -0.4, 1.0),
                     target_pos=(0.0, 0.2, 0.9), mesh_dir=mesh_dir)
    model = mujoco.MjModel.from_xml_string(xml, {})

A reconstructed room is spliced in afterwards with
`prosthesis_rl.sim.room_asset.inject_into(xml, scene_dir)`; this env carries the
floor the room fragment does not.
"""

from __future__ import annotations

import math
from pathlib import Path

from prosthesis_rl.contracts import DesignParams, JointDef, LinkDef

# Default print-material density (Nylon PA12-CF, the STRESS_TEST_PLAN default), kg/m^3.
DEFAULT_DENSITY = 1010.0

EE_SITE = "ee"
TARGET_SITE = "target"

# Default arm joint names (the canonical 4-DoF chain) — kept for callers that
# import a module-level ARM. Prefer arm_joint_names(design) for the real model.
ARM = tuple(j.name for link in DesignParams().links for j in link.joints)


def _fmt(*vals: float) -> str:
    return " ".join(f"{v:.5g}" for v in vals)


def _joint_range_rad(joint: JointDef) -> tuple[float, float]:
    """Compiled (radian) range for a joint declared in degrees (hinge) or m (slide)."""
    lo, hi = joint.range_deg
    if joint.type == "slide":
        return float(lo), float(hi)
    return math.radians(lo), math.radians(hi)


def joint_ranges(design: DesignParams) -> dict[str, tuple[float, float]]:
    """Compiled (radian) range per joint, in actuator order — shared with the IK/RL controller."""
    return {
        j.name: _joint_range_rad(j)
        for link in design.links
        for j in link.joints
    }


def arm_joint_names(design: DesignParams) -> list[str]:
    """Joint names in actuator/qpos order for this design."""
    return design.joint_names


# ── MJCF fragments ───────────────────────────────────────────────────────────


def _joint_tag(joint: JointDef) -> str:
    lo, hi = _joint_range_rad(joint)
    jtype = "slide" if joint.type == "slide" else "hinge"
    return (f'<joint name="{joint.name}" type="{jtype}" axis="{_fmt(*joint.axis)}" '
            f'range="{_fmt(lo, hi)}" damping="0.5" armature="0.01"/>')


def _link_geom(link: LinkDef, mesh_dir: Path | None, density: float) -> str:
    name = f"{link.name}_geom"
    rgba = _fmt(*link.rgba)
    if mesh_dir is not None and (mesh_dir / f"{link.name}.stl").exists():
        return (f'<geom name="{name}" type="mesh" mesh="{link.name}_mesh" '
                f'rgba="{rgba}" density="{density}"/>')
    return (f'<geom name="{name}" type="capsule" '
            f'fromto="0 0 0 0 0 {_fmt(-link.length)}" size="{link.radius:.5g}" '
            f'rgba="{rgba}" density="{density}"/>')


def _body_tree(links: tuple[LinkDef, ...], mesh_dir: Path | None,
               density: float) -> str:
    """Nested <body> chain: link[i] is a child of link[i-1], offset by its length."""
    def body(i: int) -> str:
        link = links[i]
        joints = "\n        ".join(_joint_tag(j) for j in link.joints)
        geom = _link_geom(link, mesh_dir, density)
        if i + 1 < len(links):
            child = body(i + 1)
        else:
            child = (f'<site name="{EE_SITE}" pos="0 0 {_fmt(-link.length)}" '
                     f'size="0.012" rgba="0.95 0.2 0.2 1" group="1"/>')
        pos = (0.0, 0.0, 0.0) if i == 0 else (0.0, 0.0, -links[i - 1].length)
        return (f'<body name="{link.name}" pos="{_fmt(*pos)}">\n'
                f'        {joints}\n        {geom}\n        {child}\n'
                f'      </body>')
    return body(0)


def build_mjcf(
    design: DesignParams,
    scene=None,  # reserved: Scene perturbations (STRESS_TEST_PLAN Phase 1)
    *,
    mount_pos: tuple[float, float, float] = (0.0, -0.40, 1.00),
    target_pos: tuple[float, float, float] = (0.0, 0.18, 0.88),
    density: float = DEFAULT_DENSITY,
    add_floor: bool = True,
    add_human: bool = True,
    human_side: str = "left",
    human_collide: bool = False,
    mesh_dir: str | Path | None = None,
    name: str = "prosthesis_env",
) -> str:
    """Generate a complete MJCF env string for the design's arm + a reach target.

    mount_pos:  world pose of the fixed shoulder bracket (a person's shoulder).
    target_pos: world pose of the reach-target marker (the ADL goal point).
    add_human:  add a seated wearer whose `human_side` shoulder is bare at the mount.
    human_collide: make the wearer solid so the arm can't phase through the body
                (default False keeps the original visual-only, physics-neutral wearer).
    mesh_dir:   cad.bridge export_arm() dir; skins each link with its STL.
    """
    del scene  # v1 demo ignores scene perturbations; room is injected separately

    links = design.links
    if not links:
        raise ValueError("design has no links to build")
    mdir = Path(mesh_dir).resolve() if mesh_dir is not None else None
    stiffness = max(0.1, float(design.joint_stiffness))

    # Mesh assets (absolute paths, like room_asset, so there's no meshdir coupling).
    mesh_assets = ""
    if mdir is not None:
        for link in links:
            p = mdir / f"{link.name}.stl"
            if p.exists():
                mesh_assets += f'    <mesh name="{link.name}_mesh" file="{p}"/>\n'

    body_tree = _body_tree(links, mdir, density)
    worldbody = f"""
    <light pos="0 0 4" dir="0 0 -1" diffuse="0.9 0.9 0.9"/>{_floor() if add_floor else ""}
    <site name="{TARGET_SITE}" pos="{_fmt(*target_pos)}" size="0.025"
          rgba="0.1 0.9 0.2 0.9" group="1"/>
    <body name="mount" pos="{_fmt(*mount_pos)}">
      <geom name="mount_geom" type="box" size="0.04 0.04 0.04" rgba="0.3 0.3 0.35 1" density="{density}"/>
      {body_tree}
    </body>{_human(mount_pos, human_side, collide=human_collide) if add_human else ""}"""

    actuators = "\n    ".join(
        f'<position name="act_{j.name}" joint="{j.name}" '
        f'kp="{220.0 * stiffness:.4g}" kv="{18.0 * stiffness:.4g}" '
        f'ctrlrange="{_fmt(*_joint_range_rad(j))}" forcerange="-80 80"/>'
        for link in links for j in link.joints
    )

    # Exclude parent/child self-contacts (mount->link0, link[i]->link[i+1]).
    excludes = f'<exclude body1="mount" body2="{links[0].name}"/>'
    for a, b in zip(links, links[1:]):
        excludes += f'\n    <exclude body1="{a.name}" body2="{b.name}"/>'
    # When the wearer is solid, the proximal arm (mount + first link) shares the
    # shoulder volume, so exclude it from the body to avoid a start-pose self-jam.
    # The distal links (forearm/gripper) — the ones that visibly phased through —
    # still collide and get blocked.
    if add_human and human_collide:
        excludes += '\n    <exclude body1="mount" body2="human"/>'
        excludes += f'\n    <exclude body1="{links[0].name}" body2="human"/>'

    return f"""<mujoco model="{name}">
  <compiler angle="radian" autolimits="true"/>
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
    <material name="grid" texture="grid" texrepeat="14 14" reflectance="0.05"/>
{mesh_assets}  </asset>
  <worldbody>{worldbody}
  </worldbody>
  <contact>
    {excludes}
  </contact>
  <actuator>
    {actuators}
  </actuator>
</mujoco>
"""


def _floor() -> str:
    return ('\n    <geom name="floor" type="plane" size="0 0 0.05" '
            'material="grid"/>')


# ── Decorative seated human (the prosthesis "wearer") ─────────────────────────
# A capsule mannequin whose LEFT shoulder is bare, aligned to the arm `mount`, so
# the prosthesis reads as attached where the missing arm would be. Purely visual
# (contype/conaffinity 0, group 2) — it never generates contacts, so reach
# physics, energy/self-collision metrics, and room-clearance target sampling are
# all unchanged. Person faces +y (toward the reach target); built from the same
# primitive vocabulary as the arm.

_SKIN = "0.86 0.66 0.52 1"
_SHIRT = "0.27 0.42 0.55 1"
_PANTS = "0.20 0.22 0.28 1"
_SEAT = "0.34 0.30 0.27 1"


def _vgeom(body: str, collide: bool = False) -> str:
    """A body geom line. Visual-only by default; `collide` makes it block the arm.

    Collidable wearer geoms keep group 2 (still rendered) but get real
    contype/conaffinity so the prosthesis physically cannot pass through the torso,
    lap, or legs. All wearer geoms live in the single `human` body, so MuJoCo
    auto-excludes their mutual contacts — only arm↔wearer contacts are generated.
    """
    cc = 'contype="1" conaffinity="1"' if collide else 'contype="0" conaffinity="0"'
    return f'      <geom {body} {cc} group="2"/>'


def _cap(p1, p2, size: float, rgba: str, collide: bool = False) -> str:
    return _vgeom(f'type="capsule" fromto="{_fmt(*p1, *p2)}" '
                  f'size="{size:.4g}" rgba="{rgba}"', collide)


def _sph(p, size: float, rgba: str, collide: bool = False) -> str:
    return _vgeom(f'type="sphere" pos="{_fmt(*p)}" size="{size:.4g}" rgba="{rgba}"', collide)


def _box(p, half, rgba: str, collide: bool = False) -> str:
    return _vgeom(f'type="box" pos="{_fmt(*p)}" size="{_fmt(*half)}" rgba="{rgba}"', collide)


def _human(mount_pos: tuple[float, float, float], side: str = "left",
           shoulder_half: float = 0.26, collide: bool = False) -> str:
    """Seated capsule human with the `side` shoulder bare for the prosthesis.

    With `collide=True` the wearer's mass (torso, head, lap, legs, intact arm) is
    solid so the prosthesis cannot phase through it. The bare shoulder-line capsule
    and the stool stay non-colliding: the shoulder is the arm's own attachment
    (it would self-jam at the mount) and the stool is decorative ground furniture.
    """
    mx, my, mz = mount_pos
    s = 1.0 if side == "left" else -1.0   # +x toward the spine from the bare shoulder
    cx = mx + s * shoulder_half           # spine / body centerline
    rsx = mx + s * 2 * shoulder_half      # remaining (intact) shoulder
    hip = 0.47                            # seat height
    c = collide
    geoms = [
        # shoulder line (attachment — never collide) + torso (shirt)
        _cap((mx, my, mz), (rsx, my, mz), 0.05, _SHIRT, collide=False),
        _cap((cx, my, hip + 0.02), (cx, my, mz + 0.01), 0.12, _SHIRT, collide=c),
        # neck + head (skin)
        _cap((cx, my, mz + 0.02), (cx, my, mz + 0.09), 0.045, _SKIN, collide=c),
        _sph((cx, my, mz + 0.21), 0.105, _SKIN, collide=c),
        # remaining arm: upper -> fore -> hand, resting forward in the lap (skin)
        _cap((rsx, my, mz), (rsx + s * 0.04, my + 0.10, mz - 0.28), 0.05, _SKIN, collide=c),
        _cap((rsx + s * 0.04, my + 0.10, mz - 0.28),
             (rsx - s * 0.02, my + 0.30, hip + 0.10), 0.045, _SKIN, collide=c),
        _sph((rsx - s * 0.02, my + 0.30, hip + 0.10), 0.05, _SKIN, collide=c),
        # pelvis (pants)
        _cap((cx - 0.10, my, hip), (cx + 0.10, my, hip), 0.10, _PANTS, collide=c),
        # thighs forward to the knees, then shins down to the floor (pants)
        _cap((cx - 0.09, my, hip), (cx - 0.09, my + 0.38, hip - 0.02), 0.07, _PANTS, collide=c),
        _cap((cx + 0.09, my, hip), (cx + 0.09, my + 0.38, hip - 0.02), 0.07, _PANTS, collide=c),
        _cap((cx - 0.09, my + 0.38, hip - 0.02), (cx - 0.09, my + 0.40, 0.07), 0.055, _PANTS, collide=c),
        _cap((cx + 0.09, my + 0.38, hip - 0.02), (cx + 0.09, my + 0.40, 0.07), 0.055, _PANTS, collide=c),
        # feet (skin)
        _cap((cx - 0.09, my + 0.40, 0.05), (cx - 0.09, my + 0.52, 0.04), 0.04, _SKIN, collide=c),
        _cap((cx + 0.09, my + 0.40, 0.05), (cx + 0.09, my + 0.52, 0.04), 0.04, _SKIN, collide=c),
        # solid stool to the floor (decorative — never collide)
        _box((cx, my, 0.19), (0.22, 0.20, 0.19), _SEAT, collide=False),
    ]
    return ('\n    <body name="human" pos="0 0 0">\n'
            + "\n".join(geoms)
            + "\n    </body>")
