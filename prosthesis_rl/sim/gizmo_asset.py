"""Splice task objects into the prosthesis env — the deferred Gizmo injector.

This is the consumer side of `scripts/recon/gizmo_assets.py`. That script bakes a
natural-language prompt into a physics-ready, articulated MJCF under
`assets/objects/<name>/`; this module drops that object (a drawer, a shoe, a jar)
into the arm scene at the scenario's world position, mirroring how
`room_asset.inject_into` splices a reconstructed room.

Two object sources, chosen per object:

  * **Gizmo MJCF** — when `SceneObject.mjcf_dir` points at a real bake, its
    `<asset>` and `<worldbody>` are merged into the env, wrapped in a positioned
    body. Mesh paths are rewritten absolute so there's no `meshdir` coupling.
  * **Fallback box** — until a bake lands (Gizmo generation is slow), a coloured,
    collidable primitive box stands in at the same position, so every scenario is
    visible and trainable immediately.

Waypoint markers (small green sites) are added so the viewer/renderer shows where
the hand is meant to go.

    from prosthesis_rl.sim.gizmo_asset import build_scenario_xml
    xml = build_scenario_xml(design, scenario, mesh_dir=arm_mesh_dir)
    model = mujoco.MjModel.from_xml_string(xml, {})
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from prosthesis_rl.contracts import DesignParams, SceneObject, ScenarioSpec, TaskWaypoint
from prosthesis_rl.sim.mjcf_builder import build_mjcf

_OBJECTS_ROOT = Path("assets/objects")


def _fmt(*vals: float) -> str:
    return " ".join(f"{v:.5g}" for v in vals)


# --------------------------------------------------------------------------- #
# Per-object MJCF.                                                             #
# --------------------------------------------------------------------------- #
def _fallback_body(obj: SceneObject, *, collide: bool = False) -> str:
    """A coloured box standing in for an un-baked object.

    Visual-only by default (contype/conaffinity 0, group 2 — like the wearer
    mannequin): for *reach* training the object marks where the task is, and the
    policy learns to bring the hand to it; a solid box would just wall the hand
    off at its surface. Pass ``collide=True`` once a task needs real contact
    (grasping/manipulation), or when a real articulated Gizmo asset replaces it.
    """
    hx, hy, hz = obj.fallback_half
    phys = ('friction="1 0.05 0.001" density="400"'
            if collide else 'contype="0" conaffinity="0" group="2"')
    return (
        f'    <body name="obj_{obj.name}" pos="{_fmt(*obj.pos)}" euler="{_fmt(*obj.euler)}">\n'
        f'      <geom name="obj_{obj.name}_geom" type="box" '
        f'size="{_fmt(hx, hy, hz)}" rgba="{_fmt(*obj.rgba)}" {phys}/>\n'
        f'    </body>'
    )


def _gizmo_object(obj: SceneObject) -> tuple[str, str] | None:
    """Parse a baked Gizmo MJCF into (asset_fragment, positioned_body_fragment).

    Defensive: any parsing problem returns None and the caller uses the fallback
    box, so a malformed/foreign export can never break scene assembly.
    """
    if not obj.mjcf_dir:
        return None
    mdir = Path(obj.mjcf_dir)
    if not mdir.is_absolute():
        mdir = (_OBJECTS_ROOT / obj.name) if not mdir.exists() else mdir
    candidates = sorted(mdir.rglob("*.xml")) + sorted(mdir.rglob("*.mjcf"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: (len(p.relative_to(mdir).parts), p.name))
    model_path = candidates[0]
    try:
        root = ET.parse(model_path).getroot()
    except (ET.ParseError, OSError):
        return None

    # Merge <asset>, rewriting mesh/texture file paths to absolute and **namespacing
    # every asset name** with the object slug. Gizmo exports reuse generic names
    # (mat_0, bake_…) so without a prefix two Gizmo exports (e.g. a room + a shoe)
    # collide on compile. Drop the export's own skybox texture (the env has one).
    prefix = f"{obj.name}__"
    name_map: dict[str, str] = {}
    asset_els: list = []
    for asset in root.findall("asset"):
        for el in list(asset):
            if el.tag == "texture":
                continue  # drop ALL textures: the web FS has no PNGs; flat material rgba is used
            nm = el.get("name")
            if nm is not None:
                name_map[nm] = prefix + nm
            asset_els.append(el)
    asset_parts: list[str] = []
    for el in asset_els:
        nm = el.get("name")
        if nm in name_map:
            el.set("name", name_map[nm])
        if el.tag == "material":  # textures dropped -> strip refs, keep flat rgba
            el.attrib.pop("texture", None)
            for layer in el.findall("layer"):
                el.remove(layer)
        asset_parts.append("    " + ET.tostring(el, encoding="unicode").strip())

    # Pull bodies/geoms out of <worldbody>, dropping world scaffolding (lights,
    # floor planes, cameras) so only the object itself is injected. Remap each
    # geom's material/mesh reference to the namespaced asset names.
    wb = root.find("worldbody")
    if wb is None:
        return None
    inner: list[str] = []
    n_geoms = 0
    for el in list(wb):
        if el.tag in {"light", "camera"}:
            continue
        if el.tag == "geom" and el.get("type") == "plane":
            continue
        geoms = list(el.iter("geom"))
        n_geoms += len(geoms)
        for g in geoms:
            for attr in ("material", "mesh"):
                if g.get(attr) in name_map:
                    g.set(attr, name_map[g.get(attr)])
        inner.append("      " + ET.tostring(el, encoding="unicode").strip())
    # No actual geometry (e.g. a cancelled/placeholder Gizmo asset = an empty body)
    # -> fall back to the coloured marker box so the object is never invisible.
    if not inner or n_geoms == 0:
        return None
    body = (
        f'    <body name="obj_{obj.name}" pos="{_fmt(*obj.pos)}" euler="{_fmt(*obj.euler)}">\n'
        + "\n".join(inner)
        + "\n    </body>"
    )
    return "\n".join(asset_parts), body


def object_fragments(obj: SceneObject, *, collide: bool = False) -> tuple[str, str]:
    """(asset_fragment, body_fragment) for one object — Gizmo bake or fallback box."""
    parsed = _gizmo_object(obj)
    if parsed is not None:
        return parsed
    return "", _fallback_body(obj, collide=collide)


# --------------------------------------------------------------------------- #
# Injection into a built env.                                                  #
# --------------------------------------------------------------------------- #
def _splice_before(env_xml: str, tag: str, fragment: str) -> str:
    if not fragment.strip():
        return env_xml
    idx = env_xml.rfind(tag)
    if idx == -1:
        raise ValueError(f"env_xml has no {tag} to splice before")
    return env_xml[:idx] + fragment.rstrip("\n") + "\n  " + env_xml[idx:]


def inject_objects(env_xml: str, objects: list[SceneObject], *, collide: bool = False) -> str:
    """Merge every object's assets (before </asset>) and body (before </worldbody>)."""
    for obj in objects:
        assets, body = object_fragments(obj, collide=collide)
        if assets.strip():
            env_xml = _splice_before(env_xml, "</asset>", assets + "\n")
        env_xml = _splice_before(env_xml, "</worldbody>", body + "\n")
    return env_xml


def waypoint_markers(waypoints: list[TaskWaypoint]) -> str:
    """Small non-colliding sites marking each reach waypoint (viewer/debug aid)."""
    lines = []
    for i, wp in enumerate(waypoints):
        # Terminal/grasp goals (higher weight) render larger and greener.
        size = 0.018 + 0.006 * min(wp.weight, 3.0)
        lines.append(
            f'    <site name="wp_{i}_{wp.name}" pos="{_fmt(*wp.pos)}" '
            f'size="{size:.4g}" rgba="0.15 0.85 0.25 0.65" group="1"/>'
        )
    return "\n".join(lines)


def inject_waypoint_markers(env_xml: str, waypoints: list[TaskWaypoint]) -> str:
    return _splice_before(env_xml, "</worldbody>", waypoint_markers(waypoints) + "\n")


# --------------------------------------------------------------------------- #
# One-call scene assembly.                                                     #
# --------------------------------------------------------------------------- #
def build_scenario_xml(
    design: DesignParams,
    scenario: ScenarioSpec,
    *,
    mesh_dir: str | Path | None = None,
    add_human: bool = True,
    add_markers: bool = True,
) -> str:
    """Build the full env MJCF for a scenario: arm at the posture mount + objects.

    The reach `target` site is placed at the scenario's primary (most task-defining)
    waypoint, and the rest are added as markers.
    """
    primary = scenario.primary_waypoint()
    human_side = "left" if "right" not in (scenario.primary_action or "").lower() else "left"
    xml = build_mjcf(
        design,
        mount_pos=tuple(scenario.mount_pos),
        target_pos=tuple(primary.pos),
        mesh_dir=mesh_dir,
        add_human=add_human,
        human_side=human_side,
        name=f"scenario_{scenario.task_id or 'task'}",
    )
    xml = inject_objects(xml, scenario.objects)
    if add_markers:
        extra = [w for w in scenario.waypoints if w is not primary]
        if extra:
            xml = inject_waypoint_markers(xml, extra)
    return xml
