"""Merge a whole Gizmo *scene* MJCF into the prosthesis arm env.

`gizmo_asset.py` injects ONE baked object into a hand-authored layout. This module
does the other thing the demo now needs: take a **complete Gizmo scene** (the
floor, walls, table and task objects the AI generated from the problem text) and
drop the arm into it, so the policy acts inside a real environment instead of next
to a single floating object.

What a Gizmo scene export looks like (learned from a real export):
  * `<compiler angle="radian">` — same convention as our arm env (no unit clash).
  * inline meshes (`<mesh vertex=... face=...>`), so there are **no external mesh
    files** — the geometry travels inside the XML.
  * every geom is a mesh that references a `<material>`; every material carries an
    `rgba` and references **no** texture, so the export's texture PNGs (tens of MB)
    are unused and can be dropped — leaving a self-contained, much smaller model.
  * movable objects use `<freejoint>`, which MuJoCo only allows on direct children
    of <worldbody> — so the scene bodies are injected at top level (never wrapped),
    and the whole scene is aligned to the arm by translating each top-level body.
  * a `<default>` with global geom/joint settings — isolated here under a named
    class so it tunes only the Gizmo geometry, never the arm.

The merge keeps the arm env as the base (its compiler/option/contacts and the
EE/target-site conventions the RL env + viewer depend on) and drops the scene's
own `<option>/<sensor>/<keyframe>/<size>/<visual>` (the keyframe especially —
its qpos snapshot would not match the arm-augmented model).

    from prosthesis_rl.sim.gizmo_scene_merge import build_scenario_scene_xml
    xml = build_scenario_scene_xml(design, scenario, scene_xml_path, mesh_dir=arm_dir)
    model = mujoco.MjModel.from_xml_string(xml, {})   # arm + Gizmo room, one model
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from prosthesis_rl.contracts import DesignParams, ScenarioSpec
from prosthesis_rl.sim.gizmo_asset import inject_objects, waypoint_markers
from prosthesis_rl.sim.mjcf_builder import build_mjcf

# Top-level scene sections we deliberately DROP when merging into the arm env
# (the arm env supplies its own, or they would conflict with the augmented model).
_DROP_SECTIONS = {"compiler", "option", "size", "sensor", "keyframe", "custom",
                  "statistic", "visual"}


def _vec(s: str) -> list[float]:
    return [float(x) for x in s.split()]


def inject_gizmo_scene(
    env_xml: str,
    scene_xml_path: str | Path,
    *,
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    childclass: str = "gizmo",
    drop_textures: bool = True,
) -> str:
    """Return `env_xml` with the Gizmo scene at `scene_xml_path` merged in.

    `offset` translates the whole scene (added to every top-level body's pos) so
    the room can be aligned to the arm. `drop_textures` removes the unused texture
    PNG references (keeping flat material colours) so the model is self-contained
    and small. Gizmo geoms get their material's `rgba` baked on so the THREE-based
    web viewer (which reads geom_rgba, not materials) colours them correctly.
    """
    env = ET.fromstring(env_xml)
    scene = ET.parse(str(scene_xml_path)).getroot()
    sasset = scene.find("asset")
    swb = scene.find("worldbody")
    if sasset is None or swb is None:
        return env_xml  # not a scene we understand — leave the arm env untouched

    # Carry the Gizmo compiler's mass/inertia safety bounds onto the env compiler:
    # Gizmo scenes have decorative bodies with ~zero mass + a joint that only compile
    # because Gizmo sets boundmass/boundinertia/balanceinertia. The env compiler wins
    # the merge, so without these the merged model errors ("mass ... must be > 0").
    scompiler, ecompiler = scene.find("compiler"), env.find("compiler")
    if ecompiler is not None:
        ecompiler.set("balanceinertia", "true")
        ecompiler.set("boundmass",
                      (scompiler.get("boundmass") if scompiler is not None else None) or "1e-6")
        ecompiler.set("boundinertia",
                      (scompiler.get("boundinertia") if scompiler is not None else None) or "1e-8")

    mat_rgba = {m.get("name"): m.get("rgba")
                for m in sasset.findall("material") if m.get("rgba")}

    # 1) Isolate the scene's physics defaults under a named class so they affect
    #    only the Gizmo geometry (the arm keeps MuJoCo's built-in defaults).
    sdef = scene.find("default")
    if sdef is not None and len(sdef):
        env_default = env.find("default")
        if env_default is None:
            env_default = ET.Element("default")
            env.insert(_section_index(env, ("compiler", "option", "size")), env_default)
        cls = ET.SubElement(env_default, "default")
        cls.set("class", childclass)
        for child in list(sdef):
            cls.append(child)

    # 2) Merge assets (meshes + materials). Drop the textures (tens of MB of PNGs)
    #    and, since materials reference them via MuJoCo 3.x PBR <layer> children
    #    (base/normal/roughness/metallic), strip those layer refs + any texture=
    #    attr so each material keeps only its flat `rgba` and still compiles with no
    #    PNG files present.
    env_asset = env.find("asset")
    if env_asset is None:
        env_asset = ET.Element("asset")
        env.insert(_section_index(env, ("default", "compiler", "option")), env_asset)
    for el in list(sasset):
        if el.tag == "texture":
            if drop_textures:
                continue
        elif el.tag == "material" and drop_textures:
            el.attrib.pop("texture", None)
            for layer in el.findall("layer"):
                el.remove(layer)
        env_asset.append(el)

    # 3) Merge worldbody: inject each top-level child at top level (freejoints
    #    forbid wrapping). Translate top-level bodies by `offset`, tag them with the
    #    isolated class, and bake material rgba onto their geoms.
    env_wb = env.find("worldbody")
    ox, oy, oz = offset
    for el in list(swb):
        if el.tag == "body":
            if any(offset):
                p = _vec(el.get("pos", "0 0 0"))
                el.set("pos", f"{p[0] + ox:.6g} {p[1] + oy:.6g} {p[2] + oz:.6g}")
            if childclass:
                el.set("childclass", childclass)
            for g in el.iter("geom"):
                if not g.get("rgba") and g.get("material") in mat_rgba:
                    g.set("rgba", mat_rgba[g.get("material")])
            env_wb.append(el)
        elif el.tag in ("light", "site"):
            # keep scene lights (nicer MuJoCo render) + the spawn site (alignment aid)
            if el.tag == "site" and any(offset):
                p = _vec(el.get("pos", "0 0 0"))
                el.set("pos", f"{p[0] + ox:.6g} {p[1] + oy:.6g} {p[2] + oz:.6g}")
            env_wb.append(el)
        # cameras / planes from the scene are dropped (env owns its floor/view)

    return ET.tostring(env, encoding="unicode")


def _section_index(parent: ET.Element, after_tags: tuple[str, ...]) -> int:
    """Index just past the last of `after_tags` present in parent (for ordered insert)."""
    idx = 0
    for i, child in enumerate(parent):
        if child.tag in after_tags:
            idx = i + 1
    return idx


_STRUCTURAL = ("floor", "wall", "ceiling", "door", "window", "crown", "sill",
               "jamb", "header", "baseboard", "molding", "frame", "glass")


_MANIPULAND = ("handle", "knob", "drawer", "lid", "cap")  # the actual graspable part


def interactable_pos(scene_xml_path: str | Path) -> tuple[float, float, float] | None:
    """World position of the scene's graspable task part (for aligning the arm).

    Walks the body tree accumulating world positions (a Gizmo cabinet's top-level
    origin sits at its back-against-the-wall, so we must descend to the actual
    drawer/handle sub-body). Prefers a manipuland (handle/drawer/...), else any
    interactable/articulated body, else the first non-structural body. Returns None
    if the scene is only structure (floor/walls/etc.)."""
    root = ET.parse(str(scene_xml_path)).getroot()
    wb = root.find("worldbody")
    if wb is None:
        return None
    best: tuple[int, tuple[float, float, float]] | None = None

    def walk(el, acc):
        nonlocal best
        for b in el.findall("body"):
            p = _vec(b.get("pos", "0 0 0"))
            wp = (acc[0] + p[0], acc[1] + p[1], acc[2] + p[2])
            n = (b.get("name") or "").lower()
            if not any(s in n for s in _STRUCTURAL):
                if any(k in n for k in _MANIPULAND):
                    score = 3
                elif "interactable" in n or "articulat" in n:
                    score = 2
                else:
                    score = 1
                if best is None or score > best[0]:
                    best = (score, wp)
            walk(b, wp)

    walk(wb, (0.0, 0.0, 0.0))
    return best[1] if best else None


def spawn_site_pos(scene_xml_path: str | Path) -> tuple[float, float, float] | None:
    """The Gizmo `*Robot_Spawn` site position, if present — where the AI expects a
    robot to stand. Useful for auto-aligning the arm to the generated scene."""
    root = ET.parse(str(scene_xml_path)).getroot()
    wb = root.find("worldbody")
    if wb is None:
        return None
    for site in wb.findall("site"):
        if "spawn" in (site.get("name") or "").lower():
            return tuple(_vec(site.get("pos", "0 0 0")))  # type: ignore[return-value]
    return None


def align_offset(scene_xml_path: str | Path, scenario, *, front: float = 0.8
                 ) -> tuple[float, float, float]:
    """Translate offset that lines the scene's task up with the arm (x/y only).

    Prefer Gizmo's `*Robot_Spawn` site (under the arm's shoulder); else put the main
    graspable object `front` metres in front of the arm. Z is never shifted so the
    scene stays on its floor. Shared by the live trainer and the scene viewer."""
    mx, my, _ = scenario.mount_pos
    spawn = spawn_site_pos(scene_xml_path)
    if spawn:
        return (mx - spawn[0], my - spawn[1], 0.0)
    anchor = interactable_pos(scene_xml_path)
    if anchor:
        return (mx - anchor[0], (my + front) - anchor[1], 0.0)
    return (0.0, 0.0, 0.0)


def room_center_offset(scene_xml_path: str | Path, scenario) -> tuple[float, float, float]:
    """Offset that drops the arm into the centre of the scene's floor (x/y; z kept).

    For a scene used as an *environment* (a room/house) rather than the task object,
    centre the arm on the floor body so it stands inside the room with the walls
    around it."""
    mx, my, _ = scenario.mount_pos
    # Use MuJoCo's computed geom AABB to find the floor's true TOP surface (Gizmo
    # floors are thick boxes/meshes): centre the arm on the floor (x/y) and drop the
    # room so the floor's TOP sits at z=0 — otherwise a thick floor swallows objects
    # resting on it. Falls back to the floor body's origin if anything goes wrong.
    try:
        import mujoco
        m = mujoco.MjModel.from_xml_path(str(scene_xml_path))
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        for g in range(m.ngeom):
            if "floor" in (m.body(m.geom_bodyid[g]).name or "").lower():
                cx, cy, cz = (float(x) for x in d.geom_xpos[g])
                ab = m.geom_aabb[g]  # local [cx,cy,cz, hx,hy,hz]
                top = cz + float(ab[2]) + float(ab[5])
                return (mx - cx, my - cy, -top)
    except Exception:
        pass
    root = ET.parse(str(scene_xml_path)).getroot()
    wb = root.find("worldbody")
    fx = fy = fz = 0.0
    if wb is not None:
        floor = next((b for b in wb.findall("body")
                      if "floor" in (b.get("name") or "").lower()), None)
        if floor is not None:
            fx, fy, fz = _vec(floor.get("pos", "0 0 0"))[:3]
    return (mx - fx, my - fy, -fz)


def publish_merged_scene(arm_xml_path: str | Path, scene_xml_path: str | Path,
                         scenario, out_path: str | Path, *, center: bool = True,
                         front: float = 0.8) -> Path:
    """Write `out_path` = the web arm scene with the Gizmo scene + waypoint markers
    merged in, aligned to the task. Atomic (tmp + rename) so a viewer polling the
    file never reads a half-written scene. Used by both train_live.py (live PPO in
    the scene) and scene_server.py (static scene viewer).

    Two modes:
      * **room** (`center=True`, default) — the Gizmo scene is the *environment*
        (a room/house): centre the arm in it and drop it onto z=0. The task objects
        (e.g. the shoe) are expected to already be in `arm_xml` (write_web_scene
        injects the decimated shoe), so they are NOT injected here.
      * **object** (`center=False`) — the Gizmo scene *is* the task object (a
        cabinet/table): align it to where the arm reaches.
    """
    arm_xml = Path(arm_xml_path).read_text()
    offset = (room_center_offset(scene_xml_path, scenario) if center
              else align_offset(scene_xml_path, scenario, front=front))
    merged = inject_gizmo_scene(arm_xml, scene_xml_path, offset=offset)
    merged = merged.replace("</worldbody>",
                            waypoint_markers(scenario.waypoints) + "\n</worldbody>", 1)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(merged)
    tmp.replace(out)
    return out


def build_scenario_scene_xml(
    design: DesignParams,
    scenario: ScenarioSpec,
    scene_xml_path: str | Path,
    *,
    mesh_dir: str | Path | None = None,
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    add_human: bool = True,
    add_markers: bool = True,
) -> str:
    """Full merged MJCF: the arm at the scenario's posture, inside the Gizmo scene.

    The arm's reach target is the scenario's primary (most task-defining) waypoint;
    the remaining authored waypoints are added as markers. The Gizmo geometry is
    the environment; the authored waypoints stay the RL targets (reachable by
    construction), per the chosen design.
    """
    primary = scenario.primary_waypoint()
    xml = build_mjcf(
        design,
        mount_pos=tuple(scenario.mount_pos),
        target_pos=tuple(primary.pos),
        mesh_dir=mesh_dir,
        add_human=add_human,
        name=f"scenario_{scenario.task_id or 'task'}",
    )
    xml = inject_gizmo_scene(xml, scene_xml_path, offset=offset)
    if add_markers:
        extra = [w for w in scenario.waypoints if w is not primary]
        if extra:
            # markers fragment -> splice before </worldbody>
            frag = waypoint_markers(extra)
            xml = xml.replace("</worldbody>", frag + "\n</worldbody>", 1)
    return xml
