"""Inject a reconstructed room scene into a prosthesis MuJoCo env.

Bridges scripts/recon output (assets/scenes/<name>/) into the `Scene` world-geom
the STRESS_TEST_PLAN Phase-1 verifier needs. The recon writer emits `room.xml`
as a `<mujocoinclude>` fragment with ABSOLUTE mesh paths, so it merges into any
parent model with no `meshdir` coupling.

Typical use from a (future) sim/scenes.py:

    from prosthesis_rl.sim import room_asset
    env_xml = room_asset.inject_into(robot_xml, "assets/scenes/kitchen")
    model = mujoco.MjModel.from_xml_string(env_xml)

Or just load the standalone scene the recon step produced:

    model = room_asset.load_standalone("assets/scenes/kitchen")
"""

from __future__ import annotations

from pathlib import Path


def room_xml_path(scene_dir: str | Path) -> Path:
    """Path to the injectable `<mujocoinclude>` fragment for a scene dir."""
    p = Path(scene_dir) / "room.xml"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found — run mesh_to_mjcf.py to build the scene first")
    return p.resolve()


def include_snippet(scene_dir: str | Path) -> str:
    """`<include .../>` line to splice the room into a parent `<mujoco>` model."""
    return f'<include file="{room_xml_path(scene_dir)}"/>'


def inject_into(env_xml: str, scene_dir: str | Path) -> str:
    """Insert the room include just before the closing </mujoco> of `env_xml`."""
    snippet = "  " + include_snippet(scene_dir) + "\n"
    idx = env_xml.rfind("</mujoco>")
    if idx == -1:
        raise ValueError("env_xml has no </mujoco> close tag")
    return env_xml[:idx] + snippet + env_xml[idx:]


def standalone_path(scene_dir: str | Path) -> Path:
    p = Path(scene_dir) / "scene.xml"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found — run mesh_to_mjcf.py first")
    return p.resolve()


def load_standalone(scene_dir: str | Path):
    """Load the self-contained scene.xml (room + floor + probe) as an MjModel."""
    import mujoco

    return mujoco.MjModel.from_xml_path(str(standalone_path(scene_dir)))


# Minimal env used by the demo / as an integration smoke test. A real env comes
# from sim/mjcf_builder.py (the prosthesis arm) per STRESS_TEST_PLAN.
_DEMO_ENV = """<mujoco model="env">
  <option gravity="0 0 -9.81" integrator="implicitfast"/>
  <worldbody>
    <light pos="0 0 4" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="0 0 0.05" rgba="0.3 0.3 0.3 1"/>
  </worldbody>
</mujoco>
"""


if __name__ == "__main__":
    import sys

    import mujoco

    scene_dir = sys.argv[1] if len(sys.argv) > 1 else "assets/scenes/room"
    env_xml = inject_into(_DEMO_ENV, scene_dir)
    model = mujoco.MjModel.from_xml_string(env_xml)
    n_col = sum(1 for i in range(model.ngeom) if model.geom_group[i] == 3)
    print(f"injected '{scene_dir}' -> env with {model.ngeom} geoms "
          f"({n_col} room collision pieces), {model.nbody} bodies. OK")
