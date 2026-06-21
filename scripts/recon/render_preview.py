"""Render visual proof of the recon pipeline.

Outputs (repo root):
  - alley_preview.mp4 / .png : the ACTUAL mesh the alley video produced (honest:
    it's sparse, because the footage was unsuitable).
  - demo_room.mp4   / .png : MuJoCo simulating the verified synthetic room with a
    ball dropping + colliding (proof the engine works on good geometry).

    python3 scripts/recon/render_preview.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def render_mesh_orbit(ply: Path, out_mp4: Path, out_png: Path, frames: int = 90):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio
    import trimesh

    m = trimesh.load(str(ply), force="mesh")
    v = np.asarray(m.vertices)
    colors = (np.asarray(m.visual.vertex_colors)[:, :3] / 255.0
              if getattr(m.visual, "vertex_colors", None) is not None else None)

    imgs = []
    for i in range(frames):
        fig = plt.figure(figsize=(6, 5), dpi=100)
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(v[:, 0], v[:, 1], v[:, 2], c=colors, s=8)
        if len(m.faces):
            ax.plot_trisurf(v[:, 0], v[:, 1], v[:, 2], triangles=m.faces,
                            color=(0.6, 0.6, 0.65, 0.3), linewidth=0)
        ax.set_title(f"alley video -> COLMAP mesh\n{len(v)} verts, {len(m.faces)} faces")
        ax.view_init(elev=20, azim=i * 360 / frames)
        ax.set_axis_off()
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3]
        imgs.append(buf.copy())
        if i == frames // 3:
            imageio.imwrite(out_png, buf)
        plt.close(fig)
    imageio.mimsave(out_mp4, imgs, fps=30)
    print(f"[render] {out_mp4.name}  ({len(v)} verts)")


def render_mujoco_sim(scene_xml: Path, out_mp4: Path, out_png: Path,
                      seconds: float = 4.0):
    import mujoco
    import imageio

    # Brighten: inject extra lights into the generated scene before loading.
    # from_xml_string resolves meshes against cwd, so pin meshdir to the abs dir.
    xml = scene_xml.read_text()
    xml = xml.replace('meshdir="."', f'meshdir="{scene_xml.parent.resolve()}"')
    extra = ('    <light pos="2 2 3" dir="-1 -1 -1" diffuse="0.6 0.6 0.6"/>\n'
             '    <light pos="-2 -2 3" dir="1 1 -1" diffuse="0.5 0.5 0.5"/>\n')
    xml = xml.replace("<worldbody>", "<worldbody>\n" + extra, 1)

    model = mujoco.MjModel.from_xml_string(xml, {})
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = model.stat.center
    cam.distance = 2.7 * model.stat.extent
    cam.elevation = -40  # look down into the room so all four walls are visible

    fps = 30
    substeps = max(1, int(round((1.0 / fps) / model.opt.timestep)))
    n_frames = int(seconds * fps)
    imgs = []
    for i in range(n_frames):
        for _ in range(substeps):
            mujoco.mj_step(model, data)
        cam.azimuth = 110 + 50 * i / n_frames  # slow orbit
        renderer.update_scene(data, camera=cam)
        frame = renderer.render()
        imgs.append(frame)
        if i == n_frames // 2:
            imageio.imwrite(out_png, frame)
    imageio.mimsave(out_mp4, imgs, fps=fps)
    print(f"[render] {out_mp4.name}  ({n_frames} frames, ball drop + collide)")


def main():
    alley = ROOT / "room_mesh.ply"
    if alley.exists():
        render_mesh_orbit(alley, ROOT / "alley_preview.mp4",
                          ROOT / "alley_preview.png")
    demo = ROOT / "assets/scenes/demo_room/scene.xml"
    if demo.exists():
        render_mujoco_sim(demo, ROOT / "demo_room.mp4", ROOT / "demo_room.png")


if __name__ == "__main__":
    main()
