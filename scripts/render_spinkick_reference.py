"""Render a processed Microduck BeyondMimic reference motion.

The renderer reads the exact ``motion.npz`` consumed by MJLab training, so the
preview includes the standing transitions, repeated 1.0x action, final hold,
and the final trajectory-dynamics projection rather than an intermediate UMR
result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
SCENE_XML = (
    REPO_ROOT / "src/mjlab_microduck/robot/microduck/scene.xml"
)


def _joint_qpos_addresses(model: mujoco.MjModel, names: list[str]) -> np.ndarray:
    addresses: list[int] = []
    for name in names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"motion joint {name!r} is missing from {SCENE_XML}")
        addresses.append(int(model.jnt_qposadr[joint_id]))
    return np.asarray(addresses, dtype=np.int32)


def _phase(frame: int, action_start: int, action_end: int, hold_start: int) -> str:
    if frame < action_start:
        return "STAND -> ACTION"
    if frame < action_end:
        return "MIMICKIT SPINKICK  1.0x"
    if frame < hold_start:
        return "ACTION -> RL ZERO"
    return "RL ZERO HOLD"


def _overlay(
    frame_rgb: np.ndarray,
    *,
    frame: int,
    total: int,
    fps: float,
    action_start: int,
    action_end: int,
    hold_start: int,
) -> np.ndarray:
    image = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(image, "RGBA")
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    bold_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    font = ImageFont.truetype(font_path, 22)
    bold = ImageFont.truetype(bold_path, 27)
    width, height = image.size

    draw.rounded_rectangle((24, 22, 410, 103), radius=14, fill=(5, 9, 18, 190))
    draw.text((43, 33), "MICRODUCK  /  RETARGET REFERENCE", font=bold, fill="white")
    draw.text(
        (43, 69),
        f"{_phase(frame, action_start, action_end, hold_start)}    {frame / fps:4.2f}s",
        font=font,
        fill=(171, 214, 255),
    )

    left, right, top, bottom = 30, width - 30, height - 46, height - 30
    draw.rounded_rectangle((left, top, right, bottom), radius=8, fill=(4, 8, 16, 185))
    span = right - left
    segments = (
        (0, action_start, (116, 190, 255, 255)),
        (action_start, action_end, (255, 183, 77, 255)),
        (action_end, hold_start, (165, 126, 255, 255)),
        (hold_start, total, (88, 224, 163, 255)),
    )
    for start, end, color in segments:
        x0 = left + round(span * start / total)
        x1 = left + round(span * end / total)
        draw.rectangle((x0, top, x1, bottom), fill=color)
    cursor = left + round(span * min(frame + 1, total) / total)
    draw.rectangle((cursor - 2, top - 5, cursor + 2, bottom + 5), fill="white")
    return np.asarray(image)


def render_reference(
    motion_path: Path,
    output_path: Path,
    *,
    width: int = 960,
    height: int = 720,
) -> Path:
    metadata_path = motion_path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(motion_path, allow_pickle=False) as motion:
        joint_pos = np.asarray(motion["joint_pos"], dtype=np.float64)
        body_pos = np.asarray(motion["body_pos_w"], dtype=np.float64)
        body_quat = np.asarray(motion["body_quat_w"], dtype=np.float64)

    names = [str(name) for name in metadata["joint_names"]]
    body_names = [str(name) for name in metadata["body_names"]]
    root_index = body_names.index("trunk_base")
    fps = float(metadata["output_fps"])
    action_start = int(metadata["action_start_frame"])
    action_end = action_start + int(metadata["action_frame_count"])
    hold_start = int(metadata["stand_hold_start_frame"])

    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
    data = mujoco.MjData(model)
    joint_addresses = _joint_qpos_addresses(model, names)
    free_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint"
    )
    free_address = int(model.jnt_qposadr[free_id])

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [
        float(np.mean(body_pos[:, root_index, 0])),
        float(np.mean(body_pos[:, root_index, 1])),
        0.12,
    ]
    camera.distance = 0.72
    camera.azimuth = 135.0
    camera.elevation = -13.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        output_path,
        fps=round(fps),
        codec="libx264",
        quality=9,
        macro_block_size=1,
    )
    try:
        with mujoco.Renderer(model, height=height, width=width) as renderer:
            for frame in range(joint_pos.shape[0]):
                data.qpos[:] = model.qpos0
                data.qpos[free_address : free_address + 3] = body_pos[
                    frame, root_index
                ]
                data.qpos[free_address + 3 : free_address + 7] = body_quat[
                    frame, root_index
                ]
                data.qpos[joint_addresses] = joint_pos[frame]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera)
                rgb = renderer.render()
                writer.append_data(
                    _overlay(
                        rgb,
                        frame=frame,
                        total=joint_pos.shape[0],
                        fps=fps,
                        action_start=action_start,
                        action_end=action_end,
                        hold_start=hold_start,
                    )
                )
    finally:
        writer.close()
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()
    print(
        render_reference(
            args.motion,
            args.output,
            width=args.width,
            height=args.height,
        )
    )


if __name__ == "__main__":
    main()
