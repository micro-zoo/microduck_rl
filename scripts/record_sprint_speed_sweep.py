"""Record a closed-loop speed sweep for a MicroDuck sprint checkpoint.

The rendered labels report the command sent to the policy and the simulated
body-frame forward velocity.  They intentionally do not label command speed as
achieved speed: a termination or a large tracking error remains visible in the
video and in the adjacent JSON summary.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import mediapy as media
import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from PIL import Image, ImageDraw, ImageFont

TASK_ID = "Mjlab-Sprint2mps-Flat-MicroDuck"
DEFAULT_SPEEDS_M_S = (0.7, 1.0, 1.5, 2.0)
DEFAULT_CONTROL_STEPS_PER_SEGMENT = 250  # Five seconds at the 50 Hz policy rate.
DEFAULT_FRAME_STRIDE = 2  # Render at 25 fps without changing policy/control rate.
WARMUP_STEPS = 50


def _force_forward_command(env: ManagerBasedRlEnv, speed_m_s: float) -> None:
    """Set a fixed straight-ahead command without changing physics state."""
    command = env.command_manager.get_term("twist")
    command.vel_command_b[:, 0] = speed_m_s
    command.vel_command_b[:, 1:] = 0.0
    command.vel_command_w[:, 0] = speed_m_s
    command.vel_command_w[:, 1:] = 0.0
    command.is_heading_env[:] = False
    command.is_standing_env[:] = False
    command.is_world_env[:] = False
    command.is_forward_env[:] = True
    command.time_left[:] = 1.0e6


def _label_frame(
    frame: np.ndarray,
    *,
    checkpoint_name: str,
    command_m_s: float,
    actual_m_s: float,
    segment_step: int,
    fell: bool,
) -> np.ndarray:
    """Put inspectable command/velocity telemetry directly in the video."""
    image = Image.fromarray(frame.astype(np.uint8, copy=False))
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()
    draw.rounded_rectangle((8, 8, 306, 70), radius=5, fill=(0, 0, 0, 185))
    draw.text((16, 15), f"checkpoint: {checkpoint_name}", fill="white", font=font)
    draw.text(
        (16, 31),
        f"command: {command_m_s:.2f} m/s | actual: {actual_m_s:.2f} m/s",
        fill="white",
        font=font,
    )
    draw.text(
        (16, 47),
        f"segment time: {segment_step * 0.02:.1f}s",
        fill="white",
        font=font,
    )
    if fell:
        draw.rounded_rectangle((8, 78, 126, 98), radius=4, fill=(190, 20, 20, 210))
        draw.text((16, 82), "RESET / FALL", fill="white", font=font)
    return np.asarray(image)


def _social_label_frame(frame: np.ndarray, actual_m_s: float) -> np.ndarray:
    """Add only the current measured speed, centered for a social-video cut."""
    image = Image.fromarray(frame.astype(np.uint8, copy=False))
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=42
    )
    label = f"{actual_m_s:.2f} m/s"
    bounds = draw.textbbox((0, 0), label, font=font)
    x = (image.width - (bounds[2] - bounds[0])) // 2
    y = 30
    draw.text((x, y), label, fill=(255, 255, 255, 255), font=font)
    return np.asarray(image)


def _configure_presentation_style(env: ManagerBasedRlEnv) -> Callable[[], None]:
    """Set up a studio look that changes rendering, never the physical terrain."""
    model = env.sim.mj_model

    # These are visual fields only; contact geometry, friction, height, and
    # physics parameters are untouched.
    model.tex_data[:] = np.uint8(255)
    model.vis.rgba.haze[:] = (1.0, 1.0, 1.0, 1.0)
    model.vis.headlight.ambient[:] = (0.42, 0.42, 0.42)
    model.vis.headlight.diffuse[:] = (0.70, 0.70, 0.70)
    model.vis.headlight.specular[:] = (0.20, 0.20, 0.20)
    terrain_id = model.geom("terrain").id
    terrain_material_id = model.geom_matid[terrain_id]
    model.geom_rgba[terrain_id] = (0.92, 0.92, 0.90, 1.0)
    if terrain_material_id >= 0:
        model.mat_rgba[terrain_material_id] = (0.92, 0.92, 0.90, 1.0)
        model.mat_reflectance[terrain_material_id] = 0.0

    # Use a free camera whose look-at point follows X/Y only.  Its Z coordinate
    # is intentionally fixed, so body bobbing does not make the camera bob.
    camera = env._offline_renderer._cam
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE.value
    camera.fixedcamid = -1
    camera.trackbodyid = -1
    grid_origin = np.zeros(3, dtype=np.float64)

    # The grid is static in world space.  It is deliberately *not* re-anchored
    # every frame: the horizontal tracking camera makes it move beneath the
    # robot as it should in a real render.
    def _draw_grid(visualizer) -> None:
        half_x = 12.0
        half_y = 3.5
        spacing = 0.25
        for offset in np.arange(-half_x, half_x + spacing / 2, spacing):
            visualizer.add_cylinder(
                grid_origin + np.array([offset, -half_y, 0.002]),
                grid_origin + np.array([offset, half_y, 0.002]),
                radius=0.0011,
                color=(0.38, 0.38, 0.38, 1.0),
            )
        for offset in np.arange(-half_y, half_y + spacing / 2, spacing):
            visualizer.add_cylinder(
                grid_origin + np.array([-half_x, offset, 0.002]),
                grid_origin + np.array([half_x, offset, 0.002]),
                radius=0.0011,
                color=(0.38, 0.38, 0.38, 1.0),
            )

    env.update_visualizers = _draw_grid

    def _anchor_grid_to_reset() -> None:
        root_pos = env.scene["robot"].data.root_link_pos_w[0].detach().cpu().numpy()
        grid_origin[:] = (root_pos[0], root_pos[1], 0.0)

    return _anchor_grid_to_reset


def _render_frame(env: ManagerBasedRlEnv, *, presentation: bool) -> np.ndarray:
    """Render RGB, replacing only the studio sky and terrain colour in post."""
    if presentation:
        root_pos = env.scene["robot"].data.root_link_pos_w[0].detach().cpu().numpy()
        camera = env._offline_renderer._cam
        camera.lookat[:] = (root_pos[0], root_pos[1], 0.10)
    frame = env.render()
    if frame is None:
        raise RuntimeError("RGB renderer returned no frame")
    if frame.ndim == 4:
        frame = frame[0]
    if not presentation:
        return frame

    # MuJoCo exposes per-pixel object ids through its segmentation pass.  The
    # empty sky is id -1.  Recolour just the physical terrain to a uniform,
    # neutral studio floor; the world-space visualizer grid and the robot keep
    # their own pixels.  This avoids an environment texture leaking a blue or
    # checkerboard appearance into the presentation render.
    renderer = env._offline_renderer.renderer
    renderer.enable_segmentation_rendering()
    try:
        segmentation = renderer.render()
    finally:
        renderer.disable_segmentation_rendering()
    studio_frame = frame.copy()
    studio_frame[segmentation[..., 0] == -1] = 255
    terrain_id = env.sim.mj_model.geom("terrain").id
    terrain_pixels = segmentation[..., 0] == terrain_id
    studio_frame[terrain_pixels] = (234, 234, 230)
    # The supplied terrain material carries a blue checker/grid texture.  Its
    # bright strokes are retained as a neutral gray world-space grid, while the
    # rest of the terrain becomes one flat studio colour.
    source_brightness = frame.astype(np.uint16).sum(axis=-1)
    grid_strokes = terrain_pixels & (source_brightness >= 320)
    studio_frame[grid_strokes] = (112, 112, 112)
    return studio_frame


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Local RSL-RL .pt checkpoint")
    parser.add_argument(
        "--task-id",
        default=TASK_ID,
        help="Registered task used to rebuild the checkpoint environment.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--speeds", type=float, nargs="+", default=DEFAULT_SPEEDS_M_S)
    parser.add_argument(
        "--control-steps-per-segment", type=int, default=DEFAULT_CONTROL_STEPS_PER_SEGMENT
    )
    parser.add_argument("--frame-stride", type=int, default=DEFAULT_FRAME_STRIDE)
    parser.add_argument(
        "--width",
        type=int,
        help="Output width without changing the MuJoCo scene or camera.",
    )
    parser.add_argument(
        "--height",
        type=int,
        help="Output height without changing the MuJoCo scene or camera.",
    )
    parser.add_argument(
        "--camera-distance",
        type=float,
        help="Override only the default MuJoCo camera distance.",
    )
    parser.add_argument(
        "--presentation",
        action="store_true",
        help="Render a 1280x720 white studio with a fixed world-space gray grid.",
    )
    parser.add_argument(
        "--social",
        action="store_true",
        help="Show only centered current-speed text in the presentation video.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Importing mjlab.tasks populates both upstream and project task registries.
    import mjlab.tasks  # noqa: F401

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env_cfg = load_env_cfg(args.task_id, play=True)
    agent_cfg = load_rl_cfg(args.task_id)
    env_cfg.scene.num_envs = 1
    env_cfg.seed = agent_cfg.seed
    env_cfg.viewer.width = args.width or (1280 if args.presentation else 512)
    env_cfg.viewer.height = args.height or (720 if args.presentation else 384)
    if args.camera_distance is not None:
        env_cfg.viewer.distance = args.camera_distance
    if args.presentation:
        env_cfg.viewer.distance = 1.0
        env_cfg.viewer.elevation = -10.0
        env_cfg.viewer.azimuth = 105.0
        env_cfg.viewer.lookat = (0.0, 0.0, 0.10)
    # The velocity-command arrow is a debug overlay, not part of the MuJoCo
    # scene.  Keep it out of the social cut so the only overlay is actual speed.
    env_cfg.commands["twist"].debug_vis = False
    env_cfg.commands["twist"].resampling_time_range = (1.0e6, 1.0e6)

    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
    anchor_grid_to_reset = _configure_presentation_style(raw_env) if args.presentation else None
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(args.task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
        str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = runner.get_inference_policy(device=device)

    frames: list[np.ndarray] = []
    summary: list[dict[str, float | int]] = []
    try:
        for speed_m_s in args.speeds:
            # The environment's delay buffers are initialized under inference
            # mode, so resets and command writes must use that same context.
            with torch.inference_mode():
                _obs, _ = env.reset()
                _force_forward_command(raw_env, speed_m_s)
                obs = env.get_observations()
            if anchor_grid_to_reset is not None:
                anchor_grid_to_reset()
            segment_velocities: list[float] = []
            terminations = 0

            for step in range(args.control_steps_per_segment):
                with torch.inference_mode():
                    actions = policy(obs)
                    obs, _rewards, dones, _extras = env.step(actions)
                    actual_m_s = float(raw_env.scene["robot"].data.root_link_lin_vel_b[0, 0])
                    fell = bool(dones[0].item())

                    # A termination auto-resets the environment.  Reassert the
                    # sweep command before selecting the next policy action.
                    _force_forward_command(raw_env, speed_m_s)
                    obs = env.get_observations()
                terminations += int(fell)
                if step >= WARMUP_STEPS and not fell:
                    segment_velocities.append(actual_m_s)

                if step % args.frame_stride == 0:
                    frame = _render_frame(raw_env, presentation=args.presentation)
                    if args.social:
                        frames.append(_social_label_frame(frame, actual_m_s))
                    else:
                        frames.append(
                            _label_frame(
                                frame,
                                checkpoint_name=checkpoint.name,
                                command_m_s=speed_m_s,
                                actual_m_s=actual_m_s,
                                segment_step=step,
                                fell=fell,
                            )
                        )

            valid = np.asarray(segment_velocities, dtype=np.float64)
            summary.append(
                {
                    "command_m_s": speed_m_s,
                    "mean_actual_m_s_after_warmup": float(valid.mean()) if len(valid) else float("nan"),
                    "p95_actual_m_s_after_warmup": float(np.percentile(valid, 95)) if len(valid) else float("nan"),
                    "peak_actual_m_s_after_warmup": float(valid.max()) if len(valid) else float("nan"),
                    "terminations": terminations,
                }
            )
    finally:
        env.close()

    video_path = output_dir / "sprint_speed_sweep.mp4"
    media.write_video(video_path, frames, fps=50 / args.frame_stride)
    summary_path = output_dir / "sprint_speed_sweep.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"video={video_path}")
    print(f"summary={summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
