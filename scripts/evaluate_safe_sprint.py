"""Qualify a fixed MicroDuck safe-sprint policy under bounded disturbances.

This is deliberately separate from training-time statistics.  It evaluates one
checkpoint with a deterministic 0.25 -> 0.85 -> 0.25 m/s command profile and
four explicit +/-0.05 m/s velocity pushes per speed segment.  The JSON result
reports simulated body speed, failure causes, and camera-horizon metrics; it
does not treat a command value as an achieved velocity.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TASK_ID = "Mjlab-SafeSprint-Flat-MicroDuck"
SPEED_PROFILE_M_S = (0.25, 0.45, 0.65, 0.85, 0.65, 0.45, 0.25)
PUSH_DIRECTIONS_M_S = ((0.05, 0.0), (-0.05, 0.0), (0.0, 0.05), (0.0, -0.05))
DEFAULT_CONTROL_STEPS_PER_SEGMENT = 250  # Five seconds at the 50 Hz policy rate.
DEFAULT_WARMUP_STEPS = 50
MIN_QUALIFICATION_TRIALS = 8


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Local RSL-RL .pt checkpoint")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-id", default=TASK_ID)
    parser.add_argument("--speeds", type=float, nargs="+", default=SPEED_PROFILE_M_S)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--seed", type=int, default=440)
    parser.add_argument(
        "--control-steps-per-segment", type=int, default=DEFAULT_CONTROL_STEPS_PER_SEGMENT
    )
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    return parser.parse_args()


def _force_forward_command(env: ManagerBasedRlEnv, speed_m_s: float) -> None:
    """Apply an exact fixed command, including the slew term's hidden target."""
    command = env.command_manager.get_term("twist")
    command.vel_command_b[:, 0] = speed_m_s
    command.vel_command_b[:, 1:] = 0.0
    if hasattr(command, "target_vel_command_b"):
        command.target_vel_command_b[:, 0] = speed_m_s
        command.target_vel_command_b[:, 1:] = 0.0
    command.vel_command_w[:, 0] = speed_m_s
    command.vel_command_w[:, 1:] = 0.0
    command.is_heading_env[:] = False
    command.is_standing_env[:] = False
    command.is_world_env[:] = False
    command.is_forward_env[:] = True
    command.time_left[:] = 1.0e6


def _apply_velocity_push(env: ManagerBasedRlEnv, *, x_m_s: float, y_m_s: float) -> None:
    """Apply one exact world-frame velocity perturbation through the task event."""
    event_cfg = env.event_manager.get_term_cfg("push_robot")
    event_cfg.func(
        env,
        torch.tensor([0], device=env.device, dtype=torch.long),
        velocity_range={
            "x": (x_m_s, x_m_s),
            "y": (y_m_s, y_m_s),
        },
        asset_cfg=event_cfg.params["asset_cfg"],
    )


def _camera_axes(quat_wxyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the camera site's +X forward and +Z up vectors in world coordinates."""
    w, x, y, z = quat_wxyz.unbind(dim=-1)
    forward = torch.stack(
        (
            1.0 - 2.0 * (y.square() + z.square()),
            2.0 * (x * y + w * z),
            2.0 * (x * z - w * y),
        ),
        dim=-1,
    )
    up = torch.stack(
        (
            2.0 * (x * z + w * y),
            2.0 * (y * z - w * x),
            1.0 - 2.0 * (x.square() + y.square()),
        ),
        dim=-1,
    )
    return forward, up


def _camera_metrics(
    env: ManagerBasedRlEnv,
    *,
    head_site_id: int,
    previous_axes: torch.Tensor | None,
) -> tuple[dict[str, float], torch.Tensor]:
    robot = env.scene["robot"]
    forward, up = _camera_axes(robot.data.site_quat_w[0, head_site_id])
    axes = torch.cat((forward, up))

    root_q = robot.data.root_link_quat_w[0]
    w, x, y, z = root_q.unbind(dim=-1)
    body_forward_xy = torch.stack(
        (1.0 - 2.0 * (y.square() + z.square()), 2.0 * (x * y + w * z))
    )
    body_forward_xy = torch.nn.functional.normalize(body_forward_xy, dim=-1, eps=1e-6)
    camera_forward_xy = torch.nn.functional.normalize(forward[:2], dim=-1, eps=1e-6)
    heading_error_deg = torch.rad2deg(
        torch.acos(torch.clamp(torch.dot(body_forward_xy, camera_forward_xy), -1.0, 1.0))
    )
    horizon_tilt_deg = torch.rad2deg(torch.asin(torch.clamp(forward[2].abs(), 0.0, 1.0)))
    up_error_deg = torch.rad2deg(torch.acos(torch.clamp(up[2], -1.0, 1.0)))

    camera_rate_deg_s = torch.tensor(0.0, device=env.device)
    if previous_axes is not None:
        forward_delta = torch.acos(torch.clamp(torch.dot(forward, previous_axes[:3]), -1.0, 1.0))
        up_delta = torch.acos(torch.clamp(torch.dot(up, previous_axes[3:]), -1.0, 1.0))
        camera_rate_deg_s = torch.rad2deg(0.5 * (forward_delta + up_delta)) / env.step_dt

    return (
        {
            "horizon_tilt_deg": float(horizon_tilt_deg.cpu()),
            "heading_error_deg": float(heading_error_deg.cpu()),
            "camera_up_error_deg": float(up_error_deg.cpu()),
            "camera_rate_deg_s": float(camera_rate_deg_s.cpu()),
        },
        axes.detach(),
    )


def _percentile_or_nan(values: list[float], percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values else float("nan")


def main() -> None:
    args = _parse_args()
    if args.trials < 1:
        raise ValueError("--trials must be positive")
    if not 0 <= args.warmup_steps < args.control_steps_per_segment:
        raise ValueError("--warmup-steps must be in [0, control-steps-per-segment)")

    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Importing registers project tasks before resolving the task configuration.
    import mjlab.tasks  # noqa: F401

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env_cfg = load_env_cfg(args.task_id, play=True)
    env_cfg.seed = args.seed
    env_cfg.scene.num_envs = 1
    env_cfg.auto_reset = False
    env_cfg.commands["twist"].debug_vis = False
    env_cfg.commands["twist"].resampling_time_range = (1.0e6, 1.0e6)
    # Qualification uses only the explicit, documented pushes below; stochastic
    # interval pushes would make trials incomparable and hide their direction.
    env_cfg.events["push_robot"].interval_range_s = (1.0e6, 1.0e6)

    agent_cfg = load_rl_cfg(args.task_id)
    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(args.task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)
    head_site_id = raw_env.sim.mj_model.site("robot/head_camera").id

    push_steps = (
        args.control_steps_per_segment // 3,
        2 * args.control_steps_per_segment // 3,
    )
    segments: list[dict[str, float | int]] = []
    total_falls = 0
    total_nan_terminations = 0
    push_index = 0

    try:
        for trial in range(args.trials):
            raw_env.seed(args.seed + trial)
            with torch.inference_mode():
                obs, _ = env.reset()
            for segment_index, command_m_s in enumerate(args.speeds):
                with torch.inference_mode():
                    _force_forward_command(raw_env, command_m_s)
                    obs = env.get_observations()

                actual_speeds: list[float] = []
                horizon_tilts: list[float] = []
                heading_errors: list[float] = []
                up_errors: list[float] = []
                camera_rates: list[float] = []
                segment_falls = 0
                segment_nan_terminations = 0
                previous_axes: torch.Tensor | None = None

                for step in range(args.control_steps_per_segment):
                    if step in push_steps:
                        push_x, push_y = PUSH_DIRECTIONS_M_S[push_index % len(PUSH_DIRECTIONS_M_S)]
                        push_index += 1
                        _apply_velocity_push(raw_env, x_m_s=push_x, y_m_s=push_y)

                    with torch.inference_mode():
                        actions = policy(obs)
                        obs, _rewards, dones, _extras = env.step(actions)
                        fell = bool(raw_env.termination_manager.get_term("fell_over")[0].item())
                        nan_state = bool(raw_env.termination_manager.get_term("nan_state")[0].item())
                        done = bool(dones[0].item())
                        if not done:
                            camera, previous_axes = _camera_metrics(
                                raw_env,
                                head_site_id=head_site_id,
                                previous_axes=previous_axes,
                            )
                            if step >= args.warmup_steps:
                                actual_speeds.append(
                                    float(raw_env.scene["robot"].data.root_link_lin_vel_b[0, 0])
                                )
                                horizon_tilts.append(camera["horizon_tilt_deg"])
                                heading_errors.append(camera["heading_error_deg"])
                                up_errors.append(camera["camera_up_error_deg"])
                                camera_rates.append(camera["camera_rate_deg_s"])
                        if done:
                            segment_falls += int(fell)
                            segment_nan_terminations += int(nan_state)
                            with torch.inference_mode():
                                obs, _ = env.reset()
                                _force_forward_command(raw_env, command_m_s)
                                obs = env.get_observations()
                            previous_axes = None

                total_falls += segment_falls
                total_nan_terminations += segment_nan_terminations
                segments.append(
                    {
                        "trial": trial,
                        "segment_index": segment_index,
                        "command_m_s": command_m_s,
                        "mean_actual_m_s_after_warmup": float(np.mean(actual_speeds)) if actual_speeds else float("nan"),
                        "p95_actual_m_s_after_warmup": _percentile_or_nan(actual_speeds, 95),
                        "peak_actual_m_s_after_warmup": float(max(actual_speeds)) if actual_speeds else float("nan"),
                        "fell_over_terminations": segment_falls,
                        "nan_state_terminations": segment_nan_terminations,
                        "horizon_tilt_p95_deg": _percentile_or_nan(horizon_tilts, 95),
                        "heading_error_p95_deg": _percentile_or_nan(heading_errors, 95),
                        "camera_up_error_p95_deg": _percentile_or_nan(up_errors, 95),
                        "camera_rate_p95_deg_s": _percentile_or_nan(camera_rates, 95),
                    }
                )
    finally:
        env.close()

    safe_commands = [
        command_m_s
        for command_m_s in sorted(set(args.speeds))
        if all(
            segment["command_m_s"] != command_m_s
            or (
                segment["fell_over_terminations"] == 0
                and segment["nan_state_terminations"] == 0
                and math.isfinite(float(segment["mean_actual_m_s_after_warmup"]))
            )
            for segment in segments
        )
    ]
    protocol_complete = (
        args.trials >= MIN_QUALIFICATION_TRIALS
        and args.control_steps_per_segment >= DEFAULT_CONTROL_STEPS_PER_SEGMENT
        and args.warmup_steps >= DEFAULT_WARMUP_STEPS
    )
    result = {
        "checkpoint": str(checkpoint),
        "task_id": args.task_id,
        "trials": args.trials,
        "speed_profile_m_s": list(args.speeds),
        "explicit_pushes_m_s": [list(push) for push in PUSH_DIRECTIONS_M_S],
        "total_fell_over_terminations": total_falls,
        "total_nan_state_terminations": total_nan_terminations,
        "highest_command_with_zero_recorded_failures_m_s": max(safe_commands, default=float("nan")),
        "protocol_complete": protocol_complete,
        "safety_qualified": protocol_complete and total_falls == 0 and total_nan_terminations == 0,
        "segments": segments,
    }
    output_path = output_dir / "safe_sprint_qualification.json"
    output_path.write_text(json.dumps(result, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    print(f"result={output_path}")
    print(json.dumps(result, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
