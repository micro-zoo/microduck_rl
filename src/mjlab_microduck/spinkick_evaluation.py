"""Local, full-horizon acceptance test for the Microduck Spin policy.

Unlike the stock W&B-only evaluator, this command accepts a local checkpoint
and reports phase-specific full-body tracking and final standing behavior.  It
is deliberately independent of the training loop: a high PPO reward alone is
not considered evidence that the one-shot motion was learned.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp.commands import MotionCommand
from mjlab.tasks.tracking.mdp.metrics import (
    compute_ee_orientation_error,
    compute_ee_position_error,
    compute_mpkpe,
    compute_root_relative_mpkpe,
)
from mjlab.utils.lab_api.math import quat_apply_inverse, quat_error_magnitude
from mjlab.utils.torch import configure_torch_backends

DEFAULT_TASK_ID = "Mjlab-Spinkick-Mimic-MicroDuck"


@dataclass(frozen=True)
class SpinkickEvaluationCfg:
    checkpoint_file: str
    task_id: str = DEFAULT_TASK_ID
    motion_file: str = (
        "artifacts/motions/mimickit_spinkick_microduck_umr60_g1pad/motion.npz"
    )
    num_envs: int = 64
    device: str | None = None
    hold_steps: int = 50
    start_frame: int = 0
    robust: bool = False
    show_reference: bool = True
    diagnostic_no_early_termination: bool = False
    output_file: str | None = None
    video_file: str | None = None


@dataclass
class _Mean:
    total: float = 0.0
    count: int = 0

    def add(self, value: torch.Tensor, mask: torch.Tensor) -> None:
        selected = value[mask]
        self.total += float(selected.sum().item())
        self.count += int(selected.numel())

    def result(self) -> float:
        return self.total / max(self.count, 1)


def _yaw_from_wxyz(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat.unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrapped_angle_delta(current: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
    delta = current - previous
    return torch.atan2(torch.sin(delta), torch.cos(delta))


def evaluate_acceptance(metrics: dict[str, float | int | bool]) -> dict[str, bool]:
    """Return morphology-tolerant semantic and physical acceptance gates.

    UMR maps a human and Microduck with very different proportions and degrees
    of freedom, so this is not a literal joint-for-joint replay.  It is still
    not enough to merely traverse the clip without falling: a valid policy
    must reproduce most of the turning kick and aerial excursion, track the
    full body closely, and settle back to the static tail instead of walking
    away.  These are evaluation gates only; no motion-specific reward is used
    while training the general tracking pipeline.
    """

    gates = {
        "full_horizon_success": float(metrics["success_rate"]) >= 0.95,
        "action_full_body_position": float(metrics["action_r_mpkpe_m"]) <= 0.05,
        "action_joint_position": float(metrics["action_joint_pos_mae_rad"]) <= 0.30,
        "action_body_orientation": float(metrics["action_body_ori_mae_rad"])
        <= 0.40,
        "spin_direction": bool(metrics["spin_direction_matches_reference"]),
        "spin_semantic_coverage": (
            0.70 <= float(metrics["spin_coverage_ratio"]) <= 1.30
        ),
        "kick_semantic_coverage": (
            float(metrics["kick_amplitude_coverage_ratio"]) >= 0.75
        ),
        "jump_semantic_coverage": (
            float(metrics["jump_height_coverage_ratio"]) >= 0.70
        ),
        "standing_position": float(metrics["final_joint_pos_mae_rad"]) <= 0.35,
        "standing_speed": float(metrics["final_planar_speed_m_s"]) <= 0.20,
        "standing_drift": float(metrics["hold_planar_drift_m"]) <= 0.10,
    }
    return gates


def run_evaluation(cfg: SpinkickEvaluationCfg) -> dict[str, object]:
    configure_torch_backends()
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    checkpoint = Path(cfg.checkpoint_file).resolve()
    motion_file = Path(cfg.motion_file).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not motion_file.is_file():
        raise FileNotFoundError(f"Motion not found: {motion_file}")

    metadata_path = motion_file.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text())
    action_start = int(metadata["action_start_frame"])
    action_end = action_start + int(metadata["action_frame_count"])
    hold_start = int(metadata["stand_hold_start_frame"])

    env_cfg = load_env_cfg(cfg.task_id, play=True)
    agent_cfg = load_rl_cfg(cfg.task_id)
    env_cfg.scene.num_envs = cfg.num_envs
    env_cfg.commands["motion"].motion_file = str(motion_file)
    env_cfg.commands["motion"].debug_vis = cfg.show_reference
    ee_names = tuple(
        env_cfg.terminations["ee_body_pos"].params["body_names"]
    )
    if cfg.diagnostic_no_early_termination:
        # Preserve the complete physical rollout for diagnosis.  This mode can
        # never pass acceptance: it deliberately removes the guards which
        # distinguish a tracked motion from a fallen robot.
        env_cfg.terminations = {
            name: term
            for name, term in env_cfg.terminations.items()
            if name == "time_out"
        }
    if not cfg.robust:
        # The actuator needs its startup field expansion.  All stochastic
        # perturbations are removed for the nominal acceptance test.
        env_cfg.events = {
            "expand_bam_friction_fields": env_cfg.events[
                "expand_bam_friction_fields"
            ]
        }

        # Evaluate one deterministic nominal compound-physics setting.  The
        # robust pass leaves the complete BAM voltage, sag, friction, delay,
        # encoder, mass, friction, and push randomization enabled.
        robot_cfg = env_cfg.scene.entities["robot"]
        assert robot_cfg.articulation is not None
        for actuator_cfg in robot_cfg.articulation.actuators:
            if hasattr(actuator_cfg, "vin_range"):
                actuator_cfg.vin_range = (7.4, 7.4)
            if hasattr(actuator_cfg, "vin_drop_gain_range"):
                actuator_cfg.vin_drop_gain_range = (0.1, 0.1)
            if hasattr(actuator_cfg, "delay_min_lag"):
                actuator_cfg.delay_min_lag = 4
            if hasattr(actuator_cfg, "delay_max_lag"):
                actuator_cfg.delay_max_lag = 4

    base_env = ManagerBasedRlEnv(
        cfg=env_cfg,
        device=device,
        render_mode="rgb_array" if cfg.video_file else None,
    )
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(cfg.task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
        str(checkpoint),
        load_cfg={"actor": True},
        strict=True,
        map_location=device,
    )
    policy = runner.get_inference_policy(device=device)
    command = cast(MotionCommand, env.unwrapped.command_manager.get_term("motion"))
    if not 0 <= cfg.start_frame < command.motion.time_step_total:
        raise ValueError(
            f"start_frame must be in [0, {command.motion.time_step_total - 1}]"
        )
    if cfg.start_frame:
        env_ids = torch.arange(cfg.num_envs, device=device)
        command.reset_to_frame(env_ids, cfg.start_frame)
        env.unwrapped.scene.write_data_to_sim()
        env.unwrapped.sim.forward()
        command.update_relative_body_poses()

    active = torch.ones(cfg.num_envs, dtype=torch.bool, device=device)
    survival_steps = torch.zeros(cfg.num_envs, dtype=torch.long, device=device)
    termination_counts = {
        name: 0
        for name in env.unwrapped.termination_manager.active_terms
        if name != "time_out"
    }
    means = {
        "all_mpkpe_m": _Mean(),
        "all_r_mpkpe_m": _Mean(),
        "all_joint_pos_mae_rad": _Mean(),
        "all_joint_vel_mae_rad_s": _Mean(),
        "all_body_ori_mae_rad": _Mean(),
        "all_ee_pos_mae_m": _Mean(),
        "all_ee_ori_mae_rad": _Mean(),
        "action_r_mpkpe_m": _Mean(),
        "action_joint_pos_mae_rad": _Mean(),
        "action_body_ori_mae_rad": _Mean(),
        "hold_r_mpkpe_m": _Mean(),
        "hold_joint_pos_mae_rad": _Mean(),
    }

    obs = env.get_observations()
    robot_yaw = _yaw_from_wxyz(command.robot_anchor_quat_w)
    reference_yaw = _yaw_from_wxyz(command.anchor_quat_w)
    robot_yaw_total = torch.zeros(cfg.num_envs, device=device)
    reference_yaw_total = torch.zeros(cfg.num_envs, device=device)
    body_name_to_index = {
        name: index for index, name in enumerate(command.cfg.body_names)
    }
    try:
        left_ankle_index = body_name_to_index["ankle_left"]
        right_ankle_index = body_name_to_index["ankle_right"]
    except KeyError as exc:
        raise ValueError(
            "Spin-kick evaluation requires ankle_left and ankle_right in "
            "the motion command's tracked body names"
        ) from exc
    robot_kick_foot_separation_max = torch.zeros(cfg.num_envs, device=device)
    reference_kick_foot_separation_max = torch.zeros(
        cfg.num_envs, device=device
    )
    robot_root_z_min = torch.full(
        (cfg.num_envs,), torch.inf, device=device
    )
    robot_root_z_max = torch.full(
        (cfg.num_envs,), -torch.inf, device=device
    )
    reference_root_z_min = torch.full(
        (cfg.num_envs,), torch.inf, device=device
    )
    reference_root_z_max = torch.full(
        (cfg.num_envs,), -torch.inf, device=device
    )
    hold_xy = torch.full((cfg.num_envs, 2), torch.nan, device=device)
    failure_frame = torch.full(
        (cfg.num_envs,), -1, dtype=torch.long, device=device
    )
    failure_anchor_ori_error = torch.full(
        (cfg.num_envs,), torch.nan, device=device
    )
    failure_anchor_tilt_z_error = torch.full(
        (cfg.num_envs,), torch.nan, device=device
    )
    failure_body_z_error = torch.full(
        (cfg.num_envs, len(command.cfg.body_names)),
        torch.nan,
        device=device,
    )

    video_writer = None
    if cfg.video_file:
        import imageio.v2 as imageio

        video_path = Path(cfg.video_file)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_writer = imageio.get_writer(
            video_path, fps=round(1.0 / env.unwrapped.step_dt), codec="libx264"
        )

    total_steps = command.motion.time_step_total - cfg.start_frame + cfg.hold_steps
    try:
        for _ in range(total_steps):
            if video_writer is not None:
                video_writer.append_data(base_env.render())

            # Keep a snapshot from immediately before the transition that
            # triggers auto-reset.  The post-step tensors already contain the
            # reset state and previously made failure videos/metrics look as
            # if the policy had recovered to standing.
            pre_frame = command.time_steps.clone()
            pre_anchor_ori_error = quat_error_magnitude(
                command.anchor_quat_w, command.robot_anchor_quat_w
            )
            reference_gravity_b = quat_apply_inverse(
                command.anchor_quat_w, command.robot.data.gravity_vec_w
            )
            robot_gravity_b = quat_apply_inverse(
                command.robot_anchor_quat_w, command.robot.data.gravity_vec_w
            )
            pre_anchor_tilt_z_error = (
                reference_gravity_b[:, 2] - robot_gravity_b[:, 2]
            ).abs()
            pre_body_z_error = (
                command.body_pos_relative_w[..., 2]
                - command.robot_body_pos_w[..., 2]
            ).abs()
            with torch.no_grad():
                actions = policy(obs)
            obs, _, dones, _ = env.step(actions)

            # Count the just-executed step before removing failed worlds.  The
            # environment auto-resets failures inside ``step()``, so the
            # command's frame index can no longer identify where they fell.
            survival_steps += active.long()
            newly_done = dones.bool() & active
            failure_frame[newly_done] = pre_frame[newly_done]
            failure_anchor_ori_error[newly_done] = pre_anchor_ori_error[
                newly_done
            ]
            failure_anchor_tilt_z_error[newly_done] = pre_anchor_tilt_z_error[
                newly_done
            ]
            failure_body_z_error[newly_done] = pre_body_z_error[newly_done]
            for name in termination_counts:
                term = env.unwrapped.termination_manager.get_term(name)
                termination_counts[name] += int((term & newly_done).sum().item())
            active &= ~newly_done

            current_robot_yaw = _yaw_from_wxyz(command.robot_anchor_quat_w)
            current_reference_yaw = _yaw_from_wxyz(command.anchor_quat_w)
            robot_yaw_total += torch.where(
                active, _wrapped_angle_delta(current_robot_yaw, robot_yaw), 0.0
            )
            reference_yaw_total += torch.where(
                active,
                _wrapped_angle_delta(current_reference_yaw, reference_yaw),
                0.0,
            )
            robot_yaw = current_robot_yaw
            reference_yaw = current_reference_yaw

            frame = command.time_steps
            action_mask = active & (frame >= action_start) & (frame < action_end)
            hold_mask = active & (frame >= hold_start)
            robot_foot_separation = torch.abs(
                command.robot_body_pos_w[:, left_ankle_index, 2]
                - command.robot_body_pos_w[:, right_ankle_index, 2]
            )
            reference_foot_separation = torch.abs(
                command.body_pos_relative_w[:, left_ankle_index, 2]
                - command.body_pos_relative_w[:, right_ankle_index, 2]
            )
            robot_kick_foot_separation_max = torch.where(
                action_mask,
                torch.maximum(
                    robot_kick_foot_separation_max, robot_foot_separation
                ),
                robot_kick_foot_separation_max,
            )
            reference_kick_foot_separation_max = torch.where(
                action_mask,
                torch.maximum(
                    reference_kick_foot_separation_max,
                    reference_foot_separation,
                ),
                reference_kick_foot_separation_max,
            )
            robot_root_z = command.robot_anchor_pos_w[:, 2]
            reference_root_z = command.anchor_pos_w[:, 2]
            robot_root_z_min = torch.where(
                action_mask,
                torch.minimum(robot_root_z_min, robot_root_z),
                robot_root_z_min,
            )
            robot_root_z_max = torch.where(
                action_mask,
                torch.maximum(robot_root_z_max, robot_root_z),
                robot_root_z_max,
            )
            reference_root_z_min = torch.where(
                action_mask,
                torch.minimum(reference_root_z_min, reference_root_z),
                reference_root_z_min,
            )
            reference_root_z_max = torch.where(
                action_mask,
                torch.maximum(reference_root_z_max, reference_root_z),
                reference_root_z_max,
            )
            first_hold = hold_mask & torch.isnan(hold_xy[:, 0])
            hold_xy[first_hold] = command.robot_anchor_pos_w[first_hold, :2]

            mpkpe = compute_mpkpe(command)
            r_mpkpe = compute_root_relative_mpkpe(command)
            joint_pos_mae = (
                command.joint_pos - command.robot_joint_pos
            ).abs().mean(dim=-1)
            joint_vel_mae = (
                command.joint_vel - command.robot_joint_vel
            ).abs().mean(dim=-1)
            body_ori_mae = quat_error_magnitude(
                command.body_quat_relative_w, command.robot_body_quat_w
            ).mean(dim=-1)
            ee_pos_mae = compute_ee_position_error(command, ee_names)
            ee_ori_mae = compute_ee_orientation_error(command, ee_names)

            means["all_mpkpe_m"].add(mpkpe, active)
            means["all_r_mpkpe_m"].add(r_mpkpe, active)
            means["all_joint_pos_mae_rad"].add(joint_pos_mae, active)
            means["all_joint_vel_mae_rad_s"].add(joint_vel_mae, active)
            means["all_body_ori_mae_rad"].add(body_ori_mae, active)
            means["all_ee_pos_mae_m"].add(ee_pos_mae, active)
            means["all_ee_ori_mae_rad"].add(ee_ori_mae, active)
            means["action_r_mpkpe_m"].add(r_mpkpe, action_mask)
            means["action_joint_pos_mae_rad"].add(joint_pos_mae, action_mask)
            means["action_body_ori_mae_rad"].add(body_ori_mae, action_mask)
            means["hold_r_mpkpe_m"].add(r_mpkpe, hold_mask)
            means["hold_joint_pos_mae_rad"].add(joint_pos_mae, hold_mask)
    finally:
        if video_writer is not None:
            video_writer.close()

    successful = active & ~torch.isnan(hold_xy[:, 0])
    final_xy = command.robot_anchor_pos_w[:, :2]
    hold_drift = torch.linalg.vector_norm(final_xy - hold_xy, dim=-1)
    final_planar_speed = torch.linalg.vector_norm(
        command.robot_anchor_lin_vel_w[:, :2], dim=-1
    )
    final_joint_pos_mae = (
        command.joint_pos - command.robot_joint_pos
    ).abs().mean(dim=-1)
    robot_root_z_excursion = robot_root_z_max - robot_root_z_min
    reference_root_z_excursion = reference_root_z_max - reference_root_z_min

    def successful_mean(value: torch.Tensor) -> float:
        selected = value[successful]
        # Keep failure reports valid JSON while ensuring every numeric gate fails.
        return float(selected.mean().item()) if selected.numel() else 1.0e9

    failed = failure_frame >= 0

    def failed_mean(value: torch.Tensor) -> float:
        selected = value[failed]
        return float(selected.mean().item()) if selected.numel() else 0.0

    def failed_finite_mean(value: torch.Tensor) -> float:
        selected = value[failed & torch.isfinite(value)]
        return float(selected.mean().item()) if selected.numel() else 0.0

    failure_body_z_by_name = {
        name: failed_mean(failure_body_z_error[:, body_index])
        for body_index, name in enumerate(command.cfg.body_names)
    }
    robot_kick_before_failure = failed_finite_mean(
        robot_kick_foot_separation_max
    )
    reference_kick_before_failure = failed_finite_mean(
        reference_kick_foot_separation_max
    )
    robot_jump_before_failure = failed_finite_mean(robot_root_z_excursion)
    reference_jump_before_failure = failed_finite_mean(
        reference_root_z_excursion
    )

    reference_yaw_mean = successful_mean(reference_yaw_total)
    robot_yaw_mean = successful_mean(robot_yaw_total)
    if successful.any():
        spin_coverage_ratio = abs(robot_yaw_mean) / max(
            abs(reference_yaw_mean), 1.0e-6
        )
        spin_direction_matches_reference = (
            robot_yaw_mean * reference_yaw_mean > 0.0
        )
        reference_kick_amplitude = successful_mean(
            reference_kick_foot_separation_max
        )
        robot_kick_amplitude = successful_mean(
            robot_kick_foot_separation_max
        )
        kick_amplitude_coverage_ratio = robot_kick_amplitude / max(
            reference_kick_amplitude, 1.0e-6
        )
        reference_jump_height = successful_mean(reference_root_z_excursion)
        robot_jump_height = successful_mean(robot_root_z_excursion)
        jump_height_coverage_ratio = robot_jump_height / max(
            reference_jump_height, 1.0e-6
        )
    else:
        # Do not let successful-only sentinels turn into plausible ratios.
        spin_coverage_ratio = 0.0
        spin_direction_matches_reference = False
        reference_kick_amplitude = successful_mean(
            reference_kick_foot_separation_max
        )
        robot_kick_amplitude = successful_mean(
            robot_kick_foot_separation_max
        )
        kick_amplitude_coverage_ratio = 0.0
        reference_jump_height = successful_mean(reference_root_z_excursion)
        robot_jump_height = successful_mean(robot_root_z_excursion)
        jump_height_coverage_ratio = 0.0

    metrics: dict[str, float | int | bool | dict[str, int] | dict[str, bool]] = {
        "checkpoint": str(checkpoint),
        "task_id": cfg.task_id,
        "motion": str(motion_file),
        "num_envs": cfg.num_envs,
        "robust": cfg.robust,
        "diagnostic_no_early_termination": (
            cfg.diagnostic_no_early_termination
        ),
        "start_frame": cfg.start_frame,
        "evaluated_steps": total_steps,
        "successful_envs": int(successful.sum().item()),
        "success_rate": float(active.float().mean().item()),
        "mean_survival_steps": float(survival_steps.float().mean().item()),
        "max_survival_steps": int(survival_steps.max().item()),
        "mean_reference_progress": float(
            (
                survival_steps.float()
                / max(command.motion.time_step_total - cfg.start_frame, 1)
            )
            .clamp(max=1.0)
            .mean()
            .item()
        ),
        "termination_counts": termination_counts,
        "failure_frame_before_terminal_mean": failed_mean(
            failure_frame.float()
        ),
        "failure_anchor_orientation_error_rad": failed_mean(
            failure_anchor_ori_error
        ),
        "failure_anchor_tilt_z_error": failed_mean(
            failure_anchor_tilt_z_error
        ),
        "failure_body_z_error_by_name_m": failure_body_z_by_name,
        **{name: mean.result() for name, mean in means.items()},
        "reference_root_yaw_rad": reference_yaw_mean,
        "robot_root_yaw_rad": robot_yaw_mean,
        "root_yaw_error_rad": successful_mean(
            torch.abs(robot_yaw_total - reference_yaw_total)
        ),
        "spin_direction_matches_reference": spin_direction_matches_reference,
        "spin_coverage_ratio": spin_coverage_ratio,
        "reference_kick_foot_separation_m": reference_kick_amplitude,
        "robot_kick_foot_separation_m": robot_kick_amplitude,
        "kick_amplitude_coverage_ratio": kick_amplitude_coverage_ratio,
        "reference_root_z_excursion_m": reference_jump_height,
        "robot_root_z_excursion_m": robot_jump_height,
        "jump_height_coverage_ratio": jump_height_coverage_ratio,
        # Always report the rotation accumulated before failure.  Successful-
        # only values above remain the acceptance metrics; these diagnostics
        # make an all-failed checkpoint actionable instead of returning only
        # sentinel values.
        "robot_root_yaw_before_failure_rad": float(robot_yaw_total.mean().item()),
        "reference_root_yaw_before_failure_rad": float(
            reference_yaw_total.mean().item()
        ),
        "root_yaw_error_before_failure_rad": float(
            torch.abs(robot_yaw_total - reference_yaw_total).mean().item()
        ),
        "robot_kick_foot_separation_before_failure_m": (
            robot_kick_before_failure
        ),
        "reference_kick_foot_separation_before_failure_m": (
            reference_kick_before_failure
        ),
        "kick_coverage_before_failure_ratio": (
            robot_kick_before_failure
            / max(reference_kick_before_failure, 1.0e-6)
        ),
        "robot_root_z_excursion_before_failure_m": robot_jump_before_failure,
        "reference_root_z_excursion_before_failure_m": (
            reference_jump_before_failure
        ),
        "jump_coverage_before_failure_ratio": (
            robot_jump_before_failure
            / max(reference_jump_before_failure, 1.0e-6)
        ),
        "hold_planar_drift_m": successful_mean(hold_drift),
        "final_planar_speed_m_s": successful_mean(final_planar_speed),
        "final_joint_pos_mae_rad": successful_mean(final_joint_pos_mae),
    }
    gates = evaluate_acceptance(metrics)
    if cfg.diagnostic_no_early_termination:
        gates["full_horizon_success"] = False
    metrics["acceptance_gates"] = gates
    metrics["accepted"] = all(gates.values())

    if cfg.output_file:
        output_path = Path(cfg.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")

    print(json.dumps(metrics, indent=2, allow_nan=False))
    env.close()
    return metrics


def main() -> None:
    import mjlab_microduck.tasks  # noqa: F401

    run_evaluation(tyro.cli(SpinkickEvaluationCfg))


if __name__ == "__main__":
    main()
