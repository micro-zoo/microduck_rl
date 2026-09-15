"""Regression checks for the fresh high-speed turning sprint task."""

from types import SimpleNamespace

import pytest
import torch
from mjlab.tasks.registry import list_tasks

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_turning_sprint_env_cfg import (
    TURNING_SPRINT_MAX_SPEED_M_S,
    TURNING_SPRINT_SPEED_STAGES,
    make_microduck_turning_sprint_env_cfg,
)
from mjlab_microduck.tasks.mdp import SlewedTurningVelocityCommandCfg, slew_twist_velocity_command


def test_turning_sprint_keeps_recovery_constraints_and_trains_yaw():
    cfg = make_microduck_turning_sprint_env_cfg()
    command = cfg.commands["twist"]

    assert isinstance(command, SlewedTurningVelocityCommandCfg)
    assert command.rel_turn_in_place_envs == 0.20
    assert command.ranges.lin_vel_y == (0.0, 0.0)
    assert command.ranges.ang_vel_z == (-0.65, 0.65)
    assert command.yaw_acceleration_rad_s2 == 1.60
    assert command.yaw_deceleration_rad_s2 == 2.40
    assert command.high_speed_yaw_fraction == 0.65
    assert "heading_hold" not in cfg.rewards
    assert cfg.rewards["track_angular_velocity"].weight == 3.25
    assert TURNING_SPRINT_SPEED_STAGES[-1]["max_speed"] == TURNING_SPRINT_MAX_SPEED_M_S == 0.85
    assert TURNING_SPRINT_SPEED_STAGES[-1]["max_yaw"] == 0.35
    assert "turning_sprint_command" in cfg.curriculum
    assert "sprint_forward_speed" not in cfg.curriculum


def test_turning_sprint_curriculum_preserves_yaw_at_the_final_speed_stage():
    ranges = SimpleNamespace(lin_vel_x=(0.0, 0.0), lin_vel_y=(1.0, 1.0), ang_vel_z=(0.0, 0.0))
    command = SimpleNamespace(cfg=SimpleNamespace(ranges=ranges))
    env = SimpleNamespace(
        common_step_counter=TURNING_SPRINT_SPEED_STAGES[-1]["step"] + 1,
        command_manager=SimpleNamespace(get_term=lambda name: command if name == "twist" else None),
    )

    cap = microduck_mdp.turning_sprint_command_ranges_curriculum(
        env, torch.empty(0, dtype=torch.long), "twist", list(TURNING_SPRINT_SPEED_STAGES)
    )

    assert ranges.lin_vel_x == (0.68, 0.85)
    assert ranges.lin_vel_y == (0.0, 0.0)
    assert ranges.ang_vel_z == (-0.35, 0.35)
    assert cap.item() == pytest.approx(0.85)


def test_signed_yaw_slew_limits_acceleration_braking_and_reversal():
    current = torch.tensor([0.0, 0.50, -0.30])
    target = torch.tensor([1.0, 0.0, 0.30])
    slewed = slew_twist_velocity_command(
        current, target, dt=0.02, acceleration=0.40, deceleration=0.60
    )

    assert torch.allclose(slewed, torch.tensor([0.008, 0.488, -0.288]), atol=1e-6)


def test_turning_sprint_is_registered_as_an_independent_task():
    import mjlab_microduck.tasks  # noqa: F401  (registration side effect)

    assert "Mjlab-TurningSprint-Flat-MicroDuck" in list_tasks()
