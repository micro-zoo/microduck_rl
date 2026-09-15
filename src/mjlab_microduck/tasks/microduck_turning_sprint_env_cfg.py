"""Fresh high-speed MicroDuck sprint that can track safe yaw commands.

This task is intentionally independent of the straight-only recovery sprint.
It retains its fixed-gaze, acceleration/braking, and push-recovery constraints,
but trains the policy on yaw commands and a turn-in-place bucket. At higher
forward speeds the command term narrows yaw automatically instead of asking a
new gait to make abrupt full-rate turns.
"""

from __future__ import annotations

import dataclasses
from copy import deepcopy

from mjlab.managers import CurriculumTermCfg

from mjlab_microduck.tasks import mdp as microduck_mdp

from .microduck_recovery_sprint_env_cfg import (
    MicroduckRecoverySprintRlCfg,
    make_microduck_recovery_sprint_env_cfg,
)
from .microduck_velocity_env_cfg import NUM_STEPS_PER_ENV


TURNING_SPRINT_MAX_SPEED_M_S = 0.85
TURNING_SPRINT_SPEED_STAGES = (
    {"step": 0 * NUM_STEPS_PER_ENV, "min_speed": 0.25, "max_speed": 0.35, "max_yaw": 0.65},
    {"step": 2500 * NUM_STEPS_PER_ENV, "min_speed": 0.30, "max_speed": 0.48, "max_yaw": 0.60},
    {"step": 6000 * NUM_STEPS_PER_ENV, "min_speed": 0.40, "max_speed": 0.62, "max_yaw": 0.50},
    {"step": 11000 * NUM_STEPS_PER_ENV, "min_speed": 0.50, "max_speed": 0.72, "max_yaw": 0.45},
    {"step": 17000 * NUM_STEPS_PER_ENV, "min_speed": 0.60, "max_speed": 0.78, "max_yaw": 0.40},
    {"step": 23000 * NUM_STEPS_PER_ENV, "min_speed": 0.68, "max_speed": TURNING_SPRINT_MAX_SPEED_M_S, "max_yaw": 0.35},
)


def make_microduck_turning_sprint_env_cfg(play: bool = False):
    """Build a fresh v8 speed-and-turning curriculum without a checkpoint."""
    cfg = make_microduck_recovery_sprint_env_cfg(play=play)

    command = deepcopy(cfg.commands["twist"])
    command.ranges.lin_vel_x = (
        TURNING_SPRINT_SPEED_STAGES[0]["min_speed"],
        TURNING_SPRINT_SPEED_STAGES[0]["max_speed"],
    )
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (
        -TURNING_SPRINT_SPEED_STAGES[0]["max_yaw"],
        TURNING_SPRINT_SPEED_STAGES[0]["max_yaw"],
    )
    command.rel_turn_in_place_envs = 0.20
    cfg.commands["twist"] = microduck_mdp.SlewedTurningVelocityCommandCfg(
        **vars(command),
        yaw_acceleration_rad_s2=1.60,
        yaw_deceleration_rad_s2=2.40,
        high_speed_yaw_fraction=0.65,
    )

    # A static-heading reward directly conflicts with a commanded turn. The
    # head camera remains level in world space, which is yaw-invariant.
    cfg.rewards.pop("heading_hold", None)
    cfg.rewards["track_angular_velocity"].weight = 3.25

    cfg.curriculum.pop("sprint_forward_speed", None)
    cfg.curriculum["turning_sprint_command"] = CurriculumTermCfg(
        func=microduck_mdp.turning_sprint_command_ranges_curriculum,
        params={"command_name": "twist", "speed_stages": list(TURNING_SPRINT_SPEED_STAGES)},
    )
    return cfg


MicroduckTurningSprintRlCfg = dataclasses.replace(
    MicroduckRecoverySprintRlCfg,
    experiment_name="turning_sprint_stable_head",
    run_name="turning_sprint_stable_head",
    max_iterations=42_000,
)
