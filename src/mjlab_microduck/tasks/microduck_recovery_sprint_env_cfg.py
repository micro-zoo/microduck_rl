"""Fresh v7 sprint that separates fast-gait discovery from push recovery.

v6 improved actual speed through 0.85 m/s command but repeatedly fell at the
0.95 m/s stage.  This task does not load that policy.  It first gives a
slightly lower 0.90 m/s ceiling time to discover a cadence, then introduces
the same final +/-0.05 m/s disturbance envelope in later stages.  High-speed
action smoothing is relaxed only after the gait has formed; head-level and
upright objectives remain in force throughout.
"""

from __future__ import annotations

import dataclasses

from .microduck_fast_safe_sprint_env_cfg import (
    MicroduckFastSafeSprintRlCfg,
    make_microduck_fast_safe_sprint_env_cfg,
)
from .microduck_velocity_env_cfg import NUM_STEPS_PER_ENV

RECOVERY_SPRINT_MAX_SPEED_M_S = 0.90

# The final speed is deliberately below v6's observed collapse point.  Its
# first high-speed period has only +/-0.02 m/s pushes, so PPO can establish a
# nominal gait before its recovery basin is challenged at +/-0.05 m/s.
RECOVERY_SPRINT_SPEED_STAGES = (
    {"step": 0 * NUM_STEPS_PER_ENV, "min_speed": 0.25, "max_speed": 0.35},
    {"step": 2000 * NUM_STEPS_PER_ENV, "min_speed": 0.30, "max_speed": 0.48},
    {"step": 5000 * NUM_STEPS_PER_ENV, "min_speed": 0.40, "max_speed": 0.62},
    {"step": 9000 * NUM_STEPS_PER_ENV, "min_speed": 0.50, "max_speed": 0.72},
    {"step": 14000 * NUM_STEPS_PER_ENV, "min_speed": 0.60, "max_speed": 0.80},
    {"step": 18000 * NUM_STEPS_PER_ENV, "min_speed": 0.68, "max_speed": RECOVERY_SPRINT_MAX_SPEED_M_S},
)

RECOVERY_SPRINT_PUSH_STAGES = (
    {"step": 0 * NUM_STEPS_PER_ENV, "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)}},
    {"step": 6000 * NUM_STEPS_PER_ENV, "velocity_range": {"x": (-0.01, 0.01), "y": (-0.01, 0.01)}},
    {"step": 12000 * NUM_STEPS_PER_ENV, "velocity_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02)}},
    {"step": 21000 * NUM_STEPS_PER_ENV, "velocity_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03)}},
    {"step": 27000 * NUM_STEPS_PER_ENV, "velocity_range": {"x": (-0.04, 0.04), "y": (-0.04, 0.04)}},
    {"step": 32000 * NUM_STEPS_PER_ENV, "velocity_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05)}},
)


def make_microduck_recovery_sprint_env_cfg(play: bool = False):
    """Build the v7 speed-and-recovery curriculum without weakening its test."""

    cfg = make_microduck_fast_safe_sprint_env_cfg(play=play)
    command = cfg.commands["twist"]
    command.ranges.lin_vel_x = (
        RECOVERY_SPRINT_SPEED_STAGES[0]["min_speed"],
        RECOVERY_SPRINT_SPEED_STAGES[0]["max_speed"],
    )
    command.forward_acceleration_m_s2 = 0.38
    command.forward_deceleration_m_s2 = 0.55
    command.slowdown_probability = 0.30
    command.slowdown_speed_range = (0.25, 0.40)

    # v6's 7.0 / 0.60 tracking pressure improved 0.65 m/s speed but repeatedly
    # drove the top command into falls.  Keep velocity tracking well above v5,
    # while widening the useful gradient enough for balance to compete.
    cfg.rewards["track_linear_velocity"].weight = 5.5
    cfg.rewards["track_linear_velocity"].params["std"] = 0.65
    cfg.rewards["upright"].weight = 3.5

    # Fast cadence needs less late action-rate suppression than v6.  This does
    # not remove smoothing, and it leaves the head-camera reward schedule intact.
    cfg.rewards["action_rate_l2"].weight = -0.08
    cfg.curriculum["action_rate_weight"].params["weight_stages"] = [
        {"step": 0 * NUM_STEPS_PER_ENV, "weight": -0.08},
        {"step": 12000 * NUM_STEPS_PER_ENV, "weight": -0.065},
        {"step": 18000 * NUM_STEPS_PER_ENV, "weight": -0.055},
        {"step": 30000 * NUM_STEPS_PER_ENV, "weight": -0.045},
    ]

    cfg.curriculum["sprint_forward_speed"].params["speed_stages"] = list(
        RECOVERY_SPRINT_SPEED_STAGES
    )
    cfg.events["push_robot"].interval_range_s = (3.0, 5.0)
    cfg.curriculum["push_magnitude"].params["push_stages"] = list(
        RECOVERY_SPRINT_PUSH_STAGES
    )
    return cfg


MicroduckRecoverySprintRlCfg = dataclasses.replace(
    MicroduckFastSafeSprintRlCfg,
    experiment_name="recovery_sprint_stable_head",
    run_name="recovery_sprint_stable_head",
    max_iterations=38_000,
)
