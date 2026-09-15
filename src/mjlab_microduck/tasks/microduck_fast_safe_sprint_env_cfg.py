"""Safety-preserving sprint task with stronger velocity-tracking pressure.

This is a fresh task for improving the v5 policy's measured speed, not a
continuation of its checkpoint.  It keeps v5's bounded disturbances, strong
upright reward, and horizon stabilization, while making speed error expensive
enough that a high command cannot be satisfied by a conservative low-speed
walk.
"""

from __future__ import annotations

import dataclasses

from .microduck_safe_sprint_env_cfg import (
    SAFE_PUSH_STAGES,
    MicroduckSafeSprintRlCfg,
    make_microduck_safe_sprint_env_cfg,
)

FAST_SAFE_SPRINT_MAX_SPEED_M_S = 0.95

# v5 qualified zero simulated falls under its 0.85 m/s fixed-policy protocol,
# but its 0.85 m/s command averaged only 0.49 m/s.  Promote in smaller bands
# and reserve a long final stage for speed tracking at the new 0.95 m/s ceiling.
FAST_SAFE_SPRINT_SPEED_STAGES = (
    {"step": 0, "min_speed": 0.25, "max_speed": 0.35},
    {"step": 1500 * 24, "min_speed": 0.30, "max_speed": 0.50},
    {"step": 3500 * 24, "min_speed": 0.40, "max_speed": 0.65},
    {"step": 6000 * 24, "min_speed": 0.50, "max_speed": 0.75},
    {"step": 8500 * 24, "min_speed": 0.60, "max_speed": 0.85},
    {"step": 11000 * 24, "min_speed": 0.70, "max_speed": FAST_SAFE_SPRINT_MAX_SPEED_M_S},
)


def make_microduck_fast_safe_sprint_env_cfg(play: bool = False):
    """Build a faster, but still bounded and safety-shaped, sprint task."""

    cfg = make_microduck_safe_sprint_env_cfg(play=play)
    command = cfg.commands["twist"]
    command.ranges.lin_vel_x = (
        FAST_SAFE_SPRINT_SPEED_STAGES[0]["min_speed"],
        FAST_SAFE_SPRINT_SPEED_STAGES[0]["max_speed"],
    )
    command.forward_acceleration_m_s2 = 0.40
    command.forward_deceleration_m_s2 = 0.55
    # v5 spent 45% of re-samples in its low-speed braking band.  Retain those
    # transitions for fall-safe braking, but expose more high-speed practice.
    command.slowdown_probability = 0.30
    command.slowdown_speed_range = (0.25, 0.40)

    # v5's broad 0.80 m/s tracking scale accepted a large steady-state speed
    # error.  A higher weight and 0.60 m/s scale reward true forward progress
    # while preserved uprightness keeps falling more costly than a fast stride.
    cfg.rewards["track_linear_velocity"].weight = 7.0
    cfg.rewards["track_linear_velocity"].params["std"] = 0.60
    cfg.rewards["upright"].weight = 3.5
    cfg.rewards["action_rate_l2"].weight = -0.08

    cfg.curriculum["sprint_forward_speed"].params["speed_stages"] = list(
        FAST_SAFE_SPRINT_SPEED_STAGES
    )
    cfg.curriculum["action_rate_weight"].params["weight_stages"] = [
        {"step": 0, "weight": -0.08},
        {"step": 8500 * 24, "weight": -0.07},
        {"step": 11000 * 24, "weight": -0.065},
    ]
    cfg.curriculum["head_camera_level_weight"].params["weight_stages"] = [
        {"step": 0, "weight": 0.35},
        {"step": 6000 * 24, "weight": 0.70},
        {"step": 11000 * 24, "weight": 1.00},
    ]
    cfg.curriculum["head_camera_world_rate_weight"].params["weight_stages"] = [
        {"step": 0, "weight": -0.002},
        {"step": 6000 * 24, "weight": -0.004},
        {"step": 11000 * 24, "weight": -0.006},
    ]

    # Keep exactly the v5 push ceiling rather than buying speed by weakening
    # disturbance recovery.  The longer final stage teaches the 0.95 m/s gait
    # under the same qualification magnitude.
    cfg.events["push_robot"].interval_range_s = (3.0, 5.0)
    cfg.curriculum["push_magnitude"].params["push_stages"] = list(SAFE_PUSH_STAGES)
    return cfg


MicroduckFastSafeSprintRlCfg = dataclasses.replace(
    MicroduckSafeSprintRlCfg,
    experiment_name="fast_safe_sprint_stable_head",
    run_name="fast_safe_sprint_stable_head",
    max_iterations=24_000,
)
