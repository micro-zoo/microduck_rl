"""Safety-first, straight-line MicroDuck sprint task.

This task deliberately separates the safety target from the historical 2 m/s
research task.  It learns the fastest gait inside the currently evidenced safe
band first; a later fixed-policy qualification, rather than an iteration count,
decides whether a new task may raise that ceiling.
"""

from __future__ import annotations

import dataclasses

from .microduck_sprint2mps_stable_head_env_cfg import (
    MicroduckSprint2mpsStableHeadRlCfg,
    make_microduck_sprint2mps_stable_head_env_cfg,
)


SAFE_SPRINT_MAX_SPEED_M_S = 0.85

# v4 showed clean 0.75 m/s windows but falls after the 1.05 m/s promotion.
# Hold the new policy below that observed failure boundary and give it a long
# final-stage budget before any separate faster-policy experiment is allowed.
SAFE_SPRINT_SPEED_STAGES = (
    {"step": 0, "min_speed": 0.25, "max_speed": 0.35},
    {"step": 1000 * 24, "min_speed": 0.30, "max_speed": 0.45},
    {"step": 2500 * 24, "min_speed": 0.35, "max_speed": 0.55},
    {"step": 4500 * 24, "min_speed": 0.40, "max_speed": 0.65},
    {"step": 6500 * 24, "min_speed": 0.50, "max_speed": 0.75},
    {"step": 8500 * 24, "min_speed": 0.60, "max_speed": SAFE_SPRINT_MAX_SPEED_M_S},
)

SAFE_PUSH_STAGES = (
    {"step": 0, "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)}},
    {"step": 3000 * 24, "velocity_range": {"x": (-0.01, 0.01), "y": (-0.01, 0.01)}},
    {"step": 6000 * 24, "velocity_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02)}},
    {"step": 9000 * 24, "velocity_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03)}},
    {"step": 12000 * 24, "velocity_range": {"x": (-0.04, 0.04), "y": (-0.04, 0.04)}},
    {"step": 15000 * 24, "velocity_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05)}},
)


def make_microduck_safe_sprint_env_cfg(play: bool = False):
    """Build a safety-first flat sprint environment.

    The 0.85 m/s ceiling is an explicit qualification boundary, not a claim
    that this task guarantees zero falls or that it is safe on hardware.
    """

    cfg = make_microduck_sprint2mps_stable_head_env_cfg(play=play)

    command = cfg.commands["twist"]
    command.ranges.lin_vel_x = (
        SAFE_SPRINT_SPEED_STAGES[0]["min_speed"],
        SAFE_SPRINT_SPEED_STAGES[0]["max_speed"],
    )
    command.forward_acceleration_m_s2 = 0.35
    command.forward_deceleration_m_s2 = 0.50
    command.slowdown_probability = 0.45
    command.slowdown_speed_range = (0.25, 0.40)

    # Safety dominates speed: uprightness and smooth actions stay strongly
    # shaped throughout the task, rather than being relaxed for a fast gait.
    cfg.rewards["track_linear_velocity"].weight = 3.0
    cfg.rewards["upright"].weight = 3.5
    cfg.rewards["action_rate_l2"].weight = -0.08

    cfg.curriculum["sprint_forward_speed"].params["speed_stages"] = list(
        SAFE_SPRINT_SPEED_STAGES
    )
    cfg.curriculum["action_rate_weight"].params["weight_stages"] = [
        {"step": 0, "weight": -0.08},
        {"step": 8500 * 24, "weight": -0.07},
    ]
    cfg.curriculum["head_camera_level_weight"].params["weight_stages"] = [
        {"step": 0, "weight": 0.35},
        {"step": 4500 * 24, "weight": 0.60},
        {"step": 8500 * 24, "weight": 1.00},
    ]
    cfg.curriculum["head_camera_world_rate_weight"].params["weight_stages"] = [
        {"step": 0, "weight": -0.002},
        {"step": 4500 * 24, "weight": -0.004},
        {"step": 8500 * 24, "weight": -0.006},
    ]

    cfg.events["push_robot"].interval_range_s = (3.0, 5.0)
    cfg.curriculum["push_magnitude"].params["push_stages"] = list(SAFE_PUSH_STAGES)
    return cfg


MicroduckSafeSprintRlCfg = dataclasses.replace(
    MicroduckSprint2mpsStableHeadRlCfg,
    experiment_name="safe_sprint_stable_head",
    run_name="safe_sprint_stable_head",
    max_iterations=20_000,
)
