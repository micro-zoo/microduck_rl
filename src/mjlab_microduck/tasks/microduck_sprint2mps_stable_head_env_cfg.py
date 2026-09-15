"""Flat-ground 2 m/s sprint with a horizon-stabilised head camera.

This task deliberately differs from the general velocity task:

* every command is a positive, straight-ahead speed command, gradually
  scheduled from 0.3 m/s to a final 1.8--2.0 m/s band;
* head/body pose-command slots remain in the shared 61-D observation contract,
  but are zero-padded -- this policy owns a fixed forward view rather than an
  operator-commanded head pose;
* the head objective is expressed in the ``head_camera`` WORLD frame, so the
  policy learns to counter torso pitch/roll instead of merely holding neck
  joints near HOME.

``2.0 m/s`` is the command ceiling and a simulation-training target, not a
hardware capability claim.  It must pass closed-loop checkpoint evaluation and
then separate actuator, thermal, and hardware safety validation before use on a
robot.
"""

import dataclasses
import math
from copy import deepcopy

from mjlab.managers import CurriculumTermCfg, ObservationTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    NUM_STEPS_PER_ENV,
    MicroduckRlCfg,
    make_microduck_velocity_env_cfg,
)


SPRINT_MAX_SPEED_M_S = 2.0
_HEAD_CAMERA_CFG = SceneEntityCfg("robot", site_names=("head_camera",))

# All commands are forward-only.  The final band starts early enough to spend
# thousands of iterations jointly training max-speed tracking, braking, pushes,
# and gaze stability.  The v2 schedule exposed the complete 2 m/s + 0.30 m/s
# push combination for only its last 400 iterations, which was not enough for
# the policy to discover a recoverable fast gait.
SPRINT_SPEED_STAGES = (
    {"step": 0 * NUM_STEPS_PER_ENV, "min_speed": 0.30, "max_speed": 0.40},
    {"step": 1000 * NUM_STEPS_PER_ENV, "min_speed": 0.35, "max_speed": 0.50},
    {"step": 2500 * NUM_STEPS_PER_ENV, "min_speed": 0.50, "max_speed": 0.75},
    {"step": 4500 * NUM_STEPS_PER_ENV, "min_speed": 0.70, "max_speed": 1.05},
    {"step": 6500 * NUM_STEPS_PER_ENV, "min_speed": 0.85, "max_speed": 1.25},
    {"step": 8500 * NUM_STEPS_PER_ENV, "min_speed": 1.00, "max_speed": 1.45},
    {"step": 10500 * NUM_STEPS_PER_ENV, "min_speed": 1.15, "max_speed": 1.65},
    {"step": 12500 * NUM_STEPS_PER_ENV, "min_speed": 1.30, "max_speed": 1.80},
    {"step": 14500 * NUM_STEPS_PER_ENV, "min_speed": 1.55, "max_speed": SPRINT_MAX_SPEED_M_S},
)


def make_microduck_sprint2mps_stable_head_env_cfg(
    play: bool = False,
):
    """Build the flat, forward-only sprint environment."""
    cfg = make_microduck_velocity_env_cfg(play=play, rough=False)

    # The velocity factory is a shared base recipe.  Replace the command config
    # with an independent copy before narrowing it to a straight sprint.
    command = deepcopy(cfg.commands["twist"])
    command.rel_standing_envs = 0.0
    command.rel_heading_envs = 0.0
    command.rel_world_envs = 0.0
    command.rel_forward_envs = 1.0
    command.rel_turn_in_place_envs = 0.0
    command.heading_command = False
    command.ranges.heading = None
    command.ranges.lin_vel_x = (
        SPRINT_SPEED_STAGES[0]["min_speed"],
        SPRINT_SPEED_STAGES[0]["max_speed"],
    )
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (0.0, 0.0)
    # Runtime throttle changes are rate-limited.  Training therefore sees the
    # same acceleration and braking envelope used by the controller instead of
    # discontinuous velocity-target jumps.
    command.resampling_time_range = (1.5, 3.5)
    command.init_velocity_prob = 0.0
    cfg.commands["twist"] = microduck_mdp.SlewedForwardVelocityCommandCfg(
        **vars(command),
        forward_acceleration_m_s2=0.50,
        forward_deceleration_m_s2=0.75,
        # Keep 35% low-speed targets throughout every stage.  This rehearses
        # braking and prevents the final high-speed distribution from erasing
        # the gait needed to settle safely at 0.25--0.45 m/s.
        slowdown_probability=0.35,
        slowdown_speed_range=(0.25, 0.45),
    )

    # A fixed spawn yaw makes the forward sprint and the run-time command frame
    # identical.  Position remains random so parallel worlds do not share a
    # contact patch.
    cfg.events["reset_base"].params["pose_range"]["yaw"] = (0.0, 0.0)

    # This is a fixed-gaze policy.  Keep the 61-D command slots, but do not ask
    # it to optimise arbitrary operator head/body offsets which conflict with a
    # world-level horizon objective.
    for name in ("head_pose", "body_pose"):
        cfg.commands.pop(name, None)
    for name in ("head_pose_tracking", "head_pose_bias", "body_pose_tracking"):
        cfg.rewards.pop(name, None)
    for name in ("head_pose_range", "head_pose_bias_weight", "body_pose_range"):
        cfg.curriculum.pop(name, None)
    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding,
            params={"dim": 4},
        )
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding,
            params={"dim": 6},
        )

    # A fast gait needs more flight time freedom and weaker late action-rate
    # regularisation than the low-speed walking policy.  The task still keeps
    # the parent action-over-limit, torque, contact, noise, and DR protections.
    cfg.rewards["track_linear_velocity"].weight = 4.0
    # Keep a tracking gradient at the early high-speed stages rather than
    # letting a large error collapse the Gaussian reward to zero.
    cfg.rewards["track_linear_velocity"].params["std"] = 0.80
    cfg.rewards["track_angular_velocity"].weight = 3.0
    cfg.rewards["track_angular_velocity"].params["std"] = math.sqrt(0.30)
    cfg.rewards["upright"].weight = 2.5
    cfg.rewards["air_time"].params["threshold_min"] = 0.07
    cfg.rewards["air_time"].params["threshold_max"] = 0.22
    cfg.rewards["foot_swing_height"].params["target_height"] = 0.025
    cfg.rewards["action_rate_l2"].weight = -0.06

    # Keep a straight body heading while the camera suppresses pitch/roll view
    # motion.  A light camera objective starts with gait discovery, so it is
    # not suddenly introduced at the same time as max speed and disturbance.
    cfg.rewards["heading_hold"] = RewardTermCfg(
        func=microduck_mdp.heading_hold_reward,
        weight=0.5,
        params={"std": math.radians(15.0)},
    )
    cfg.rewards["head_camera_level_forward"] = RewardTermCfg(
        func=microduck_mdp.head_camera_level_forward_reward,
        weight=0.0,
        params={"asset_cfg": _HEAD_CAMERA_CFG, "forward_std": 0.35, "up_std": 0.35},
    )
    cfg.rewards["head_camera_world_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.head_camera_world_angular_rate_l2,
        weight=0.0,
        params={"asset_cfg": _HEAD_CAMERA_CFG},
    )

    # No standing fraction in a dedicated sprint task.
    cfg.curriculum.pop("standing_envs", None)
    cfg.curriculum["sprint_forward_speed"] = CurriculumTermCfg(
        func=microduck_mdp.forward_speed_command_ranges_curriculum,
        params={"command_name": "twist", "speed_stages": list(SPRINT_SPEED_STAGES)},
    )
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0 * NUM_STEPS_PER_ENV, "weight": -0.06},
                {"step": 6500 * NUM_STEPS_PER_ENV, "weight": -0.05},
                {"step": 10500 * NUM_STEPS_PER_ENV, "weight": -0.04},
                {"step": 14500 * NUM_STEPS_PER_ENV, "weight": -0.03},
            ],
        },
    )
    cfg.curriculum["head_camera_level_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "head_camera_level_forward",
            "weight_stages": [
                {"step": 0 * NUM_STEPS_PER_ENV, "weight": 0.15},
                {"step": 3000 * NUM_STEPS_PER_ENV, "weight": 0.25},
                {"step": 7000 * NUM_STEPS_PER_ENV, "weight": 0.50},
                {"step": 11000 * NUM_STEPS_PER_ENV, "weight": 0.75},
                {"step": 14000 * NUM_STEPS_PER_ENV, "weight": 1.00},
            ],
        },
    )
    cfg.curriculum["head_camera_world_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "head_camera_world_rate_l2",
            "weight_stages": [
                {"step": 0 * NUM_STEPS_PER_ENV, "weight": -0.001},
                {"step": 5000 * NUM_STEPS_PER_ENV, "weight": -0.003},
                {"step": 10000 * NUM_STEPS_PER_ENV, "weight": -0.005},
                {"step": 14000 * NUM_STEPS_PER_ENV, "weight": -0.006},
            ],
        },
    )

    # A fall ends the episode, so the policy loses all future survival and
    # tracking reward.  Train that response explicitly under bounded push
    # disturbances, after clean acceleration/deceleration has emerged.
    cfg.events["push_robot"].interval_range_s = (2.0, 4.0)
    cfg.curriculum["push_magnitude"] = CurriculumTermCfg(
        func=microduck_mdp.push_curriculum,
        params={
            "event_name": "push_robot",
            "push_stages": [
                {"step": 0 * NUM_STEPS_PER_ENV, "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)}},
                {"step": 3500 * NUM_STEPS_PER_ENV, "velocity_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03)}},
                {"step": 6500 * NUM_STEPS_PER_ENV, "velocity_range": {"x": (-0.06, 0.06), "y": (-0.06, 0.06)}},
                {"step": 9000 * NUM_STEPS_PER_ENV, "velocity_range": {"x": (-0.10, 0.10), "y": (-0.10, 0.10)}},
                {"step": 11500 * NUM_STEPS_PER_ENV, "velocity_range": {"x": (-0.15, 0.15), "y": (-0.15, 0.15)}},
                {"step": 13500 * NUM_STEPS_PER_ENV, "velocity_range": {"x": (-0.18, 0.18), "y": (-0.18, 0.18)}},
                {"step": 15000 * NUM_STEPS_PER_ENV, "velocity_range": {"x": (-0.22, 0.22), "y": (-0.22, 0.22)}},
                {"step": 16500 * NUM_STEPS_PER_ENV, "velocity_range": {"x": (-0.25, 0.25), "y": (-0.25, 0.25)}},
                {"step": 17500 * NUM_STEPS_PER_ENV, "velocity_range": {"x": (-0.30, 0.30), "y": (-0.30, 0.30)}},
            ],
        },
    )

    return cfg


MicroduckSprint2mpsStableHeadRlCfg = dataclasses.replace(
    MicroduckRlCfg,
    experiment_name="sprint2mps_stable_head",
    run_name="sprint2mps_stable_head",
    max_iterations=20_000,
)
