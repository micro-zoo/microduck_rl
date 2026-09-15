import math

import pytest
from mjlab.tasks.registry import load_runner_cls
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg

from mjlab_microduck.tasks.microduck_spinkick_mimic_env_cfg import (
    MICRODUCK_ACTION_JOINTS,
    MICRODUCK_ACTION_SCALE,
    MICRODUCK_INTERMEDIATE_TRACKING_TERMINATION_DISTANCE,
    MICRODUCK_LINEAR_TRACKING_SCALE,
    MICRODUCK_MORPHOLOGY_GRAVITY,
    MICRODUCK_REFERENCE_LOOKAHEAD_STEPS,
    MICRODUCK_RESIDUAL_ACTION_SCALE,
    MICRODUCK_SPIN_MAX_ANG_VEL,
    MICRODUCK_STRICT_TRACKING_TERMINATION_DISTANCE,
    MICRODUCK_TRACKED_BODIES,
    MICRODUCK_TRACKING_TERMINATION_DISTANCE,
    HoldLastMotionCommandCfg,
    ReferenceResidualJointPositionActionCfg,
    make_microduck_spinkick_mimic_env_cfg,
    microduck_spinkick_mimic_runner_cfg,
)


def test_spinkick_is_stock_beyondmimic_with_microduck_adapter():
    stock = make_tracking_env_cfg()
    cfg = make_microduck_spinkick_mimic_env_cfg()

    assert list(cfg.rewards) == list(stock.rewards)
    assert cfg.commands["motion"].anchor_body_name == "trunk_base"
    assert isinstance(cfg.commands["motion"], HoldLastMotionCommandCfg)
    assert cfg.commands["motion"].body_names == MICRODUCK_TRACKED_BODIES
    assert cfg.actions["joint_pos"].scale == MICRODUCK_ACTION_SCALE
    assert cfg.actions["joint_pos"].actuator_names == MICRODUCK_ACTION_JOINTS
    assert cfg.actions["joint_pos"].preserve_order is True
    assert "base_ang_vel_exceed" in cfg.terminations
    assert "protected_body_ground_contact" in cfg.terminations
    assert "protected_body_ground_contact" in {
        sensor.name for sensor in cfg.scene.sensors
    }
    assert cfg.terminations["motion_complete"].time_out is True
    assert math.isclose(
        cfg.terminations["base_ang_vel_exceed"].params["threshold"],
        MICRODUCK_SPIN_MAX_ANG_VEL,
    )
    # The final 1.0x reference peaks at 562.5 deg/s.  The guard must not make
    # exact reference tracking an impossible episode.
    assert MICRODUCK_SPIN_MAX_ANG_VEL > math.radians(562.5)
    assert "expand_bam_friction_fields" in cfg.events
    assert cfg.episode_length_s == 5.0
    assert math.isclose(
        cfg.rewards["motion_global_root_pos"].params["std"],
        stock.rewards["motion_global_root_pos"].params["std"]
        * MICRODUCK_LINEAR_TRACKING_SCALE,
    )
    assert math.isclose(
        cfg.rewards["motion_body_pos"].params["std"],
        stock.rewards["motion_body_pos"].params["std"]
        * MICRODUCK_LINEAR_TRACKING_SCALE,
    )
    assert math.isclose(
        cfg.rewards["motion_body_lin_vel"].params["std"],
        stock.rewards["motion_body_lin_vel"].params["std"]
        * MICRODUCK_LINEAR_TRACKING_SCALE,
    )
    assert math.isclose(
        cfg.terminations["anchor_pos"].params["threshold"],
        MICRODUCK_TRACKING_TERMINATION_DISTANCE,
    )
    assert math.isclose(
        cfg.terminations["ee_body_pos"].params["threshold"],
        MICRODUCK_TRACKING_TERMINATION_DISTANCE,
    )
    for axis in ("x", "y", "z"):
        assert cfg.commands["motion"].pose_range[axis] == tuple(
            value * MICRODUCK_LINEAR_TRACKING_SCALE
            for value in stock.commands["motion"].pose_range[axis]
        )
        assert cfg.commands["motion"].velocity_range[axis] == tuple(
            value * MICRODUCK_LINEAR_TRACKING_SCALE
            for value in stock.commands["motion"].velocity_range[axis]
        )


def test_spinkick_actor_matches_no_state_estimation_example():
    cfg = make_microduck_spinkick_mimic_env_cfg()
    terms = cfg.observations["actor"].terms

    assert "motion_anchor_pos_b" not in terms
    assert "base_lin_vel" not in terms
    assert "motion_anchor_ori_b" in terms
    assert "command" in terms
    assert "joint_pos" in terms
    assert "joint_vel" in terms


def test_official_spinkick_training_keeps_stock_command_lifecycle():
    cfg = make_microduck_spinkick_mimic_env_cfg(
        official_training_lifecycle=True
    )

    assert not isinstance(cfg.commands["motion"], HoldLastMotionCommandCfg)
    assert cfg.commands["motion"].sampling_mode == "adaptive"
    assert cfg.episode_length_s == 10.0
    assert "motion_complete" not in cfg.terminations
    assert "protected_body_ground_contact" not in cfg.terminations
    assert "protected_body_ground_contact" not in {
        sensor.name for sensor in cfg.scene.sensors
    }
    assert len(cfg.rewards) == 9


def test_scaled_gravity_is_only_a_generic_physics_curriculum():
    scaled = make_microduck_spinkick_mimic_env_cfg(
        official_training_lifecycle=True,
        morphology_scaled_gravity=True,
    )
    earth = make_microduck_spinkick_mimic_env_cfg(
        official_training_lifecycle=True
    )

    assert scaled.sim.mujoco.gravity == (
        0.0,
        0.0,
        -MICRODUCK_MORPHOLOGY_GRAVITY,
    )
    assert earth.sim.mujoco.gravity == (0.0, 0.0, -9.81)
    assert list(scaled.rewards) == list(earth.rewards)
    assert list(scaled.terminations) == list(earth.terminations)


def test_explicit_gravity_curriculum_only_changes_gravity():
    stage = make_microduck_spinkick_mimic_env_cfg(
        start_training=True,
        dense_tracking_guidance=True,
        gravity_magnitude=5.5,
    )
    earth = make_microduck_spinkick_mimic_env_cfg(
        start_training=True,
        dense_tracking_guidance=True,
    )

    assert stage.sim.mujoco.gravity == (0.0, 0.0, -5.5)
    assert earth.sim.mujoco.gravity == (0.0, 0.0, -9.81)
    assert list(stage.rewards) == list(earth.rewards)
    assert list(stage.terminations) == list(earth.terminations)
    assert stage.commands["motion"].sampling_mode == "start"

    with pytest.raises(ValueError, match="mutually exclusive"):
        make_microduck_spinkick_mimic_env_cfg(
            morphology_scaled_gravity=True,
            gravity_magnitude=5.5,
        )
    with pytest.raises(ValueError, match="must be positive"):
        make_microduck_spinkick_mimic_env_cfg(gravity_magnitude=0.0)


def test_spinkick_residual_variant_centers_actions_on_reference():
    cfg = make_microduck_spinkick_mimic_env_cfg(residual_actions=True)
    action = cfg.actions["joint_pos"]

    assert isinstance(action, ReferenceResidualJointPositionActionCfg)
    assert action.scale == MICRODUCK_RESIDUAL_ACTION_SCALE
    assert action.use_default_offset is False
    assert (
        action.reference_lookahead_steps
        == MICRODUCK_REFERENCE_LOOKAHEAD_STEPS
    )

    runner = microduck_spinkick_mimic_runner_cfg(residual_actions=True)
    assert runner.experiment_name == "microduck_spinkick_residual_mimic"
    assert runner.actor.distribution_cfg["init_std"] == 0.2


def test_spinkick_nominal_curriculum_preserves_bam_and_removes_noise():
    cfg = make_microduck_spinkick_mimic_env_cfg(
        residual_actions=True, nominal_training=True
    )
    motion = cfg.commands["motion"]

    assert list(cfg.events) == ["expand_bam_friction_fields"]
    assert motion.sampling_mode == "adaptive"
    assert motion.pose_range == {}
    assert motion.velocity_range == {}
    assert motion.joint_position_range == (0.0, 0.0)
    actuator = cfg.scene.entities["robot"].articulation.actuators[0]
    assert actuator.vin_range == (7.4, 7.4)
    assert actuator.vin_drop_gain_range == (0.1, 0.1)
    assert actuator.delay_min_lag == actuator.delay_max_lag == 4

    robust = make_microduck_spinkick_mimic_env_cfg(residual_actions=True)
    robust_actuator = robust.scene.entities["robot"].articulation.actuators[0]
    assert robust_actuator.vin_range == (6.5, 8.2)
    assert robust_actuator.delay_min_lag == 3
    assert robust_actuator.delay_max_lag == 6

    runner = microduck_spinkick_mimic_runner_cfg(
        residual_actions=True, nominal_training=True
    )
    assert (
        runner.experiment_name
        == "microduck_spinkick_residual_nominal_mimic"
    )


def test_spinkick_strict_curriculum_tightens_failure_distance():
    cfg = make_microduck_spinkick_mimic_env_cfg(
        residual_actions=True,
        nominal_training=True,
        strict_tracking=True,
    )

    assert math.isclose(
        cfg.terminations["anchor_pos"].params["threshold"],
        MICRODUCK_STRICT_TRACKING_TERMINATION_DISTANCE,
    )
    assert math.isclose(
        cfg.terminations["ee_body_pos"].params["threshold"],
        MICRODUCK_STRICT_TRACKING_TERMINATION_DISTANCE,
    )

    focused = make_microduck_spinkick_mimic_env_cfg(
        residual_actions=True,
        nominal_training=True,
        strict_tracking=True,
        start_training=True,
        dense_tracking_guidance=True,
    )
    assert focused.commands["motion"].sampling_mode == "start"
    assert "motion_anchor_ang_vel_dense" in focused.rewards
    assert "motion_full_body_tracking_dense" in focused.rewards
    assert (
        focused.rewards["motion_anchor_ang_vel_dense"].params[
            "angular_velocity_scale"
        ]
        == 8.0
    )

    mixed = make_microduck_spinkick_mimic_env_cfg(
        residual_actions=True,
        nominal_training=True,
        dense_tracking_guidance=True,
        adaptive_start_probability=0.25,
        terminate_protected_contact=False,
    )
    assert mixed.commands["motion"].sampling_mode == "adaptive"
    assert mixed.commands["motion"].adaptive_start_probability == 0.25
    assert "protected_body_ground_contact" not in mixed.terminations
    assert mixed.rewards["protected_body_ground_contact"].weight == -2.0

    intermediate = make_microduck_spinkick_mimic_env_cfg(
        residual_actions=True,
        nominal_training=True,
        tracking_termination_distance=(
            MICRODUCK_INTERMEDIATE_TRACKING_TERMINATION_DISTANCE
        ),
    )
    assert intermediate.terminations["anchor_pos"].params["threshold"] == 0.15
    assert intermediate.terminations["ee_body_pos"].params["threshold"] == 0.15


def test_spinkick_play_starts_at_reference_frame_zero():
    cfg = make_microduck_spinkick_mimic_env_cfg(play=True)
    motion = cfg.commands["motion"]

    assert motion.sampling_mode == "start"
    assert motion.pose_range == {}
    assert motion.velocity_range == {}
    assert cfg.observations["actor"].enable_corruption is False
    assert "push_robot" not in cfg.events
    assert "motion_complete" not in cfg.terminations


def test_spinkick_registration_uses_stock_motion_runner():
    # Importing mjlab_microduck.tasks (performed by package entry points) owns
    # registration; this is not the custom velocity or PBHC runner.
    import mjlab_microduck.tasks  # noqa: F401

    assert (
        load_runner_cls("Mjlab-Spinkick-Mimic-MicroDuck")
        is MotionTrackingOnPolicyRunner
    )
    assert (
        load_runner_cls("Mjlab-Spinkick-Official-BeyondMimic-MicroDuck")
        is MotionTrackingOnPolicyRunner
    )
    for gravity_name in ("3p3", "5p5", "7p7"):
        assert (
            load_runner_cls(
                f"Mjlab-Spinkick-Dense-Gravity{gravity_name}-MicroDuck"
            )
            is MotionTrackingOnPolicyRunner
        )
    assert (
        load_runner_cls("Mjlab-Spinkick-Official-ScaledGravity-MicroDuck")
        is MotionTrackingOnPolicyRunner
    )
    assert (
        load_runner_cls("Mjlab-Spinkick-Dense-Start-MicroDuck")
        is MotionTrackingOnPolicyRunner
    )
    assert (
        load_runner_cls("Mjlab-Spinkick-Residual-MicroDuck")
        is MotionTrackingOnPolicyRunner
    )
    assert (
        load_runner_cls("Mjlab-Spinkick-Residual-Nominal-MicroDuck")
        is MotionTrackingOnPolicyRunner
    )
    assert (
        load_runner_cls("Mjlab-Spinkick-Residual-Nominal-Guided-MicroDuck")
        is MotionTrackingOnPolicyRunner
    )
    assert (
        load_runner_cls("Mjlab-Spinkick-Residual-Nominal-Mid-MicroDuck")
        is MotionTrackingOnPolicyRunner
    )
    assert (
        load_runner_cls("Mjlab-Spinkick-Residual-Nominal-Strict-MicroDuck")
        is MotionTrackingOnPolicyRunner
    )
    assert (
        load_runner_cls(
            "Mjlab-Spinkick-Residual-Nominal-Strict-Start-MicroDuck"
        )
        is MotionTrackingOnPolicyRunner
    )
    assert (
        load_runner_cls("Mjlab-Spinkick-Residual-Strict-MicroDuck")
        is MotionTrackingOnPolicyRunner
    )
    runner = microduck_spinkick_mimic_runner_cfg()
    assert runner.experiment_name == "microduck_spinkick_mimic"
    assert runner.max_iterations == 20_000
