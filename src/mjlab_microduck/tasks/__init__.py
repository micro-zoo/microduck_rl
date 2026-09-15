from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .tracking_feedback_runner import TrackingFeedbackMotionTrackingOnPolicyRunner


class MicroduckOnPolicyRunner(VelocityOnPolicyRunner):
    def __init__(self, env, train_cfg: dict, log_dir=None, device="cpu", **kwargs):
        super().__init__(env, train_cfg, log_dir, device, **kwargs)
        # resolve_symmetry_config injects _env into train_cfg["algorithm"]["symmetry_cfg"]
        # in-place, sharing the same dict object with self.alg.symmetry.  Replace the
        # train_cfg reference with a copy that omits _env so dump_yaml can serialize the
        # config (MjSpec is not picklable), without touching the PPO's internal reference.
        alg = train_cfg.get("algorithm", {})
        sym = alg.get("symmetry_cfg") if isinstance(alg, dict) else None
        if isinstance(sym, dict) and "_env" in sym:
            alg["symmetry_cfg"] = {k: v for k, v in sym.items() if k != "_env"}


from .microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
    MicroduckRlCfg,
)
from .microduck_standup_env_cfg import (
    make_microduck_standup_env_cfg,
    MicroduckStandUpRlCfg,
)
from .microduck_velstand_env_cfg import (
    make_microduck_velstand_env_cfg,
    MicroduckVelStandRlCfg,
)
from .microduck_ground_pick_env_cfg import (
    make_microduck_ground_pick_env_cfg,
    MicroduckGroundPickRlCfg,
)
from .microduck_ball_kick_env_cfg import (
    make_microduck_ball_kick_env_cfg,
    MicroduckBallKickRlCfg,
)
from .microduck_sitstand_env_cfg import (
    make_microduck_sitstand_env_cfg,
    MicroduckSitStandRlCfg,
)
from .microduck_velocity_rollers_env_cfg import (
    make_microduck_velocity_rollers_env_cfg,
    MicroduckRollersRlCfg,
)
from .microduck_velocity_swizzle_env_cfg import (
    make_microduck_velocity_swizzle_env_cfg,
    MicroduckSwizzleRlCfg,
)
from .microduck_roller_crouch_env_cfg import (
    make_microduck_roller_crouch_env_cfg,
    MicroduckRollerCrouchRlCfg,
)
from .microduck_roller_slope_env_cfg import (
    make_microduck_roller_slope_env_cfg,
    MicroduckRollerSlopeRlCfg,
)
from .microduck_roller_standup_env_cfg import (
    make_microduck_roller_standup_env_cfg,
    MicroduckRollerStandUpRlCfg,
)
from .microduck_spin_env_cfg import (
    make_microduck_spin_env_cfg,
    MicroduckSpinRlCfg,
)
from .microduck_spinkick_mimic_env_cfg import (
    MICRODUCK_INTERMEDIATE_TRACKING_TERMINATION_DISTANCE,
    make_microduck_spinkick_mimic_env_cfg,
    microduck_spinkick_mimic_runner_cfg,
)
from .microduck_roulade_env_cfg import (
    make_microduck_roulade_env_cfg,
    MicroduckRouladeRlCfg,
)
from .microduck_dance_env_cfg import (
    make_microduck_dance_env_cfg,
    MicroduckDanceRlCfg,
)
from .microduck_fast_safe_sprint_env_cfg import (
    make_microduck_fast_safe_sprint_env_cfg,
    MicroduckFastSafeSprintRlCfg,
)
from .microduck_recovery_sprint_env_cfg import (
    make_microduck_recovery_sprint_env_cfg,
    MicroduckRecoverySprintRlCfg,
)
from .microduck_running_env_cfg import (
    make_microduck_running_env_cfg,
    MicroduckRunningRlCfg,
)
from .microduck_running_stable_head_env_cfg import (
    make_microduck_running_stable_head_env_cfg,
    MicroduckRunningStableHeadRlCfg,
)
from .microduck_safe_sprint_env_cfg import (
    make_microduck_safe_sprint_env_cfg,
    MicroduckSafeSprintRlCfg,
)
from .microduck_sprint2mps_stable_head_env_cfg import (
    make_microduck_sprint2mps_stable_head_env_cfg,
    MicroduckSprint2mpsStableHeadRlCfg,
)
from .microduck_sprint_env_cfg import (
    make_microduck_sprint_env_cfg,
    MicroduckSprintRlCfg,
)
from .microduck_turning_sprint_env_cfg import (
    make_microduck_turning_sprint_env_cfg,
    MicroduckTurningSprintRlCfg,
)
from .backlash import make_backlash_variant

# Standard velocity task
register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-MicroDuck",
    env_cfg=make_microduck_velocity_env_cfg(),
    play_env_cfg=make_microduck_velocity_env_cfg(play=True),
    rl_cfg=MicroduckRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
# Canonical top-speed task. DuckEMW's released 2.196 m/s result ultimately used
# Hannes/Vottivott's Running reward and curriculum. Sprint keeps that proven
# recipe, adds world-frame head stabilization, and always trains it from random
# initialization rather than treating the fixed gaze as a fine-tuning stage.
register_mjlab_task(
    task_id="Mjlab-Sprint-Flat-MicroDuck",
    env_cfg=make_microduck_running_stable_head_env_cfg(),
    play_env_cfg=make_microduck_running_stable_head_env_cfg(play=True),
    rl_cfg=MicroduckRunningStableHeadRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Explicit name for the fixed-gaze variant used by Sprint.
register_mjlab_task(
    task_id="Mjlab-RunningStableHead-Flat-MicroDuck",
    env_cfg=make_microduck_running_stable_head_env_cfg(),
    play_env_cfg=make_microduck_running_stable_head_env_cfg(play=True),
    rl_cfg=MicroduckRunningStableHeadRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Upstream-compatible name used by the deterministic running evaluator.
register_mjlab_task(
    task_id="Mjlab-Running-Flat-MicroDuck",
    env_cfg=make_microduck_running_env_cfg(),
    play_env_cfg=make_microduck_running_env_cfg(play=True),
    rl_cfg=MicroduckRunningRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Retain the earlier DuckEMW v4 environment only as an explicit comparison.
register_mjlab_task(
    task_id="Mjlab-SprintV4-Flat-MicroDuck",
    env_cfg=make_microduck_sprint_env_cfg(),
    play_env_cfg=make_microduck_sprint_env_cfg(play=True),
    rl_cfg=MicroduckSprintRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
register_mjlab_task(
    task_id="Mjlab-Velocity-Rough-MicroDuck",
    env_cfg=make_microduck_velocity_env_cfg(rough=True),
    play_env_cfg=make_microduck_velocity_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
# Flat, straight 2 m/s command ceiling with a world-level, forward-looking
# head camera.  This is intentionally separate from the general walk task.
register_mjlab_task(
    task_id="Mjlab-Sprint2mps-Flat-MicroDuck",
    env_cfg=make_microduck_sprint2mps_stable_head_env_cfg(),
    play_env_cfg=make_microduck_sprint2mps_stable_head_env_cfg(play=True),
    rl_cfg=MicroduckSprint2mpsStableHeadRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Safety-first sprint: an independent low-fall qualification task.  Its speed
# ceiling is intentionally lower than the 2 m/s research target.
register_mjlab_task(
    task_id="Mjlab-SafeSprint-Flat-MicroDuck",
    env_cfg=make_microduck_safe_sprint_env_cfg(),
    play_env_cfg=make_microduck_safe_sprint_env_cfg(play=True),
    rl_cfg=MicroduckSafeSprintRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# A fresh speed-optimised variant that retains the qualified v5 disturbance and
# horizon constraints while increasing the pressure to track the target speed.
register_mjlab_task(
    task_id="Mjlab-FastSafeSprint-Flat-MicroDuck",
    env_cfg=make_microduck_fast_safe_sprint_env_cfg(),
    play_env_cfg=make_microduck_fast_safe_sprint_env_cfg(play=True),
    rl_cfg=MicroduckFastSafeSprintRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# v7: stage nominal high-speed gait formation before applying the final
# disturbance envelope, while retaining fixed-gaze and upright constraints.
register_mjlab_task(
    task_id="Mjlab-RecoverySprint-Flat-MicroDuck",
    env_cfg=make_microduck_recovery_sprint_env_cfg(),
    play_env_cfg=make_microduck_recovery_sprint_env_cfg(play=True),
    rl_cfg=MicroduckRecoverySprintRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Fresh v8 sprint: retains recovery constraints but learns bounded yaw control.
register_mjlab_task(
    task_id="Mjlab-TurningSprint-Flat-MicroDuck",
    env_cfg=make_microduck_turning_sprint_env_cfg(),
    play_env_cfg=make_microduck_turning_sprint_env_cfg(play=True),
    rl_cfg=MicroduckTurningSprintRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
# VelStand — walking + fall recovery + body pose control in one policy.
register_mjlab_task(
    task_id="Mjlab-VelStand-Flat-MicroDuck",
    env_cfg=make_microduck_velstand_env_cfg(),
    play_env_cfg=make_microduck_velstand_env_cfg(play=True),
    rl_cfg=MicroduckVelStandRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-VelStand-Rough-MicroDuck",
    env_cfg=make_microduck_velstand_env_cfg(rough=True),
    play_env_cfg=make_microduck_velstand_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckVelStandRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Stand-up task — robot starts inverted (lying on back) and must stand up
register_mjlab_task(
    task_id="Mjlab-StandUp-Flat-MicroDuck",
    env_cfg=make_microduck_standup_env_cfg(),
    play_env_cfg=make_microduck_standup_env_cfg(play=True),
    rl_cfg=MicroduckStandUpRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-StandUp-Rough-MicroDuck",
    env_cfg=make_microduck_standup_env_cfg(rough=True),
    play_env_cfg=make_microduck_standup_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckStandUpRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# SitStand task — commanded sit ↔ stand in one policy, gently, head commandable
register_mjlab_task(
    task_id="Mjlab-SitStand-Flat-MicroDuck",
    env_cfg=make_microduck_sitstand_env_cfg(),
    play_env_cfg=make_microduck_sitstand_env_cfg(play=True),
    rl_cfg=MicroduckSitStandRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-SitStand-Rough-MicroDuck",
    env_cfg=make_microduck_sitstand_env_cfg(rough=True),
    play_env_cfg=make_microduck_sitstand_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckSitStandRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Ground-pick task — crouch, touch the ground with the mouth tip, return to stand
register_mjlab_task(
    task_id="Mjlab-GroundPick-Flat-MicroDuck",
    env_cfg=make_microduck_ground_pick_env_cfg(),
    play_env_cfg=make_microduck_ground_pick_env_cfg(play=True),
    rl_cfg=MicroduckGroundPickRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# BallKick task — kick a 70mm/15g ball forward hard with the right foot from a
# standing start (flat terrain only — a ball on rough terrain is another task).
register_mjlab_task(
    task_id="Mjlab-BallKick-Flat-MicroDuck",
    env_cfg=make_microduck_ball_kick_env_cfg(),
    play_env_cfg=make_microduck_ball_kick_env_cfg(play=True),
    rl_cfg=MicroduckBallKickRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-GroundPick-Rough-MicroDuck",
    env_cfg=make_microduck_ground_pick_env_cfg(rough=True),
    play_env_cfg=make_microduck_ground_pick_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckGroundPickRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Roller skate velocity task (passive-wheel model; historical task id kept)
register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-MicroDuck-Rollers",
    env_cfg=make_microduck_velocity_rollers_env_cfg(),
    play_env_cfg=make_microduck_velocity_rollers_env_cfg(play=True),
    rl_cfg=MicroduckRollersRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Roller SWIZZLE task — clean classic swizzle (symmetric, feet grounded).
register_mjlab_task(
    task_id="Mjlab-Velocity-Swizzle-MicroDuck",
    env_cfg=make_microduck_velocity_swizzle_env_cfg(),
    play_env_cfg=make_microduck_velocity_swizzle_env_cfg(play=True),
    rl_cfg=MicroduckSwizzleRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-RollerCrouch-Flat-MicroDuck",
    env_cfg=make_microduck_roller_crouch_env_cfg(),
    play_env_cfg=make_microduck_roller_crouch_env_cfg(play=True),
    rl_cfg=MicroduckRollerCrouchRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-RollerSlope-Flat-MicroDuck",
    env_cfg=make_microduck_roller_slope_env_cfg(),
    play_env_cfg=make_microduck_roller_slope_env_cfg(play=True),
    rl_cfg=MicroduckRollerSlopeRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Roller STANDUP — se relever sur rollers (policy dédiée, départ au sol).
register_mjlab_task(
    task_id="Mjlab-RollerStandUp-Flat-MicroDuck",
    env_cfg=make_microduck_roller_standup_env_cfg(),
    play_env_cfg=make_microduck_roller_standup_env_cfg(play=True),
    rl_cfg=MicroduckRollerStandUpRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Spin task — rotation rapide sur place, sur rollers (slot ground-pick).
register_mjlab_task(
    task_id="Mjlab-Spin-Flat-MicroDuck",
    env_cfg=make_microduck_spin_env_cfg(),
    play_env_cfg=make_microduck_spin_env_cfg(play=True),
    rl_cfg=MicroduckSpinRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
# Pure finite-motion imitation ported from mujocolab/g1_spinkick_example.
# This deliberately uses MJLab's stock BeyondMimic runner and is unrelated to
# the yaw-rate ``Mjlab-Spin-Flat-MicroDuck`` locomotion policy above.
register_mjlab_task(
    task_id="Mjlab-Spinkick-Mimic-MicroDuck",
    env_cfg=make_microduck_spinkick_mimic_env_cfg(),
    play_env_cfg=make_microduck_spinkick_mimic_env_cfg(play=True),
    rl_cfg=microduck_spinkick_mimic_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

# Direct port of mujocolab/g1_spinkick_example's training lifecycle.  The
# training environment deliberately keeps stock MotionCommand resampling and
# the stock 10-second episode; its play environment uses the finite command so
# the policy holds the final RL-zero frame after the reference ends.
register_mjlab_task(
    task_id="Mjlab-Spinkick-Official-BeyondMimic-MicroDuck",
    env_cfg=make_microduck_spinkick_mimic_env_cfg(
        official_training_lifecycle=True
    ),
    play_env_cfg=make_microduck_spinkick_mimic_env_cfg(play=True),
    rl_cfg=microduck_spinkick_mimic_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Spinkick-Official-ScaledGravity-MicroDuck",
    env_cfg=make_microduck_spinkick_mimic_env_cfg(
        official_training_lifecycle=True,
        morphology_scaled_gravity=True,
    ),
    play_env_cfg=make_microduck_spinkick_mimic_env_cfg(
        play=True,
        morphology_scaled_gravity=True,
    ),
    rl_cfg=microduck_spinkick_mimic_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

# Generic escape from the stationary local optimum of periodic quaternion and
# saturated exponential tracking rewards.  Every rollout follows the finite
# command from frame zero, while the two additional scores track the complete
# three-axis angular velocity and averaged full-body state.  No phase, spin
# axis, kick leg, keyframe, or motion-specific target is encoded here.
register_mjlab_task(
    task_id="Mjlab-Spinkick-Dense-Start-MicroDuck",
    env_cfg=make_microduck_spinkick_mimic_env_cfg(
        start_training=True,
        dense_tracking_guidance=True,
    ),
    play_env_cfg=make_microduck_spinkick_mimic_env_cfg(
        play=True,
        dense_tracking_guidance=True,
    ),
    rl_cfg=microduck_spinkick_mimic_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

# Generic high-fidelity continuation: every body and joint is tracked more
# tightly, with no spin-, kick-, or clip-specific shaping.  Train the
# deterministic Earth-physics stage first, then resume the same policy in the
# robust task below.
register_mjlab_task(
    task_id="Mjlab-Spinkick-Dense-Fidelity-Nominal-MicroDuck",
    env_cfg=make_microduck_spinkick_mimic_env_cfg(
        start_training=True,
        dense_tracking_guidance=True,
        high_fidelity_guidance=True,
        nominal_training=True,
    ),
    play_env_cfg=make_microduck_spinkick_mimic_env_cfg(
        play=True,
        dense_tracking_guidance=True,
        high_fidelity_guidance=True,
        nominal_training=True,
    ),
    rl_cfg=microduck_spinkick_mimic_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Spinkick-Dense-Fidelity-MicroDuck",
    env_cfg=make_microduck_spinkick_mimic_env_cfg(
        start_training=True,
        dense_tracking_guidance=True,
        high_fidelity_guidance=True,
    ),
    play_env_cfg=make_microduck_spinkick_mimic_env_cfg(
        play=True,
        dense_tracking_guidance=True,
        high_fidelity_guidance=True,
    ),
    rl_cfg=microduck_spinkick_mimic_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

# A closed-loop version of the generic high-fidelity task.  It adds only the
# already-available root-position error and base linear velocity to the actor;
# no motion-specific reward, limb, axis, or keyframe is introduced.
register_mjlab_task(
    task_id="Mjlab-Spinkick-Dense-Fidelity-Feedback-Nominal-MicroDuck",
    env_cfg=make_microduck_spinkick_mimic_env_cfg(
        start_training=True,
        dense_tracking_guidance=True,
        high_fidelity_guidance=True,
        actor_tracking_feedback=True,
        nominal_training=True,
    ),
    play_env_cfg=make_microduck_spinkick_mimic_env_cfg(
        play=True,
        dense_tracking_guidance=True,
        high_fidelity_guidance=True,
        actor_tracking_feedback=True,
        nominal_training=True,
    ),
    rl_cfg=microduck_spinkick_mimic_runner_cfg(),
    runner_cls=TrackingFeedbackMotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Spinkick-Dense-Fidelity-Feedback-MicroDuck",
    env_cfg=make_microduck_spinkick_mimic_env_cfg(
        start_training=True,
        dense_tracking_guidance=True,
        high_fidelity_guidance=True,
        actor_tracking_feedback=True,
    ),
    play_env_cfg=make_microduck_spinkick_mimic_env_cfg(
        play=True,
        dense_tracking_guidance=True,
        high_fidelity_guidance=True,
        actor_tracking_feedback=True,
    ),
    rl_cfg=microduck_spinkick_mimic_runner_cfg(),
    runner_cls=TrackingFeedbackMotionTrackingOnPolicyRunner,
)

# Optional PPO continuation curriculum, not part of UMR.  UMR preserves time
# and never modifies gravity.  All three stages keep the same start-state
# distribution, observations, actions and generic full-body objective as the
# Earth task above; only gravity changes to keep a moving initialization alive.
# Their checkpoints are intermediate only: final training, evaluation and
# videos must use the Earth task above at 9.81 m/s^2.
for _gravity_name, _gravity_magnitude in (
    ("3p3", 3.3),
    ("5p5", 5.5),
    ("7p7", 7.7),
):
    register_mjlab_task(
        task_id=(
            f"Mjlab-Spinkick-Dense-Gravity{_gravity_name}-MicroDuck"
        ),
        env_cfg=make_microduck_spinkick_mimic_env_cfg(
            start_training=True,
            dense_tracking_guidance=True,
            gravity_magnitude=_gravity_magnitude,
        ),
        play_env_cfg=make_microduck_spinkick_mimic_env_cfg(
            play=True,
            dense_tracking_guidance=True,
            gravity_magnitude=_gravity_magnitude,
        ),
        rl_cfg=microduck_spinkick_mimic_runner_cfg(),
        runner_cls=MotionTrackingOnPolicyRunner,
    )

# Same full-body BeyondMimic objective with a reference-centered residual
# action adapter.  This starts from retarget playback, not another learned
# policy/teacher, and learns the correction needed by delayed BAM actuators.
register_mjlab_task(
    task_id="Mjlab-Spinkick-Residual-MicroDuck",
    env_cfg=make_microduck_spinkick_mimic_env_cfg(residual_actions=True),
    play_env_cfg=make_microduck_spinkick_mimic_env_cfg(
        play=True, residual_actions=True
    ),
    rl_cfg=microduck_spinkick_mimic_runner_cfg(residual_actions=True),
    runner_cls=MotionTrackingOnPolicyRunner,
)

# Curriculum stage 1: identical residual policy and full-body objective under
# deterministic nominal BAM physics.  Its checkpoint can be resumed directly
# in the robust residual task above because observations/actions are identical.
register_mjlab_task(
    task_id="Mjlab-Spinkick-Residual-Nominal-MicroDuck",
    env_cfg=make_microduck_spinkick_mimic_env_cfg(
        residual_actions=True, nominal_training=True
    ),
    play_env_cfg=make_microduck_spinkick_mimic_env_cfg(
        play=True, residual_actions=True, nominal_training=True
    ),
    rl_cfg=microduck_spinkick_mimic_runner_cfg(
        residual_actions=True, nominal_training=True
    ),
    runner_cls=MotionTrackingOnPolicyRunner,
)

# Broad adaptive-RSI stage with the signed spin-rate guidance retained.  This
# follows the focused frame-zero stage: it exposes every part of the finite
# clip without dropping the signal that fixed the stationary-spin failure.
register_mjlab_task(
    task_id="Mjlab-Spinkick-Residual-Nominal-Guided-MicroDuck",
    env_cfg=make_microduck_spinkick_mimic_env_cfg(
        residual_actions=True,
        nominal_training=True,
        dense_tracking_guidance=True,
        adaptive_start_probability=0.25,
        terminate_protected_contact=False,
    ),
    play_env_cfg=make_microduck_spinkick_mimic_env_cfg(
        play=True,
        residual_actions=True,
        nominal_training=True,
        dense_tracking_guidance=True,
        adaptive_start_probability=0.25,
        terminate_protected_contact=False,
    ),
    rl_cfg=microduck_spinkick_mimic_runner_cfg(
        residual_actions=True, nominal_training=True
    ),
    runner_cls=MotionTrackingOnPolicyRunner,
)

# Intermediate guard used between broad phase learning and the final 0.10 m
# strict pass.  It preserves exploration across the takeoff while rejecting
# the curled/head-supported trajectory long before motion completion.
register_mjlab_task(
    task_id="Mjlab-Spinkick-Residual-Nominal-Mid-MicroDuck",
    env_cfg=make_microduck_spinkick_mimic_env_cfg(
        residual_actions=True,
        nominal_training=True,
        dense_tracking_guidance=True,
        adaptive_start_probability=0.5,
        tracking_termination_distance=(
            MICRODUCK_INTERMEDIATE_TRACKING_TERMINATION_DISTANCE
        ),
    ),
    play_env_cfg=make_microduck_spinkick_mimic_env_cfg(
        play=True,
        residual_actions=True,
        nominal_training=True,
        dense_tracking_guidance=True,
        tracking_termination_distance=(
            MICRODUCK_INTERMEDIATE_TRACKING_TERMINATION_DISTANCE
        ),
    ),
    rl_cfg=microduck_spinkick_mimic_runner_cfg(
        residual_actions=True, nominal_training=True
    ),
    runner_cls=MotionTrackingOnPolicyRunner,
)

# Curriculum stage 2: resume the broad residual policy with Microduck-scale
# failure distances so lying through the final hold is no longer a successful
# episode.  Train nominal first, then use the robust strict task below.
register_mjlab_task(
    task_id="Mjlab-Spinkick-Residual-Nominal-Strict-MicroDuck",
    env_cfg=make_microduck_spinkick_mimic_env_cfg(
        residual_actions=True,
        nominal_training=True,
        strict_tracking=True,
        dense_tracking_guidance=True,
    ),
    play_env_cfg=make_microduck_spinkick_mimic_env_cfg(
        play=True,
        residual_actions=True,
        nominal_training=True,
        strict_tracking=True,
        dense_tracking_guidance=True,
    ),
    rl_cfg=microduck_spinkick_mimic_runner_cfg(
        residual_actions=True, nominal_training=True
    ),
    runner_cls=MotionTrackingOnPolicyRunner,
)

# Focused stage 2a: all training rollouts start at frame zero until the policy
# can enter and survive the first spin from the commanded standing state.
register_mjlab_task(
    task_id="Mjlab-Spinkick-Residual-Nominal-Strict-Start-MicroDuck",
    env_cfg=make_microduck_spinkick_mimic_env_cfg(
        residual_actions=True,
        nominal_training=True,
        strict_tracking=True,
        start_training=True,
        dense_tracking_guidance=True,
    ),
    play_env_cfg=make_microduck_spinkick_mimic_env_cfg(
        play=True,
        residual_actions=True,
        nominal_training=True,
        strict_tracking=True,
        start_training=True,
        dense_tracking_guidance=True,
    ),
    rl_cfg=microduck_spinkick_mimic_runner_cfg(
        residual_actions=True, nominal_training=True
    ),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Spinkick-Residual-Strict-MicroDuck",
    env_cfg=make_microduck_spinkick_mimic_env_cfg(
        residual_actions=True,
        strict_tracking=True,
        dense_tracking_guidance=True,
    ),
    play_env_cfg=make_microduck_spinkick_mimic_env_cfg(
        play=True,
        residual_actions=True,
        strict_tracking=True,
        dense_tracking_guidance=True,
    ),
    rl_cfg=microduck_spinkick_mimic_runner_cfg(residual_actions=True),
    runner_cls=MotionTrackingOnPolicyRunner,
)
# Roulade — forward roll over the flat head top, land back on the feet.
register_mjlab_task(
    task_id="Mjlab-Roulade-Flat-MicroDuck",
    env_cfg=make_microduck_roulade_env_cfg(),
    play_env_cfg=make_microduck_roulade_env_cfg(play=True),
    rl_cfg=MicroduckRouladeRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Dance — beat-conditioned dancing on the spot.
register_mjlab_task(
    task_id="Mjlab-Dance-Flat-MicroDuck",
    env_cfg=make_microduck_dance_env_cfg(),
    play_env_cfg=make_microduck_dance_env_cfg(play=True),
    rl_cfg=MicroduckDanceRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Backlash variants — ±1° serial gear play per servo + encoder-through-backlash
# actuator feedback and joint obs (see tasks/backlash.py). Each family keeps its
# base task's collision model: Velocity → robot_walk_backlash.xml,
# VelStand/StandUp → robot_allcollisions_backlash.xml. Obs/action dims are
# unchanged vs the base tasks.
from mjlab_microduck.robot.microduck_constants import (
    MICRODUCK_BACKLASH_ROBOT_CFG,
    MICRODUCK_ROLLERS_BACKLASH_ROBOT_CFG,
    MICRODUCK_WALK_BACKLASH_ROBOT_CFG,
)

# (task_id, make_fn, make_kwargs, rl_cfg, backlash robot cfg). Task ids mirror
# the base ids with "-Backlash" inserted. Walk-model tasks get the walk
# backlash robot, roller tasks the wheels+backlash robot, the rest the
# allcollisions backlash robot — same model as their base task in each case.
_BL_ALLCOL = MICRODUCK_BACKLASH_ROBOT_CFG
_BL_WALK = MICRODUCK_WALK_BACKLASH_ROBOT_CFG
_BL_ROLLERS = MICRODUCK_ROLLERS_BACKLASH_ROBOT_CFG
_BACKLASH_TASKS = (
    ("Mjlab-Velocity-Flat-Backlash-MicroDuck", make_microduck_velocity_env_cfg, {}, MicroduckRlCfg, _BL_WALK),
    ("Mjlab-Velocity-Rough-Backlash-MicroDuck", make_microduck_velocity_env_cfg, {"rough": True}, MicroduckRlCfg, _BL_WALK),
    ("Mjlab-VelStand-Flat-Backlash-MicroDuck", make_microduck_velstand_env_cfg, {}, MicroduckVelStandRlCfg, _BL_ALLCOL),
    ("Mjlab-VelStand-Rough-Backlash-MicroDuck", make_microduck_velstand_env_cfg, {"rough": True}, MicroduckVelStandRlCfg, _BL_ALLCOL),
    ("Mjlab-StandUp-Flat-Backlash-MicroDuck", make_microduck_standup_env_cfg, {}, MicroduckStandUpRlCfg, _BL_ALLCOL),
    ("Mjlab-StandUp-Rough-Backlash-MicroDuck", make_microduck_standup_env_cfg, {"rough": True}, MicroduckStandUpRlCfg, _BL_ALLCOL),
    ("Mjlab-SitStand-Flat-Backlash-MicroDuck", make_microduck_sitstand_env_cfg, {}, MicroduckSitStandRlCfg, _BL_ALLCOL),
    ("Mjlab-SitStand-Rough-Backlash-MicroDuck", make_microduck_sitstand_env_cfg, {"rough": True}, MicroduckSitStandRlCfg, _BL_ALLCOL),
    ("Mjlab-GroundPick-Flat-Backlash-MicroDuck", make_microduck_ground_pick_env_cfg, {}, MicroduckGroundPickRlCfg, _BL_ALLCOL),
    ("Mjlab-GroundPick-Rough-Backlash-MicroDuck", make_microduck_ground_pick_env_cfg, {"rough": True}, MicroduckGroundPickRlCfg, _BL_ALLCOL),
    ("Mjlab-BallKick-Flat-Backlash-MicroDuck", make_microduck_ball_kick_env_cfg, {}, MicroduckBallKickRlCfg, _BL_ALLCOL),
    ("Mjlab-Velocity-Flat-Backlash-MicroDuck-Rollers", make_microduck_velocity_rollers_env_cfg, {}, MicroduckRollersRlCfg, _BL_ROLLERS),
    ("Mjlab-Velocity-Swizzle-Backlash-MicroDuck", make_microduck_velocity_swizzle_env_cfg, {}, MicroduckSwizzleRlCfg, _BL_ROLLERS),
    ("Mjlab-RollerCrouch-Flat-Backlash-MicroDuck", make_microduck_roller_crouch_env_cfg, {}, MicroduckRollerCrouchRlCfg, _BL_ROLLERS),
    ("Mjlab-RollerSlope-Flat-Backlash-MicroDuck", make_microduck_roller_slope_env_cfg, {}, MicroduckRollerSlopeRlCfg, _BL_ROLLERS),
    ("Mjlab-Dance-Flat-Backlash-MicroDuck", make_microduck_dance_env_cfg, {}, MicroduckDanceRlCfg, _BL_WALK),
)
for _task_id, _make_cfg, _kw, _rl_cfg, _robot_cfg in _BACKLASH_TASKS:
    register_mjlab_task(
        task_id=_task_id,
        env_cfg=make_backlash_variant(_make_cfg(**_kw), _robot_cfg),
        play_env_cfg=make_backlash_variant(_make_cfg(play=True, **_kw), _robot_cfg),
        rl_cfg=_rl_cfg,
        runner_cls=MicroduckOnPolicyRunner,
    )
