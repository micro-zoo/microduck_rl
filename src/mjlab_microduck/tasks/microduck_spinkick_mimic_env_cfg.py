"""BeyondMimic spin-kick tasks for Microduck.

This is the Microduck port of mujocolab/g1_spinkick_example.  The task starts
from MJLab's stock tracking environment and keeps its observations, rewards,
randomisation, adaptive reference-state initialisation and PPO algorithm.  The
finite Spin clip is held at its final standing frame instead of being treated
as a cyclic locomotion clip.  The remaining changes describe Microduck's robot
model, body names and motion-safe angular-velocity guard.

This module is intentionally independent of ``microduck_spin_env_cfg``.  That
module learns a commanded yaw-rate locomotion skill; this one learns a complete
finite full-body reference motion.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, fields
from typing import cast

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from mjlab.managers import EventTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.mdp.commands import MotionCommand
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from mjlab.utils.lab_api.math import quat_error_magnitude

from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp

# The G1 example uses 500 deg/s, but Microduck's *reference* trunk reaches
# 562.5 deg/s during the spin.  A 500 deg/s guard therefore terminates even
# perfect tracking.  Keep the same numerical-instability guard with enough
# margin for the intended motion instead of changing the motion's speed.
MICRODUCK_SPIN_MAX_ANG_VEL = 800.0 * math.pi / 180.0

# UMR's measured target/source height ratio for this exact pair
# (0.27154 m / 1.61146 m).  BeyondMimic's stock metric widths and termination
# distances are expressed in metres and tuned for a human-sized G1.  Keeping
# 0.30 m / 0.25 m unchanged makes a 0.27 m Microduck receive almost full pose
# reward while sitting still.  Scale only linear quantities; angular terms,
# joint-space noise, observations, rewards, and PPO remain stock BeyondMimic.
MICRODUCK_LINEAR_TRACKING_SCALE = 0.168508
MICRODUCK_MORPHOLOGY_GRAVITY = 9.81 * MICRODUCK_LINEAR_TRACKING_SCALE

# Reward widths should follow morphology scale, but termination thresholds are
# exploration guards rather than pose-accuracy tolerances.  A literal scaling
# of G1's 0.25 m guard to 0.042 m (and a 0.10 m compromise) ended most
# rollouts after only 14--16 control steps, before the delayed actuators could
# enter the spin.  Preserve the stock G1 exploration guard; the evaluator
# enforces much tighter full-horizon pose, yaw and final-stand criteria.
MICRODUCK_TRACKING_TERMINATION_DISTANCE = 0.25
MICRODUCK_INTERMEDIATE_TRACKING_TERMINATION_DISTANCE = 0.15
MICRODUCK_STRICT_TRACKING_TERMINATION_DISTANCE = 0.10

# G1's robot-specific action scales lie mostly in the 0.35--0.55 rad/action
# range.  Keep MJLab's 0.5 rad/action default for Microduck as the closest
# morphology-independent adapter.  Larger reference excursions remain fully
# representable by actions outside [-1, 1]; this scale only avoids destructive
# one-radian target noise from the stock initial policy standard deviation.
MICRODUCK_ACTION_SCALE = 0.5

# The residual variant starts from the retargeted joint-space playback and asks
# PPO to learn only the dynamically necessary correction.  One control-frame
# lookahead compensates the nominal 20 ms BAM command delay.  This is still a
# BeyondMimic motion-conditioned policy; no PBHC policy or optimized teacher is
# loaded.
MICRODUCK_RESIDUAL_ACTION_SCALE = 0.25
MICRODUCK_REFERENCE_LOOKAHEAD_STEPS = 1

# Track the complete actuated kinematic structure.  MotionCommand still loads
# body state for every body in the NPZ; these are the ten task bodies used by
# the stock BeyondMimic body-pose/body-velocity rewards.
MICRODUCK_TRACKED_BODIES = (
    "trunk_base",
    "upper_leg_left",
    "leg",
    "ankle_left",
    "upper_leg_right",
    "leg_2",
    "ankle_right",
    "neck",
    "neck_pitch",
    "yaw_roll_motion",
)

MICRODUCK_ACTION_JOINTS = (
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
)


class HoldLastMotionCommand(MotionCommand):
    """Advance a finite motion once, then keep commanding its final frame.

    Stock ``MotionCommand`` jumps back to frame zero at the end.  That is useful
    for cyclic locomotion, but it is incorrect for a one-shot spin followed by
    an explicit RL-zero standing tail.
    """

    cfg: HoldLastMotionCommandCfg

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        # Retain a deployment-start cohort while adaptive RSI concentrates the
        # remaining worlds around difficult later phases.  Pure adaptive RSI
        # improved the first landing but forgot the learned frame-zero rollout.
        super()._resample_command(env_ids)
        probability = self.cfg.adaptive_start_probability
        if self.cfg.sampling_mode != "adaptive" or probability <= 0.0:
            return
        start_mask = torch.rand(len(env_ids), device=self.device) < probability
        start_ids = env_ids[start_mask]
        if start_ids.numel() > 0:
            self.reset_to_frame(start_ids, 0)
            self.update_relative_body_poses()

    def _update_command(self) -> None:
        self.time_steps.add_(1).clamp_(max=self.motion.time_step_total - 1)
        self.update_relative_body_poses()

        # Preserve stock BeyondMimic adaptive-start feedback exactly.
        if self.cfg.sampling_mode == "adaptive":
            self.bin_failed_count = (
                self.cfg.adaptive_alpha * self._current_bin_failed
                + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
            )
            self._current_bin_failed.zero_()


@dataclass(kw_only=True)
class HoldLastMotionCommandCfg(MotionCommandCfg):
    """Configuration for a finite, non-looping BeyondMimic motion command."""

    adaptive_start_probability: float = 0.0

    def build(self, env: ManagerBasedRlEnv) -> MotionCommand:
        return HoldLastMotionCommand(self, env)


class ReferenceResidualJointPositionAction(JointPositionAction):
    """Apply a learned residual around the finite motion's joint target."""

    cfg: ReferenceResidualJointPositionActionCfg

    def __init__(
        self,
        cfg: ReferenceResidualJointPositionActionCfg,
        env: ManagerBasedRlEnv,
    ) -> None:
        super().__init__(cfg, env)
        self._command = cast(
            MotionCommand, env.command_manager.get_term(cfg.command_name)
        )

    def apply_actions(self) -> None:
        frame = torch.clamp(
            self._command.time_steps + self.cfg.reference_lookahead_steps,
            max=self._command.motion.time_step_total - 1,
        )
        reference = self._command.motion.joint_pos[frame][:, self._target_ids]
        encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
        target = reference + self._processed_actions - encoder_bias
        self._entity.set_joint_position_target(
            target, joint_ids=self._target_ids
        )


@dataclass(kw_only=True)
class ReferenceResidualJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for reference-centered residual position actions."""

    command_name: str = "motion"
    reference_lookahead_steps: int = MICRODUCK_REFERENCE_LOOKAHEAD_STEPS
    use_default_offset: bool = False

    def build(
        self, env: ManagerBasedRlEnv
    ) -> ReferenceResidualJointPositionAction:
        return ReferenceResidualJointPositionAction(self, env)


def base_ang_vel_exceed(
    env: ManagerBasedRlEnv,
    threshold: float,
) -> torch.Tensor:
    """Terminate worlds whose floating base has become numerically unstable."""

    asset: Entity = env.scene["robot"]
    return torch.any(asset.data.root_link_ang_vel_b.abs() > threshold, dim=-1)


def protected_body_ground_contact(
    env: ManagerBasedRlEnv,
    sensor_name: str,
) -> torch.Tensor:
    """Reject head/neck/trunk-supported poses that evade the root tilt guard."""

    sensor = env.scene[sensor_name]
    found = sensor.data.found
    if found is None:
        raise RuntimeError(f"contact sensor {sensor_name!r} has no found field")
    return found.reshape(env.num_envs, -1).any(dim=-1)


def protected_body_ground_contact_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str,
) -> torch.Tensor:
    """Generic soft penalty used before enabling the hard contact guard."""

    return protected_body_ground_contact(env, sensor_name).float()


def motion_complete(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    """End training after the finite reference, including its standing tail."""

    command = env.command_manager.get_term(command_name)
    assert isinstance(command, MotionCommand)
    return command.time_steps >= command.motion.time_step_total - 1


def motion_anchor_angular_velocity_tracking_dense(
    env: ManagerBasedRlEnv,
    command_name: str,
    angular_velocity_scale: float,
) -> torch.Tensor:
    """Generic non-saturating floating-base angular-velocity tracking.

    Quaternion pose rewards are periodic and their exponential kernel is nearly
    flat once the robot is far behind a fast reference.  Tracking the complete
    angular-velocity vector preserves rotation direction and magnitude for any
    dynamic clip; it contains no motion-specific phase, axis, or joint target.
    """

    command = cast(
        MotionCommand, env.command_manager.get_term(command_name)
    )
    error = torch.linalg.vector_norm(
        command.robot_anchor_ang_vel_w - command.anchor_ang_vel_w, dim=-1
    )
    # Keep a signed slope after the policy falls behind.  Clamping at zero
    # makes a stopped body and one rotating in the wrong direction
    # indistinguishable on fast clips; [-1, 1] stays bounded without losing
    # that generic directional signal.
    return (1.0 - error / angular_velocity_scale).clamp_(-1.0, 1.0)


def motion_full_body_tracking_dense(
    env: ManagerBasedRlEnv,
    command_name: str,
    position_scale: float,
    orientation_scale: float,
    joint_scale: float,
    linear_velocity_scale: float,
    angular_velocity_scale: float,
) -> torch.Tensor:
    """Non-saturating guidance that rejects curled or otherwise fake solutions.

    The stock exponential kernels are retained as the primary BeyondMimic
    objective.  Once the robot is far from a small reference, however, those
    kernels are numerically close to zero for both a recoverable deviation and
    a completely wrong pose.  This bounded linear score preserves that ordering
    across every tracked body and actuated joint.
    """

    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    body_position_error = torch.linalg.vector_norm(
        command.body_pos_relative_w - command.robot_body_pos_w, dim=-1
    ).mean(dim=-1)
    body_orientation_error = quat_error_magnitude(
        command.body_quat_relative_w, command.robot_body_quat_w
    ).mean(dim=-1)
    joint_position_error = (
        command.joint_pos - command.robot_joint_pos
    ).abs().mean(dim=-1)
    body_linear_velocity_error = torch.linalg.vector_norm(
        command.body_lin_vel_w - command.robot_body_lin_vel_w, dim=-1
    ).mean(dim=-1)
    body_angular_velocity_error = torch.linalg.vector_norm(
        command.body_ang_vel_w - command.robot_body_ang_vel_w, dim=-1
    ).mean(dim=-1)

    position_score = 1.0 - body_position_error / position_scale
    orientation_score = 1.0 - body_orientation_error / orientation_scale
    joint_score = 1.0 - joint_position_error / joint_scale
    linear_velocity_score = (
        1.0 - body_linear_velocity_error / linear_velocity_scale
    )
    angular_velocity_score = (
        1.0 - body_angular_velocity_error / angular_velocity_scale
    )
    return torch.stack(
        (
            position_score,
            orientation_score,
            joint_score,
            linear_velocity_score,
            angular_velocity_score,
        ),
        dim=-1,
    ).clamp_(min=0.0, max=1.0).mean(dim=-1)


def make_microduck_spinkick_mimic_env_cfg(
    play: bool = False,
    residual_actions: bool = False,
    nominal_training: bool = False,
    official_training_lifecycle: bool = False,
    morphology_scaled_gravity: bool = False,
    gravity_magnitude: float | None = None,
    strict_tracking: bool = False,
    start_training: bool = False,
    dense_tracking_guidance: bool = False,
    high_fidelity_guidance: bool = False,
    actor_tracking_feedback: bool = False,
    adaptive_start_probability: float = 0.0,
    tracking_termination_distance: float | None = None,
    terminate_protected_contact: bool = True,
) -> ManagerBasedRlEnvCfg:
    """Create the stock MJLab/BeyondMimic tracking task for Microduck."""

    cfg = make_tracking_env_cfg()
    if morphology_scaled_gravity and gravity_magnitude is not None:
        raise ValueError(
            "morphology_scaled_gravity and gravity_magnitude are mutually exclusive"
        )
    if gravity_magnitude is not None and gravity_magnitude <= 0.0:
        raise ValueError("gravity_magnitude must be positive")
    if morphology_scaled_gravity or gravity_magnitude is not None:
        # UMR itself never changes gravity: it is a geometric retargeter which
        # preserves the source clock.  These non-Earth values belong only to an
        # optional PPO continuation curriculum used to avoid losing an already
        # moving policy during optimization.  They are not UMR output and are
        # never valid for final training, evaluation, or video acceptance.
        magnitude = (
            MICRODUCK_MORPHOLOGY_GRAVITY
            if gravity_magnitude is None
            else gravity_magnitude
        )
        cfg.sim.mujoco.gravity = (0.0, 0.0, -magnitude)
    # Each registered variant must own its actuator config.  The nominal stage
    # fixes BAM ranges below; sharing the module-level EntityCfg would silently
    # leak those values into the robust task registered earlier.
    cfg.scene.entities = {
        "robot": copy.deepcopy(MICRODUCK_STANDUP_ROBOT_CFG)
    }

    # Mirror the stock G1 tracking adapter: self contacts are exposed to the
    # unchanged BeyondMimic self-collision penalty.
    sensors = [
        ContactSensorCfg(
            name="self_collision",
            primary=ContactMatch(
                mode="subtree", pattern="trunk_base", entity="robot"
            ),
            secondary=ContactMatch(
                mode="subtree", pattern="trunk_base", entity="robot"
            ),
            fields=("found", "force"),
            reduce="none",
            num_slots=1,
            history_length=4,
        )
    ]
    if not official_training_lifecycle:
        sensors.append(ContactSensorCfg(
            name="protected_body_ground_contact",
            primary=ContactMatch(
                mode="body",
                pattern=(
                    r"^(trunk_base|yaw2roll|neck|neck_pitch|"
                    r"yaw_roll_motion|jaw_soft|bearing_roll)$"
                ),
                entity="robot",
            ),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found",),
            reduce="none",
            num_slots=1,
        ))
    cfg.scene.sensors = tuple(sensors)

    if residual_actions:
        # Zero action exactly replays the retargeted full-body reference.  PPO
        # controls a bounded-scale correction, so the learned policy remains
        # responsible for balance, contact timing and angular momentum.
        cfg.actions["joint_pos"] = ReferenceResidualJointPositionActionCfg(
            entity_name="robot",
            actuator_names=MICRODUCK_ACTION_JOINTS,
            preserve_order=True,
            scale=MICRODUCK_RESIDUAL_ACTION_SCALE,
            offset=0.0,
        )
    else:
        action = cfg.actions["joint_pos"]
        assert isinstance(action, JointPositionActionCfg)
        # Match BeyondMimic's robot-specific action adapter with Microduck's
        # BAM convention: target = RL zero + 1.0 * action.
        action.actuator_names = MICRODUCK_ACTION_JOINTS
        action.preserve_order = True
        action.scale = MICRODUCK_ACTION_SCALE

    stock_motion = cfg.commands["motion"]
    assert isinstance(stock_motion, MotionCommandCfg)
    if official_training_lifecycle:
        # Match g1_spinkick_example: stock MotionCommand, adaptive RSI, and a
        # 10-second episode.  The command resamples when it reaches the end of
        # the reference.  Finite hold behavior belongs to play/deployment, not
        # to the official training lifecycle.
        motion = stock_motion
    else:
        motion = HoldLastMotionCommandCfg(
            **{
                field.name: copy.deepcopy(getattr(stock_motion, field.name))
                for field in fields(stock_motion)
            }
        )
        cfg.commands["motion"] = motion
    if not 0.0 <= adaptive_start_probability <= 1.0:
        raise ValueError("adaptive_start_probability must be in [0, 1]")
    if isinstance(motion, HoldLastMotionCommandCfg):
        motion.adaptive_start_probability = adaptive_start_probability
    motion.anchor_body_name = "trunk_base"
    motion.body_names = MICRODUCK_TRACKED_BODIES

    # Preserve the dimensionless reset-noise contract of the G1 task.
    for axis in ("x", "y", "z"):
        low, high = motion.pose_range[axis]
        motion.pose_range[axis] = (
            low * MICRODUCK_LINEAR_TRACKING_SCALE,
            high * MICRODUCK_LINEAR_TRACKING_SCALE,
        )
        low, high = motion.velocity_range[axis]
        motion.velocity_range[axis] = (
            low * MICRODUCK_LINEAR_TRACKING_SCALE,
            high * MICRODUCK_LINEAR_TRACKING_SCALE,
        )

    # The functions and weights are unchanged stock BeyondMimic.  Only metre
    # and metre/second kernel widths are converted to Microduck's scale.
    for name in ("motion_global_root_pos", "motion_body_pos"):
        cfg.rewards[name].params["std"] *= MICRODUCK_LINEAR_TRACKING_SCALE
    cfg.rewards["motion_body_lin_vel"].params[
        "std"
    ] *= MICRODUCK_LINEAR_TRACKING_SCALE
    if dense_tracking_guidance:
        # A generic continuation for dynamic motions whose first successful
        # policy has learned a stable but visibly reduced-amplitude version of
        # the reference.  These terms remain averages over every tracked body
        # and every actuator; there are no phase, limb, axis, or keyframe
        # rewards.  The tighter, still non-saturating scales give PPO a usable
        # gradient toward higher-fidelity motion rather than merely rewarding
        # the first broadly stable traversal.
        if high_fidelity_guidance:
            angular_velocity_reward_weight = 3.0
            angular_velocity_reward_scale = 6.0
            full_body_reward_weight = 4.0
            full_body_scales = {
                "position_scale": 0.09,
                "orientation_scale": 0.75,
                "joint_scale": 0.45,
                "linear_velocity_scale": 0.75,
                "angular_velocity_scale": 8.0,
            }
        else:
            angular_velocity_reward_weight = 2.0
            angular_velocity_reward_scale = 8.0
            full_body_reward_weight = 2.0
            full_body_scales = {
                "position_scale": 0.15,
                "orientation_scale": 1.5,
                "joint_scale": 0.75,
                "linear_velocity_scale": 1.0,
                "angular_velocity_scale": 12.0,
            }
        cfg.rewards["motion_anchor_ang_vel_dense"] = RewardTermCfg(
            func=motion_anchor_angular_velocity_tracking_dense,
            weight=angular_velocity_reward_weight,
            params={
                "command_name": "motion",
                "angular_velocity_scale": angular_velocity_reward_scale,
            },
        )
        cfg.rewards["motion_full_body_tracking_dense"] = RewardTermCfg(
            func=motion_full_body_tracking_dense,
            weight=full_body_reward_weight,
            params={
                "command_name": "motion",
                **full_body_scales,
            },
        )

    cfg.events["foot_friction"].params[
        "asset_cfg"
    ].geom_names = r"^(left|right)_foot_collision$"
    cfg.events["base_com"].params["asset_cfg"].body_names = ("trunk_base",)

    # BAM writes per-world friction/damping fields.  Registering this no-op is
    # required by the maintained Microduck actuator implementation and does not
    # change BeyondMimic's training algorithm.
    cfg.events["expand_bam_friction_fields"] = EventTermCfg(
        func=microduck_mdp.expand_bam_friction_fields,
        mode="startup",
    )

    cfg.terminations["ee_body_pos"].params["body_names"] = (
        "ankle_left",
        "ankle_right",
        "yaw_roll_motion",
    )
    if tracking_termination_distance is not None:
        if tracking_termination_distance <= 0.0:
            raise ValueError("tracking_termination_distance must be positive")
        termination_distance = tracking_termination_distance
    else:
        termination_distance = (
            MICRODUCK_STRICT_TRACKING_TERMINATION_DISTANCE
            if strict_tracking
            else MICRODUCK_TRACKING_TERMINATION_DISTANCE
        )
    for name in ("anchor_pos", "ee_body_pos"):
        cfg.terminations[name].params[
            "threshold"
        ] = termination_distance
    cfg.terminations["base_ang_vel_exceed"] = TerminationTermCfg(
        func=base_ang_vel_exceed,
        params={"threshold": MICRODUCK_SPIN_MAX_ANG_VEL},
    )
    if official_training_lifecycle:
        # The reference G1 task has only its z-position, orientation,
        # end-effector and angular-velocity guards.  Keep this baseline clean
        # so its result is a valid test of stock BeyondMimic rather than of a
        # motion-specific curriculum.
        pass
    elif terminate_protected_contact:
        cfg.terminations["protected_body_ground_contact"] = TerminationTermCfg(
            func=protected_body_ground_contact,
            params={"sensor_name": "protected_body_ground_contact"},
        )
    else:
        cfg.rewards["protected_body_ground_contact"] = RewardTermCfg(
            func=protected_body_ground_contact_penalty,
            weight=-2.0,
            params={"sensor_name": "protected_body_ground_contact"},
        )
    if not official_training_lifecycle:
        cfg.terminations["motion_complete"] = TerminationTermCfg(
            func=motion_complete,
            params={"command_name": "motion"},
            time_out=True,
        )

    cfg.viewer.body_name = "trunk_base"
    cfg.viewer.distance = 0.72
    cfg.viewer.elevation = -8.0

    # Match the G1 example's 4.65 s processed reference: a 2.65 s repeated
    # action at 1.0x, 0.5 s transitions on both sides, and a 1.0 s standing
    # hold.  The motion-complete term ends the episode at the reference tail.
    if not official_training_lifecycle:
        cfg.episode_length_s = 5.0

    if not actor_tracking_feedback:
        # Match g1_spinkick_example's no-state-estimation variant.  The critic
        # keeps privileged state, while the actor does not receive global root
        # position or base linear velocity.
        actor_terms = {
            name: term
            for name, term in cfg.observations["actor"].terms.items()
            if name not in {"motion_anchor_pos_b", "base_lin_vel"}
        }
        cfg.observations["actor"] = ObservationGroupCfg(
            terms=actor_terms,
            concatenate_terms=True,
            enable_corruption=True,
        )

    if nominal_training:
        # Stage 1 curriculum: learn the dynamically corrected action under one
        # reproducible compound-physics setting before widening to the full
        # sim-to-real distribution.  The same BAM motor/friction model remains
        # active; only stochastic parameter/reset perturbations are removed.
        cfg.events = {
            "expand_bam_friction_fields": cfg.events[
                "expand_bam_friction_fields"
            ]
        }
        robot_cfg = cfg.scene.entities["robot"]
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
        motion.pose_range = {}
        motion.velocity_range = {}
        motion.joint_position_range = (0.0, 0.0)

    if start_training:
        # Focused curriculum for the deployment-critical rollout from frame 0.
        # Adaptive RSI can otherwise spend nearly all samples on a later bin
        # while the policy still fails at the first spin entry.
        motion.sampling_mode = "start"

    if play:
        cfg.episode_length_s = int(1e9)
        # Playback/deployment must remain at the RL-zero final command forever;
        # only training episodes use motion completion as a successful timeout.
        cfg.terminations.pop("motion_complete")
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        motion.pose_range = {}
        motion.velocity_range = {}
        motion.joint_position_range = (0.0, 0.0)
        motion.sampling_mode = "start"

    return cfg


def microduck_spinkick_mimic_runner_cfg(
    residual_actions: bool = False,
    nominal_training: bool = False,
):
    """Return the unchanged stock G1 BeyondMimic PPO configuration."""

    from mjlab.tasks.tracking.config.g1.rl_cfg import (
        unitree_g1_tracking_ppo_runner_cfg,
    )

    cfg = unitree_g1_tracking_ppo_runner_cfg()
    if residual_actions and nominal_training:
        cfg.experiment_name = "microduck_spinkick_residual_nominal_mimic"
    elif residual_actions:
        cfg.experiment_name = "microduck_spinkick_residual_mimic"
    else:
        cfg.experiment_name = "microduck_spinkick_mimic"
    if residual_actions:
        # The useful zero-residual controller already executes the kinematic
        # reference.  Start exploration at 0.05 rad physical target noise
        # instead of destroying that initialization with a 0.25 rad spread.
        cfg.actor.distribution_cfg["init_std"] = 0.2
    cfg.max_iterations = 20_000
    return cfg
