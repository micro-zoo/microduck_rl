"""DuckEMW record-running recipe with a horizon-stabilised head camera.

The underlying gait, speed curriculum, BAM actuator model, domain
randomisation, and 61D actor contract are unchanged.  From random
initialisation, the fixed-gaze variant gradually introduces two world-frame
objectives alongside the speed curriculum:

* keep the camera optical axis level and its up axis vertical;
* suppress camera-frame angular rate while allowing compensating neck motion.

The head/body command slots remain in their original observation positions but
are zero padded, because this policy owns a fixed forward gaze.
"""

from copy import deepcopy

from mjlab.managers import CurriculumTermCfg, ObservationTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_running_env_cfg import (
    RUNNING_CURRICULUM_DIVISOR,
    MicroduckRunningRlCfg,
    make_microduck_running_env_cfg,
)
from mjlab_microduck.tasks.microduck_velocity_env_cfg import NUM_STEPS_PER_ENV

RUNNING_STABLE_HEAD_START_ITERATION = 0
RUNNING_HEAD_LEVEL_FINAL_WEIGHT = 1.0
RUNNING_HEAD_RATE_FINAL_WEIGHT = -0.012
RUNNING_HEAD_COARSE_FINAL_WEIGHT = 0.25
RUNNING_NECK_ACTION_RATE_FINAL_WEIGHT = -0.06
# Keep PPO's mini-batch size constant when the synchronous rollout is enlarged.
# At 4096 envs (divisor 8), 32 mini-batches contain the same ~3072 samples as
# 4 mini-batches at 512 envs.  This preserves both samples and optimizer updates
# per curriculum stage while still exploiting parallel simulation throughput.
RUNNING_NUM_MINI_BATCHES = 4 * RUNNING_CURRICULUM_DIVISOR

_HEAD_CAMERA_CFG = SceneEntityCfg("robot", site_names=("head_camera",))


def _training_step(iteration: int) -> int:
    return round(iteration / RUNNING_CURRICULUM_DIVISOR) * NUM_STEPS_PER_ENV


def make_microduck_running_stable_head_env_cfg(play: bool = False):
    """Build the max-speed task with a fixed, world-stabilised camera view."""
    cfg = make_microduck_running_env_cfg(play=play)

    # A fixed-gaze policy must not receive random operator pose commands.  Keep
    # their 4D + 6D slots exactly where they were in the 61D contract.
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

    # Split leg and neck regularisation.  Applying the full late leg weight to
    # the neck made slowing the whole gait cheaper than counter-motion; removing
    # neck smoothing entirely produced measured 4.6x higher head-action jitter
    # than leg-action jitter.  A lighter neck term permits smooth compensation
    # while suppressing the high-frequency policy chatter seen in real rollouts.
    cfg.rewards["action_rate_l2"].func = microduck_mdp.leg_action_rate_l2
    cfg.rewards["neck_action_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.neck_action_rate_l2,
        weight=RUNNING_NECK_ACTION_RATE_FINAL_WEIGHT if play else -0.01,
    )

    cfg.rewards["head_camera_level_forward"] = RewardTermCfg(
        func=microduck_mdp.head_camera_level_forward_reward,
        weight=RUNNING_HEAD_LEVEL_FINAL_WEIGHT if play else 0.0,
        params={
            "asset_cfg": _HEAD_CAMERA_CFG,
            "forward_std": 0.35,
            "up_std": 0.35,
        },
    )
    cfg.rewards["head_camera_world_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.head_camera_world_angular_rate_l2,
        weight=RUNNING_HEAD_RATE_FINAL_WEIGHT if play else 0.0,
        params={"asset_cfg": _HEAD_CAMERA_CFG},
    )
    # A 0.35-rad Gaussian is effectively zero for a camera more than 90 degrees
    # from level, so keep a broad acquisition basin throughout training while
    # the fine term progressively tightens the final alignment.
    cfg.rewards["head_camera_level_coarse"] = RewardTermCfg(
        func=microduck_mdp.head_camera_level_forward_reward,
        weight=RUNNING_HEAD_COARSE_FINAL_WEIGHT,
        params={
            "asset_cfg": _HEAD_CAMERA_CFG,
            "forward_std": 1.5,
            "up_std": 1.5,
        },
    )

    if not play:
        # The reference Running recipe progressively tightens action rate for a
        # clean gait.  At the highest commands, relax only the leg term again so
        # required sprint cadence is not taxed while head smoothness remains.
        cfg.curriculum["action_rate_weight"].params["weight_stages"] = [
            {"step": _training_step(0), "weight": -0.02},
            {"step": _training_step(2_500), "weight": -0.05},
            {"step": _training_step(6_000), "weight": -0.06},
            {"step": _training_step(9_000), "weight": -0.05},
            {"step": _training_step(12_000), "weight": -0.04},
        ]
        cfg.curriculum["neck_action_rate_weight"] = CurriculumTermCfg(
            func=microduck_mdp.reward_weight,
            params={
                "reward_name": "neck_action_rate_l2",
                "weight_stages": [
                    {"step": _training_step(0), "weight": -0.01},
                    {"step": _training_step(3_000), "weight": -0.02},
                    {"step": _training_step(6_000), "weight": -0.03},
                    {"step": _training_step(9_000), "weight": -0.05},
                    {
                        "step": _training_step(12_000),
                        "weight": RUNNING_NECK_ACTION_RATE_FINAL_WEIGHT,
                    },
                ],
            },
        )
        cfg.curriculum["head_camera_level_weight"] = CurriculumTermCfg(
            func=microduck_mdp.reward_weight,
            params={
                "reward_name": "head_camera_level_forward",
                "weight_stages": [
                    {"step": _training_step(0), "weight": 0.15},
                    {"step": _training_step(3_000), "weight": 0.25},
                    {"step": _training_step(7_000), "weight": 0.50},
                    {"step": _training_step(10_000), "weight": 0.75},
                    {
                        "step": _training_step(12_000),
                        "weight": RUNNING_HEAD_LEVEL_FINAL_WEIGHT,
                    },
                ],
            },
        )
        cfg.curriculum["head_camera_world_rate_weight"] = CurriculumTermCfg(
            func=microduck_mdp.reward_weight,
            params={
                "reward_name": "head_camera_world_rate_l2",
                "weight_stages": [
                    {"step": _training_step(0), "weight": -0.001},
                    {"step": _training_step(3_000), "weight": -0.002},
                    {"step": _training_step(6_000), "weight": -0.004},
                    {"step": _training_step(9_000), "weight": -0.008},
                    {
                        "step": _training_step(12_000),
                        "weight": RUNNING_HEAD_RATE_FINAL_WEIGHT,
                    },
                ],
            },
        )

    return cfg


MicroduckRunningStableHeadRlCfg = deepcopy(MicroduckRunningRlCfg)
MicroduckRunningStableHeadRlCfg.experiment_name = "running"
MicroduckRunningStableHeadRlCfg.run_name = "running-max-speed-stable-head-scratch"
MicroduckRunningStableHeadRlCfg.max_iterations = 13_500
# The running base uses 0.02 for broad gait exploration.  With a fixed-gaze
# objective that left a large train/deploy gap: stochastic actions supplied
# much of the measured training progress while the deterministic actor slowed
# down.  The velocity-family default still explores, but better aligns the
# learned mean action with deployment.
MicroduckRunningStableHeadRlCfg.algorithm.entropy_coef = 0.01
MicroduckRunningStableHeadRlCfg.algorithm.num_mini_batches = RUNNING_NUM_MINI_BATCHES
