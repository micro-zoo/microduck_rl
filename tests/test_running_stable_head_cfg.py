"""Regression checks for head-stabilised DuckEMW Running from scratch."""

from itertools import pairwise

from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_running_env_cfg import (
    make_microduck_running_env_cfg,
)
from mjlab_microduck.tasks.microduck_running_stable_head_env_cfg import (
    RUNNING_HEAD_COARSE_FINAL_WEIGHT,
    RUNNING_HEAD_LEVEL_FINAL_WEIGHT,
    RUNNING_HEAD_RATE_FINAL_WEIGHT,
    RUNNING_NECK_ACTION_RATE_FINAL_WEIGHT,
    RUNNING_NUM_MINI_BATCHES,
    RUNNING_STABLE_HEAD_START_ITERATION,
    MicroduckRunningStableHeadRlCfg,
    make_microduck_running_stable_head_env_cfg,
)


def test_stable_head_variant_preserves_running_objective_and_zero_pads_pose_slots():
    base = make_microduck_running_env_cfg()
    cfg = make_microduck_running_stable_head_env_cfg()
    assert (
        cfg.rewards["forward_progress"].weight
        == base.rewards["forward_progress"].weight
    )
    assert cfg.rewards["heading_hold"].weight == base.rewards["heading_hold"].weight
    for name in ("head_pose", "body_pose"):
        assert name not in cfg.commands
    for group in ("actor", "critic"):
        head = cfg.observations[group].terms["head_command"]
        body = cfg.observations[group].terms["body_command"]
        assert head.func is mdp.zero_command_padding
        assert head.params["dim"] == 4
        assert body.func is mdp.zero_command_padding
        assert body.params["dim"] == 6


def test_stable_head_uses_world_camera_objectives_with_gentle_curricula():
    cfg = make_microduck_running_stable_head_env_cfg()
    level = cfg.rewards["head_camera_level_forward"]
    rate = cfg.rewards["head_camera_world_rate_l2"]
    coarse = cfg.rewards["head_camera_level_coarse"]
    neck_rate = cfg.rewards["neck_action_rate_l2"]
    assert level.func is mdp.head_camera_level_forward_reward
    assert rate.func is mdp.head_camera_world_angular_rate_l2
    assert coarse.func is mdp.head_camera_level_forward_reward
    assert neck_rate.func is mdp.neck_action_rate_l2
    assert cfg.rewards["action_rate_l2"].func is mdp.leg_action_rate_l2
    assert level.weight == 0.0
    assert rate.weight == 0.0
    assert coarse.weight == RUNNING_HEAD_COARSE_FINAL_WEIGHT
    assert neck_rate.weight == -0.01

    level_stages = cfg.curriculum["head_camera_level_weight"].params["weight_stages"]
    rate_stages = cfg.curriculum["head_camera_world_rate_weight"].params[
        "weight_stages"
    ]
    neck_rate_stages = cfg.curriculum["neck_action_rate_weight"].params[
        "weight_stages"
    ]
    assert level_stages[0]["step"] == RUNNING_STABLE_HEAD_START_ITERATION * 24
    assert rate_stages[0]["step"] == RUNNING_STABLE_HEAD_START_ITERATION * 24
    assert level_stages[-1]["weight"] == RUNNING_HEAD_LEVEL_FINAL_WEIGHT
    assert rate_stages[-1]["weight"] == RUNNING_HEAD_RATE_FINAL_WEIGHT
    assert neck_rate_stages[-1]["weight"] == RUNNING_NECK_ACTION_RATE_FINAL_WEIGHT
    assert all(a["weight"] < b["weight"] for a, b in pairwise(level_stages))
    assert all(a["weight"] > b["weight"] for a, b in pairwise(rate_stages))
    assert all(a["weight"] > b["weight"] for a, b in pairwise(neck_rate_stages))
    assert level_stages[-1]["step"] == 12_000 * 24
    assert rate_stages[-1]["step"] == 12_000 * 24
    assert neck_rate_stages[-1]["step"] == 12_000 * 24


def test_play_cfg_applies_final_head_constraint_and_keeps_61d_layout():
    cfg = make_microduck_running_stable_head_env_cfg(play=True)
    assert (
        cfg.rewards["head_camera_level_forward"].weight
        == RUNNING_HEAD_LEVEL_FINAL_WEIGHT
    )
    assert (
        cfg.rewards["head_camera_world_rate_l2"].weight
        == RUNNING_HEAD_RATE_FINAL_WEIGHT
    )
    assert (
        cfg.rewards["head_camera_level_coarse"].weight
        == RUNNING_HEAD_COARSE_FINAL_WEIGHT
    )
    assert (
        cfg.rewards["neck_action_rate_l2"].weight
        == RUNNING_NECK_ACTION_RATE_FINAL_WEIGHT
    )
    assert "head_camera_level_weight" not in cfg.curriculum
    assert "head_camera_world_rate_weight" not in cfg.curriculum
    assert "neck_action_rate_weight" not in cfg.curriculum
    assert list(cfg.observations["actor"].terms) == [
        "base_ang_vel",
        "projected_gravity",
        "joint_pos",
        "joint_vel",
        "actions",
        "command",
        "head_command",
        "body_command",
    ]


def test_stable_head_runner_is_full_from_scratch_recipe():
    assert MicroduckRunningStableHeadRlCfg.experiment_name == "running"
    assert (
        MicroduckRunningStableHeadRlCfg.run_name
        == "running-max-speed-stable-head-scratch"
    )
    assert MicroduckRunningStableHeadRlCfg.max_iterations == 13_500
    assert MicroduckRunningStableHeadRlCfg.algorithm.entropy_coef == 0.01
    assert RUNNING_NUM_MINI_BATCHES == 4
    assert MicroduckRunningStableHeadRlCfg.algorithm.num_mini_batches == 4
