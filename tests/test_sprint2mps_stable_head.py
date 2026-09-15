"""Regression checks for the 2 m/s sprint and its camera-stability contract."""

from types import SimpleNamespace

import torch

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_sprint2mps_stable_head_env_cfg import (
    SPRINT_MAX_SPEED_M_S,
    SPRINT_SPEED_STAGES,
    make_microduck_sprint2mps_stable_head_env_cfg,
)
from mjlab_microduck.tasks.mdp import (
    SlewedForwardVelocityCommandCfg,
    slew_forward_velocity_command,
)


class _Asset:
    def __init__(self, site_quat_w, root_link_quat_w=None):
        if root_link_quat_w is None:
            root_link_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        self.data = SimpleNamespace(
            site_quat_w=torch.as_tensor(site_quat_w, dtype=torch.float32),
            root_link_quat_w=torch.as_tensor(root_link_quat_w, dtype=torch.float32),
        )


class _Env:
    def __init__(self, site_quat_w, root_link_quat_w=None):
        self.device = "cpu"
        self.num_envs = 1
        self.step_dt = 0.02
        self.episode_length_buf = torch.tensor([1])
        self._asset = _Asset(site_quat_w, root_link_quat_w)
        self.scene = self

    def __getitem__(self, _name):
        return self._asset


_HEAD_CAMERA = SimpleNamespace(name="robot", site_ids=[0])


def test_level_forward_camera_scores_one_and_tilt_or_roll_costs_reward():
    identity = [[[1.0, 0.0, 0.0, 0.0]]]
    env = _Env(identity)
    level = microduck_mdp.head_camera_level_forward_reward(env, asset_cfg=_HEAD_CAMERA)
    assert torch.allclose(level, torch.ones(1), atol=1e-6)

    # 90 degrees around world Y tips the optical +X axis vertical.
    tipped = [[[2**-0.5, 0.0, 2**-0.5, 0.0]]]
    assert microduck_mdp.head_camera_level_forward_reward(
        _Env(tipped), asset_cfg=_HEAD_CAMERA
    ).item() < 1e-4

    # 90 degrees around world X rolls the camera-up axis sideways.
    rolled = [[[2**-0.5, 2**-0.5, 0.0, 0.0]]]
    assert microduck_mdp.head_camera_level_forward_reward(
        _Env(rolled), asset_cfg=_HEAD_CAMERA
    ).item() < 1e-4


def test_camera_rate_penalty_measures_world_view_motion_not_joint_motion():
    env = _Env([[[1.0, 0.0, 0.0, 0.0]]])
    assert microduck_mdp.head_camera_world_angular_rate_l2(
        env, asset_cfg=_HEAD_CAMERA
    ).item() == 0.0

    env.episode_length_buf[:] = 2
    env._asset.data.site_quat_w[:] = torch.tensor(
        [[[0.9950042, 0.0, 0.0998334, 0.0]]]
    )
    assert microduck_mdp.head_camera_world_angular_rate_l2(
        env, asset_cfg=_HEAD_CAMERA
    ).item() > 1.0


def test_sprint_cfg_is_forward_only_and_reaches_two_meters_per_second():
    cfg = make_microduck_sprint2mps_stable_head_env_cfg()
    cmd = cfg.commands["twist"]
    assert cmd.rel_standing_envs == 0.0
    assert cmd.rel_forward_envs == 1.0
    assert cmd.rel_turn_in_place_envs == 0.0
    assert cmd.heading_command is False
    assert cmd.ranges.heading is None
    assert cmd.ranges.lin_vel_y == (0.0, 0.0)
    assert cmd.ranges.ang_vel_z == (0.0, 0.0)
    assert SPRINT_SPEED_STAGES[-1]["min_speed"] == 1.55
    assert SPRINT_SPEED_STAGES[-1]["max_speed"] == SPRINT_MAX_SPEED_M_S == 2.0
    assert "sprint_forward_speed" in cfg.curriculum
    assert "standing_envs" not in cfg.curriculum


def test_live_sprint_curriculum_reaches_the_final_high_speed_band():
    ranges = SimpleNamespace(
        lin_vel_x=(0.0, 0.0), lin_vel_y=(1.0, 1.0), ang_vel_z=(1.0, 1.0)
    )
    command = SimpleNamespace(cfg=SimpleNamespace(ranges=ranges))
    env = SimpleNamespace(
        common_step_counter=SPRINT_SPEED_STAGES[-1]["step"] + 1,
        command_manager=SimpleNamespace(get_term=lambda name: command if name == "twist" else None),
    )
    cap = microduck_mdp.forward_speed_command_ranges_curriculum(
        env, torch.empty(0, dtype=torch.long), "twist", list(SPRINT_SPEED_STAGES)
    )
    assert ranges.lin_vel_x == (1.55, 2.0)
    assert ranges.lin_vel_y == (0.0, 0.0)
    assert ranges.ang_vel_z == (0.0, 0.0)
    assert cap.item() == 2.0


def test_sprint_cfg_uses_camera_world_terms_and_zero_command_slots():
    cfg = make_microduck_sprint2mps_stable_head_env_cfg()
    assert "head_camera_level_forward" in cfg.rewards
    assert "head_camera_world_rate_l2" in cfg.rewards
    for old_term in ("head_pose_tracking", "head_pose_bias", "body_pose_tracking"):
        assert old_term not in cfg.rewards
    for old_command in ("head_pose", "body_pose"):
        assert old_command not in cfg.commands

    for group in ("actor", "critic"):
        head = cfg.observations[group].terms["head_command"]
        body = cfg.observations[group].terms["body_command"]
        assert head.func is microduck_mdp.zero_command_padding
        assert head.params["dim"] == 4
        assert body.func is microduck_mdp.zero_command_padding
        assert body.params["dim"] == 6


def test_sprint_command_is_rate_bounded_and_uses_a_push_curriculum():
    cfg = make_microduck_sprint2mps_stable_head_env_cfg()
    command = cfg.commands["twist"]
    assert isinstance(command, SlewedForwardVelocityCommandCfg)
    assert command.resampling_time_range == (1.5, 3.5)
    assert command.init_velocity_prob == 0.0
    assert command.forward_acceleration_m_s2 == 0.50
    assert command.forward_deceleration_m_s2 == 0.75
    assert command.slowdown_probability == 0.35
    assert command.slowdown_speed_range == (0.25, 0.45)
    pushes = cfg.curriculum["push_magnitude"].params["push_stages"]
    assert pushes[0]["velocity_range"]["x"] == (0.0, 0.0)
    assert pushes[-1]["velocity_range"]["x"] == (-0.30, 0.30)
    assert pushes[-1]["step"] == 17500 * 24


def test_forward_command_slew_bounds_acceleration_and_braking():
    current = torch.tensor([0.0, 2.0])
    target = torch.tensor([2.0, 0.0])
    slewed = slew_forward_velocity_command(
        current, target, dt=0.02, acceleration_m_s2=0.50, deceleration_m_s2=0.75
    )
    assert torch.allclose(slewed, torch.tensor([0.010, 1.985]), atol=1e-6)
