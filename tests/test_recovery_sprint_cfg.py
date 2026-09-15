"""Regression checks for v7's separated speed and recovery curriculum."""

from mjlab.tasks.registry import list_tasks

from mjlab_microduck.tasks.microduck_recovery_sprint_env_cfg import (
    RECOVERY_SPRINT_MAX_SPEED_M_S,
    RECOVERY_SPRINT_PUSH_STAGES,
    RECOVERY_SPRINT_SPEED_STAGES,
    make_microduck_recovery_sprint_env_cfg,
)


def test_recovery_sprint_keeps_final_push_test_after_high_speed_gait_training():
    cfg = make_microduck_recovery_sprint_env_cfg()
    command = cfg.commands["twist"]

    assert RECOVERY_SPRINT_SPEED_STAGES[-1]["max_speed"] == RECOVERY_SPRINT_MAX_SPEED_M_S == 0.90
    assert command.forward_acceleration_m_s2 == 0.38
    assert command.forward_deceleration_m_s2 == 0.55
    assert cfg.rewards["track_linear_velocity"].weight == 5.5
    assert cfg.rewards["track_linear_velocity"].params["std"] == 0.65
    assert cfg.rewards["upright"].weight == 3.5
    assert cfg.curriculum["action_rate_weight"].params["weight_stages"][-1]["weight"] == -0.045
    assert RECOVERY_SPRINT_PUSH_STAGES[-1]["velocity_range"]["x"] == (-0.05, 0.05)
    assert RECOVERY_SPRINT_PUSH_STAGES[-1]["step"] > RECOVERY_SPRINT_SPEED_STAGES[-1]["step"]


def test_recovery_sprint_is_registered_as_an_independent_task():
    import mjlab_microduck.tasks  # noqa: F401  (registration side effect)

    assert "Mjlab-RecoverySprint-Flat-MicroDuck" in list_tasks()
