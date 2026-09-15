"""Regression checks for the speed-optimised safety sprint task."""

from mjlab.tasks.registry import list_tasks

from mjlab_microduck.tasks.microduck_fast_safe_sprint_env_cfg import (
    FAST_SAFE_SPRINT_MAX_SPEED_M_S,
    FAST_SAFE_SPRINT_SPEED_STAGES,
    make_microduck_fast_safe_sprint_env_cfg,
)


def test_fast_safe_sprint_increases_tracking_without_weakening_safety_limits():
    cfg = make_microduck_fast_safe_sprint_env_cfg()
    command = cfg.commands["twist"]

    assert FAST_SAFE_SPRINT_SPEED_STAGES[-1]["max_speed"] == FAST_SAFE_SPRINT_MAX_SPEED_M_S == 0.95
    assert command.forward_acceleration_m_s2 == 0.40
    assert command.forward_deceleration_m_s2 == 0.55
    assert command.slowdown_probability == 0.30
    assert cfg.rewards["track_linear_velocity"].weight == 7.0
    assert cfg.rewards["track_linear_velocity"].params["std"] == 0.60
    assert cfg.rewards["upright"].weight == 3.5
    assert cfg.events["push_robot"].interval_range_s == (3.0, 5.0)


def test_fast_safe_sprint_is_registered_as_an_independent_task():
    import mjlab_microduck.tasks  # noqa: F401  (registration side effect)

    assert "Mjlab-FastSafeSprint-Flat-MicroDuck" in list_tasks()
