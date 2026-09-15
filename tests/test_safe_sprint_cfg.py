"""Regression checks for the safety-first MicroDuck sprint task."""

from mjlab.tasks.registry import list_tasks

from mjlab_microduck.tasks.microduck_safe_sprint_env_cfg import (
    SAFE_PUSH_STAGES,
    SAFE_SPRINT_MAX_SPEED_M_S,
    SAFE_SPRINT_SPEED_STAGES,
    make_microduck_safe_sprint_env_cfg,
)


def test_safe_sprint_has_a_conservative_speed_and_push_ceiling():
    cfg = make_microduck_safe_sprint_env_cfg()
    command = cfg.commands["twist"]

    assert SAFE_SPRINT_SPEED_STAGES[-1]["max_speed"] == SAFE_SPRINT_MAX_SPEED_M_S == 0.85
    assert command.forward_acceleration_m_s2 == 0.35
    assert command.forward_deceleration_m_s2 == 0.50
    assert command.slowdown_probability == 0.45
    assert command.slowdown_speed_range == (0.25, 0.40)
    assert cfg.rewards["upright"].weight == 3.5
    assert cfg.rewards["track_linear_velocity"].weight == 3.0
    assert cfg.events["push_robot"].interval_range_s == (3.0, 5.0)
    assert SAFE_PUSH_STAGES[-1]["velocity_range"]["x"] == (-0.05, 0.05)


def test_safe_sprint_is_registered_as_an_independent_task():
    import mjlab_microduck.tasks  # noqa: F401  (registration side effect)

    assert "Mjlab-SafeSprint-Flat-MicroDuck" in list_tasks()
