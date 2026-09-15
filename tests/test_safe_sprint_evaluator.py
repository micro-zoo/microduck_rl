"""Static guardrails for the fixed-policy safe-sprint qualification script."""

from scripts.evaluate_safe_sprint import (
    DEFAULT_CONTROL_STEPS_PER_SEGMENT,
    DEFAULT_WARMUP_STEPS,
    MIN_QUALIFICATION_TRIALS,
    PUSH_DIRECTIONS_M_S,
    SPEED_PROFILE_M_S,
    TASK_ID,
)


def test_qualification_profile_accelerates_then_brakes_with_bounded_pushes():
    assert SPEED_PROFILE_M_S == (0.25, 0.45, 0.65, 0.85, 0.65, 0.45, 0.25)
    assert set(PUSH_DIRECTIONS_M_S) == {
        (0.05, 0.0),
        (-0.05, 0.0),
        (0.0, 0.05),
        (0.0, -0.05),
    }
    assert DEFAULT_CONTROL_STEPS_PER_SEGMENT == 250
    assert DEFAULT_WARMUP_STEPS == 50
    assert MIN_QUALIFICATION_TRIALS == 8
    assert TASK_ID == "Mjlab-SafeSprint-Flat-MicroDuck"
