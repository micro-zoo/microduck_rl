import numpy as np

from mjlab_microduck.trajectory_dynamics import project_joint_trajectory_dynamics


def test_position_violation_is_projected_even_when_dynamics_are_smooth():
    source = np.full((8, 1), 1.2)

    projected, diagnostics = project_joint_trajectory_dynamics(
        source,
        fps=50.0,
        lower_limits=np.array([-1.0]),
        upper_limits=np.array([1.0]),
        velocity_limits=np.array([10.0]),
        acceleration_limits=np.array([200.0]),
        jerk_limits=np.array([6000.0]),
        anchor_start_frames=0,
        anchor_end_frames=0,
    )

    assert diagnostics["projection_required"] is True
    assert diagnostics["source_position_violation_count"] == 8
    assert diagnostics["position_violation_count"] == 0
    assert np.all(projected <= 1.0 + 1.0e-9)
