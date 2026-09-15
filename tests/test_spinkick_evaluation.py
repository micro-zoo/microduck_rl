from mjlab_microduck.spinkick_evaluation import evaluate_acceptance


def _passing_metrics() -> dict[str, float]:
    return {
        "success_rate": 1.0,
        "action_r_mpkpe_m": 0.03,
        "action_joint_pos_mae_rad": 0.12,
        "action_body_ori_mae_rad": 0.25,
        "root_yaw_error_rad": 0.30,
        "spin_direction_matches_reference": True,
        "spin_coverage_ratio": 0.90,
        "kick_amplitude_coverage_ratio": 0.80,
        "jump_height_coverage_ratio": 0.75,
        "final_joint_pos_mae_rad": 0.08,
        "final_planar_speed_m_s": 0.05,
        "hold_planar_drift_m": 0.02,
    }


def test_acceptance_requires_every_full_body_and_standing_gate():
    gates = evaluate_acceptance(_passing_metrics())

    assert gates
    assert all(gates.values())


def test_acceptance_rejects_stationary_non_spin_policy():
    metrics = _passing_metrics()
    metrics["spin_coverage_ratio"] = 0.05

    gates = evaluate_acceptance(metrics)

    assert gates["spin_semantic_coverage"] is False
    assert not all(gates.values())


def test_acceptance_rejects_wrong_spin_direction():
    metrics = _passing_metrics()
    metrics["spin_direction_matches_reference"] = False

    gates = evaluate_acceptance(metrics)

    assert gates["spin_direction"] is False
    assert not all(gates.values())


def test_acceptance_rejects_foot_shuffle_without_kick_amplitude():
    metrics = _passing_metrics()
    metrics["kick_amplitude_coverage_ratio"] = 0.20

    gates = evaluate_acceptance(metrics)

    assert gates["kick_semantic_coverage"] is False
    assert not all(gates.values())


def test_acceptance_rejects_grounded_motion_that_removes_the_jump():
    metrics = _passing_metrics()
    metrics["jump_height_coverage_ratio"] = 0.10

    gates = evaluate_acceptance(metrics)

    assert gates["jump_semantic_coverage"] is False
    assert not all(gates.values())
