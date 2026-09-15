from pathlib import Path

import numpy as np

from mjlab_microduck.motion_conversion import (
    MICRODUCK_REFERENCE_Z_OFFSET,
    _apply_robot_floor_calibration,
    _load_robot_motion,
)


def test_loads_official_umr_result_without_gem_intermediate(tmp_path: Path):
    source = tmp_path / "humanoid_spinkick.pkl"
    source.write_bytes(b"official-mimickit-motion")
    qpos = np.zeros((3, 9), dtype=np.float32)
    qpos[:, 3] = 1.0
    result = tmp_path / "umr_result.npz"
    np.savez_compressed(
        result,
        qpos=qpos,
        fps=np.asarray([60.0], dtype=np.float32),
        robot_name=np.asarray("microduck"),
        robot_joint_names=np.asarray(["joint_a", "joint_b"], dtype=object),
        source_data=np.asarray(str(source)),
        root_orientation_cost=np.asarray([1000.0], dtype=np.float32),
        root_orientation_source_body=np.asarray("pelvis"),
    )

    motion = _load_robot_motion(result)

    assert motion.fps == 60.0
    assert motion.joint_names == ("joint_a", "joint_b")
    assert motion.root_position.shape == (3, 3)
    np.testing.assert_array_equal(motion.root_rotation_xyzw[0], [0, 0, 0, 1])
    assert motion.diagnostics["source_data"] == str(source)
    assert motion.diagnostics["root_orientation_cost"] == 1000.0
    assert motion.diagnostics["root_orientation_source_body"] == "pelvis"


def test_floor_calibration_is_a_rigid_translation_that_preserves_jump(tmp_path: Path):
    source = tmp_path / "humanoid_spinkick.pkl"
    source.write_bytes(b"official-mimickit-motion")
    qpos = np.zeros((3, 9), dtype=np.float32)
    qpos[:, 2] = [0.10, 0.18, 0.11]
    qpos[:, 3] = 1.0
    result = tmp_path / "umr_result.npz"
    np.savez_compressed(
        result,
        qpos=qpos,
        fps=np.asarray([60.0], dtype=np.float32),
        robot_name=np.asarray("microduck"),
        robot_joint_names=np.asarray(["joint_a", "joint_b"], dtype=object),
        source_data=np.asarray(str(source)),
    )

    original = _load_robot_motion(result)
    calibrated = _apply_robot_floor_calibration(original)

    np.testing.assert_allclose(
        calibrated.root_position[:, 2],
        original.root_position[:, 2] + MICRODUCK_REFERENCE_Z_OFFSET,
    )
    np.testing.assert_allclose(
        np.diff(calibrated.root_position, axis=0),
        np.diff(original.root_position, axis=0),
    )
    assert np.ptp(calibrated.root_position[:, 2]) == np.ptp(
        original.root_position[:, 2]
    )
