"""Convert a Gem/UMR Microduck RobotMotion to MJLab's tracking NPZ.

The preprocessing contract follows ``mujocolab/g1_spinkick_example``:

* repeat the original motion to the requested action duration;
* prepend a safe standing-to-motion transition;
* append a motion-to-standing transition and a standing hold;
* sample the resulting reference at MJLab's 50 Hz control rate.

The action itself is always played at 1.0x.  ``transition_duration`` is a
blend duration, not a time-scaling factor for the source action.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import mjlab  # noqa: F401  # Load registered task packages before robot constants.
import mujoco
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, RotationSpline, Slerp

from mjlab_microduck.robot.microduck_constants import (
    HOME_FRAME,
    MICRODUCK_ALLCOLLISIONS_XML,
    MICRODUCK_STANDUP_ROBOT_CFG,
)
from mjlab_microduck.trajectory_dynamics import (
    audit_joint_trajectory_dynamics,
    project_joint_trajectory_dynamics,
)

MJLAB_FPS = 50.0
# The historical 0.115 m root height places the lowest vertices of both sole
# collision meshes 2.18 mm below a z=0 plane in the RL-zero pose.  The deepest
# penetration over the unmodified UMR spin-kick is 2.48 mm.  Apply one
# robot-specific world-Z calibration to the whole motion instead of grounding
# individual frames; this removes the mesh/floor offset while preserving every
# take-off, landing and root-velocity difference exactly.
MICRODUCK_REFERENCE_Z_OFFSET = 0.0025
RL_ZERO_ROOT_HEIGHT = 0.115 + MICRODUCK_REFERENCE_Z_OFFSET


@dataclass(frozen=True)
class RobotMotion:
    fps: float
    joint_names: tuple[str, ...]
    root_position: np.ndarray
    root_rotation_xyzw: np.ndarray
    joint_positions: np.ndarray
    diagnostics: dict[str, object]

    @property
    def frame_count(self) -> int:
        return int(self.root_position.shape[0])


def _load_robot_motion(path: Path) -> RobotMotion:
    # Accept the official UMR result directly.  This keeps the complete
    # MimicKit -> UMR -> MJLab conversion inside this repository instead of
    # requiring a Gem-specific intermediate conversion script.
    if path.is_file() and path.suffix == ".npz":
        with np.load(path, allow_pickle=True) as data:
            if "qpos" in data and "robot_joint_names" in data:
                qpos = np.asarray(data["qpos"], dtype=np.float64)
                fps = float(np.asarray(data["fps"]).reshape(-1)[0])
                joint_names = tuple(
                    str(name) for name in data["robot_joint_names"].tolist()
                )
                if str(np.asarray(data["robot_name"]).item()) != "microduck":
                    raise ValueError("the UMR result must target microduck")
                expected_shape = (qpos.shape[0], 7 + len(joint_names))
                if qpos.ndim != 2 or qpos.shape != expected_shape:
                    raise ValueError("UMR qpos has an invalid shape")
                source_data = str(np.asarray(data["source_data"]).item())
                diagnostics: dict[str, object] = {
                    "retargeter": "official UMR character surface correspondence",
                    "source_data": source_data,
                    "source_action_speed": 1.0,
                    "source_frames": int(qpos.shape[0]),
                    "source_fps": fps,
                    "target_zero": "microduck_rl_zero",
                    "umr_result": str(path.resolve()),
                }
                if "root_orientation_cost" in data:
                    diagnostics["root_orientation_cost"] = float(
                        np.asarray(data["root_orientation_cost"]).reshape(-1)[0]
                    )
                if "root_orientation_source_body" in data:
                    diagnostics["root_orientation_source_body"] = str(
                        np.asarray(data["root_orientation_source_body"]).item()
                    )
                return RobotMotion(
                    fps=fps,
                    joint_names=joint_names,
                    root_position=qpos[:, :3],
                    root_rotation_xyzw=qpos[:, [4, 5, 6, 3]],
                    joint_positions=qpos[:, 7:],
                    diagnostics=diagnostics,
                )

    directory = path if path.is_dir() else path.parent
    manifest_path = directory / "motion.json"
    arrays_path = directory / "motion.npz"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "gem.robot-motion":
        raise ValueError(f"unsupported RobotMotion schema in {manifest_path}")
    if manifest.get("robot_id") != "microduck":
        raise ValueError("the input RobotMotion must target microduck")
    fps = float(manifest["fps"])
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("RobotMotion fps must be finite and positive")

    with np.load(arrays_path, allow_pickle=False) as data:
        root_position = np.asarray(data["root_position"], dtype=np.float64)
        root_rotation = np.asarray(data["root_rotation_xyzw"], dtype=np.float64)
        joint_positions = np.asarray(data["joint_positions"], dtype=np.float64)
    joint_names = tuple(str(name) for name in manifest["joint_names"])
    frames = root_position.shape[0]
    if (
        root_position.shape != (frames, 3)
        or root_rotation.shape != (frames, 4)
        or joint_positions.shape != (frames, len(joint_names))
        or frames < 2
    ):
        raise ValueError("RobotMotion arrays have invalid shapes")
    if not all(
        np.isfinite(value).all()
        for value in (root_position, root_rotation, joint_positions)
    ):
        raise ValueError("RobotMotion contains NaN or infinity")
    root_rotation /= np.linalg.norm(root_rotation, axis=1, keepdims=True)
    return RobotMotion(
        fps=fps,
        joint_names=joint_names,
        root_position=root_position,
        root_rotation_xyzw=root_rotation,
        joint_positions=joint_positions,
        diagnostics=dict(manifest.get("diagnostics", {})),
    )


def _model_joint_names(model: mujoco.MjModel) -> tuple[str, ...]:
    return tuple(
        str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id))
        for joint_id in range(model.njnt)
        if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE)
    )


def _rl_zero(joint_names: tuple[str, ...]) -> np.ndarray:
    values = []
    for name in joint_names:
        matches = [
            float(value)
            for pattern, value in HOME_FRAME.joint_pos.items()
            if re.fullmatch(pattern, name)
        ]
        if len(matches) != 1:
            raise ValueError(f"RL zero does not resolve exactly once for joint {name!r}")
        values.append(matches[0])
    return np.asarray(values, dtype=np.float64)


def _yaw_upright(quaternion_xyzw: np.ndarray) -> np.ndarray:
    yaw = Rotation.from_quat(quaternion_xyzw).as_euler("ZYX")[0]
    return Rotation.from_euler("Z", yaw).as_quat()


def _repeat_action(motion: RobotMotion, duration: float | None) -> RobotMotion:
    if duration is None:
        return motion
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("duration must be finite and positive")
    target_frames = max(round(duration * motion.fps), 2)
    if target_frames <= motion.frame_count:
        cycles = 1
    else:
        cycles = math.ceil(target_frames / motion.frame_count)
    displacement = motion.root_position[-1] - motion.root_position[0]
    displacement[2] = 0.0
    positions = []
    rotations = []
    joints = []
    for cycle in range(cycles):
        cycle_positions = motion.root_position.copy()
        cycle_positions += cycle * displacement
        positions.append(cycle_positions)
        rotations.append(motion.root_rotation_xyzw)
        joints.append(motion.joint_positions)
    return RobotMotion(
        fps=motion.fps,
        joint_names=motion.joint_names,
        root_position=np.concatenate(positions, axis=0)[:target_frames],
        root_rotation_xyzw=np.concatenate(rotations, axis=0)[:target_frames],
        joint_positions=np.concatenate(joints, axis=0)[:target_frames],
        diagnostics=dict(motion.diagnostics),
    )


def _apply_robot_floor_calibration(motion: RobotMotion) -> RobotMotion:
    """Apply Microduck's constant sole-to-root height calibration.

    This is deliberately a rigid translation, not per-frame contact fitting.
    Consequently it cannot flatten an aerial trajectory or alter velocities,
    accelerations, timing, joint poses, or root rotations.
    """

    root_position = motion.root_position.copy()
    root_position[:, 2] += MICRODUCK_REFERENCE_Z_OFFSET
    diagnostics = dict(motion.diagnostics)
    diagnostics["microduck_reference_z_offset_m"] = (
        MICRODUCK_REFERENCE_Z_OFFSET
    )
    return RobotMotion(
        fps=motion.fps,
        joint_names=motion.joint_names,
        root_position=root_position,
        root_rotation_xyzw=motion.root_rotation_xyzw,
        joint_positions=motion.joint_positions,
        diagnostics=diagnostics,
    )


def _blend(
    start: np.ndarray,
    end: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    return (1.0 - alpha[:, None]) * start + alpha[:, None] * end


def _add_safe_transitions(
    motion: RobotMotion,
    *,
    transition_duration: float,
    hold_duration: float,
) -> tuple[RobotMotion, dict[str, int]]:
    if transition_duration <= 0.0 or hold_duration < 0.0:
        raise ValueError("transition duration must be positive and hold non-negative")
    transition_frames = max(round(transition_duration * motion.fps), 2)
    hold_frames = max(round(hold_duration * motion.fps), 0)
    zero = _rl_zero(motion.joint_names)

    # Match the reference implementation's cubic ease-in/ease-out convention.
    phase = np.linspace(0.0, 1.0, transition_frames, dtype=np.float64)
    entry_alpha = phase**3
    exit_alpha = 1.0 - (1.0 - phase) ** 3

    entry_pos = motion.root_position[0].copy()
    entry_pos[2] = RL_ZERO_ROOT_HEIGHT
    entry_quat = _yaw_upright(motion.root_rotation_xyzw[0])
    exit_pos = motion.root_position[-1].copy()
    exit_pos[2] = RL_ZERO_ROOT_HEIGHT
    exit_quat = _yaw_upright(motion.root_rotation_xyzw[-1])

    start_positions = _blend(entry_pos, motion.root_position[0], entry_alpha)
    start_joints = _blend(zero, motion.joint_positions[0], entry_alpha)
    start_rotations = Slerp(
        (0.0, 1.0), Rotation.from_quat((entry_quat, motion.root_rotation_xyzw[0]))
    )(entry_alpha).as_quat()

    end_positions = _blend(motion.root_position[-1], exit_pos, exit_alpha)
    end_joints = _blend(motion.joint_positions[-1], zero, exit_alpha)
    end_rotations = Slerp(
        (0.0, 1.0), Rotation.from_quat((motion.root_rotation_xyzw[-1], exit_quat))
    )(exit_alpha).as_quat()

    positions = np.concatenate(
        (
            start_positions,
            motion.root_position,
            end_positions,
            np.repeat(exit_pos[None, :], hold_frames, axis=0),
        ),
        axis=0,
    )
    rotations = np.concatenate(
        (
            start_rotations,
            motion.root_rotation_xyzw,
            end_rotations,
            np.repeat(exit_quat[None, :], hold_frames, axis=0),
        ),
        axis=0,
    )
    joints = np.concatenate(
        (
            start_joints,
            motion.joint_positions,
            end_joints,
            np.repeat(zero[None, :], hold_frames, axis=0),
        ),
        axis=0,
    )
    action_start = transition_frames
    return (
        RobotMotion(
            fps=motion.fps,
            joint_names=motion.joint_names,
            root_position=positions,
            root_rotation_xyzw=rotations,
            joint_positions=joints,
            diagnostics=dict(motion.diagnostics),
        ),
        {
            "action_start_frame_source_fps": action_start,
            "action_frame_count_source_fps": motion.frame_count,
            "stand_hold_start_frame_source_fps": (
                transition_frames + motion.frame_count + transition_frames
            ),
        },
    )


def _resample(motion: RobotMotion, output_fps: float) -> RobotMotion:
    if output_fps <= 0.0:
        raise ValueError("output_fps must be positive")
    source_time = np.arange(motion.frame_count, dtype=np.float64) / motion.fps
    target_count = round(source_time[-1] * output_fps) + 1
    target_time = np.arange(target_count, dtype=np.float64) / output_fps
    target_time = np.minimum(target_time, source_time[-1])
    root_position = CubicSpline(source_time, motion.root_position, axis=0)(target_time)
    joint_positions = CubicSpline(source_time, motion.joint_positions, axis=0)(target_time)
    root_rotation = RotationSpline(
        source_time, Rotation.from_quat(motion.root_rotation_xyzw)
    )(target_time).as_quat()
    return RobotMotion(
        fps=output_fps,
        joint_names=motion.joint_names,
        root_position=root_position,
        root_rotation_xyzw=root_rotation,
        joint_positions=joint_positions,
        diagnostics=dict(motion.diagnostics),
    )


def _angular_velocity(
    before_wxyz: np.ndarray,
    after_wxyz: np.ndarray,
    duration: float,
) -> np.ndarray:
    conjugate = before_wxyz.copy()
    conjugate[1:] *= -1.0
    relative = np.empty(4, dtype=np.float64)
    mujoco.mju_mulQuat(relative, after_wxyz, conjugate)
    value = np.empty(3, dtype=np.float64)
    mujoco.mju_quat2Vel(value, relative, duration)
    return value


def _finite_kinematics(
    model: mujoco.MjModel,
    motion: RobotMotion,
) -> dict[str, np.ndarray | tuple[str, ...]]:
    model_joint_names = _model_joint_names(model)
    if set(motion.joint_names) != set(model_joint_names):
        missing = sorted(set(model_joint_names) - set(motion.joint_names))
        extra = sorted(set(motion.joint_names) - set(model_joint_names))
        raise ValueError(f"RobotMotion joint mismatch: missing={missing}, extra={extra}")
    source_index = {name: index for index, name in enumerate(motion.joint_names)}
    joint_pos = np.column_stack(
        [motion.joint_positions[:, source_index[name]] for name in model_joint_names]
    )

    qpos = np.repeat(model.qpos0[None, :], motion.frame_count, axis=0)
    qpos[:, :3] = motion.root_position
    qpos[:, 3:7] = motion.root_rotation_xyzw[:, [3, 0, 1, 2]]
    qpos_addresses = np.asarray(
        [
            model.jnt_qposadr[
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            ]
            for name in model_joint_names
        ],
        dtype=np.int32,
    )
    dof_addresses = np.asarray(
        [
            model.jnt_dofadr[
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            ]
            for name in model_joint_names
        ],
        dtype=np.int32,
    )
    qpos[:, qpos_addresses] = joint_pos

    body_ids = np.arange(1, model.nbody, dtype=np.int32)
    body_names = tuple(
        str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(body_id)))
        for body_id in body_ids
    )
    body_pos = np.empty((motion.frame_count, len(body_ids), 3), dtype=np.float64)
    body_quat = np.empty((motion.frame_count, len(body_ids), 4), dtype=np.float64)
    data = mujoco.MjData(model)
    for frame in range(motion.frame_count):
        data.qpos[:] = qpos[frame]
        mujoco.mj_forward(model, data)
        body_pos[frame] = data.xpos[body_ids]
        body_quat[frame] = data.xquat[body_ids]

    dt = 1.0 / motion.fps
    joint_vel = np.empty_like(joint_pos)
    body_lin_vel = np.gradient(body_pos, dt, axis=0, edge_order=2)
    body_ang_vel = np.empty_like(body_pos)
    qvel = np.empty(model.nv, dtype=np.float64)
    for frame in range(motion.frame_count):
        left = max(frame - 1, 0)
        right = min(frame + 1, motion.frame_count - 1)
        duration = (right - left) * dt
        mujoco.mj_differentiatePos(model, qvel, duration, qpos[left], qpos[right])
        joint_vel[frame] = qvel[dof_addresses]
        for body in range(len(body_ids)):
            body_ang_vel[frame, body] = _angular_velocity(
                body_quat[left, body], body_quat[right, body], duration
            )
    return {
        "joint_names": model_joint_names,
        "body_names": body_names,
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "body_pos_w": body_pos,
        "body_quat_w": body_quat,
        "body_lin_vel_w": body_lin_vel,
        "body_ang_vel_w": body_ang_vel,
    }


def _enforce_microduck_dynamics(
    model: mujoco.MjModel,
    motion: RobotMotion,
) -> tuple[RobotMotion, dict[str, object], dict[str, object]]:
    """Project the final 50 Hz samples, including transition boundaries."""

    model_names = _model_joint_names(model)
    if set(model_names) != set(motion.joint_names):
        raise ValueError("RobotMotion joints do not match the Microduck MJCF")
    source_index = {name: index for index, name in enumerate(motion.joint_names)}
    ordered = np.column_stack(
        [motion.joint_positions[:, source_index[name]] for name in model_names]
    )
    hard_lower = []
    hard_upper = []
    for name in model_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        hard_lower.append(float(model.jnt_range[joint_id, 0]))
        hard_upper.append(float(model.jnt_range[joint_id, 1]))
    articulation = MICRODUCK_STANDUP_ROBOT_CFG.articulation
    if articulation is None:
        raise RuntimeError("Microduck mimic robot must be articulated")
    soft_factor = float(articulation.soft_joint_pos_limit_factor)
    hard_lower_array = np.asarray(hard_lower)
    hard_upper_array = np.asarray(hard_upper)
    midpoint = 0.5 * (hard_lower_array + hard_upper_array)
    half_range = 0.5 * (hard_upper_array - hard_lower_array) * soft_factor
    # Keep a tiny float32 serialization margin inside MJLab's reset clamp.
    position_margin = 1.0e-6
    soft_lower = midpoint - half_range
    soft_upper = midpoint + half_range
    lower = soft_lower + position_margin
    upper = soft_upper - position_margin
    joint_count = len(model_names)
    velocity = np.full(joint_count, 10.0, dtype=np.float64)
    acceleration = np.full(joint_count, 200.0, dtype=np.float64)
    jerk = np.full(joint_count, 6000.0, dtype=np.float64)
    raw_audit = audit_joint_trajectory_dynamics(
        ordered,
        fps=motion.fps,
        velocity_limits=velocity,
        acceleration_limits=acceleration,
        jerk_limits=jerk,
    )
    # Leave a small serialization margin so float32 NPZ samples remain inside
    # the public limits after the QP's float64 solution is rounded.
    projection_margin = 0.999
    projected, projection = project_joint_trajectory_dynamics(
        ordered,
        fps=motion.fps,
        lower_limits=lower,
        upper_limits=upper,
        velocity_limits=velocity * projection_margin,
        acceleration_limits=acceleration * projection_margin,
        jerk_limits=jerk * projection_margin,
        anchor_start_frames=1,
        anchor_end_frames=1,
    )
    serialized = projected.astype(np.float32).astype(np.float64)
    final_audit = audit_joint_trajectory_dynamics(
        serialized,
        fps=motion.fps,
        velocity_limits=velocity,
        acceleration_limits=acceleration,
        jerk_limits=jerk,
    )
    final_position_violation_count = int(
        np.count_nonzero(
            (serialized < soft_lower[None, :])
            | (serialized > soft_upper[None, :])
        )
    )
    if not final_audit.accepted or final_position_violation_count:
        raise RuntimeError(
            "final MJLab reference failed the Microduck hard audit: "
            f"dynamics={final_audit.as_dict()}, "
            f"position_violations={final_position_violation_count}"
        )
    return (
        RobotMotion(
            fps=motion.fps,
            joint_names=model_names,
            root_position=motion.root_position,
            root_rotation_xyzw=motion.root_rotation_xyzw,
            joint_positions=serialized,
            diagnostics=dict(motion.diagnostics),
        ),
        raw_audit.as_dict(),
        {
            **projection,
            "soft_joint_position_limit_factor": soft_factor,
            "serialized_position_violation_count": final_position_violation_count,
            "projection_limit_margin": projection_margin,
            "serialized_final_audit": final_audit.as_dict(),
        },
    )


def convert_umr_motion(
    input_path: Path,
    output_path: Path,
    *,
    duration: float | None = None,
    transition_duration: float = 0.5,
    hold_duration: float = 1.0,
    output_fps: float = MJLAB_FPS,
) -> Path:
    """Convert one UMR motion while preserving the source action at 1.0x."""

    source = _apply_robot_floor_calibration(
        _load_robot_motion(Path(input_path))
    )
    model = mujoco.MjModel.from_xml_path(str(MICRODUCK_ALLCOLLISIONS_XML))
    action = _repeat_action(source, duration)
    padded, frame_metadata = _add_safe_transitions(
        action,
        transition_duration=transition_duration,
        hold_duration=hold_duration,
    )
    motion = _resample(padded, output_fps)
    motion, raw_dynamics, dynamics_projection = _enforce_microduck_dynamics(
        model, motion
    )
    kinematics = _finite_kinematics(model, motion)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        joint_pos=np.asarray(kinematics["joint_pos"], dtype=np.float32),
        joint_vel=np.asarray(kinematics["joint_vel"], dtype=np.float32),
        body_pos_w=np.asarray(kinematics["body_pos_w"], dtype=np.float32),
        body_quat_w=np.asarray(kinematics["body_quat_w"], dtype=np.float32),
        body_lin_vel_w=np.asarray(kinematics["body_lin_vel_w"], dtype=np.float32),
        body_ang_vel_w=np.asarray(kinematics["body_ang_vel_w"], dtype=np.float32),
    )
    scale = output_fps / source.fps
    # These are segment *boundaries*.  Use the first output sample at or after
    # each source boundary; round-to-even mislabeled the G1-compatible 3.65 s
    # hold boundary as frame 182 instead of frame 183 at 50 Hz.
    action_start = math.ceil(
        frame_metadata["action_start_frame_source_fps"] * scale - 1.0e-12
    )
    action_end = math.ceil(
        (
            frame_metadata["action_start_frame_source_fps"]
            + frame_metadata["action_frame_count_source_fps"]
        )
        * scale
        - 1.0e-12
    )
    hold_start = math.ceil(
        frame_metadata["stand_hold_start_frame_source_fps"] * scale - 1.0e-12
    )
    original_source = Path(
        str(source.diagnostics.get("source_data", Path(input_path).resolve()))
    )
    source_sha256 = None
    if original_source.is_file():
        source_sha256 = hashlib.sha256(original_source.read_bytes()).hexdigest()
    metadata = {
        "schema": "mjlab-microduck-beyondmimic-motion-v1",
        "source": str(Path(input_path).resolve()),
        "original_source": str(original_source.resolve()),
        "original_source_sha256": source_sha256,
        "reference_pipeline": "mujocolab/g1_spinkick_example",
        "source_action_speed": 1.0,
        "source_fps": source.fps,
        "output_fps": output_fps,
        "output_frames": motion.frame_count,
        "requested_action_duration_seconds": duration,
        "action_duration_seconds": action.frame_count / action.fps,
        "transition_duration_seconds": transition_duration,
        "hold_duration_seconds": hold_duration,
        "joint_names": list(kinematics["joint_names"]),
        "body_names": list(kinematics["body_names"]),
        "action_start_frame": action_start,
        "action_frame_count": action_end - action_start,
        "stand_hold_start_frame": hold_start,
        "rl_zero_root_height": RL_ZERO_ROOT_HEIGHT,
        "microduck_reference_z_offset_m": MICRODUCK_REFERENCE_Z_OFFSET,
        "raw_50hz_dynamics": raw_dynamics,
        "final_50hz_dynamics": dynamics_projection,
        "umr_diagnostics": source.diagnostics,
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Optional repeated action duration; omit for one source Spin at 1.0x.",
    )
    parser.add_argument("--transition-duration", type=float, default=0.5)
    parser.add_argument("--hold-duration", type=float, default=1.0)
    parser.add_argument("--output-fps", type=float, default=MJLAB_FPS)
    args = parser.parse_args()
    output = convert_umr_motion(
        args.input,
        args.output,
        duration=args.duration,
        transition_duration=args.transition_duration,
        hold_duration=args.hold_duration,
        output_fps=args.output_fps,
    )
    print(output)


if __name__ == "__main__":
    main()
