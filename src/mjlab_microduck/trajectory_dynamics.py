"""Hard position/velocity/acceleration/jerk contract for motion references.

UMR's smoothing objectives are soft costs.  This module projects each scalar
joint onto the nearest trajectory that satisfies Microduck's declared temporal
limits, then audits the exact samples consumed by MJLab.
"""

from __future__ import annotations

from dataclasses import dataclass

import clarabel
import numpy as np
from scipy import sparse


@dataclass(frozen=True, slots=True)
class TrajectoryDynamicsAudit:
    velocity_peak_rad_s: float
    acceleration_peak_rad_s2: float
    jerk_peak_rad_s3: float
    maximum_velocity_ratio: float
    maximum_acceleration_ratio: float
    maximum_jerk_ratio: float
    velocity_violation_count: int
    acceleration_violation_count: int
    jerk_violation_count: int

    @property
    def accepted(self) -> bool:
        return not (
            self.velocity_violation_count
            or self.acceleration_violation_count
            or self.jerk_violation_count
        )

    def as_dict(self) -> dict[str, float | int | bool]:
        return {
            "accepted": self.accepted,
            "velocity_peak_rad_s": self.velocity_peak_rad_s,
            "acceleration_peak_rad_s2": self.acceleration_peak_rad_s2,
            "jerk_peak_rad_s3": self.jerk_peak_rad_s3,
            "maximum_velocity_ratio": self.maximum_velocity_ratio,
            "maximum_acceleration_ratio": self.maximum_acceleration_ratio,
            "maximum_jerk_ratio": self.maximum_jerk_ratio,
            "velocity_violation_count": self.velocity_violation_count,
            "acceleration_violation_count": self.acceleration_violation_count,
            "jerk_violation_count": self.jerk_violation_count,
        }


def _limits(values: np.ndarray, joint_count: int, label: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if (
        result.shape != (joint_count,)
        or not np.isfinite(result).all()
        or np.any(result <= 0.0)
    ):
        raise ValueError(f"{label} must contain one finite positive value per joint")
    return result


def _difference(values: np.ndarray, order: int, fps: float) -> np.ndarray:
    if len(values) <= order:
        return np.zeros((0, values.shape[1]), dtype=np.float64)
    return np.diff(values, n=order, axis=0) * fps**order


def audit_joint_trajectory_dynamics(
    joint_positions: np.ndarray,
    *,
    fps: float,
    velocity_limits: np.ndarray,
    acceleration_limits: np.ndarray,
    jerk_limits: np.ndarray,
    tolerance: float = 1.0e-6,
) -> TrajectoryDynamicsAudit:
    values = np.asarray(joint_positions, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("joint_positions must be a finite [frames, joints] array")
    if not np.isfinite(fps) or fps <= 0.0 or tolerance < 0.0:
        raise ValueError("fps must be positive and tolerance non-negative")
    joint_count = values.shape[1]
    limits = (
        _limits(velocity_limits, joint_count, "velocity_limits"),
        _limits(acceleration_limits, joint_count, "acceleration_limits"),
        _limits(jerk_limits, joint_count, "jerk_limits"),
    )
    derivatives = tuple(_difference(values, order, fps) for order in (1, 2, 3))

    def peak(values: np.ndarray) -> float:
        return float(np.max(np.abs(values), initial=0.0))

    def ratio(values: np.ndarray, limit: np.ndarray) -> float:
        return (
            float(np.max(np.abs(values) / limit[None, :])) if values.size else 0.0
        )

    def violation_count(values: np.ndarray, limit: np.ndarray) -> int:
        return int(
            np.count_nonzero(np.abs(values) > limit[None, :] * (1.0 + tolerance))
        )

    return TrajectoryDynamicsAudit(
        velocity_peak_rad_s=peak(derivatives[0]),
        acceleration_peak_rad_s2=peak(derivatives[1]),
        jerk_peak_rad_s3=peak(derivatives[2]),
        maximum_velocity_ratio=ratio(derivatives[0], limits[0]),
        maximum_acceleration_ratio=ratio(derivatives[1], limits[1]),
        maximum_jerk_ratio=ratio(derivatives[2], limits[2]),
        velocity_violation_count=violation_count(derivatives[0], limits[0]),
        acceleration_violation_count=violation_count(derivatives[1], limits[1]),
        jerk_violation_count=violation_count(derivatives[2], limits[2]),
    )


def _difference_matrix(frame_count: int, order: int) -> sparse.csc_matrix:
    coefficients = {
        1: (-1.0, 1.0),
        2: (1.0, -2.0, 1.0),
        3: (-1.0, 3.0, -3.0, 1.0),
    }[order]
    rows = max(frame_count - order, 0)
    return sparse.diags(
        coefficients,
        offsets=range(order + 1),
        shape=(rows, frame_count),
        format="csc",
    )


def project_joint_trajectory_dynamics(
    joint_positions: np.ndarray,
    *,
    fps: float,
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
    velocity_limits: np.ndarray,
    acceleration_limits: np.ndarray,
    jerk_limits: np.ndarray,
    anchor_start_frames: int = 1,
    anchor_end_frames: int = 1,
) -> tuple[np.ndarray, dict[str, object]]:
    """Project to the nearest feasible trajectory using independent convex QPs."""

    source = np.asarray(joint_positions, dtype=np.float64)
    if source.ndim != 2 or not np.isfinite(source).all():
        raise ValueError("joint_positions must be a finite [frames, joints] array")
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError("fps must be finite and positive")
    if anchor_start_frames < 0 or anchor_end_frames < 0:
        raise ValueError("trajectory anchor counts must be non-negative")
    frame_count, joint_count = source.shape
    lower = np.asarray(lower_limits, dtype=np.float64)
    upper = np.asarray(upper_limits, dtype=np.float64)
    if (
        lower.shape != (joint_count,)
        or upper.shape != (joint_count,)
        or not np.isfinite(lower).all()
        or not np.isfinite(upper).all()
        or np.any(lower >= upper)
    ):
        raise ValueError("joint position limits must be finite with lower < upper")
    velocity = _limits(velocity_limits, joint_count, "velocity_limits")
    acceleration = _limits(acceleration_limits, joint_count, "acceleration_limits")
    jerk = _limits(jerk_limits, joint_count, "jerk_limits")
    source_audit = audit_joint_trajectory_dynamics(
        source,
        fps=fps,
        velocity_limits=velocity,
        acceleration_limits=acceleration,
        jerk_limits=jerk,
    )
    position_tolerance = 1.0e-9
    source_position_violation_count = int(
        np.count_nonzero(
            (source < lower[None, :] - position_tolerance)
            | (source > upper[None, :] + position_tolerance)
        )
    )
    if source_audit.accepted and source_position_violation_count == 0:
        return source.copy(), {
            "algorithm": "clarabel-nearest-trajectory-hard-v1",
            "projection_required": False,
            "frames_preserved": frame_count,
            "fps_preserved": float(fps),
            "changed_values": 0,
            "joint_position_rmse_rad": 0.0,
            "maximum_joint_correction_rad": 0.0,
            "source_position_violation_count": 0,
            "position_violation_count": 0,
            "audit": source_audit.as_dict(),
        }
    if frame_count < 2:
        raise ValueError("a single-frame trajectory violates its position limits")

    differences = tuple(_difference_matrix(frame_count, order) for order in (1, 2, 3))
    identity = sparse.eye(frame_count, format="csc", dtype=np.float64)
    anchor_indices = sorted(
        set(range(min(anchor_start_frames, frame_count)))
        | set(range(max(frame_count - anchor_end_frames, 0), frame_count))
    )
    anchor_matrix = identity[anchor_indices]
    inequality_matrix = sparse.vstack(
        [
            identity,
            -identity,
            differences[0],
            -differences[0],
            differences[1],
            -differences[1],
            differences[2],
            -differences[2],
        ],
        format="csc",
    )
    constraint_matrix = sparse.vstack(
        [anchor_matrix, inequality_matrix], format="csc"
    )
    cones: list[object] = [
        clarabel.ZeroConeT(len(anchor_indices)),
        clarabel.NonnegativeConeT(inequality_matrix.shape[0]),
    ]
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_iter = 300
    settings.tol_gap_abs = 1.0e-10
    settings.tol_gap_rel = 1.0e-10
    settings.tol_feas = 1.0e-10
    settings.presolve_enable = True
    result = np.empty_like(source)

    for joint in range(joint_count):
        bounds = [
            np.full(frame_count, upper[joint]),
            np.full(frame_count, -lower[joint]),
        ]
        for matrix, limit in zip(
            differences,
            (
                velocity[joint] / fps,
                acceleration[joint] / fps**2,
                jerk[joint] / fps**3,
            ),
            strict=True,
        ):
            bounds.extend(
                [np.full(matrix.shape[0], limit), np.full(matrix.shape[0], limit)]
            )
        right_hand_side = np.concatenate(
            [source[anchor_indices, joint], *bounds]
        )
        solution = clarabel.DefaultSolver(
            identity,
            -source[:, joint],
            constraint_matrix,
            right_hand_side,
            cones,
            settings,
        ).solve()
        if solution.status != clarabel.SolverStatus.Solved:
            raise RuntimeError(
                f"trajectory dynamics projection failed for joint {joint}: "
                f"{solution.status}"
            )
        result[:, joint] = np.asarray(solution.x, dtype=np.float64)

    audit = audit_joint_trajectory_dynamics(
        result,
        fps=fps,
        velocity_limits=velocity,
        acceleration_limits=acceleration,
        jerk_limits=jerk,
        tolerance=2.0e-6,
    )
    if not audit.accepted:
        raise RuntimeError(f"trajectory projection returned violations: {audit.as_dict()}")
    correction = result - source
    position_violation_count = int(
        np.count_nonzero(
            (result < lower[None, :] - position_tolerance)
            | (result > upper[None, :] + position_tolerance)
        )
    )
    if position_violation_count:
        raise RuntimeError("trajectory projection did not satisfy position limits")
    return result, {
        "algorithm": "clarabel-nearest-trajectory-hard-v1",
        "projection_required": True,
        "frames_preserved": frame_count,
        "fps_preserved": float(fps),
        "anchor_start_frames": anchor_start_frames,
        "anchor_end_frames": anchor_end_frames,
        "changed_values": int(np.count_nonzero(np.abs(correction) > 1.0e-9)),
        "joint_position_rmse_rad": float(np.sqrt(np.mean(correction**2))),
        "maximum_joint_correction_rad": float(
            np.max(np.abs(correction), initial=0.0)
        ),
        "source_position_violation_count": source_position_violation_count,
        "position_violation_count": position_violation_count,
        "audit": audit.as_dict(),
    }


__all__ = [
    "TrajectoryDynamicsAudit",
    "audit_joint_trajectory_dynamics",
    "project_joint_trajectory_dynamics",
]
