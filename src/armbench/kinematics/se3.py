"""Minimal SE(3) helpers (numpy only) shared by the kinematics code."""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def rot_x(angle: float) -> np.ndarray:
    """Rotation matrix about the x axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(angle: float) -> np.ndarray:
    """Rotation matrix about the y axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(angle: float) -> np.ndarray:
    """Rotation matrix about the z axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF roll-pitch-yaw to rotation matrix: ``R_z(yaw) R_y(pitch) R_x(roll)``."""
    rot: np.ndarray = rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)
    return rot


def transform(xyz: tuple[float, float, float], rpy: tuple[float, float, float]) -> np.ndarray:
    """Homogeneous transform ``Trans(xyz) * Rot_rpy(rpy)`` (URDF ``<origin>`` convention)."""
    t = np.eye(4)
    t[:3, :3] = rpy_to_matrix(*rpy)
    t[:3, 3] = xyz
    return t


def rot_z_4x4(angle: float) -> np.ndarray:
    """Homogeneous pure rotation about the local z axis (revolute joint motion)."""
    t = np.eye(4)
    t[:3, :3] = rot_z(angle)
    return t


def _angle_and_vee(rot: np.ndarray) -> tuple[float, np.ndarray]:
    vee = np.array([rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]])
    sin_angle = float(np.linalg.norm(vee)) / 2.0
    cos_angle = (float(np.trace(rot)) - 1.0) / 2.0
    return float(np.arctan2(sin_angle, cos_angle)), vee


def rotation_vector(rot: np.ndarray) -> np.ndarray:
    """Logarithmic map SO(3) -> R^3 (axis times angle, angle in [0, pi]).

    The angle comes from ``atan2(|vee| / 2, (trace - 1) / 2)``, which is accurate for small
    angles; the near-pi case (where ``vee`` vanishes) takes the axis from ``R + I``.
    """
    angle, vee = _angle_and_vee(rot)
    if angle < 1e-12:
        return 0.5 * vee
    if np.pi - angle < 1e-6:
        sym = rot + np.eye(3)
        column = sym[:, int(np.argmax(np.diag(sym)))]
        axis = column / max(float(np.linalg.norm(column)), _EPS)
        result: np.ndarray = angle * axis
        return result
    scaled: np.ndarray = (angle / (2.0 * np.sin(angle))) * vee
    return scaled


def rotation_angle(rot: np.ndarray) -> float:
    """Rotation angle in [0, pi] of a rotation matrix."""
    return _angle_and_vee(rot)[0]


def is_rotation_matrix(rot: np.ndarray, *, atol: float = 1e-9) -> bool:
    """True if ``rot`` is orthonormal with determinant +1 (within ``atol``)."""
    if rot.shape != (3, 3):
        return False
    orthonormal = bool(np.allclose(rot.T @ rot, np.eye(3), atol=atol))
    return orthonormal and bool(abs(float(np.linalg.det(rot)) - 1.0) < atol)
