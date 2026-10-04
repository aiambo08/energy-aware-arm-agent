"""Pinhole model and camera extrinsics (numpy only).

Frames: the *sensor* frame follows Gazebo (x forward, y left, z up) and is placed with the SDF
``xyz``/``rpy``; the *optical* frame (x right, y down, z forward) is what the images and
``sensor_msgs/CameraInfo`` refer to. Points are ``(N, 3)`` arrays, pixels ``(N, 2)`` ``(u, v)``.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from armbench.kinematics.se3 import transform
from armbench.perception.params import CameraSpec

# Columns: optical axes in the sensor frame (x_opt = -y_s, y_opt = -z_s, z_opt = x_s).
OPTICAL_IN_SENSOR: Final = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


class Intrinsics(BaseModel):
    """Pinhole intrinsics without distortion (the simulated camera is ``plumb_bob`` with zeros)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fx: float = Field(gt=0)
    fy: float = Field(gt=0)
    cx: float
    cy: float

    @classmethod
    def from_fov(cls, width: int, height: int, horizontal_fov_rad: float) -> Intrinsics:
        """Gazebo convention: ``fx = fy = (width / 2) / tan(hfov / 2)``, centred principal point."""
        f = (width / 2.0) / float(np.tan(horizontal_fov_rad / 2.0))
        return cls(width=width, height=height, fx=f, fy=f, cx=width / 2.0, cy=height / 2.0)

    @classmethod
    def from_k(cls, k: tuple[float, ...], width: int, height: int) -> Intrinsics:
        """From the row-major 3x3 ``K`` of ``sensor_msgs/CameraInfo``."""
        if len(k) != 9:
            msg = f"K must have 9 entries, got {len(k)}"
            raise ValueError(msg)
        return cls(width=width, height=height, fx=k[0], fy=k[4], cx=k[2], cy=k[5])

    def project(self, points_optical: np.ndarray) -> np.ndarray:
        """Optical-frame points ``(N, 3)`` with ``z > 0`` to pixels ``(N, 2)``."""
        p = np.asarray(points_optical, dtype=float).reshape(-1, 3)
        if np.any(p[:, 2] <= 0):
            msg = "cannot project points behind the camera"
            raise ValueError(msg)
        u = self.fx * p[:, 0] / p[:, 2] + self.cx
        v = self.fy * p[:, 1] / p[:, 2] + self.cy
        return np.stack([u, v], axis=1)

    def backproject(self, pixels: np.ndarray, depth_m: np.ndarray) -> np.ndarray:
        """Pixels ``(N, 2)`` and z-depth ``(N,)`` to optical-frame points ``(N, 3)``."""
        px = np.asarray(pixels, dtype=float).reshape(-1, 2)
        z = np.asarray(depth_m, dtype=float).reshape(-1)
        if px.shape[0] != z.shape[0]:
            msg = f"{px.shape[0]} pixels but {z.shape[0]} depths"
            raise ValueError(msg)
        x = (px[:, 0] - self.cx) / self.fx * z
        y = (px[:, 1] - self.cy) / self.fy * z
        return np.stack([x, y, z], axis=1)

    def in_image(self, pixels: np.ndarray, margin_px: float = 0.0) -> np.ndarray:
        px = np.asarray(pixels, dtype=float).reshape(-1, 2)
        ok: np.ndarray = (
            (px[:, 0] >= margin_px)
            & (px[:, 0] <= self.width - 1 - margin_px)
            & (px[:, 1] >= margin_px)
            & (px[:, 1] <= self.height - 1 - margin_px)
        )
        return ok


class Extrinsics:
    """Rigid transform between the optical frame and the reference (base_link) frame."""

    def __init__(self, xyz: tuple[float, float, float], rpy: tuple[float, float, float]) -> None:
        sensor_in_base = transform(xyz, rpy)
        optical_in_sensor = np.eye(4)
        optical_in_sensor[:3, :3] = OPTICAL_IN_SENSOR
        self.base_from_optical: np.ndarray = sensor_in_base @ optical_in_sensor
        self.optical_from_base: np.ndarray = np.linalg.inv(self.base_from_optical)

    @classmethod
    def from_spec(cls, spec: CameraSpec) -> Extrinsics:
        return cls(spec.xyz, spec.rpy)

    @staticmethod
    def _apply(t: np.ndarray, points: np.ndarray) -> np.ndarray:
        p = np.asarray(points, dtype=float).reshape(-1, 3)
        out: np.ndarray = p @ t[:3, :3].T + t[:3, 3]
        return out

    def to_base(self, points_optical: np.ndarray) -> np.ndarray:
        return self._apply(self.base_from_optical, points_optical)

    def to_optical(self, points_base: np.ndarray) -> np.ndarray:
        return self._apply(self.optical_from_base, points_base)


class Camera:
    """Intrinsics + extrinsics: base_link points <-> pixels with metric depth."""

    def __init__(self, intrinsics: Intrinsics, extrinsics: Extrinsics) -> None:
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics

    @classmethod
    def from_spec(cls, spec: CameraSpec, k: tuple[float, ...] | None = None) -> Camera:
        """Camera from the config; ``k`` (``CameraInfo.K``) overrides the FOV-derived intrinsics."""
        intr = (
            Intrinsics.from_k(k, spec.width, spec.height)
            if k is not None
            else Intrinsics.from_fov(spec.width, spec.height, spec.horizontal_fov_rad)
        )
        return cls(intr, Extrinsics.from_spec(spec))

    def project_base(self, points_base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Base-frame points to ``(pixels (N, 2), depth (N,))``."""
        opt = self.extrinsics.to_optical(points_base)
        return self.intrinsics.project(opt), opt[:, 2]

    def pixel_to_base(self, pixels: np.ndarray, depth_m: np.ndarray) -> np.ndarray:
        return self.extrinsics.to_base(self.intrinsics.backproject(pixels, depth_m))
