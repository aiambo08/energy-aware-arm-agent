"""Synthetic top-down RGB-D renders of cube scenes (table plane + cube faces).

Used by the perception unit tests and by the kinematic (ROS-free) robot backend so that the
whole observe -> detect -> move -> grasp loop can run on the host. Colours are the mean RGB
Gazebo Harmonic renders for each palette entry (headless, sun without shadows).
"""

from __future__ import annotations

import math
from typing import Final

import cv2
import numpy as np

from armbench.perception.camera import Camera
from armbench.scene import Cube

RENDERED_RGB: Final[dict[str, tuple[int, int, int]]] = {
    "red": (224, 83, 83),
    "green": (80, 197, 97),
    "blue": (76, 118, 206),
    "yellow": (209, 204, 75),
}
TABLE_RGB: Final = (232, 232, 229)
ARM_RGB: Final = (65, 65, 65)
TABLE_Z: Final = 0.0
SUBPIXEL_SHIFT: Final = 4  # fillConvexPoly fixed-point bits: corners placed at 1/16 px
SIDE_SHADE: Final = 0.6


def cube_corners_top(cube: Cube) -> np.ndarray:
    """Top-face corners (4, 3) in ``base_link``."""
    h = cube.size / 2.0
    c, s = math.cos(cube.yaw), math.sin(cube.yaw)
    local = np.array([[-h, -h], [h, -h], [h, h], [-h, h]])
    xy = local @ np.array([[c, s], [-s, c]]) + np.array([cube.x, cube.y])
    return np.column_stack([xy, np.full(4, cube.z + h)])


def _fill(
    rgb: np.ndarray,
    depth: np.ndarray,
    camera: Camera,
    quad: np.ndarray,
    color: tuple[int, int, int],
) -> None:
    px, dz = camera.project_base(quad)
    poly = np.round(px * 2**SUBPIXEL_SHIFT).astype(np.int32)
    cv2.fillConvexPoly(rgb, poly, color, shift=SUBPIXEL_SHIFT)
    cv2.fillConvexPoly(depth, poly, float(dz.mean()), shift=SUBPIXEL_SHIFT)


def render(
    cubes: list[Cube],
    camera: Camera,
    *,
    noise_sigma: float = 0.0,
    seed: int = 0,
    side_faces: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """RGB ``(H, W, 3) uint8`` and depth ``(H, W) float32`` metres as the Gazebo camera would.

    Cubes are drawn in order, so a cube listed later (e.g. one lifted by the gripper) occludes
    earlier ones; callers that need correct occlusion sort by distance to the camera.
    """
    intr = camera.intrinsics
    h, w = intr.height, intr.width
    rgb = np.empty((h, w, 3), dtype=np.uint8)
    rgb[:] = TABLE_RGB
    _, d_table = camera.project_base(np.array([[0.0, 0.0, TABLE_Z]]))
    depth = np.full((h, w), float(d_table[0]), dtype=np.float32)
    rng = np.random.default_rng(seed)
    for cube in cubes:
        top = cube_corners_top(cube)
        color = RENDERED_RGB[cube.color]
        if side_faces:
            bottom = top - np.array([0.0, 0.0, cube.size])
            shade = (
                int(color[0] * SIDE_SHADE),
                int(color[1] * SIDE_SHADE),
                int(color[2] * SIDE_SHADE),
            )
            for i in range(4):
                quad = np.array([top[i], top[(i + 1) % 4], bottom[(i + 1) % 4], bottom[i]])
                _fill(rgb, depth, camera, quad, shade)
        _fill(rgb, depth, camera, top, color)
    if noise_sigma > 0:
        noisy = rgb.astype(float) + rng.normal(0.0, noise_sigma, rgb.shape)
        rgb = np.clip(noisy, 0, 255).astype(np.uint8)
    return rgb, depth
