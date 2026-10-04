"""``detect()``: colour cubes from one RGB-D frame to base_link poses (HSV + depth).

Pipeline per colour: HSV hue/saturation/value mask -> morphological opening -> connected
components -> keep the pixels within ``top_face_band_m`` of the component's nearest depth (the
top face; side faces are farther) -> sub-pixel centroid and median depth -> back-projection ->
cube centre ``top - size / 2``. Yaw comes from the top face's minimum-area rectangle, folded into
``[-pi/4, pi/4)`` (a square is symmetric under 90 degree turns).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Final

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from armbench.perception.camera import Camera
from armbench.perception.params import PerceptionParams

ANY_TARGET: Final = "any"
_CENTRE_DEPTH_PERCENTILE: Final = 5.0
_QUARTER_TURN: Final = math.pi / 2.0


class UnknownTargetError(ValueError):
    """``target`` is not one of the configured colours (nor ``"any"``)."""


class Detection(BaseModel):
    """One cube in the reference frame of the camera config (``base_link``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    color: str
    position: tuple[float, float, float] = Field(description="cube centre [m]")
    top_center: tuple[float, float, float] = Field(description="top-face centre [m]")
    yaw_rad: float = Field(ge=-math.pi / 4, lt=math.pi / 4)
    pixel: tuple[float, float] = Field(description="top-face centroid (u, v)")
    bbox: tuple[int, int, int, int] = Field(description="component bbox (x, y, w, h)")
    area_px: int = Field(gt=0, description="top-face pixel count")
    top_face_fraction: float = Field(
        gt=0, description="area_px over the pixel area a full top face has at depth_m"
    )
    complete: bool = Field(
        description="top face fully visible (fraction >= min_top_face_fraction); a partial face "
        "(occluded by the arm or cut by the image border) biases the centroid towards what is seen"
    )
    depth_m: float = Field(gt=0, description="median z-depth of the top face")

    @property
    def xy(self) -> tuple[float, float]:
        return self.position[0], self.position[1]


class DetectResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    detections: tuple[Detection, ...]
    latency_ms: float = Field(ge=0)


def _validate_frame(rgb: np.ndarray, depth: np.ndarray, camera: Camera) -> None:
    h, w = camera.intrinsics.height, camera.intrinsics.width
    if rgb.shape != (h, w, 3) or rgb.dtype != np.uint8:
        msg = f"rgb must be uint8 ({h}, {w}, 3); got {rgb.dtype} {rgb.shape}"
        raise ValueError(msg)
    if depth.shape != (h, w):
        msg = f"depth must be ({h}, {w}); got {depth.shape}"
        raise ValueError(msg)
    if not np.issubdtype(depth.dtype, np.floating):
        msg = f"depth must be float metres; got {depth.dtype}"
        raise ValueError(msg)


def _hue_mask(
    hsv: np.ndarray, ranges: tuple[tuple[int, int], ...], s_min: int, v_min: int
) -> np.ndarray:
    mask: np.ndarray = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in ranges:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, (lo, s_min, v_min), (hi, 255, 255)))
    return mask


def _fold_yaw(yaw: float) -> float:
    folded = (yaw + _QUARTER_TURN / 2.0) % _QUARTER_TURN - _QUARTER_TURN / 2.0
    return _QUARTER_TURN / -2.0 if folded >= _QUARTER_TURN / 2.0 else folded


def yaw_difference(a: float, b: float) -> float:
    """Signed ``a - b`` for square yaws, wrapped into ``[-pi/4, pi/4)`` (90 degree symmetry)."""
    return _fold_yaw(a - b)


def _top_face_yaw(top: np.ndarray, depth_m: float, camera: Camera) -> float:
    """Yaw of the top face in the base frame from its minimum-area rectangle in the image."""
    ys, xs = np.nonzero(top)
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    (cx, cy), _size, angle_deg = cv2.minAreaRect(pts)
    a = math.radians(angle_deg)
    px = np.array([[cx, cy], [cx + math.cos(a), cy + math.sin(a)]])
    p = camera.pixel_to_base(px, np.array([depth_m, depth_m]))
    d = p[1] - p[0]
    return _fold_yaw(math.atan2(d[1], d[0]))


@dataclass(frozen=True)
class _Frame:
    """One validated RGB-D frame plus everything shared by the per-colour passes."""

    hsv: np.ndarray
    depth: np.ndarray
    depth_ok: np.ndarray
    params: PerceptionParams
    camera: Camera
    kernel: np.ndarray

    @classmethod
    def build(
        cls, rgb: np.ndarray, depth: np.ndarray, params: PerceptionParams, camera: Camera
    ) -> _Frame:
        depth_ok = (
            np.isfinite(depth) & (depth >= params.depth.min_m) & (depth <= params.depth.max_m)
        ).astype(np.uint8) * 255
        k = params.segmentation.open_kernel_px
        return cls(
            hsv=cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV),
            depth=depth,
            depth_ok=depth_ok,
            params=params,
            camera=camera,
            kernel=np.ones((k, k), dtype=np.uint8),
        )

    def _detection(self, colour: str, comp: np.ndarray, stats_row: np.ndarray) -> Detection | None:
        d_near = float(np.percentile(self.depth[comp], _CENTRE_DEPTH_PERCENTILE))
        top = comp & (self.depth <= d_near + self.params.depth.top_face_band_m)
        area = int(np.count_nonzero(top))
        if area < self.params.segmentation.min_area_px:
            return None
        ys, xs = np.nonzero(top)
        u, v = float(xs.mean()), float(ys.mean())
        d_top = float(np.median(self.depth[top]))
        top_c = self.camera.pixel_to_base(np.array([[u, v]]), np.array([d_top]))[0]
        centre = top_c - np.array([0.0, 0.0, self.params.cube_size_m / 2.0])
        x, y, w, h = (
            int(stats_row[k])
            for k in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT)
        )
        fraction = area / self.expected_top_face_px(d_top)
        return Detection(
            color=colour,
            position=(float(centre[0]), float(centre[1]), float(centre[2])),
            top_center=(float(top_c[0]), float(top_c[1]), float(top_c[2])),
            yaw_rad=_top_face_yaw(top, d_top, self.camera),
            pixel=(u, v),
            bbox=(x, y, w, h),
            area_px=area,
            top_face_fraction=fraction,
            complete=fraction >= self.params.segmentation.min_top_face_fraction,
            depth_m=d_top,
        )

    def expected_top_face_px(self, depth_m: float) -> float:
        """Pixel area of a whole ``cube_size_m`` top face seen face-on at ``depth_m``."""
        intr = self.camera.intrinsics
        size = self.params.cube_size_m
        return (size * intr.fx / depth_m) * (size * intr.fy / depth_m)

    def for_colour(self, colour: str) -> list[Detection]:
        seg = self.params.segmentation
        mask = _hue_mask(self.hsv, seg.hue_ranges[colour], seg.min_saturation, seg.min_value)
        mask &= self.depth_ok
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        out: list[Detection] = []
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < seg.min_area_px:
                continue
            det = self._detection(colour, labels == i, stats[i])
            if det is not None:
                out.append(det)
        return out


def detect(
    rgb: np.ndarray,
    depth: np.ndarray,
    target: str = ANY_TARGET,
    *,
    params: PerceptionParams,
    camera: Camera,
) -> DetectResult:
    """Detect cubes of colour ``target`` (``"any"`` for all configured colours).

    Returns an empty list when nothing matches; raises ``UnknownTargetError`` for a colour that is
    not configured and ``ValueError`` for malformed frames. Non-finite or out-of-range depths are
    ignored. Detections are sorted by ``(color, x, y)`` so output is deterministic.
    """
    t0 = time.perf_counter()
    _validate_frame(rgb, depth, camera)
    colours = params.segmentation.colours
    if target != ANY_TARGET:
        if target not in colours:
            msg = f"unknown target {target!r}; expected one of {colours} or {ANY_TARGET!r}"
            raise UnknownTargetError(msg)
        colours = (target,)
    frame = _Frame.build(rgb, depth, params, camera)
    dets: list[Detection] = []
    for colour in colours:
        dets.extend(frame.for_colour(colour))
    dets.sort(key=lambda d: (d.color, d.position[0], d.position[1]))
    return DetectResult(detections=tuple(dets), latency_ms=(time.perf_counter() - t0) * 1e3)
