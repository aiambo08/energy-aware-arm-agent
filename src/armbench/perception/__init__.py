"""RGB-D cube perception: ``detect()`` (HSV + depth) and the simulated camera model."""

from armbench.perception.camera import OPTICAL_IN_SENSOR, Camera, Extrinsics, Intrinsics
from armbench.perception.detector import (
    ANY_TARGET,
    Detection,
    DetectResult,
    UnknownTargetError,
    detect,
    yaw_difference,
)
from armbench.perception.params import (
    DEFAULT_PERCEPTION_FILE,
    CameraSpec,
    DepthSpec,
    PerceptionParams,
    SegmentationSpec,
    load_perception_params,
)

__all__ = [
    "ANY_TARGET",
    "DEFAULT_PERCEPTION_FILE",
    "OPTICAL_IN_SENSOR",
    "Camera",
    "CameraSpec",
    "DepthSpec",
    "DetectResult",
    "Detection",
    "Extrinsics",
    "Intrinsics",
    "PerceptionParams",
    "SegmentationSpec",
    "UnknownTargetError",
    "detect",
    "load_perception_params",
    "yaw_difference",
]
