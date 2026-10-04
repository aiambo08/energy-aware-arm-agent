"""Perception parameters (``configs/perception.yaml``)."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from armbench.paths import CONFIG_DIR

DEFAULT_PERCEPTION_FILE: Final = CONFIG_DIR / "perception.yaml"
HUE_MAX: Final = 179

HueRange = tuple[int, int]


class CameraSpec(BaseModel):
    """Image size, horizontal field of view and sensor pose (URDF/SDF xyz + rpy)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    width: int = Field(gt=0)
    height: int = Field(gt=0)
    horizontal_fov_rad: float = Field(gt=0, lt=3.141592653589793)
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]


class DepthSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    min_m: float = Field(gt=0)
    max_m: float = Field(gt=0)
    top_face_band_m: float = Field(gt=0)

    @model_validator(mode="after")
    def _ordered(self) -> DepthSpec:
        if self.max_m <= self.min_m:
            msg = "depth.max_m must exceed depth.min_m"
            raise ValueError(msg)
        return self


class SegmentationSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    min_saturation: int = Field(ge=0, le=255)
    min_value: int = Field(ge=0, le=255)
    min_area_px: int = Field(gt=0)
    min_top_face_fraction: float = Field(gt=0, le=1)
    open_kernel_px: int = Field(ge=1)
    hue_ranges: dict[str, tuple[HueRange, ...]]

    @field_validator("hue_ranges")
    @classmethod
    def _valid_ranges(
        cls, value: dict[str, tuple[HueRange, ...]]
    ) -> dict[str, tuple[HueRange, ...]]:
        if not value:
            msg = "hue_ranges must name at least one colour"
            raise ValueError(msg)
        for colour, ranges in value.items():
            if not ranges:
                msg = f"hue_ranges[{colour!r}] is empty"
                raise ValueError(msg)
            for lo, hi in ranges:
                if not 0 <= lo <= hi <= HUE_MAX:
                    msg = (
                        f"hue_ranges[{colour!r}]: bad range {(lo, hi)} (0 <= lo <= hi <= {HUE_MAX})"
                    )
                    raise ValueError(msg)
        return value

    @property
    def colours(self) -> tuple[str, ...]:
        return tuple(self.hue_ranges)


class PerceptionParams(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int
    frame: str
    camera: CameraSpec
    depth: DepthSpec
    segmentation: SegmentationSpec
    cube_size_m: float = Field(gt=0)


def load_perception_params(path: Path = DEFAULT_PERCEPTION_FILE) -> PerceptionParams:
    with path.open() as fh:
        raw = yaml.safe_load(fh)
    return PerceptionParams.model_validate(raw)
