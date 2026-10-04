"""Validated parameters of the primitives (``configs/primitives.yaml``)."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from armbench.paths import CONFIG_DIR

DEFAULT_PRIMITIVES_FILE: Final = CONFIG_DIR / "primitives.yaml"


class WorkspaceBox(BaseModel):
    """TCP positions the agent may request (``base_link``); outside -> ``OutOfReach``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    @model_validator(mode="after")
    def _ordered(self) -> WorkspaceBox:
        if not (self.x_min < self.x_max and self.y_min < self.y_max and self.z_min < self.z_max):
            msg = "workspace bounds must satisfy min < max on every axis"
            raise ValueError(msg)
        return self

    def contains(self, x: float, y: float, z: float) -> bool:
        return (
            self.x_min <= x <= self.x_max
            and self.y_min <= y <= self.y_max
            and self.z_min <= z <= self.z_max
        )


class MotionSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    max_joint_speed_rad_s: float = Field(gt=0)
    """Average joint speed of the slowest-limiting joint at ``speed_scale = 1``."""
    min_duration_s: float = Field(gt=0)
    speed_scale_min: float = Field(gt=0, le=1)
    goal_tolerance_rad: float = Field(gt=0)
    settle_timeout_s: float = Field(ge=0)
    """Simulated seconds to wait after the controller reports success for the joints to stop."""
    settle_qd_rad_s: float = Field(gt=0)
    """Joint speed below which the arm counts as settled."""
    timeout_factor: float = Field(ge=1)
    timeout_margin_s: float = Field(ge=0)


class IKSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    position_tol_m: float = Field(gt=0)
    orientation_tol_rad: float = Field(gt=0)
    max_iters: int = Field(gt=0)
    min_singular_value: float = Field(gt=0)
    """Smallest allowed singular value of the geometric Jacobian at the target."""
    branch: dict[str, tuple[float, float]]
    """Joint name -> ``(lo, hi)`` interval every IK solution must satisfy (one elbow branch)."""

    @model_validator(mode="after")
    def _branch_ordered(self) -> IKSpec:
        for name, (lo, hi) in self.branch.items():
            if hi <= lo:
                msg = f"ik.branch.{name}: hi must exceed lo"
                raise ValueError(msg)
        return self


class CollisionSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    table_z_m: float
    link_clearance_m: float = Field(ge=0)
    """Minimum height above the table of every arm frame origin along the path."""
    tip_clearance_m: float = Field(ge=0)
    """Minimum height above the table of the finger tips (TCP minus ``finger_tip_below_tcp_m``)."""
    path_samples: int = Field(ge=2)


class GripperSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    stroke_m: float = Field(gt=0)
    """Travel of each finger from fully open to fully closed."""
    open_effort_n: float = Field(lt=0)
    close_effort_n: float = Field(gt=0)
    settle_s: float = Field(gt=0)
    min_object_width_m: float = Field(gt=0)
    """Pad gap below which the fingers are considered closed on nothing."""

    @property
    def max_opening_m(self) -> float:
        return 2.0 * self.stroke_m


class PrimitiveParams(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int
    frame: str
    tcp_offset_m: float = Field(ge=0)
    """``tool0`` -> TCP along the tool z axis (gripper xacro)."""
    finger_tip_below_tcp_m: float = Field(ge=0)
    workspace: WorkspaceBox
    motion: MotionSpec
    ik: IKSpec
    collision: CollisionSpec
    gripper: GripperSpec
    camera_timeout_s: float = Field(gt=0)
    ready_pose: tuple[float, float, float, float]
    """TCP ``(x, y, z, yaw)`` the arm returns to on ``reset()``; seeds every IK from its branch."""
    ready_q: tuple[float, float, float, float, float, float]
    """Joint configuration realising ``ready_pose`` (checked against FK at load time)."""


def load_primitive_params(path: Path = DEFAULT_PRIMITIVES_FILE) -> PrimitiveParams:
    with path.open("rb") as fh:
        raw = yaml.safe_load(fh)
    return PrimitiveParams.model_validate(raw)
