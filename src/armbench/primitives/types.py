"""Pydantic contracts shared by every primitive (inputs, outputs and observations).

All poses are top-down tool-centre-point (TCP) poses in ``base_link``: the tool z axis points
straight down onto the table and ``yaw`` rotates the gripper about the vertical. Square objects
make yaw observable only modulo 90 degrees, so the agent never needs more than ``[-pi/4, pi/4)``,
but any finite yaw is accepted and used as given.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from armbench.perception import Detection

Vec6 = tuple[float, float, float, float, float, float]
Primitive = Literal["move_to", "grasp", "release", "reset", "observe", "detect", "home"]


class Pose(BaseModel):
    """Top-down TCP pose in ``base_link`` (metres, radians)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    x: float = Field(allow_inf_nan=False)
    y: float = Field(allow_inf_nan=False)
    z: float = Field(allow_inf_nan=False)
    yaw: float = Field(default=0.0, allow_inf_nan=False)

    def above(self, dz: float) -> Pose:
        """Same x, y and yaw, ``dz`` metres higher."""
        return self.model_copy(update={"z": self.z + dz})

    def with_yaw(self, yaw: float) -> Pose:
        return self.model_copy(update={"yaw": yaw})

    def xyz(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def distance_xy(self, other: Pose) -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


class MotionPlan(BaseModel):
    """What ``move_to`` decided before touching the controller (also returned by ``plan``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    target: Pose
    q_start: Vec6
    q_target: Vec6
    duration_s: float = Field(gt=0)
    speed_scale: float = Field(gt=0, le=1)
    ik_iters: int = Field(ge=0)
    ik_position_error_m: float = Field(ge=0)
    ik_orientation_error_rad: float = Field(ge=0)
    min_singular_value: float = Field(ge=0)
    path_min_clearance_m: float
    """Lowest height above the table of any arm frame or finger tip along the joint path."""


class Result(BaseModel):
    """Outcome of a primitive that completed (failures raise :class:`PrimitiveError`)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    primitive: Primitive
    t_start_sim: float
    t_end_sim: float
    wall_s: float = Field(ge=0)

    @property
    def sim_s(self) -> float:
        return self.t_end_sim - self.t_start_sim


class MoveResult(Result):
    plan: MotionPlan
    q_final: Vec6
    tcp_final: Pose
    position_error_m: float = Field(ge=0)
    """Distance between the requested TCP position and the TCP computed from the final joints."""
    yaw_error_rad: float = Field(ge=0)
    joint_error_rad: float = Field(ge=0)


class GripperResult(Result):
    opening_m: float = Field(ge=0)
    """Gap between the finger pads after the command settled."""
    holding: bool


class Observation(BaseModel):
    """Everything the agent may look at between primitives."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    t_sim: float
    q: Vec6
    qd: Vec6
    tcp: Pose
    gripper_opening_m: float = Field(ge=0)
    holding: bool
    detections: tuple[Detection, ...]
