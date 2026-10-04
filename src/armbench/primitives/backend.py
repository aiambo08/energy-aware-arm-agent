"""The thin interface a robot runtime must implement for :class:`armbench.primitives.Robot`.

Two implementations exist: ``armbench.primitives.fake.KinematicBackend`` (pure Python, instant
motions, synthetic camera; used by the unit tests and by replay) and
``armbench_bringup.ros_backend.RosBackend`` (rclpy: ros2_control trajectory action, effort
gripper, Gazebo RGB-D camera). The contract is deliberately small so that every safety check
(reach, collision, singularity, timeouts) lives once, in :class:`Robot`.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict


class JointSnapshot(BaseModel):
    """Latest joint state: arm positions/velocities and finger joint positions (stroke used)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    t_sim: float
    q: tuple[float, float, float, float, float, float]
    qd: tuple[float, float, float, float, float, float]
    fingers: tuple[float, float]


class MotionOutcome(StrEnum):
    SUCCESS = "success"
    REJECTED = "rejected"  # the controller refused the goal
    ABORTED = "aborted"  # the controller gave up (path/goal tolerance)
    TIMEOUT = "timeout"  # no result before the deadline


class Backend(Protocol):
    def sim_time(self) -> float:
        """Current simulated time in seconds (NaN until the clock is known)."""
        ...

    def snapshot(self, timeout_s: float) -> JointSnapshot | None:
        """Latest joint state, waiting up to ``timeout_s`` wall seconds for one to exist."""
        ...

    def follow(self, q: Sequence[float], duration_s: float, timeout_s: float) -> MotionOutcome:
        """Move the arm joints to ``q`` in ``duration_s`` simulated seconds (single waypoint)."""
        ...

    def gripper(self, effort_n: float, settle_s: float) -> None:
        """Command both fingers with ``effort_n`` (positive closes) for ``settle_s`` sim seconds."""
        ...

    def frame(self, timeout_s: float) -> tuple[np.ndarray, np.ndarray] | None:
        """A fresh RGB (uint8 HxWx3) + depth (float32 HxW, metres) pair stamped after now."""
        ...
