"""ROS-free robot runtime: instant joint moves on a simulated clock, a gripper that attaches the
cube between its pads, and a synthetic RGB-D camera. The same :class:`Robot` runs on it as on
Gazebo, so contracts, scripted tasks and (later) replayed agent programs can be tested in
milliseconds. It models no dynamics: moves are exact, cubes do not fall or slide.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from armbench.kinematics import UR5eModel
from armbench.perception import Camera
from armbench.perception.synthetic import render
from armbench.primitives.backend import JointSnapshot, MotionOutcome
from armbench.primitives.params import PrimitiveParams
from armbench.scene import Cube

GRASP_XY_TOL_M = 0.012
"""Pad centre must be within this of the cube centre (in the plane) to pick it up."""
GRASP_Z_TOL_M = 0.02
SUPPORT_OVERLAP = 0.6
"""A dropped cube rests on another when their centres overlap by this fraction of a cube."""
"""Pad centre must be within this of the cube centre height to pick it up."""


@dataclass
class KinematicBackend:
    params: PrimitiveParams
    camera: Camera
    kinematics: UR5eModel = field(default_factory=UR5eModel)
    cubes: list[Cube] = field(default_factory=list)
    q: np.ndarray = field(init=False)
    fingers: tuple[float, float] = field(init=False)
    t: float = 0.0
    held: str | None = None
    tracking_noise_rad: float = 0.0
    """Uniform +/- error added to every reached joint (models controller tracking error)."""
    camera_available: bool = True
    motion_outcome: MotionOutcome = MotionOutcome.SUCCESS
    """Forced result of ``follow`` (tests use it to provoke ``Timeout``)."""
    follow_log: list[tuple[tuple[float, ...], float]] = field(default_factory=list)
    _rng: np.random.Generator = field(init=False, repr=False)
    _frame_seed: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.q = np.asarray(self.params.ready_q, dtype=float)
        self.fingers = (0.0, 0.0)
        self._rng = np.random.default_rng(0)

    # -- Backend protocol -------------------------------------------------------------------
    def sim_time(self) -> float:
        return self.t

    def snapshot(self, timeout_s: float) -> JointSnapshot:
        q = tuple(float(v) for v in self.q)
        return JointSnapshot(t_sim=self.t, q=q, qd=(0.0,) * 6, fingers=self.fingers)

    def follow(self, q: Sequence[float], duration_s: float, timeout_s: float) -> MotionOutcome:
        self.follow_log.append((tuple(float(v) for v in q), duration_s))
        if self.motion_outcome is not MotionOutcome.SUCCESS:
            return self.motion_outcome
        target = np.asarray(q, dtype=float)
        if self.tracking_noise_rad > 0:
            target = target + self._rng.uniform(
                -self.tracking_noise_rad, self.tracking_noise_rad, target.shape
            )
        self.q = target
        self.t += duration_s
        self._carry()
        return MotionOutcome.SUCCESS

    def gripper(self, effort_n: float, settle_s: float) -> None:
        self.t += settle_s
        stroke = self.params.gripper.stroke_m
        if effort_n <= 0:
            self.fingers = (0.0, 0.0)
            if self.held is not None:
                self._drop()
            return
        cube = self._cube_between_pads()
        if cube is None:
            self.fingers = (stroke, stroke)
            return
        per_finger = (self.params.gripper.max_opening_m - cube.size) / 2.0
        self.fingers = (per_finger, per_finger)
        self.held = cube.name
        self._carry()

    def frame(self, timeout_s: float) -> tuple[np.ndarray, np.ndarray] | None:
        if not self.camera_available:
            return None
        self._frame_seed += 1
        ordered = sorted(self.cubes, key=lambda c: c.z)  # higher cubes drawn last (occlude)
        return render(ordered, self.camera, seed=self._frame_seed)

    # -- scene helpers ----------------------------------------------------------------------
    def tcp(self) -> np.ndarray:
        tool0 = self.kinematics.fk(self.q)
        return np.asarray(tool0[:3, 3] + self.params.tcp_offset_m * tool0[:3, 2])

    def tcp_yaw(self) -> float:
        tool0 = self.kinematics.fk(self.q)
        return math.atan2(tool0[1, 0], tool0[0, 0])

    def cube(self, name: str) -> Cube:
        for c in self.cubes:
            if c.name == name:
                return c
        msg = f"no cube named {name!r}"
        raise KeyError(msg)

    def _replace(self, cube: Cube) -> None:
        self.cubes = [cube if c.name == cube.name else c for c in self.cubes]

    def _cube_between_pads(self) -> Cube | None:
        p = self.tcp()
        best: Cube | None = None
        best_d = math.inf
        for c in self.cubes:
            d = math.hypot(c.x - p[0], c.y - p[1])
            if d <= GRASP_XY_TOL_M and abs(c.z - p[2]) <= GRASP_Z_TOL_M and d < best_d:
                best, best_d = c, d
        return best

    def _carry(self) -> None:
        if self.held is None:
            return
        p = self.tcp()
        c = self.cube(self.held)
        self._replace(c.model_copy(update={"x": float(p[0]), "y": float(p[1]), "z": float(p[2])}))

    def _drop(self) -> None:
        """Let the held cube fall straight down onto the table or onto a cube under it."""
        if self.held is None:
            return
        c = self.cube(self.held)
        support = 0.0
        for o in self.cubes:
            if o.name == c.name:
                continue
            overlap = (c.size + o.size) / 2.0 * SUPPORT_OVERLAP
            if abs(o.x - c.x) < overlap and abs(o.y - c.y) < overlap:
                support = max(support, o.z + o.size / 2.0)
        self._replace(c.model_copy(update={"z": support + c.size / 2.0}))
        self.held = None
