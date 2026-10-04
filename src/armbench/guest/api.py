"""What a sandboxed agent program sees: ``robot`` (a proxy), ``Pose``, the primitive errors and
``math``. Standard library only — the worker process imports this module and nothing else from
``armbench``, so the child never holds numpy, pydantic, the simulator or the real robot.

The names and error codes mirror :mod:`armbench.primitives` one to one; the parent process
validates every argument against the real contracts and serialises every result to JSON, so the
objects here are read-only views of those results.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Final

PRIMITIVES: Final[tuple[str, ...]] = (
    "observe",
    "detect",
    "move_to",
    "grasp",
    "release",
    "execute_skill",
    "reset",
)
"""The only ``robot`` attributes a program may call (``home`` is the runner's, not the agent's)."""

_POSE_KEYS: Final = frozenset({"x", "y", "z", "yaw"})


class Pose:
    """Top-down TCP pose in ``base_link`` (metres, radians); the gripper points straight down."""

    __slots__ = ("x", "y", "yaw", "z")

    def __init__(self, x: float, y: float, z: float, yaw: float = 0.0) -> None:
        for name, v in (("x", x), ("y", y), ("z", z), ("yaw", yaw)):
            if isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v):
                msg = f"Pose.{name} must be a finite number, got {v!r}"
                raise TypeError(msg)
        self.x, self.y, self.z, self.yaw = float(x), float(y), float(z), float(yaw)

    def above(self, dz: float) -> Pose:
        """Same x, y and yaw, ``dz`` metres higher."""
        return Pose(self.x, self.y, self.z + dz, self.yaw)

    def with_yaw(self, yaw: float) -> Pose:
        return Pose(self.x, self.y, self.z, yaw)

    def distance_xy(self, other: Pose) -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def to_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "z": self.z, "yaw": self.yaw}

    def __repr__(self) -> str:
        return f"Pose(x={self.x:.4f}, y={self.y:.4f}, z={self.z:.4f}, yaw={self.yaw:.4f})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Pose) and self.to_dict() == other.to_dict()

    def __hash__(self) -> int:
        return hash((self.x, self.y, self.z, self.yaw))


class Record:
    """Read-only attribute view of a JSON result (``obs.tcp.x``, ``det.position[0]``, ...)."""

    __slots__ = ("_data",)

    def __init__(self, data: Mapping[str, object]) -> None:
        object.__setattr__(self, "_data", dict(data))

    def __getattr__(self, name: str) -> object:
        if name.startswith("_"):
            raise AttributeError(name)
        data: dict[str, object] = object.__getattribute__(self, "_data")
        try:
            return wrap(data[name])
        except KeyError:
            msg = f"result has no field {name!r}; fields: {', '.join(sorted(data))}"
            raise AttributeError(msg) from None

    def __setattr__(self, name: str, value: object) -> None:
        msg = "results are read-only"
        raise AttributeError(msg)

    def to_dict(self) -> dict[str, object]:
        return dict(object.__getattribute__(self, "_data"))

    def __repr__(self) -> str:
        data: dict[str, object] = object.__getattribute__(self, "_data")
        body = ", ".join(f"{k}={wrap(v)!r}" for k, v in data.items())
        return f"Record({body})"


def wrap(value: object) -> object:
    """JSON value -> program value: pose dicts become ``Pose``, other dicts ``Record``, lists
    tuples; scalars pass through."""
    if isinstance(value, dict):
        if set(value) == _POSE_KEYS and all(isinstance(v, int | float) for v in value.values()):
            return Pose(value["x"], value["y"], value["z"], value["yaw"])
        return Record(value)
    if isinstance(value, list):
        return tuple(wrap(v) for v in value)
    return value


def unwrap(value: object) -> object:
    """Program value -> JSON value, for primitive arguments; rejects anything else."""
    if isinstance(value, Pose):
        return value.to_dict()
    if isinstance(value, Record):
        return value.to_dict()
    if isinstance(value, bool | int | float | str) or value is None:
        return value
    if isinstance(value, list | tuple):
        return [unwrap(v) for v in value]
    if isinstance(value, dict):
        return {str(k): unwrap(v) for k, v in value.items()}
    msg = f"cannot pass a {type(value).__name__} to a primitive"
    raise TypeError(msg)


class PrimitiveError(Exception):
    """A primitive failed; ``code`` is stable, ``details`` is what the robot reported."""

    code = "primitive_error"

    def __init__(self, message: str, details: Mapping[str, object] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, object] = dict(details or {})


class OutOfReach(PrimitiveError):
    """Target outside the workspace box or without an inverse-kinematics solution."""

    code = "out_of_reach"


class Singularity(PrimitiveError):
    """Target configuration too close to a kinematic singularity."""

    code = "singularity"


class Collision(PrimitiveError):
    """Target or path would bring the arm or the finger tips into the table."""

    code = "collision"


class Timeout(PrimitiveError):
    """The controller did not finish the motion in time."""

    code = "timeout"


class NoObjectGrasped(PrimitiveError):
    """``grasp()`` closed the fingers completely: nothing between the pads."""

    code = "no_object_grasped"


class CameraTimeout(PrimitiveError):
    """No fresh RGB-D frame arrived in time."""

    code = "camera_timeout"


class SkillNotAvailable(PrimitiveError):
    """``execute_skill()`` named a skill that is not in the library."""

    code = "skill_not_available"


class SkillPreconditionFailed(PrimitiveError):
    """A skill precondition did not hold; the skill did not move the arm."""

    code = "skill_precondition_failed"


class SkillPostconditionFailed(PrimitiveError):
    """The skill body completed but a postcondition does not hold."""

    code = "skill_postcondition_failed"


class SkillFailed(PrimitiveError):
    """The skill body stopped before completing."""

    code = "skill_failed"


ERRORS: Final[tuple[type[PrimitiveError], ...]] = (
    OutOfReach,
    Singularity,
    Collision,
    Timeout,
    NoObjectGrasped,
    CameraTimeout,
    SkillNotAvailable,
    SkillPreconditionFailed,
    SkillPostconditionFailed,
    SkillFailed,
)
ERRORS_BY_CODE: Final[dict[str, type[PrimitiveError]]] = {e.code: e for e in ERRORS}


def error_from_code(code: str, message: str, details: Mapping[str, object]) -> PrimitiveError:
    return ERRORS_BY_CODE.get(code, PrimitiveError)(message, details)
