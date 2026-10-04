"""Typed failures of the robot primitives (docs/plan.es.md, F4).

Every error carries a stable ``code`` (what an agent or a log parser matches on) and a
``details`` dict that is JSON-serialisable, so a failed primitive can be reported verbatim in
episode logs and fed back to an LLM as text.
"""

from __future__ import annotations

from typing import ClassVar


class PrimitiveError(Exception):
    """Base class; ``code`` identifies the failure class, ``details`` the instance."""

    code: ClassVar[str] = "primitive_error"

    def __init__(self, message: str, **details: object) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, object] = dict(details)

    def to_dict(self) -> dict[str, object]:
        return {"code": self.code, "message": self.message, "details": self.details}


class OutOfReach(PrimitiveError):
    """Target pose outside the allowed workspace or without an inverse-kinematics solution."""

    code = "out_of_reach"


class Singularity(PrimitiveError):
    """Target configuration too close to a kinematic singularity (ill-conditioned Jacobian)."""

    code = "singularity"


class Collision(PrimitiveError):
    """Target or joint-space path would bring the arm or gripper into the table."""

    code = "collision"


class Timeout(PrimitiveError):
    """The controller did not finish the motion (or the gripper did not settle) in time."""

    code = "timeout"


class NoObjectGrasped(PrimitiveError):
    """``grasp()`` closed the fingers completely: nothing between the pads."""

    code = "no_object_grasped"


class CameraTimeout(PrimitiveError):
    """No fresh RGB-D pair arrived within the camera timeout."""

    code = "camera_timeout"


class SkillNotAvailable(PrimitiveError):
    """``execute_skill()`` named a skill that is not in the library the robot was given."""

    code = "skill_not_available"


class SkillPreconditionFailed(PrimitiveError):
    """A skill's precondition did not hold when ``execute_skill()`` was called; nothing moved."""

    code = "skill_precondition_failed"


class SkillPostconditionFailed(PrimitiveError):
    """The skill body ran to completion but a postcondition does not hold afterwards."""

    code = "skill_postcondition_failed"


class SkillFailed(PrimitiveError):
    """The skill body stopped before completing (sandbox limit, exception in the body)."""

    code = "skill_failed"


ERRORS: tuple[type[PrimitiveError], ...] = (
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
"""All concrete error classes, for agents' prompts and for the log schema."""
