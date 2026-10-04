"""Robot primitives with typed contracts (phase F4): the only API an agent program sees."""

from armbench.primitives.backend import Backend, JointSnapshot, MotionOutcome
from armbench.primitives.errors import (
    ERRORS,
    CameraTimeout,
    Collision,
    NoObjectGrasped,
    OutOfReach,
    PrimitiveError,
    Singularity,
    SkillFailed,
    SkillNotAvailable,
    SkillPostconditionFailed,
    SkillPreconditionFailed,
    Timeout,
)
from armbench.primitives.fake import KinematicBackend
from armbench.primitives.params import (
    DEFAULT_PRIMITIVES_FILE,
    PrimitiveParams,
    load_primitive_params,
)
from armbench.primitives.robot import Robot, SkillExecutor
from armbench.primitives.types import (
    GripperResult,
    MotionPlan,
    MoveResult,
    Observation,
    Pose,
    Result,
    SkillResult,
)

__all__ = [
    "DEFAULT_PRIMITIVES_FILE",
    "ERRORS",
    "Backend",
    "CameraTimeout",
    "Collision",
    "GripperResult",
    "JointSnapshot",
    "KinematicBackend",
    "MotionOutcome",
    "MotionPlan",
    "MoveResult",
    "NoObjectGrasped",
    "Observation",
    "OutOfReach",
    "Pose",
    "PrimitiveError",
    "PrimitiveParams",
    "Result",
    "Robot",
    "Singularity",
    "SkillExecutor",
    "SkillFailed",
    "SkillNotAvailable",
    "SkillPostconditionFailed",
    "SkillPreconditionFailed",
    "SkillResult",
    "Timeout",
    "load_primitive_params",
]
