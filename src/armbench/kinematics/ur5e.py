"""UR5e forward/inverse kinematics modelled after ``ur_description``'s ``ur_macro.xacro``.

The chain (ROS 2 Jazzy, ur_description 3.5.1) is::

    base_link
      fixed    base_link -> base_link_inertia : xyz 0 0 0, rpy 0 0 pi
      revolute shoulder_pan_joint             : origin "shoulder",  axis z
      revolute shoulder_lift_joint            : origin "upper_arm", axis z
      revolute elbow_joint                    : origin "forearm",   axis z
      revolute wrist_1_joint                  : origin "wrist_1",   axis z
      revolute wrist_2_joint                  : origin "wrist_2",   axis z
      revolute wrist_3_joint                  : origin "wrist_3",   axis z
      fixed    wrist_3_link -> flange         : xyz 0 0 0, rpy 0 -pi/2 -pi/2
      fixed    flange -> tool0                : xyz 0 0 0, rpy pi/2 0 pi/2

Every joint origin is ``Trans(xyz) * R_z(yaw) R_y(pitch) R_x(roll)`` followed by
``R_z(q_i)`` for the joint variable. All poses returned here are expressed in
``base_link``, which is also the frame the Gazebo/TF model publishes.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from armbench.kinematics.se3 import rot_x, rot_z, rot_z_4x4, rotation_vector, transform

REPO_ROOT: Final = Path(__file__).resolve().parents[3]
DEFAULT_KINEMATICS_FILE: Final = REPO_ROOT / "configs" / "ur5e_kinematics.yaml"

JOINT_NAMES: Final = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
ORIGIN_NAMES: Final = ("shoulder", "upper_arm", "forearm", "wrist_1", "wrist_2", "wrist_3")
N_JOINTS: Final = 6

HOME_Q: Final = (0.0, -1.57, 0.0, -1.57, 0.0, 0.0)
"""Initial joint configuration of the simulation (arm straight up, tool horizontal)."""

_BASE_TO_INERTIA: Final = transform((0.0, 0.0, 0.0), (0.0, 0.0, np.pi))
_WRIST3_TO_FLANGE: Final = transform((0.0, 0.0, 0.0), (0.0, -np.pi / 2, -np.pi / 2))
_FLANGE_TO_TOOL0: Final = transform((0.0, 0.0, 0.0), (np.pi / 2, 0.0, np.pi / 2))
_TWO_PI: Final = 2.0 * np.pi
_LAMBDA_FLOOR_RATIO: Final = 1e-3
_LAMBDA_MAX: Final = 1e3
_STALL_WINDOW: Final = 10
_KICK_RAD: Final = 0.3
_KICK_MAX_RAD: Final = 1.5
_STALL_RATIO: Final = 0.5


class JointOrigin(BaseModel):
    """URDF ``<origin xyz rpy>`` of a joint in its parent link frame."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float

    def matrix(self) -> np.ndarray:
        return transform((self.x, self.y, self.z), (self.roll, self.pitch, self.yaw))


class JointLimit(BaseModel):
    """Position/velocity/effort limits of one revolute joint."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    min_position: float
    max_position: float
    max_velocity: float = Field(gt=0)
    max_effort: float = Field(gt=0)

    @model_validator(mode="after")
    def _ordered(self) -> JointLimit:
        if self.max_position <= self.min_position:
            msg = f"max_position ({self.max_position}) <= min_position ({self.min_position})"
            raise ValueError(msg)
        return self


class UR5eKinematics(BaseModel):
    """Kinematic parameters loaded from ``configs/ur5e_kinematics.yaml``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int
    kinematics: dict[str, JointOrigin]
    joint_limits: dict[str, JointLimit]

    @field_validator("kinematics")
    @classmethod
    def _all_origins(cls, value: dict[str, JointOrigin]) -> dict[str, JointOrigin]:
        if tuple(value) != ORIGIN_NAMES:
            msg = f"kinematics keys must be {ORIGIN_NAMES} in order, got {tuple(value)}"
            raise ValueError(msg)
        return value

    @field_validator("joint_limits")
    @classmethod
    def _all_joints(cls, value: dict[str, JointLimit]) -> dict[str, JointLimit]:
        if tuple(value) != JOINT_NAMES:
            msg = f"joint_limits keys must be {JOINT_NAMES} in order, got {tuple(value)}"
            raise ValueError(msg)
        return value

    def lower_limits(self) -> np.ndarray:
        return np.array([self.joint_limits[n].min_position for n in JOINT_NAMES])

    def upper_limits(self) -> np.ndarray:
        return np.array([self.joint_limits[n].max_position for n in JOINT_NAMES])


def load_ur5e_kinematics(path: Path = DEFAULT_KINEMATICS_FILE) -> UR5eKinematics:
    """Load and validate the UR5e kinematics YAML."""
    with path.open("rb") as fh:
        raw = yaml.safe_load(fh)
    return UR5eKinematics.model_validate(raw)


class IKResult(BaseModel):
    """Outcome of :meth:`UR5eModel.ik`; ``q`` is the best configuration found."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    success: bool
    q: np.ndarray
    iters: int = Field(ge=0)
    position_error: float = Field(ge=0)
    orientation_error: float = Field(ge=0)

    @field_validator("q")
    @classmethod
    def _six_floats(cls, value: np.ndarray) -> np.ndarray:
        q = np.asarray(value, dtype=float)
        if q.shape != (N_JOINTS,):
            msg = f"q must have shape ({N_JOINTS},), got {q.shape}"
            raise ValueError(msg)
        q.setflags(write=False)
        return q


def _as_q(q: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(q, dtype=float)
    if arr.shape != (N_JOINTS,):
        msg = f"expected {N_JOINTS} joint values, got shape {arr.shape}"
        raise ValueError(msg)
    return arr


def _check_pose(target: np.ndarray) -> np.ndarray:
    pose = np.asarray(target, dtype=float)
    if pose.shape != (4, 4):
        msg = f"target pose must be a 4x4 homogeneous matrix, got shape {pose.shape}"
        raise ValueError(msg)
    return pose


def top_down_pose(xyz: Sequence[float] | np.ndarray, yaw: float) -> np.ndarray:
    """Pose with the tool z axis pointing along -z of ``base_link``: ``R = R_z(yaw) R_x(pi)``."""
    pose = np.eye(4)
    pose[:3, :3] = rot_z(yaw) @ rot_x(np.pi)
    pose[:3, 3] = np.asarray(xyz, dtype=float)
    return pose


class UR5eModel:
    """Forward/inverse kinematics and geometric Jacobian of the UR5e (``base_link -> tool0``)."""

    def __init__(self, params: UR5eKinematics | None = None) -> None:
        self.params = params if params is not None else load_ur5e_kinematics()
        self._origins = [self.params.kinematics[name].matrix() for name in ORIGIN_NAMES]
        self._lower = self.params.lower_limits()
        self._upper = self.params.upper_limits()

    @property
    def lower_limits(self) -> np.ndarray:
        return self._lower.copy()

    @property
    def upper_limits(self) -> np.ndarray:
        return self._upper.copy()

    def fk_all(self, q: Sequence[float] | np.ndarray) -> list[np.ndarray]:
        """Frames of the six moving links (after their joint rotation) followed by ``tool0``.

        Index ``i < 6`` is the frame of the child link of joint ``i`` (its z axis is the
        joint axis and its origin lies on it); index 6 is ``tool0``. All in ``base_link``.
        """
        qa = _as_q(q)
        frames: list[np.ndarray] = []
        t = _BASE_TO_INERTIA
        for origin, qi in zip(self._origins, qa, strict=True):
            t = t @ origin @ rot_z_4x4(float(qi))
            frames.append(t)
        frames.append(t @ _WRIST3_TO_FLANGE @ _FLANGE_TO_TOOL0)
        return frames

    def fk(self, q: Sequence[float] | np.ndarray) -> np.ndarray:
        """4x4 homogeneous transform ``base_link -> tool0``."""
        return self.fk_all(q)[-1]

    def jacobian(self, q: Sequence[float] | np.ndarray) -> np.ndarray:
        """6x6 geometric Jacobian of ``tool0`` in ``base_link`` (linear rows 0-2, angular 3-5)."""
        frames = self.fk_all(q)
        p_tool = frames[-1][:3, 3]
        jac = np.zeros((6, N_JOINTS))
        for i in range(N_JOINTS):
            axis = frames[i][:3, 2]
            jac[:3, i] = np.cross(axis, p_tool - frames[i][:3, 3])
            jac[3:, i] = axis
        return jac

    def within_limits(self, q: Sequence[float] | np.ndarray, *, atol: float = 1e-9) -> bool:
        """True if every joint value lies inside its position limits."""
        qa = _as_q(q)
        return bool(np.all(qa >= self._lower - atol) and np.all(qa <= self._upper + atol))

    def wrap_to_limits(self, q: Sequence[float] | np.ndarray) -> np.ndarray:
        """Shift out-of-range joints by the fewest multiples of 2*pi that bring them inside.

        Joints already inside their limits are untouched; when no equivalent angle fits
        (range narrower than 2*pi) the value is clipped to the nearest limit.
        """
        qa = _as_q(q).copy()
        for i in range(N_JOINTS):
            lo, hi = self._lower[i], self._upper[i]
            if qa[i] < lo:
                qa[i] += _TWO_PI * np.ceil((lo - qa[i]) / _TWO_PI)
            elif qa[i] > hi:
                qa[i] += _TWO_PI * np.floor((hi - qa[i]) / _TWO_PI)
            qa[i] = float(np.clip(qa[i], lo, hi))
        return qa

    def clamp(self, q: Sequence[float] | np.ndarray) -> np.ndarray:
        """Clip joint values to their position limits."""
        clipped: np.ndarray = np.clip(_as_q(q), self._lower, self._upper)
        return clipped

    def pose_error(self, target: np.ndarray, current: np.ndarray) -> np.ndarray:
        """6-D error: translation difference and rotation vector of ``R_target R_current^T``."""
        err = np.empty(6)
        err[:3] = target[:3, 3] - current[:3, 3]
        err[3:] = rotation_vector(target[:3, :3] @ current[:3, :3].T)
        return err

    def ik(  # noqa: PLR0913
        self,
        target: np.ndarray,
        q0: Sequence[float] | np.ndarray,
        *,
        position_tol: float = 1e-4,
        orientation_tol: float = 1e-3,
        max_iters: int = 200,
        damping: float = 1e-2,
    ) -> IKResult:
        """Damped least squares (Levenberg-Marquardt) IK for a ``tool0`` pose.

        Starts from ``q0`` (clamped to the limits) and iterates
        ``dq = J^T (J J^T + lambda^2 I)^-1 e`` on the 6-D pose error, bringing the result
        back inside the joint limits after every step with :meth:`wrap_to_limits` (a joint
        that overshoots a limit is shifted by 2*pi when the equivalent angle fits, otherwise
        clipped, so the search never gets pinned against a limit). ``lambda`` starts at
        ``damping``, is multiplied by 4 when a step does not reduce ``|e|`` and halved (down
        to ``damping * 1e-3``) when it does. If the error has not halved over a window of
        ``_STALL_WINDOW`` iterations (a local minimum near a singularity), the search restarts
        from the best configuration plus a deterministic random kick of growing amplitude
        (``numpy.random.default_rng(0)``, so results are reproducible). Never raises for an
        unreachable target: it returns ``success=False`` with the best configuration found.
        """
        pose = _check_pose(target)
        q = self.clamp(q0)
        err = self.pose_error(pose, self.fk(q))
        best_q, best_err = q, err
        lam = damping
        lam_floor = damping * _LAMBDA_FLOOR_RATIO
        kicks = 0
        rng = np.random.default_rng(0)
        window_start = float(np.linalg.norm(err))
        iters = 0
        while not self._converged(err, position_tol, orientation_tol) and iters < max_iters:
            iters += 1
            jac = self.jacobian(q)
            gain = jac @ jac.T + (lam**2) * np.eye(6)
            dq = jac.T @ np.linalg.solve(gain, err)
            q_new = self.wrap_to_limits(q + dq)
            err_new = self.pose_error(pose, self.fk(q_new))
            if np.linalg.norm(err_new) < np.linalg.norm(err):
                q, err = q_new, err_new
                lam = max(lam_floor, lam / 2.0)
                if np.linalg.norm(err) < np.linalg.norm(best_err):
                    best_q, best_err = q, err
            else:
                lam = min(lam * 4.0, _LAMBDA_MAX)
            if iters % _STALL_WINDOW == 0:
                if np.linalg.norm(err) > _STALL_RATIO * window_start:
                    kicks += 1
                    amplitude = min(_KICK_RAD * kicks, _KICK_MAX_RAD)
                    q = self.wrap_to_limits(best_q + rng.uniform(-amplitude, amplitude, N_JOINTS))
                    err = self.pose_error(pose, self.fk(q))
                    lam = damping
                window_start = float(np.linalg.norm(err))
        if np.linalg.norm(err) > np.linalg.norm(best_err):
            q, err = best_q, best_err
        return IKResult(
            success=self._converged(err, position_tol, orientation_tol),
            q=q,
            iters=iters,
            position_error=float(np.linalg.norm(err[:3])),
            orientation_error=float(np.linalg.norm(err[3:])),
        )

    def ik_top_down(  # noqa: PLR0913
        self,
        xyz: Sequence[float] | np.ndarray,
        yaw: float,
        q0: Sequence[float] | np.ndarray,
        *,
        tcp_offset_m: float = 0.0,
        position_tol: float = 1e-4,
        orientation_tol: float = 1e-3,
        max_iters: int = 200,
        damping: float = 1e-2,
    ) -> IKResult:
        """IK for a top-down grasp pose.

        Convention (shared with ``armbench_bringup/config/sim_check.yaml``): the target
        orientation is ``R = R_z(yaw) R_x(pi)``, i.e. the tool z axis points along ``-z`` of
        ``base_link`` (straight down onto the table) and ``yaw`` rotates the tool about the
        vertical. ``xyz`` is the position of the TCP, defined as ``tool0`` translated
        ``tcp_offset_m`` along the tool z axis (``0.0`` means ``xyz`` is ``tool0`` itself;
        the simulation gripper uses ``0.125``). Because the tool z axis is ``-z``, the
        ``tool0`` target is ``xyz + (0, 0, tcp_offset_m)``.
        """
        pose = top_down_pose(xyz, yaw)
        pose[:3, 3] -= tcp_offset_m * pose[:3, 2]
        return self.ik(
            pose,
            q0,
            position_tol=position_tol,
            orientation_tol=orientation_tol,
            max_iters=max_iters,
            damping=damping,
        )

    @staticmethod
    def _converged(err: np.ndarray, position_tol: float, orientation_tol: float) -> bool:
        return bool(
            np.linalg.norm(err[:3]) < position_tol and np.linalg.norm(err[3:]) < orientation_tol
        )
