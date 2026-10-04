"""The primitives an agent program may call: ``observe``, ``detect``, ``move_to``, ``grasp``,
``release``, ``execute_skill`` (F7) and ``reset``.

Every check that keeps the simulation consistent lives here, before the backend is touched:

* ``move_to``: pose inside the workspace box, IK solution on the configured elbow branch and
  within joint limits (else ``OutOfReach``), Jacobian well conditioned (else ``Singularity``),
  every arm frame and the finger tips above the table along the joint-space path (else
  ``Collision``). Only then is a single-waypoint trajectory sent; the controller must finish it
  within ``duration * timeout_factor + margin`` wall seconds (else ``Timeout``).
* ``grasp``: closes the fingers and raises ``NoObjectGrasped`` when the pads meet.
* ``observe``/``detect``: wait for a fresh RGB-D pair (else ``CameraTimeout``) and run
  :func:`armbench.perception.detect`.

``speed_scale`` scales the average joint speed, so the duration of a move is
``max(min_duration, max|dq| / (max_joint_speed * speed_scale))``: monotonically decreasing in
``speed_scale`` for a fixed start and target.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from typing import Protocol

import numpy as np

from armbench.kinematics import JOINT_NAMES, N_JOINTS, UR5eModel
from armbench.perception import (
    ANY_TARGET,
    Camera,
    Detection,
    PerceptionParams,
    load_perception_params,
)
from armbench.perception import (
    detect as detect_cubes,
)
from armbench.primitives.backend import Backend, JointSnapshot, MotionOutcome
from armbench.primitives.errors import (
    CameraTimeout,
    Collision,
    NoObjectGrasped,
    OutOfReach,
    Singularity,
    SkillNotAvailable,
    Timeout,
)
from armbench.primitives.params import PrimitiveParams, load_primitive_params
from armbench.primitives.types import (
    GripperResult,
    MotionPlan,
    MoveResult,
    Observation,
    Pose,
    Primitive,
    Result,
    SkillResult,
    Vec6,
)

_TWO_PI = 2.0 * math.pi
_READY_TOL_M = 1e-3


def _vec6(q: Sequence[float] | np.ndarray) -> Vec6:
    a, b, c, d, e, f = (float(v) for v in q)
    return (a, b, c, d, e, f)


def _wrap_angle(a: float) -> float:
    return (a + math.pi) % _TWO_PI - math.pi


class SkillExecutor(Protocol):
    """What ``execute_skill`` delegates to (implemented by :mod:`armbench.skills`)."""

    def execute(self, robot: Robot, name: str, kwargs: Mapping[str, object]) -> SkillResult: ...


def _jsonable_kwargs(kwargs: Mapping[str, object]) -> dict[str, object]:
    plain = str | int | float | bool | None
    return {k: v if isinstance(v, plain) else repr(v) for k, v in kwargs.items()}


class Robot:
    """Primitives over a :class:`Backend`; one instance per episode."""

    def __init__(  # noqa: PLR0913 - every collaborator is optional and keyword-only
        self,
        backend: Backend,
        *,
        params: PrimitiveParams | None = None,
        perception: PerceptionParams | None = None,
        camera: Camera | None = None,
        kinematics: UR5eModel | None = None,
        skills: SkillExecutor | None = None,
    ) -> None:
        self.backend = backend
        self.skills = skills
        self.params = params if params is not None else load_primitive_params()
        self.perception = perception if perception is not None else load_perception_params()
        self.camera = camera if camera is not None else Camera.from_spec(self.perception.camera)
        self.kin = kinematics if kinematics is not None else UR5eModel()
        self._branch_lo, self._branch_hi = self._branch_bounds()
        self._ready_q = np.asarray(self.params.ready_q, dtype=float)
        rx, ry, rz, ryaw = self.params.ready_pose
        self.ready_pose = Pose(x=rx, y=ry, z=rz, yaw=ryaw)
        ready_tcp = self.tcp_pose(self._ready_q)
        if ready_tcp.distance_xy(self.ready_pose) > _READY_TOL_M or (
            abs(ready_tcp.z - self.ready_pose.z) > _READY_TOL_M
        ):
            msg = f"ready_q realises {ready_tcp}, not ready_pose {self.ready_pose}"
            raise ValueError(msg)
        if not self._on_branch(self._ready_q):
            msg = "ready_q is not inside ik.branch"
            raise ValueError(msg)
        self._last_gripper_effort = 0.0

    # -- kinematic helpers ------------------------------------------------------------------
    def _branch_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lo = np.full(N_JOINTS, -np.inf)
        hi = np.full(N_JOINTS, np.inf)
        for name, (a, b) in self.params.ik.branch.items():
            if name not in JOINT_NAMES:
                msg = f"ik.branch: unknown joint {name!r}"
                raise ValueError(msg)
            i = JOINT_NAMES.index(name)
            lo[i], hi[i] = a, b
        return lo, hi

    def _on_branch(self, q: np.ndarray) -> bool:
        return bool(np.all(q >= self._branch_lo) and np.all(q <= self._branch_hi))

    def tcp_pose(self, q: Sequence[float] | np.ndarray) -> Pose:
        """TCP position and yaw (about the vertical, folded to ``[-pi, pi)``) from joints."""
        tool0 = self.kin.fk(q)
        p = tool0[:3, 3] + self.params.tcp_offset_m * tool0[:3, 2]
        # top-down tool: its x axis lies in the table plane; yaw is its direction.
        yaw = math.atan2(tool0[1, 0], tool0[0, 0])
        return Pose(x=float(p[0]), y=float(p[1]), z=float(p[2]), yaw=_wrap_angle(yaw))

    def _unwrap_towards(self, q: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """Shift joints by multiples of 2*pi to be nearest ``ref`` while staying within limits."""
        out = q.copy()
        lower, upper = self.kin.lower_limits, self.kin.upper_limits
        for i in range(N_JOINTS):
            best = out[i]
            for k in (-2, -1, 1, 2):
                cand = q[i] + k * _TWO_PI
                if lower[i] <= cand <= upper[i] and abs(cand - ref[i]) < abs(best - ref[i]):
                    best = cand
            out[i] = best
        return out

    def _solve_ik(self, target: Pose, q_now: np.ndarray) -> tuple[np.ndarray, int, float, float]:
        """IK on the configured branch, seeded from the current joints and then from ``ready_q``."""
        ik = self.params.ik
        attempts: list[tuple[np.ndarray, int, float, float]] = []
        for seed in (q_now, self._ready_q):
            res = self.kin.ik_top_down(
                target.xyz(),
                target.yaw,
                seed,
                tcp_offset_m=self.params.tcp_offset_m,
                position_tol=ik.position_tol_m,
                orientation_tol=ik.orientation_tol_rad,
                max_iters=ik.max_iters,
            )
            q = self._unwrap_towards(np.asarray(res.q, dtype=float), q_now)
            attempts.append((q, res.iters, res.position_error, res.orientation_error))
            if res.success and self._on_branch(q) and self.kin.within_limits(q):
                return q, res.iters, res.position_error, res.orientation_error
        best = min(attempts, key=lambda a: a[2] + a[3])
        raise OutOfReach(
            f"no inverse-kinematics solution on the configured branch for {target}",
            target=target.model_dump(),
            reason="ik",
            position_error_m=best[2],
            orientation_error_rad=best[3],
        )

    def _path_clearance(self, q_from: np.ndarray, q_to: np.ndarray) -> tuple[float, float]:
        """(min frame height, min finger-tip height) above the table along the joint path."""
        col = self.params.collision
        frames_min = math.inf
        tips_min = math.inf
        for a in np.linspace(0.0, 1.0, col.path_samples):
            q = q_from + a * (q_to - q_from)
            frames = self.kin.fk_all(q)
            frames_min = min(frames_min, *(float(f[2, 3]) for f in frames))
            tool0 = frames[-1]
            tcp_z = float(tool0[2, 3] + self.params.tcp_offset_m * tool0[2, 2])
            tips_min = min(tips_min, tcp_z - self.params.finger_tip_below_tcp_m)
        return frames_min - col.table_z_m, tips_min - col.table_z_m

    def _duration(self, q_from: np.ndarray, q_to: np.ndarray, speed_scale: float) -> float:
        m = self.params.motion
        dq = float(np.max(np.abs(q_to - q_from)))
        return max(m.min_duration_s, dq / (m.max_joint_speed_rad_s * speed_scale))

    def _snapshot(self) -> JointSnapshot:
        snap = self.backend.snapshot(self.params.motion.timeout_margin_s)
        if snap is None:
            raise Timeout("no joint state from the robot", what="joint_state")
        return snap

    # -- planning ---------------------------------------------------------------------------
    def plan(self, target: Pose, speed_scale: float = 1.0) -> MotionPlan:
        """Validate and solve a move without executing it (what ``move_to`` does first)."""
        m = self.params.motion
        if not (m.speed_scale_min <= speed_scale <= 1.0):
            msg = f"speed_scale must be in [{m.speed_scale_min}, 1], got {speed_scale}"
            raise ValueError(msg)
        snap = self._snapshot()
        q_now = np.asarray(snap.q, dtype=float)
        return self._plan_from(q_now, target, speed_scale)

    def _plan_from(self, q_now: np.ndarray, target: Pose, speed_scale: float) -> MotionPlan:
        col = self.params.collision
        tip_z = target.z - self.params.finger_tip_below_tcp_m
        if tip_z < col.table_z_m + col.tip_clearance_m:
            raise Collision(
                f"finger tips would reach {tip_z:.4f} m, below the table clearance",
                target=target.model_dump(),
                reason="target_below_table",
                tip_z_m=tip_z,
            )
        if not self.params.workspace.contains(target.x, target.y, target.z):
            raise OutOfReach(
                f"{target} is outside the workspace box",
                target=target.model_dump(),
                reason="outside_workspace",
                workspace=self.params.workspace.model_dump(),
            )
        q_target, iters, pos_err, ori_err = self._solve_ik(target, q_now)
        sigma_min = float(np.linalg.svd(self.kin.jacobian(q_target), compute_uv=False)[-1])
        if sigma_min < self.params.ik.min_singular_value:
            raise Singularity(
                f"target configuration is near-singular (sigma_min={sigma_min:.4f})",
                target=target.model_dump(),
                min_singular_value=sigma_min,
            )
        frames_clear, tips_clear = self._path_clearance(q_now, q_target)
        if frames_clear < col.link_clearance_m or tips_clear < col.tip_clearance_m:
            raise Collision(
                "arm or finger tips would pass below the table clearance on the way",
                target=target.model_dump(),
                reason="path",
                frames_clearance_m=frames_clear,
                tips_clearance_m=tips_clear,
            )
        return MotionPlan(
            target=target,
            q_start=_vec6(q_now),
            q_target=_vec6(q_target),
            duration_s=self._duration(q_now, q_target, speed_scale),
            speed_scale=speed_scale,
            ik_iters=iters,
            ik_position_error_m=pos_err,
            ik_orientation_error_rad=ori_err,
            min_singular_value=sigma_min,
            path_min_clearance_m=min(frames_clear, tips_clear),
        )

    # -- primitives -------------------------------------------------------------------------
    def move_to(self, target: Pose, speed_scale: float = 1.0) -> MoveResult:
        """Move the TCP to a top-down pose; raises before moving when the pose is unsafe."""
        plan = self.plan(target, speed_scale)
        return self._execute(plan, "move_to")

    def _execute(self, plan: MotionPlan, primitive: Primitive) -> MoveResult:
        m = self.params.motion
        t_wall = time.time()
        t0 = self.backend.sim_time()
        deadline = plan.duration_s * m.timeout_factor + m.timeout_margin_s
        outcome = self.backend.follow(plan.q_target, plan.duration_s, deadline)
        if outcome is not MotionOutcome.SUCCESS:
            raise Timeout(
                f"controller returned {outcome.value} for the move",
                outcome=outcome.value,
                target=plan.target.model_dump(),
                duration_s=plan.duration_s,
            )
        snap = self._settle()
        q_final = np.asarray(snap.q, dtype=float)
        joint_err = float(np.max(np.abs(q_final - np.asarray(plan.q_target))))
        if joint_err > m.goal_tolerance_rad:
            raise Timeout(
                f"move finished {joint_err:.4f} rad from the goal "
                f"(tolerance {m.goal_tolerance_rad})",
                outcome="goal_tolerance",
                joint_error_rad=joint_err,
                target=plan.target.model_dump(),
            )
        tcp = self.tcp_pose(q_final)
        pos_err = math.dist(tcp.xyz(), plan.target.xyz())
        yaw_err = abs(_wrap_angle(tcp.yaw - plan.target.yaw))
        return MoveResult(
            primitive=primitive,
            t_start_sim=t0,
            t_end_sim=snap.t_sim,
            wall_s=time.time() - t_wall,
            plan=plan,
            q_final=_vec6(q_final),
            tcp_final=tcp,
            position_error_m=pos_err,
            yaw_error_rad=yaw_err,
            joint_error_rad=joint_err,
        )

    def _settle(self) -> JointSnapshot:
        """The controller succeeds on reaching the goal tolerance while the joints may still be
        converging; wait (bounded) until they are at rest before measuring the final pose."""
        m = self.params.motion
        snap = self._snapshot()
        t_end = snap.t_sim + m.settle_timeout_s
        while max(abs(v) for v in snap.qd) > m.settle_qd_rad_s and snap.t_sim < t_end:
            snap = self._snapshot()
        return snap

    def _gripper(self, effort_n: float, primitive: Primitive) -> GripperResult:
        g = self.params.gripper
        t_wall = time.time()
        t0 = self.backend.sim_time()
        self.backend.gripper(effort_n, g.settle_s)
        self._last_gripper_effort = effort_n
        snap = self._snapshot()
        opening = self.opening_m(snap)
        holding = effort_n > 0 and opening >= g.min_object_width_m
        return GripperResult(
            primitive=primitive,
            t_start_sim=t0,
            t_end_sim=snap.t_sim,
            wall_s=time.time() - t_wall,
            opening_m=opening,
            holding=holding,
        )

    def opening_m(self, snap: JointSnapshot) -> float:
        """Gap between the pads: full opening minus both finger strokes."""
        return max(0.0, self.params.gripper.max_opening_m - snap.fingers[0] - snap.fingers[1])

    def grasp(self) -> GripperResult:
        """Close the fingers; ``NoObjectGrasped`` when they meet without an object."""
        res = self._gripper(self.params.gripper.close_effort_n, "grasp")
        if not res.holding:
            raise NoObjectGrasped(
                f"fingers closed to a {res.opening_m * 1000:.1f} mm gap: nothing grasped",
                opening_m=res.opening_m,
            )
        return res

    def release(self) -> GripperResult:
        """Open the fingers fully."""
        return self._gripper(self.params.gripper.open_effort_n, "release")

    def frame(self) -> tuple[np.ndarray, np.ndarray]:
        pair = self.backend.frame(self.params.camera_timeout_s)
        if pair is None:
            raise CameraTimeout(
                f"no RGB-D pair within {self.params.camera_timeout_s} s",
                timeout_s=self.params.camera_timeout_s,
            )
        return pair

    def detect(self, target: str = ANY_TARGET) -> list[Detection]:
        """Cubes of ``target`` colour (or all) in ``base_link`` from a fresh RGB-D frame."""
        rgb, depth = self.frame()
        return list(
            detect_cubes(rgb, depth, target, params=self.perception, camera=self.camera).detections
        )

    def observe(self) -> Observation:
        """Joint state, TCP pose, gripper state and every detected cube."""
        return self.state(tuple(self.detect()))

    def state(self, detections: tuple[Detection, ...] = ()) -> Observation:
        """Joint and gripper state without touching the camera (skill conditions use it)."""
        snap = self._snapshot()
        opening = self.opening_m(snap)
        return Observation(
            t_sim=snap.t_sim,
            q=snap.q,
            qd=snap.qd,
            tcp=self.tcp_pose(snap.q),
            gripper_opening_m=opening,
            holding=self._last_gripper_effort > 0
            and opening >= self.params.gripper.min_object_width_m,
            detections=tuple(detections),
        )

    def execute_skill(self, name: str, **kwargs: object) -> SkillResult:
        """Run a validated skill from the library the robot was built with (F7); a robot
        without a library (agents A and B) always raises ``SkillNotAvailable``."""
        if self.skills is None:
            raise SkillNotAvailable(
                f"skill {name!r} is not available: this robot has no skill library",
                name=name, kwargs=_jsonable_kwargs(kwargs),
            )  # fmt: skip
        return self.skills.execute(self, name, kwargs)

    def home(self, speed_scale: float = 1.0) -> MoveResult:
        """Return to ``ready_pose`` (joint-space, no IK: the configured ``ready_q``)."""
        snap = self._snapshot()
        q_now = np.asarray(snap.q, dtype=float)
        frames_clear, tips_clear = self._path_clearance(q_now, self._ready_q)
        plan = MotionPlan(
            target=self.ready_pose,
            q_start=_vec6(q_now),
            q_target=_vec6(self._ready_q),
            duration_s=self._duration(q_now, self._ready_q, speed_scale),
            speed_scale=speed_scale,
            ik_iters=0,
            ik_position_error_m=0.0,
            ik_orientation_error_rad=0.0,
            min_singular_value=float(
                np.linalg.svd(self.kin.jacobian(self._ready_q), compute_uv=False)[-1]
            ),
            path_min_clearance_m=min(frames_clear, tips_clear),
        )
        return self._execute(plan, "home")

    def reset(self) -> Result:
        """Open the gripper and return to ``ready_pose`` at full speed; never raises
        ``NoObjectGrasped``. The scene (cubes) is the runner's business, not the robot's."""
        t_wall = time.time()
        t0 = self.backend.sim_time()
        self.release()
        moved = self.home()
        return Result(
            primitive="reset",
            t_start_sim=t0,
            t_end_sim=moved.t_end_sim,
            wall_s=time.time() - t_wall,
        )
