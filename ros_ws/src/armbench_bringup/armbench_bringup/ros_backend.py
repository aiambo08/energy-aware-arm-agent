"""``armbench.primitives.Backend`` on the running simulation (rclpy), plus the Gazebo scene
helper used by the gates and the runner.

* arm: ``/joint_trajectory_controller/follow_joint_trajectory`` (single-waypoint goals),
* gripper: ``/gripper_controller/commands`` (effort on both fingers),
* camera: ``/camera/image`` + ``/camera/depth_image`` (fresh pair stamped after the call),
* clearance monitor: forward kinematics on every ``/joint_states`` message records the lowest
  arm frame and finger-tip height since the last ``reset_monitor()`` (the simulation has no
  table contact sensor; this is the F4 "0 table collisions" evidence).
"""

from __future__ import annotations

import math
import subprocess
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PoseStamped
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from armbench.kinematics import JOINT_NAMES, UR5eModel
from armbench.primitives import JointSnapshot, MotionOutcome, PrimitiveParams
from armbench.scene import Box, Cube, Scene, box_to_sdf_model, cube_to_box
from armbench.scene.generator import SceneConfig

FINGER_JOINTS = ("left_finger_joint", "right_finger_joint")
MAX_CUBES = 6
MODEL_NAMES = (*[f"cube_{i}" for i in range(MAX_CUBES)], "obstacle")
"""Models whose ``/model/<name>/pose`` is bridged in sim.launch.py."""
GZ_TIMEOUT_MS = 5000
GZ_ATTEMPTS = 3  # the gz CLI reply is lost now and then while the world steps
CLI_TIMEOUT_S = 90.0
SPIN_S = 0.01


def stamp_s(sec: int, nanosec: int) -> float:
    return sec + nanosec * 1e-9


def yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


@dataclass
class Frame:
    t: float
    data: np.ndarray


@dataclass
class CubePose:
    t: float
    x: float
    y: float
    z: float
    yaw: float


@dataclass
class ClearanceMonitor:
    """Lowest heights seen since ``reset`` (metres above ``base_link`` z = 0 = table top)."""

    min_frame_z: float = math.inf
    min_tip_z: float = math.inf
    n_samples: int = 0

    def reset(self) -> None:
        self.min_frame_z = math.inf
        self.min_tip_z = math.inf
        self.n_samples = 0


@dataclass
class _State:
    clock: float = float("nan")
    js: JointSnapshot | None = None
    js_seq: int = 0
    rgb: Frame | None = None
    depth: Frame | None = None
    poses: dict[str, CubePose] = field(default_factory=dict)


class RosBackend(Node):
    def __init__(self, params: PrimitiveParams, *, kinematics: UR5eModel | None = None) -> None:
        super().__init__("armbench_robot")
        self.params = params
        self.kin = kinematics if kinematics is not None else UR5eModel()
        self.monitor = ClearanceMonitor()
        self.follow_count = 0
        self.s = _State()
        self._js_index: list[int] | None = None
        self._finger_index: list[int] | None = None
        self.create_subscription(Clock, "/clock", self._on_clock, qos_profile_sensor_data)
        self.create_subscription(JointState, "/joint_states", self._on_js, 10)
        self.create_subscription(Image, "/camera/image", self._on_rgb, qos_profile_sensor_data)
        self.create_subscription(
            Image, "/camera/depth_image", self._on_depth, qos_profile_sensor_data
        )
        for name in MODEL_NAMES:
            self.create_subscription(
                PoseStamped,
                f"/model/{name}/pose",
                lambda msg, name=name: self._on_pose(name, msg),  # type: ignore[misc]
                10,
            )
        self.gripper_pub = self.create_publisher(
            Float64MultiArray, "/gripper_controller/commands", 10
        )
        self.jtc = ActionClient(
            self, FollowJointTrajectory, "/joint_trajectory_controller/follow_joint_trajectory"
        )

    # -- callbacks -------------------------------------------------------------------------
    def _on_clock(self, msg: Clock) -> None:
        self.s.clock = stamp_s(msg.clock.sec, msg.clock.nanosec)

    def _on_js(self, msg: JointState) -> None:
        if self._js_index is None:
            names = list(msg.name)
            if not all(n in names for n in (*JOINT_NAMES, *FINGER_JOINTS)):
                return
            self._js_index = [names.index(n) for n in JOINT_NAMES]
            self._finger_index = [names.index(n) for n in FINGER_JOINTS]
        if self._finger_index is None:
            return
        q = tuple(float(msg.position[i]) for i in self._js_index)
        qd = tuple(float(msg.velocity[i]) for i in self._js_index) if msg.velocity else (0.0,) * 6
        fingers = tuple(float(msg.position[i]) for i in self._finger_index)
        self.s.js = JointSnapshot(
            t_sim=stamp_s(msg.header.stamp.sec, msg.header.stamp.nanosec),
            q=q,  # type: ignore[arg-type]
            qd=qd,  # type: ignore[arg-type]
            fingers=fingers,  # type: ignore[arg-type]
        )
        self.s.js_seq += 1
        frames = self.kin.fk_all(q)
        tool0 = frames[-1]
        tcp_z = float(tool0[2, 3] + self.params.tcp_offset_m * tool0[2, 2])
        m = self.monitor
        m.min_frame_z = min(m.min_frame_z, *(float(f[2, 3]) for f in frames))
        m.min_tip_z = min(m.min_tip_z, tcp_z - self.params.finger_tip_below_tcp_m)
        m.n_samples += 1

    def _on_rgb(self, msg: Image) -> None:
        if msg.encoding != "rgb8":
            self.get_logger().error(f"unexpected RGB encoding {msg.encoding}")
            return
        arr = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(msg.height, msg.width, 3)
        self.s.rgb = Frame(stamp_s(msg.header.stamp.sec, msg.header.stamp.nanosec), arr.copy())

    def _on_depth(self, msg: Image) -> None:
        if msg.encoding != "32FC1":
            self.get_logger().error(f"unexpected depth encoding {msg.encoding}")
            return
        arr = np.frombuffer(bytes(msg.data), dtype=np.float32).reshape(msg.height, msg.width)
        self.s.depth = Frame(stamp_s(msg.header.stamp.sec, msg.header.stamp.nanosec), arr.copy())

    def _on_pose(self, name: str, msg: PoseStamped) -> None:
        p, q = msg.pose.position, msg.pose.orientation
        self.s.poses[name] = CubePose(
            stamp_s(msg.header.stamp.sec, msg.header.stamp.nanosec),
            p.x,
            p.y,
            p.z,
            yaw_from_quat(q.x, q.y, q.z, q.w),
        )

    # -- spinning helpers ------------------------------------------------------------------
    def wait_until(self, pred: Callable[[], bool], timeout_s: float) -> bool:
        deadline = time.time() + timeout_s
        while True:
            rclpy.spin_once(self, timeout_sec=SPIN_S)
            if pred():
                return True
            if time.time() >= deadline:
                return False

    def spin_sim(self, seconds: float) -> None:
        """Spin for ``seconds`` of simulated time (wall guard: 4x + 5 s)."""
        end = self.s.clock + seconds
        guard = time.time() + 4.0 * seconds + 5.0
        while self.s.clock < end and time.time() < guard:
            rclpy.spin_once(self, timeout_sec=SPIN_S)

    def ready(self, timeout_s: float) -> bool:
        """Clock, joint states, camera and trajectory action server all up."""
        if not self.wait_until(
            lambda: not math.isnan(self.s.clock) and self.s.js is not None, timeout_s
        ):
            return False
        if not self.jtc.wait_for_server(timeout_sec=timeout_s):
            return False
        return self.frame(timeout_s) is not None

    # -- Backend protocol ------------------------------------------------------------------
    def sim_time(self) -> float:
        rclpy.spin_once(self, timeout_sec=0.0)
        return self.s.clock

    def snapshot(self, timeout_s: float) -> JointSnapshot | None:
        seq = self.s.js_seq
        if not self.wait_until(lambda: self.s.js_seq > seq, timeout_s):
            return None
        return self.s.js

    def follow(self, q: Sequence[float], duration_s: float, timeout_s: float) -> MotionOutcome:
        self.follow_count += 1
        goal = FollowJointTrajectory.Goal()
        traj = JointTrajectory()
        traj.joint_names = list(JOINT_NAMES)
        pt = JointTrajectoryPoint()
        pt.positions = [float(v) for v in q]
        pt.time_from_start = Duration(sec=int(duration_s), nanosec=int((duration_s % 1) * 1e9))
        traj.points = [pt]
        goal.trajectory = traj
        if not self.jtc.wait_for_server(timeout_sec=min(timeout_s, 10.0)):
            return MotionOutcome.REJECTED
        deadline = time.time() + timeout_s
        send = self.jtc.send_goal_async(goal)
        if not self.wait_until(send.done, deadline - time.time()):
            return MotionOutcome.TIMEOUT
        handle = send.result()
        if handle is None or not handle.accepted:
            return MotionOutcome.REJECTED
        result_future = handle.get_result_async()
        if not self.wait_until(result_future.done, deadline - time.time()):
            handle.cancel_goal_async()
            self.wait_until(result_future.done, 5.0)
            return MotionOutcome.TIMEOUT
        wrapped = result_future.result()
        succeeded = (
            wrapped is not None
            and wrapped.status == GoalStatus.STATUS_SUCCEEDED
            and wrapped.result.error_code == FollowJointTrajectory.Result.SUCCESSFUL
        )
        return MotionOutcome.SUCCESS if succeeded else MotionOutcome.ABORTED

    def gripper(self, effort_n: float, settle_s: float) -> None:
        msg = Float64MultiArray()
        msg.data = [float(effort_n), float(effort_n)]
        end = self.s.clock + settle_s
        guard = time.time() + 4.0 * settle_s + 5.0
        while self.s.clock < end and time.time() < guard:
            self.gripper_pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=SPIN_S)
            self.spin_sim(0.05)

    def frame(self, timeout_s: float) -> tuple[np.ndarray, np.ndarray] | None:
        rclpy.spin_once(self, timeout_sec=0.0)
        t_after = self.s.clock

        def fresh() -> bool:
            r, d = self.s.rgb, self.s.depth
            return r is not None and d is not None and r.t >= t_after and abs(r.t - d.t) < 1e-6

        if not self.wait_until(fresh, timeout_s):
            return None
        r, d = self.s.rgb, self.s.depth
        if r is None or d is None:
            return None
        return r.data, d.data

    # -- extra observations for gates -----------------------------------------------------
    def reset_monitor(self) -> None:
        self.monitor.reset()

    def cube_poses(self) -> dict[str, CubePose]:
        rclpy.spin_once(self, timeout_sec=0.0)
        return dict(self.s.poses)


class GzScene:
    """Spawn/remove the cubes of a seeded scene through ``gz service`` (hard timeouts)."""

    def __init__(self, backend: RosBackend, world: str, config: SceneConfig) -> None:
        self.backend = backend
        self.world = world
        self.config = config
        self.sdf_dir = Path(tempfile.mkdtemp(prefix="armbench_sdf_"))
        self.leftover: set[str] = set()
        self.log = backend.get_logger()

    def _gz(self, cmd: list[str], what: str, done: Callable[[], bool]) -> bool:
        """Run a gz service call; ``done`` tells whether the world already applied it."""
        for attempt in range(1, GZ_ATTEMPTS + 1):
            proc = subprocess.run(  # noqa: S603 - fixed executable, scene-generated arguments
                cmd, capture_output=True, text=True, timeout=CLI_TIMEOUT_S, check=False
            )
            if proc.returncode == 0 and "data: true" in proc.stdout:
                return True
            self.backend.spin_sim(0.2)
            if done():
                return True
            self.log.warning(
                f"{what} attempt {attempt}/{GZ_ATTEMPTS} failed: "
                f"{proc.stdout[-200:]} {proc.stderr[-200:]}"
            )
        self.log.error(f"{what} failed after {GZ_ATTEMPTS} attempts")
        return False

    def spawn(self, model: Cube | Box) -> bool:
        box = cube_to_box(model, self.config) if isinstance(model, Cube) else model
        sdf_path = self.sdf_dir / f"{box.name}.sdf"
        sdf_path.write_text(
            '<?xml version="1.0"?><sdf version="1.8">' + box_to_sdf_model(box) + "</sdf>"
        )
        cmd = [
            "gz", "service", "-s", f"/world/{self.world}/create",
            "--reqtype", "gz.msgs.EntityFactory", "--reptype", "gz.msgs.Boolean",
            "--timeout", str(GZ_TIMEOUT_MS),
            "--req", f'sdf_filename: "{sdf_path}" name: "{box.name}"',
        ]  # fmt: skip
        return self._gz(cmd, f"spawn {box.name}", lambda: box.name in self.backend.s.poses)

    def remove(self, name: str) -> bool:
        cmd = [
            "gz", "service", "-s", f"/world/{self.world}/remove",
            "--reqtype", "gz.msgs.Entity", "--reptype", "gz.msgs.Boolean",
            "--timeout", str(GZ_TIMEOUT_MS), "--req", f'name: "{name}" type: MODEL',
        ]  # fmt: skip

        def gone() -> bool:  # a live model keeps publishing its pose
            p = self.backend.s.poses.get(name)
            return p is None or p.t < self.backend.s.clock - 0.1

        return self._gz(cmd, f"remove {name}", gone)

    def place(self, scene: Scene, obstacles: Sequence[Box] = ()) -> str | None:
        """Clear leftovers, spawn every cube and obstacle; returns an error tag or None."""
        self.leftover = {name for name in self.leftover if not self.remove(name)}
        if self.leftover:
            return f"leftover: {sorted(self.leftover)}"
        models: list[Cube | Box] = [*scene.cubes, *obstacles]
        for m in models:
            self.backend.s.poses.pop(m.name, None)
        if not all(self.spawn(m) for m in models):
            return "spawn"
        return None

    def settled(
        self, scene: Scene, t_after: float, obstacles: Sequence[Box] = (), tol_m: float = 0.005
    ) -> bool:
        for m in (*scene.cubes, *obstacles):
            p = self.backend.s.poses.get(m.name)
            if p is None or p.t < t_after:
                return False
            if max(abs(p.x - m.x), abs(p.y - m.y), abs(p.z - m.z)) > tol_m:
                return False
        return True

    def clear(self, scene: Scene, obstacles: Sequence[Box] = ()) -> bool:
        names = [m.name for m in (*scene.cubes, *obstacles)]
        removed = [self.remove(n) for n in names]
        self.leftover |= {n for n, ok in zip(names, removed, strict=True) if not ok}
        self.backend.spin_sim(0.2)
        for n in names:
            self.backend.s.poses.pop(n, None)
        return all(removed)
