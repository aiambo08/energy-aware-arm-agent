"""F1 simulation self-check (runs inside the simulation container).

Verifies, against a running ``sim.launch.py`` with ``check_scene.sdf``:

* controllers become active (time-to-ready);
* ``/joint_states`` carries position/velocity/effort for the 6 arm joints + 2 fingers;
* the RGB-D camera publishes colour and depth images (FPS in sim time and wall time);
* real-time factor while the arm moves;
* every arm joint tracks a trajectory; the gripper opens and closes;
* a fixed-waypoint pick-and-place of ``cube_0`` lifts the cube and transports it with
  slip <= 5 mm (ground-truth pose from Gazebo's PosePublisher on the cube model).

Writes a JSON report; the host script ``scripts/check_sim.py`` aggregates many runs.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field

import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from controller_manager_msgs.srv import ListControllers
from geometry_msgs.msg import PoseStamped
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64MultiArray
from tf2_msgs.msg import TFMessage
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
FINGER_JOINTS = ["left_finger_joint", "right_finger_joint"]
CONTROLLERS = ["joint_state_broadcaster", "joint_trajectory_controller", "gripper_controller"]


@dataclass
class Pose3:
    x: float
    y: float
    z: float

    def dist(self, other: Pose3) -> float:
        return math.sqrt(
            (self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2
        )

    def as_list(self) -> list[float]:
        return [round(self.x, 5), round(self.y, 5), round(self.z, 5)]


@dataclass
class Samples:
    clock_sim: list[float] = field(default_factory=list)
    clock_wall: list[float] = field(default_factory=list)
    rgb_stamps_sim: list[float] = field(default_factory=list)
    rgb_stamps_wall: list[float] = field(default_factory=list)
    depth_stamps_sim: list[float] = field(default_factory=list)
    rgb_shape: tuple[int, int, str] | None = None
    depth_shape: tuple[int, int, str] | None = None
    joint_state: JointState | None = None
    cube_pose: Pose3 | None = None
    tcp_pose: Pose3 | None = None


class SimCheck(Node):
    def __init__(self, cfg: dict, cube_name: str) -> None:
        super().__init__("armbench_sim_check")
        self.cfg = cfg
        self.cube_name = cube_name
        self.s = Samples()
        self.t_start_wall = time.time()
        sensor_qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Clock, "/clock", self._on_clock, sensor_qos)
        self.create_subscription(JointState, "/joint_states", self._on_js, 10)
        self.create_subscription(Image, "/camera/image", self._on_rgb, sensor_qos)
        self.create_subscription(Image, "/camera/depth_image", self._on_depth, sensor_qos)
        self.create_subscription(PoseStamped, f"/model/{cube_name}/pose", self._on_gz_pose, 10)
        self.create_subscription(TFMessage, "/tf", self._on_tf, 50)
        static_qos = QoSProfile(depth=100, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(TFMessage, "/tf_static", self._on_tf, static_qos)
        self.gripper_pub = self.create_publisher(
            Float64MultiArray, "/gripper_controller/commands", 10
        )
        self.jtc = ActionClient(
            self, FollowJointTrajectory, "/joint_trajectory_controller/follow_joint_trajectory"
        )
        self.list_ctrl = self.create_client(ListControllers, "/controller_manager/list_controllers")
        self._tf_chain: dict[str, tuple[str, Pose3, tuple[float, float, float, float]]] = {}

    # ---------------------------------------------------------------- callbacks
    def _on_clock(self, msg: Clock) -> None:
        self.s.clock_sim.append(msg.clock.sec + msg.clock.nanosec * 1e-9)
        self.s.clock_wall.append(time.time())

    def _on_js(self, msg: JointState) -> None:
        self.s.joint_state = msg

    def _on_rgb(self, msg: Image) -> None:
        self.s.rgb_stamps_sim.append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
        self.s.rgb_stamps_wall.append(time.time())
        self.s.rgb_shape = (msg.width, msg.height, msg.encoding)

    def _on_depth(self, msg: Image) -> None:
        self.s.depth_stamps_sim.append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
        self.s.depth_shape = (msg.width, msg.height, msg.encoding)

    def _on_gz_pose(self, msg: PoseStamped) -> None:
        p = msg.pose.position
        self.s.cube_pose = Pose3(p.x, p.y, p.z)

    def _on_tf(self, msg: TFMessage) -> None:
        for t in msg.transforms:
            p = t.transform.translation
            q = t.transform.rotation
            self._tf_chain[t.child_frame_id] = (
                t.header.frame_id,
                Pose3(p.x, p.y, p.z),
                (q.x, q.y, q.z, q.w),
            )

    # ---------------------------------------------------------------- helpers
    def spin_for(self, seconds: float) -> None:
        end = time.time() + seconds
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def sim_now(self) -> float:
        return self.s.clock_sim[-1] if self.s.clock_sim else float("nan")

    def wait_until(self, pred, timeout: float, what: str) -> bool:  # noqa: ANN001
        end = time.time() + timeout
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
            if pred():
                return True
        self.get_logger().error(f"timeout waiting for {what}")
        return False

    def controllers_active(self) -> bool:
        if not self.list_ctrl.service_is_ready():
            return False
        fut = self.list_ctrl.call_async(ListControllers.Request())
        rclpy.spin_until_future_complete(self, fut, timeout_sec=2.0)
        res = fut.result()
        if res is None:
            return False
        active = {c.name for c in res.controller if c.state == "active"}
        return all(name in active for name in CONTROLLERS)

    def joint_positions(self, names: list[str]) -> list[float] | None:
        js = self.s.joint_state
        if js is None:
            return None
        idx = {n: i for i, n in enumerate(js.name)}
        if any(n not in idx for n in names):
            return None
        return [js.position[idx[n]] for n in names]

    def move(self, q: list[float], duration_s: float) -> tuple[bool, float]:
        """Send a single-point trajectory and wait for the result; returns (ok, final_error)."""
        goal = FollowJointTrajectory.Goal()
        traj = JointTrajectory()
        traj.joint_names = ARM_JOINTS
        pt = JointTrajectoryPoint()
        pt.positions = list(q)
        pt.time_from_start = Duration(sec=int(duration_s), nanosec=int((duration_s % 1) * 1e9))
        traj.points = [pt]
        goal.trajectory = traj
        if not self.jtc.wait_for_server(timeout_sec=10.0):
            return False, float("nan")
        send = self.jtc.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send, timeout_sec=10.0)
        handle = send.result()
        if handle is None or not handle.accepted:
            return False, float("nan")
        res_fut = handle.get_result_async()
        deadline = time.time() + duration_s * 4 + 15
        while not res_fut.done() and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
        if not res_fut.done():
            return False, float("nan")
        err_code = res_fut.result().result.error_code
        self.spin_for(0.3)
        cur = self.joint_positions(ARM_JOINTS)
        err = max(abs(a - b) for a, b in zip(cur, q, strict=True)) if cur else float("nan")
        return err_code == FollowJointTrajectory.Result.SUCCESSFUL, err

    def gripper(self, effort_n: float, settle_s: float) -> list[float] | None:
        msg = Float64MultiArray()
        msg.data = [effort_n, effort_n]
        end = time.time() + settle_s
        while time.time() < end:
            self.gripper_pub.publish(msg)
            self.spin_for(0.1)
        return self.joint_positions(FINGER_JOINTS)

    def tcp_pose_world(self) -> Pose3 | None:
        # Compose world -> ... -> tcp from the latest /tf (+/tf_static) frames.
        def quat_to_mat(q):  # noqa: ANN001, ANN202
            x, y, z, w = q
            return np.array(
                [
                    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                ]
            )

        frame = "tcp"
        world_t_frame = np.eye(4)
        for _ in range(32):
            if frame not in self._tf_chain:
                return None
            parent, p, q = self._tf_chain[frame]
            parent_t_child = np.eye(4)
            parent_t_child[:3, :3] = quat_to_mat(q)
            parent_t_child[:3, 3] = [p.x, p.y, p.z]
            world_t_frame = parent_t_child @ world_t_frame
            frame = parent
            if frame in ("world", "base_link"):
                break
        else:
            return None
        return Pose3(
            float(world_t_frame[0, 3]), float(world_t_frame[1, 3]), float(world_t_frame[2, 3])
        )


def rate_hz(stamps: list[float], window_start: float) -> float | None:
    pts = [t for t in stamps if t >= window_start]
    if len(pts) < 3:
        return None
    return (len(pts) - 1) / (pts[-1] - pts[0]) if pts[-1] > pts[0] else None


def run(args: argparse.Namespace) -> dict:  # noqa: PLR0915 - linear checklist
    share = get_package_share_directory("armbench_bringup")
    with open(f"{share}/config/sim_check.yaml", "rb") as fh:
        cfg = yaml.safe_load(fh)
    rclpy.init()
    node = SimCheck(cfg, cfg["cube_name"])
    report: dict = {"launch_start_epoch": args.launch_start, "node_start_epoch": node.t_start_wall}
    try:
        # 1. simulation clock
        ok_clock = node.wait_until(lambda: len(node.s.clock_sim) > 0, args.timeout, "/clock")
        report["clock_ok"] = ok_clock
        # 2. controllers
        ok_ctrl = node.wait_until(node.controllers_active, args.timeout, "controllers active")
        report["controllers_active"] = ok_ctrl
        report["ready_epoch"] = time.time()
        report["ready_s_from_launch"] = (
            round(report["ready_epoch"] - args.launch_start, 2) if args.launch_start else None
        )
        if not (ok_clock and ok_ctrl):
            report["passed"] = False
            return report
        # 3. joint states with effort
        ok_js = node.wait_until(
            lambda: node.joint_positions(ARM_JOINTS + FINGER_JOINTS) is not None, 20, "joint_states"
        )
        js = node.s.joint_state
        report["joint_states"] = {
            "ok": ok_js,
            "names": list(js.name) if js else [],
            "effort_len": len(js.effort) if js else 0,
            "effort_for_all_joints": bool(js and len(js.effort) == len(js.name) == 8),
        }
        # 4. camera FPS + RTF measured over an 8 s wall window while idle
        t_wall0 = time.time()
        t_sim0 = node.sim_now()
        node.spin_for(8.0)
        report["camera"] = {
            "rgb_shape": node.s.rgb_shape,
            "depth_shape": node.s.depth_shape,
            "rgb_fps_sim": rate_hz(node.s.rgb_stamps_sim, t_sim0),
            "rgb_fps_wall": rate_hz(node.s.rgb_stamps_wall, t_wall0),
            "depth_fps_sim": rate_hz(node.s.depth_stamps_sim, t_sim0),
        }
        report["rtf_idle"] = round((node.sim_now() - t_sim0) / (time.time() - t_wall0), 3)
        # 5. joint sweep: each joint +0.4 rad and back, 2 s per leg
        home = cfg["home"]
        sweep = []
        t_wall0 = time.time()
        t_sim0 = node.sim_now()
        ok_home, _ = node.move(home, 3.0)
        for i, name in enumerate(ARM_JOINTS):
            q = list(home)
            q[i] += 0.4
            ok1, e1 = node.move(q, 2.0)
            ok2, e2 = node.move(home, 2.0)
            sweep.append({"joint": name, "ok": ok1 and ok2, "max_err_rad": round(max(e1, e2), 4)})
        report["rtf_moving"] = round((node.sim_now() - t_sim0) / (time.time() - t_wall0), 3)
        report["joint_sweep"] = {
            "home_ok": ok_home,
            "joints": sweep,
            "ok": all(s["ok"] for s in sweep),
        }
        # 6. gripper open/close without object
        g = cfg["gripper"]
        closed = node.gripper(g["close_effort_n"], 2.0)
        opened = node.gripper(g["open_effort_n"], 2.0)
        stroke = g["stroke_m"]
        report["gripper"] = {
            "closed_pos": closed,
            "opened_pos": opened,
            "ok": bool(
                closed
                and opened
                and all(abs(p - stroke) < 0.004 for p in closed)
                and all(abs(p) < 0.004 for p in opened)
            ),
        }
        # 7. pick and place of cube_0 with fixed waypoints
        grasp: dict = {"ok": False}
        report["grasp"] = grasp
        if not args.no_grasp:
            node.wait_until(lambda: node.s.cube_pose is not None, 10, "cube pose")
            start = node.s.cube_pose
            grasp["cube_start"] = start.as_list() if start else None
            ok_a, _ = node.move(cfg["pregrasp"], 4.0)
            ok_b, _ = node.move(cfg["grasp"], 2.5)
            fingers = node.gripper(g["close_effort_n"], 2.0)
            grasp["finger_pos_closed_on_cube"] = fingers
            expected = (stroke - g["cube_size_m"] / 2.0) + (g.get("finger_offset_m", 0.0))
            grasp["fingers_stopped_by_cube"] = bool(
                fingers and all(0.5 * expected < p < stroke - 0.001 for p in fingers)
            )
            ok_c, _ = node.move(cfg["pregrasp"], 2.5)
            node.spin_for(0.5)
            lifted = node.s.cube_pose
            tcp_lift = node.tcp_pose_world()
            grasp["cube_after_lift"] = lifted.as_list() if lifted else None
            grasp["lift_height_m"] = round(lifted.z - start.z, 4) if (lifted and start) else None
            ok_d, _ = node.move(cfg["transport"], 3.0)
            node.spin_for(0.5)
            moved = node.s.cube_pose
            tcp_moved = node.tcp_pose_world()
            slip = None
            if lifted and moved and tcp_lift and tcp_moved:
                rel0 = (lifted.x - tcp_lift.x, lifted.y - tcp_lift.y, lifted.z - tcp_lift.z)
                rel1 = (moved.x - tcp_moved.x, moved.y - tcp_moved.y, moved.z - tcp_moved.z)
                slip = math.sqrt(sum((a - b) ** 2 for a, b in zip(rel0, rel1, strict=True)))
            grasp["transport_slip_m"] = round(slip, 5) if slip is not None else None
            grasp["cube_after_transport"] = moved.as_list() if moved else None
            ok_e, _ = node.move(cfg["place"], 2.5)
            node.gripper(g["open_effort_n"], 1.5)
            ok_f, _ = node.move(cfg["pregrasp"], 3.0)
            node.spin_for(1.0)
            placed = node.s.cube_pose
            grasp["cube_after_place"] = placed.as_list() if placed else None
            grasp["moves_ok"] = all([ok_a, ok_b, ok_c, ok_d, ok_e, ok_f])
            grasp["ok"] = bool(
                grasp["moves_ok"]
                and grasp["lift_height_m"] is not None
                and grasp["lift_height_m"] > 0.08
                and slip is not None
                and slip <= 0.005
                and placed is not None
                and abs(placed.z - start.z) < 0.01
                and abs(placed.y - 0.18) < 0.03
            )
        report["sim_time_s"] = round(node.sim_now(), 2)
        report["wall_s"] = round(time.time() - node.t_start_wall, 2)
        report["passed"] = bool(
            ok_js
            and report["joint_states"]["effort_for_all_joints"]
            and report["joint_sweep"]["ok"]
            and report["gripper"]["ok"]
            and (args.no_grasp or grasp["ok"])
        )
        return report
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="/tmp/sim_check.json")  # noqa: S108
    parser.add_argument("--timeout", type=float, default=60.0, help="seconds to wait for readiness")
    parser.add_argument("--launch-start", type=float, default=None, help="epoch of `docker run`")
    parser.add_argument("--no-grasp", action="store_true")
    args = parser.parse_args()
    report = run(args)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))  # noqa: T201
    raise SystemExit(0 if report.get("passed") else 1)


if __name__ == "__main__":
    main()
