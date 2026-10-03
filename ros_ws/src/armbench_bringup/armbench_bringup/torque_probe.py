"""F2 torque-source probe (runs inside the simulation container).

Executes a scripted plan (static holds, multi-point trajectories, repeats of one
trajectory) through the ``joint_trajectory_controller`` and records every
``/joint_states`` message of each segment to ``<out-dir>/<segment>.jsonl`` as
``{"t", "q", "qd", "tau"}`` (sim time, 6 arm joints). ``torque_compare`` then
confronts ``tau`` (Gazebo joint effort) with inverse dynamics.

Plan JSON::

    {"hold_s": 2.0, "settle_s": 1.0, "move_s": 3.0,
     "static_holds": [{"name": "home", "q": [..6..]}, ...],
     "trajectories": [{"name": "traj_00", "points": [{"q": [..6..], "t": 1.5}, ...]}, ...],
     "repeat": {"trajectory": "traj_00", "n": 10}}
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from controller_manager_msgs.srv import ListControllers
from rclpy.action import ActionClient
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
HOME = [0.0, -1.57, 0.0, -1.57, 0.0, 0.0]


def _stamp(sec: int, nanosec: int) -> float:
    return sec + nanosec * 1e-9


class TorqueProbe(Node):
    def __init__(self) -> None:
        super().__init__("armbench_torque_probe")
        self.clock_sim = float("nan")
        self.recording: list[dict] | None = None
        self.last_js: JointState | None = None
        self.create_subscription(Clock, "/clock", self._on_clock, 10)
        self.create_subscription(JointState, "/joint_states", self._on_js, 200)
        self.jtc = ActionClient(
            self, FollowJointTrajectory, "/joint_trajectory_controller/follow_joint_trajectory"
        )
        self.list_ctrl = self.create_client(ListControllers, "/controller_manager/list_controllers")

    def _on_clock(self, msg: Clock) -> None:
        self.clock_sim = _stamp(msg.clock.sec, msg.clock.nanosec)

    def _on_js(self, msg: JointState) -> None:
        self.last_js = msg
        if self.recording is None:
            return
        idx = {n: i for i, n in enumerate(msg.name)}
        if any(j not in idx for j in ARM_JOINTS) or len(msg.effort) != len(msg.name):
            return
        sel = [idx[j] for j in ARM_JOINTS]
        self.recording.append(
            {
                "t": _stamp(msg.header.stamp.sec, msg.header.stamp.nanosec),
                "q": [msg.position[i] for i in sel],
                "qd": [msg.velocity[i] for i in sel],
                "tau": [msg.effort[i] for i in sel],
            }
        )

    # ------------------------------------------------------------------ helpers
    def spin_for(self, seconds: float) -> None:
        end = time.time() + seconds
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.02)

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
        return {"joint_state_broadcaster", "joint_trajectory_controller"} <= active

    def execute(self, points: list[tuple[list[float], float]]) -> bool:
        goal = FollowJointTrajectory.Goal()
        traj = JointTrajectory()
        traj.joint_names = ARM_JOINTS
        for q, t in points:
            pt = JointTrajectoryPoint()
            pt.positions = list(q)
            pt.time_from_start = Duration(sec=int(t), nanosec=int((t % 1) * 1e9))
            traj.points.append(pt)
        goal.trajectory = traj
        if not self.jtc.wait_for_server(timeout_sec=10.0):
            return False
        send = self.jtc.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send, timeout_sec=10.0)
        handle = send.result()
        if handle is None or not handle.accepted:
            return False
        res_fut = handle.get_result_async()
        deadline = time.time() + points[-1][1] * 4 + 15
        while not res_fut.done() and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.02)
        if not res_fut.done():
            return False
        return res_fut.result().result.error_code == FollowJointTrajectory.Result.SUCCESSFUL

    def move(self, q: list[float], duration_s: float) -> bool:
        return self.execute([(q, duration_s)])

    def record(self, out: Path, action) -> dict:  # noqa: ANN001
        """Run ``action`` while buffering joint states; write them to ``out``."""
        self.recording = []
        t_wall0, t_sim0 = time.time(), self.clock_sim
        ok = action()
        samples, self.recording = self.recording, None
        with out.open("w") as fh:
            for s in samples:
                fh.write(json.dumps(s) + "\n")
        wall = time.time() - t_wall0
        sim = self.clock_sim - t_sim0
        return {
            "file": out.name,
            "ok": bool(ok),
            "n_samples": len(samples),
            "sim_s": round(sim, 3),
            "wall_s": round(wall, 3),
            "rtf": round(sim / wall, 3) if wall > 0 else None,
            "js_rate_hz": round(len(samples) / sim, 1) if sim > 0 else None,
        }


def run(args: argparse.Namespace) -> dict:
    plan = json.loads(Path(args.plan).read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hold_s = float(plan.get("hold_s", 2.0))
    settle_s = float(plan.get("settle_s", 1.0))
    move_s = float(plan.get("move_s", 3.0))
    rclpy.init()
    node = TorqueProbe()
    report: dict = {"segments": [], "cpu_samples": []}
    try:
        ok = node.wait_until(node.controllers_active, args.timeout, "controllers active")
        ok = ok and node.wait_until(lambda: node.last_js is not None, 20, "joint_states")
        report["ready"] = ok
        if not ok:
            return report
        node.move(HOME, move_s)
        for hold in plan.get("static_holds", []):
            node.move(hold["q"], move_s)
            node.spin_for(settle_s)
            out = out_dir / f"static_{hold['name']}.jsonl"
            seg = node.record(out, lambda: node.spin_for(hold_s))
            seg.update({"kind": "static", "name": hold["name"], "q": hold["q"]})
            report["segments"].append(seg)
        trajs = {t["name"]: t for t in plan.get("trajectories", [])}
        order = [(t["name"], "trajectory") for t in plan.get("trajectories", [])]
        rep = plan.get("repeat")
        if rep:
            order += [(rep["trajectory"], f"rep_{i:02d}") for i in range(int(rep["n"]))]
        for name, tag in order:
            traj = trajs[name]
            points = [(p["q"], float(p["t"])) for p in traj["points"]]
            node.move(points[0][0], move_s)
            node.spin_for(settle_s)
            rest = points[1:]
            fname = f"{name}.jsonl" if tag == "trajectory" else f"{tag}_{name}.jsonl"

            def _go(rest: list = rest) -> bool:
                done = node.execute(rest)
                node.spin_for(settle_s)
                return done

            seg = node.record(out_dir / fname, _go)
            seg.update({"kind": tag if tag != "trajectory" else "trajectory", "name": name})
            report["segments"].append(seg)
            if args.cpu_pid:
                report["cpu_samples"].append(_cpu_percent(args.cpu_pid))
        node.move(HOME, move_s)
        report["all_ok"] = all(s["ok"] for s in report["segments"])
        return report
    finally:
        node.destroy_node()
        rclpy.shutdown()


def _cpu_percent(pid: int) -> float | None:
    """Lifetime CPU share of ``pid`` from /proc (single-core percent)."""
    try:
        with open(f"/proc/{pid}/stat") as fh:
            fields = fh.read().split(")")[-1].split()
        with open("/proc/uptime") as fh:
            uptime = float(fh.read().split()[0])
    except OSError:
        return None
    hz = os.sysconf("SC_CLK_TCK")
    total = (int(fields[11]) + int(fields[12])) / hz
    start = int(fields[19]) / hz
    elapsed = uptime - start
    return round(100.0 * total / elapsed, 2) if elapsed > 0 else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--out-dir", default="/tmp/f2_torque")  # noqa: S108
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--cpu-pid", type=int, default=None, help="pid of energy_meter to sample")
    args = parser.parse_args()
    report = run(args)
    Path(args.out_dir, "probe.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))  # noqa: T201
    raise SystemExit(0 if report.get("all_ok") else 1)


if __name__ == "__main__":
    main()
