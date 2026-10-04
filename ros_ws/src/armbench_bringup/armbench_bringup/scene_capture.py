"""``scene_capture`` node: RGB-D frames with ground truth for seeded scenes (F3 dataset).

For every seed it generates the scene with ``armbench.scene``, spawns the cubes into the running
world (gz ``EntityFactory`` service), waits until Gazebo reports every cube at rest at its
requested pose plus ``--settle-s`` of simulated time, records one RGB + depth pair rendered
*after* that instant (matching header stamps), then removes the cubes (``/world/<name>/remove``).
Output per seed:
``seed_NNNN/{rgb.png, depth.npy, meta.json}`` where ``meta.json`` carries ``CameraInfo.K`` and the
PosePublisher ground truth of each cube; ``capture_summary.json`` lists timings and failures.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, Image

from armbench.scene import Cube, Scene, cube_to_sdf_model, generate_scene, load_scene_config
from armbench.scene.generator import DEFAULT_SCENE_FILE, SceneConfig

MAX_CUBES = 6  # /model/cube_{0..5}/pose are bridged in sim.launch.py
POSE_TOL_M = 0.005
GZ_TIMEOUT_MS = 5000
CLI_TIMEOUT_S = 90.0  # ros2 / gz CLI start-up is slow when the host is loaded


def stamp_s(sec: int, nanosec: int) -> float:
    return sec + nanosec * 1e-9


def yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


@dataclass
class CubePose:
    t: float
    x: float
    y: float
    z: float
    yaw: float


@dataclass
class Frame:
    t: float
    data: np.ndarray


@dataclass
class State:
    clock: float = float("nan")
    k: tuple[float, ...] | None = None
    size: tuple[int, int] | None = None
    rgb: Frame | None = None
    depth: Frame | None = None
    poses: dict[str, CubePose] = field(default_factory=dict)


class SceneCaptureNode(Node):
    def __init__(self, world: str) -> None:
        super().__init__("armbench_scene_capture")
        self.world = world
        self.sdf_dir = Path(tempfile.mkdtemp(prefix="armbench_sdf_"))
        self.leftover: set[str] = set()  # cubes whose removal timed out; retried before next spawn
        self.s = State()
        self.create_subscription(Clock, "/clock", self._on_clock, qos_profile_sensor_data)
        self.create_subscription(
            CameraInfo, "/camera/camera_info", self._on_info, qos_profile_sensor_data
        )
        self.create_subscription(Image, "/camera/image", self._on_rgb, qos_profile_sensor_data)
        self.create_subscription(
            Image, "/camera/depth_image", self._on_depth, qos_profile_sensor_data
        )
        for i in range(MAX_CUBES):
            name = f"cube_{i}"
            self.create_subscription(
                PoseStamped,
                f"/model/{name}/pose",
                lambda msg, name=name: self._on_pose(name, msg),  # type: ignore[misc]
                10,
            )

    # -- callbacks -------------------------------------------------------------------------
    def _on_clock(self, msg: Clock) -> None:
        self.s.clock = stamp_s(msg.clock.sec, msg.clock.nanosec)

    def _on_info(self, msg: CameraInfo) -> None:
        self.s.k = tuple(float(v) for v in msg.k)
        self.s.size = (int(msg.width), int(msg.height))

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

    # -- helpers ---------------------------------------------------------------------------
    def wait_until(self, pred, timeout_s: float, what: str) -> bool:  # noqa: ANN001
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.02)
            if pred():
                return True
        self.get_logger().error(f"timeout waiting for {what}")
        return False

    def spin_sim(self, seconds: float) -> None:
        end = self.s.clock + seconds
        guard = time.time() + 4.0 * seconds + 5.0
        while self.s.clock < end and time.time() < guard:
            rclpy.spin_once(self, timeout_sec=0.02)

    def spawn(self, cube: Cube, config: SceneConfig) -> bool:
        # gz-transport EntityFactory with the model written to a file: ``ros2 run ros_gz_sim
        # create`` occasionally never returns (its ROS node blocks on discovery), which stalled
        # whole capture runs; the gz CLI has a hard --timeout like ``remove``.
        sdf_path = self.sdf_dir / f"{cube.name}.sdf"
        sdf_path.write_text(
            '<?xml version="1.0"?><sdf version="1.8">' + cube_to_sdf_model(cube, config) + "</sdf>"
        )
        cmd = [
            "gz", "service", "-s", f"/world/{self.world}/create",
            "--reqtype", "gz.msgs.EntityFactory", "--reptype", "gz.msgs.Boolean",
            "--timeout", str(GZ_TIMEOUT_MS),
            "--req", f'sdf_filename: "{sdf_path}" name: "{cube.name}"',
        ]  # fmt: skip
        proc = subprocess.run(  # noqa: S603 - fixed executables, scene-generated arguments
            cmd, capture_output=True, text=True, timeout=CLI_TIMEOUT_S, check=False
        )
        ok = proc.returncode == 0 and "data: true" in proc.stdout
        if not ok:
            self.get_logger().error(
                f"spawn {cube.name} failed: {proc.stdout[-300:]} {proc.stderr[-300:]}"
            )
        return ok

    def remove(self, name: str) -> bool:
        cmd = [
            "gz", "service", "-s", f"/world/{self.world}/remove",
            "--reqtype", "gz.msgs.Entity", "--reptype", "gz.msgs.Boolean",
            "--timeout", str(GZ_TIMEOUT_MS), "--req", f'name: "{name}" type: MODEL',
        ]  # fmt: skip
        proc = subprocess.run(  # noqa: S603 - fixed executables, model name from the scene
            cmd, capture_output=True, text=True, timeout=CLI_TIMEOUT_S, check=False
        )
        ok = proc.returncode == 0 and "data: true" in proc.stdout
        if not ok:
            self.get_logger().error(
                f"remove {name} failed: {proc.stdout[-300:]} {proc.stderr[-300:]}"
            )
        return ok

    def place(self, scene: Scene, config: SceneConfig) -> str | None:
        """Clear cubes left over from a failed removal, then spawn the scene; error tag or None."""
        self.leftover = {name for name in self.leftover if not self.remove(name)}
        if self.leftover:
            return f"leftover: {sorted(self.leftover)}"
        if not all(self.spawn(c, config) for c in scene.cubes):
            return "spawn"
        return None

    def settled(self, scene: Scene, t_after: float) -> bool:
        for cube in scene.cubes:
            p = self.s.poses.get(cube.name)
            if p is None or p.t < t_after:
                return False
            if max(abs(p.x - cube.x), abs(p.y - cube.y), abs(p.z - cube.z)) > POSE_TOL_M:
                return False
        return True

    def fresh_pair(self, t_after: float) -> bool:
        r, d = self.s.rgb, self.s.depth
        return r is not None and d is not None and r.t >= t_after and abs(r.t - d.t) < 1e-6

    # -- one scene -------------------------------------------------------------------------
    def capture(
        self, seed: int, config: SceneConfig, out: Path, settle_s: float, timeout_s: float
    ) -> dict:
        t_wall = time.time()
        scene = generate_scene(seed, config)
        rec: dict = {"seed": seed, "n_cubes": len(scene.cubes), "ok": False}
        try:
            if (err := self.place(scene, config)) is not None:
                rec["error"] = err
                return rec
            t_spawned = self.s.clock
            if not self.wait_until(
                lambda: self.settled(scene, t_spawned), timeout_s, "cubes at rest"
            ):
                rec["error"] = "settle"
                rec["poses"] = {n: vars(p) for n, p in self.s.poses.items()}
                return rec
            self.spin_sim(settle_s)
            t_settled = self.s.clock
            if not self.wait_until(
                lambda: self.fresh_pair(t_settled), timeout_s, "fresh RGB-D pair"
            ):
                rec["error"] = "frame"
                return rec
            if not self.wait_until(
                lambda: self.settled(scene, t_settled), timeout_s, "cubes still at rest"
            ):
                rec["error"] = "moved"
                return rec
            rgb, depth = self.s.rgb, self.s.depth
            if rgb is None or depth is None:
                rec["error"] = "frame"
                return rec
            d = out / f"seed_{seed:04d}"
            d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / "rgb.png"), cv2.cvtColor(rgb.data, cv2.COLOR_RGB2BGR))
            np.save(d / "depth.npy", depth.data)
            gt = [
                {
                    "name": c.name,
                    "color": c.color,
                    "requested": {"x": c.x, "y": c.y, "z": c.z, "yaw": c.yaw},
                    "gazebo": vars(self.s.poses[c.name]),
                }
                for c in scene.cubes
            ]
            meta = {
                "seed": seed,
                "scene_config_hash": config.config_hash(),
                "camera": {"k": self.s.k, "width": self.s.size[0] if self.s.size else None,
                           "height": self.s.size[1] if self.s.size else None},
                "frame_stamp_sim_s": rgb.t,
                "settled_sim_s": t_settled,
                "cubes": gt,
            }  # fmt: skip
            (d / "meta.json").write_text(json.dumps(meta, indent=1))
            rec["ok"] = True
        except subprocess.TimeoutExpired as exc:
            self.get_logger().error(f"seed {seed}: {exc}")
            rec["error"] = f"cli_timeout: {exc.cmd[:4]}"
        finally:
            removed = [self.remove(c.name) for c in scene.cubes]
            self.leftover |= {c.name for c, ok in zip(scene.cubes, removed, strict=True) if not ok}
            rec["removed_all"] = all(removed)
            self.spin_sim(0.2)
            for c in scene.cubes:
                self.s.poses.pop(c.name, None)
            rec["wall_s"] = round(time.time() - t_wall, 3)
        return rec


def parse_seeds(spec: str) -> list[int]:
    """``"200-399"`` or ``"1,2,3"``."""
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in spec.split(",") if s]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="200-399")
    parser.add_argument("--out", type=Path, default=Path("/tmp/f3_dataset"))  # noqa: S108
    parser.add_argument("--world", default="armbench_tabletop")
    parser.add_argument("--scene-config", type=Path, default=DEFAULT_SCENE_FILE)
    parser.add_argument("--settle-s", type=float, default=1.0, help="simulated seconds after rest")
    parser.add_argument("--timeout", type=float, default=30.0, help="wall seconds per wait")
    args = parser.parse_args()
    config = load_scene_config(args.scene_config)
    rclpy.init()
    node = SceneCaptureNode(args.world)
    records: list[dict] = []
    try:
        ready = node.wait_until(
            lambda: (
                node.s.k is not None
                and node.s.rgb is not None
                and node.s.depth is not None
                and not math.isnan(node.s.clock)
            ),
            args.timeout,
            "camera + clock",
        )
        if not ready:
            raise SystemExit(2)
        for seed in parse_seeds(args.seeds):
            rec = node.capture(seed, config, args.out, args.settle_s, args.timeout)
            if not rec["ok"]:  # one retry: transient CLI or settle timeouts under host load
                rec = node.capture(seed, config, args.out, args.settle_s, args.timeout)
                rec["retried"] = True
            records.append(rec)
            node.get_logger().info(json.dumps(rec))
    finally:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "capture_summary.json").write_text(
            json.dumps(
                {"n": len(records), "n_ok": sum(r["ok"] for r in records), "scenes": records},
                indent=1,
            )
        )
        node.destroy_node()
        rclpy.try_shutdown()
    raise SystemExit(0 if records and all(r["ok"] for r in records) else 1)


if __name__ == "__main__":
    main()
