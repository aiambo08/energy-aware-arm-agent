"""Run benchmark episodes on the live simulation (inside the container).

For every (seed, repeat) it spawns the task's scene and obstacles, lets the agent act on the
robot, judges with Gazebo ground truth, taps the 100 Hz effort broadcaster for the energy
table (all variants and eta values) and appends one :class:`EpisodeRecord` line to the JSONL
log. Raw torque samples go to ``<out dir>/samples/<episode>.jsonl.gz`` so Wh can be
recomputed offline; ``--mcap`` additionally records a rosbag (MCAP) per episode.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import numpy as np
import rclpy
from sensor_msgs.msg import JointState

from armbench import __version__
from armbench.agents import get_agent
from armbench.energy import EnergyParams, load_energy_params, sensitivity
from armbench.llm import load_llm_params, make_provider
from armbench.primitives import Robot, load_primitive_params
from armbench.runner import (
    EnergyRecord,
    EnergyRow,
    EpisodeRecord,
    Software,
    authorise_seeds,
    run_episode,
)
from armbench.scene import load_scene_config
from armbench.scene.generator import DEFAULT_SCENE_FILE
from armbench.seeds import load_seed_split
from armbench.tasks import FinalState, ModelPose, TaskInstance, get_task
from armbench_bringup.ros_backend import GzScene, RosBackend

ENERGY_TOPIC = "/energy_state_broadcaster/joint_states"
ROS2 = shutil.which("ros2") or "ros2"
BAG_TOPICS = (
    "/clock",
    "/joint_states",
    ENERGY_TOPIC,
    "/model/cube_0/pose",
    "/model/cube_1/pose",
    "/model/cube_2/pose",
    "/model/cube_3/pose",
    "/model/cube_4/pose",
    "/model/cube_5/pose",
    "/model/obstacle/pose",
)


class GazeboWorld:
    """:class:`armbench.runner.World` on Gazebo: GzScene for models, a JointState tap for
    torques, ``ros2 bag record`` for the optional MCAP."""

    def __init__(
        self,
        backend: RosBackend,
        gz: GzScene,
        energy: EnergyParams,
        out_dir: Path,
        *,
        mcap: bool,
    ) -> None:
        self.backend = backend
        self.gz = gz
        self.energy = energy
        self.out_dir = out_dir
        self.mcap = mcap
        self.joints = list(energy.joint_order)
        self.samples: list[tuple[float, list[float], list[float], list[float]]] = []
        self.recording = False
        self.episode_id = ""
        self.bag: subprocess.Popen[bytes] | None = None
        self._idx: list[int] | None = None
        self.t_spawned = 0.0
        backend.create_subscription(JointState, ENERGY_TOPIC, self._on_energy_js, 200)

    def _on_energy_js(self, msg: JointState) -> None:
        if not self.recording:
            return
        if self._idx is None:
            names = list(msg.name)
            if any(j not in names for j in self.joints) or len(msg.effort) != len(names):
                return
            self._idx = [names.index(j) for j in self.joints]
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.samples.append(
            (
                t,
                [float(msg.position[i]) for i in self._idx],
                [float(msg.velocity[i]) for i in self._idx],
                [float(msg.effort[i]) for i in self._idx],
            )
        )

    # -- World protocol ------------------------------------------------------------------
    def place(self, instance: TaskInstance) -> str | None:
        err = self.gz.place(instance.scene, instance.obstacles)
        self.t_spawned = self.backend.sim_time()
        return err

    def settled(self, instance: TaskInstance, timeout_s: float) -> bool:
        ok = self.backend.wait_until(
            lambda: self.gz.settled(instance.scene, self.t_spawned, instance.obstacles), timeout_s
        )
        if ok:
            self.backend.spin_sim(0.5)
        return ok

    def begin(self, episode_id: str) -> None:
        self.episode_id = episode_id
        self.samples = []
        self.backend.reset_monitor()
        if self.mcap:
            bag_dir = self.out_dir / "mcap" / episode_id
            bag_dir.parent.mkdir(parents=True, exist_ok=True)
            self.bag = subprocess.Popen(  # noqa: S603 - fixed executable and arguments
                [ROS2, "bag", "record", "-s", "mcap", "-o", str(bag_dir), *BAG_TOPICS],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            self.backend.spin_sim(0.5)  # let the recorder subscribe before the arm moves
        self.recording = True

    def wait_sim(self, seconds: float) -> None:
        self.backend.spin_sim(seconds)

    def final_state(self, instance: TaskInstance, sim_s: float) -> FinalState:
        poses = {
            name: ModelPose(x=p.x, y=p.y, z=p.z, yaw=p.yaw)
            for name, p in self.backend.cube_poses().items()
        }
        return FinalState(poses=poses, min_tip_z_m=self.backend.monitor.min_tip_z, sim_s=sim_s)

    def end(self) -> tuple[EnergyRecord | None, str | None]:
        self.recording = False
        mcap_path = self._stop_bag()
        if len(self.samples) < 2:
            return None, mcap_path
        t = np.array([s[0] for s in self.samples])
        qd = np.array([s[2] for s in self.samples])
        tau = np.array([s[3] for s in self.samples])
        order = np.argsort(t, kind="stable")
        t, qd, tau = t[order], qd[order], tau[order]
        keep = np.concatenate(([True], np.diff(t) > 0))  # drop duplicate stamps
        t, qd, tau = t[keep], qd[keep], tau[keep]
        rows = tuple(EnergyRow.from_breakdown(b) for b in sensitivity(t, qd, tau, self.energy))
        rel = Path("samples") / f"{self.episode_id}.jsonl.gz"
        path = self.out_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt") as fh:
            for ts, q, v, e in self.samples:
                fh.write(json.dumps({"t": ts, "q": q, "qd": v, "tau": e}) + "\n")
        span = float(t[-1] - t[0])
        record = EnergyRecord(
            source="gazebo_effort",
            topic=ENERGY_TOPIC,
            n_samples=int(t.shape[0]),
            rate_hz=(t.shape[0] - 1) / span if span > 0 else 0.0,
            rows=rows,
            samples_path=str(rel),
        )
        return record, mcap_path

    def _stop_bag(self) -> str | None:
        if self.bag is None:
            return None
        os.killpg(self.bag.pid, signal.SIGINT)
        try:
            self.bag.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(self.bag.pid, signal.SIGKILL)
            self.bag.wait(timeout=10)
        self.bag = None
        return str(Path("mcap") / self.episode_id)

    def clear(self, instance: TaskInstance) -> bool:
        return self.gz.clear(instance.scene, instance.obstacles)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, help="name@version, e.g. pick_place@1")
    parser.add_argument("--agent", default="A")
    parser.add_argument("--seeds", default="dev", help="split name, lo-hi or comma list")
    parser.add_argument("--repeat", type=int, default=1, help="episodes per seed")
    parser.add_argument("--final-eval", action="store_true")
    parser.add_argument("--protocol-hash", default=None)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--world", default="armbench_tabletop")
    parser.add_argument("--scene-config", type=Path, default=DEFAULT_SCENE_FILE)
    parser.add_argument("--timeout", type=float, default=60.0, help="sim readiness wait (s)")
    parser.add_argument("--mcap", action="store_true")
    parser.add_argument("--image", default=None, help="container image tag, for the log")
    parser.add_argument("--git-sha", default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--llm-dir",
        type=Path,
        default=None,
        help="agent B bundle: llm.yaml + cache/ pre-fetched on the host (replayed here)",
    )
    return parser.parse_args()


def make_agent(args: argparse.Namespace) -> object:
    if args.agent != "B":
        return get_agent(args.agent)
    if args.llm_dir is None:
        msg = "agent B needs --llm-dir (the container has no network; programs are replayed)"
        raise SystemExit(msg)
    llm = load_llm_params(args.llm_dir / "llm.yaml")
    provider = make_provider(llm, kind="replay", cache_dir=args.llm_dir / "cache")
    return get_agent("B", llm=llm, provider=provider, artifacts_dir=args.out_dir)


def main() -> None:
    args = parse_args()
    split = load_seed_split()
    seeds = authorise_seeds(
        args.seeds, split, final_eval=args.final_eval, protocol_hash=args.protocol_hash
    )
    task = get_task(args.task)
    agent = make_agent(args)
    params = load_primitive_params()
    scene_cfg = load_scene_config(args.scene_config)
    energy = load_energy_params()
    software = Software(armbench=__version__, git_sha=args.git_sha, image=args.image)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / "episodes.jsonl"
    status: dict[str, object] = {"ready": False, "n_episodes": 0, "n_ok": 0}

    rclpy.init()
    backend = RosBackend(params)
    t_wall = time.time()
    try:
        if not backend.ready(args.timeout):
            status["error"] = "sim_not_ready"
            return
        status["ready"] = True
        robot = Robot(backend, params=params)
        robot.reset()
        gz = GzScene(backend, args.world, scene_cfg)
        world = GazeboWorld(backend, gz, energy, args.out_dir, mcap=args.mcap)
        with log_path.open("a") as fh:
            for seed in seeds:
                instance = task.instance(seed, scene_cfg)
                for rep in range(args.repeat):
                    rec: EpisodeRecord = run_episode(
                        robot,
                        agent,
                        task,
                        instance,
                        world,
                        run_id=args.run_id,
                        repeat=rep,
                        backend="sim",
                        software=software,
                    )
                    fh.write(rec.line() + "\n")
                    fh.flush()
                    status["n_episodes"] = int(status["n_episodes"]) + 1  # type: ignore[call-overload]
                    status["n_ok"] = int(status["n_ok"]) + int(rec.ok)  # type: ignore[call-overload]
                    backend.get_logger().info(
                        f"{rec.task} seed {seed} rep {rep}: ok={rec.ok} {rec.reason} "
                        f"sim={rec.sim_s} wall={rec.wall_s:.1f}s "
                        f"wh={rec.energy.wh_a if rec.energy else None}"
                    )
    finally:
        status["wall_s_total"] = round(time.time() - t_wall, 1)
        (args.out_dir / "status.json").write_text(json.dumps(status, indent=1) + "\n")
        backend.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
