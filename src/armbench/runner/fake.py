"""In-process episodes on the kinematic backend: no physics, no torques, no energy table."""

from __future__ import annotations

import math

from armbench.primitives import KinematicBackend, Robot
from armbench.runner.schema import EnergyRecord
from armbench.tasks import FinalState, ModelPose, TaskInstance


class FakeWorld:
    """Cubes are the backend's list; obstacles are static props the fake never touches."""

    def __init__(self, backend: KinematicBackend, robot: Robot) -> None:
        self.backend = backend
        self.robot = robot
        self._log_start = 0

    def place(self, instance: TaskInstance) -> str | None:
        self.backend.cubes = list(instance.scene.cubes)
        self.backend.held = None
        return None

    def settled(self, instance: TaskInstance, timeout_s: float) -> bool:
        return True

    def begin(self, episode_id: str) -> None:
        self._log_start = len(self.backend.follow_log)

    def wait_sim(self, seconds: float) -> None:
        self.backend.t += seconds

    def final_state(self, instance: TaskInstance, sim_s: float) -> FinalState:
        poses = {c.name: ModelPose(x=c.x, y=c.y, z=c.z, yaw=c.yaw) for c in self.backend.cubes}
        for o in instance.obstacles:
            poses[o.name] = ModelPose(x=o.x, y=o.y, z=o.z, yaw=o.yaw)
        tip = self.robot.params.finger_tip_below_tcp_m
        lowest = min(
            (self.robot.tcp_pose(q).z - tip for q, _ in self.backend.follow_log[self._log_start :]),
            default=math.inf,
        )
        return FinalState(poses=poses, min_tip_z_m=lowest, sim_s=sim_s)

    def end(self) -> tuple[EnergyRecord | None, str | None]:
        return None, None

    def clear(self, instance: TaskInstance) -> bool:
        self.backend.cubes = []
        return True
