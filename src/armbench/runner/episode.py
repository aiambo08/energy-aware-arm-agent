"""One episode, backend-agnostic: place the scene, let the agent act, judge with ground truth.

The simulator-specific parts (spawning models, reading poses, tapping torques) sit behind
:class:`World`, so the same function runs on the kinematic fake in tests and on Gazebo in
the container.
"""

from __future__ import annotations

import time
import uuid
from datetime import UTC, datetime
from typing import Protocol

from armbench.agents import Agent, AgentError, AgentTrace
from armbench.primitives import CameraTimeout, PrimitiveError, Robot
from armbench.runner.schema import (
    INFRA_CODES,
    BackendName,
    EnergyRecord,
    EpisodeRecord,
    Failure,
    Software,
    instance_sha256,
)
from armbench.tasks import FinalState, Task, TaskInstance

SETTLE_TIMEOUT_S = 10.0
POST_SIM_S = 1.0
"""Simulated seconds to let released cubes come to rest before judging."""


class World(Protocol):
    def place(self, instance: TaskInstance) -> str | None:
        """Spawn the scene (and obstacles); an error tag means infrastructure failure."""
        ...

    def settled(self, instance: TaskInstance, timeout_s: float) -> bool: ...

    def begin(self, episode_id: str) -> None:
        """Start clearance monitoring, torque sampling and (optionally) bag recording."""
        ...

    def wait_sim(self, seconds: float) -> None: ...

    def final_state(self, instance: TaskInstance, sim_s: float) -> FinalState: ...

    def end(self) -> tuple[EnergyRecord | None, str | None]:
        """Stop sampling; return the energy table and the bag path, if any."""
        ...

    def clear(self, instance: TaskInstance) -> bool: ...


def _act(
    robot: Robot, agent: Agent, instance: TaskInstance
) -> tuple[AgentTrace | None, Failure | None]:
    """Run the agent; map what it raises to a failure stage (camera timeouts are the harness)."""
    try:
        return agent.solve(robot, instance), None
    except CameraTimeout as exc:
        return None, Failure(stage="infra", code=exc.code, message=exc.message)
    except PrimitiveError as exc:
        return None, Failure(stage="robot", code=exc.code, message=exc.message, details=exc.details)
    except AgentError as exc:
        return exc.trace, Failure(stage="agent", code=exc.code, message=exc.message)
    except LookupError as exc:
        return None, Failure(stage="agent", code="lookup", message=str(exc))


class _Outcome:
    """Mutable scratch for one episode; turned into a frozen record at the end."""

    def __init__(self) -> None:
        self.episode_id = uuid.uuid4().hex[:12]
        self.ok = False
        self.reason = ""
        self.metrics: dict[str, float] = {}
        self.failure: Failure | None = None
        self.trace: AgentTrace | None = None
        self.sim_s: float | None = None
        self.final: FinalState | None = None
        self.energy: EnergyRecord | None = None
        self.mcap: str | None = None
        self.reset_error: str | None = None
        self.began = False

    def judge(self, task: Task, instance: TaskInstance) -> None:
        if self.final is None:
            return
        verdict = task.check(instance, self.final)
        self.metrics = dict(verdict.metrics)
        if self.failure is None:
            self.ok, self.reason = verdict.ok, verdict.reason
            if not self.ok:
                self.failure = Failure(stage="judge", code="task_failed", message=self.reason)
        else:
            self.reason = f"{self.failure.stage}: {self.failure.message}"

    def play(
        self, robot: Robot, agent: Agent, task: Task, instance: TaskInstance, world: World
    ) -> None:
        if (err := world.place(instance)) is not None:
            self.failure = Failure(stage="infra", code="scene", message=err)
            self.reason = f"infra: {err}"
            return
        if not world.settled(instance, SETTLE_TIMEOUT_S):
            self.failure = Failure(stage="infra", code="scene", message="scene did not settle")
            self.reason = "infra: scene did not settle"
            return
        world.begin(self.episode_id)
        self.began = True
        t0 = robot.backend.sim_time()
        self.trace, self.failure = _act(robot, agent, instance)
        world.wait_sim(POST_SIM_S)
        self.sim_s = robot.backend.sim_time() - t0
        self.final = world.final_state(instance, self.sim_s)
        self.judge(task, instance)


def run_episode(  # noqa: PLR0913
    robot: Robot,
    agent: Agent,
    task: Task,
    instance: TaskInstance,
    world: World,
    *,
    run_id: str,
    repeat: int,
    backend: BackendName,
    software: Software,
) -> EpisodeRecord:
    t_wall = time.time()
    started = datetime.now(UTC).isoformat(timespec="seconds")
    o = _Outcome()
    try:
        o.play(robot, agent, task, instance, world)
    finally:
        if o.began:
            o.energy, o.mcap = world.end()
        try:
            robot.reset()
        except PrimitiveError as exc:
            o.reset_error = f"{exc.code}: {exc.message}"
        if not world.clear(instance) and o.failure is None:
            o.failure = Failure(stage="infra", code="scene", message="could not clear the scene")
            o.ok, o.reason = False, "infra: could not clear the scene"
    return EpisodeRecord(
        run_id=run_id,
        episode_id=o.episode_id,
        task=str(instance.task),
        task_version=instance.task.version,
        agent=agent.id,
        seed=instance.seed,
        repeat=repeat,
        backend=backend,
        instance_sha256=instance_sha256(instance),
        started_at=started,
        ok=o.ok,
        reason=o.reason,
        metrics=o.metrics,
        failure=o.failure,
        infra_failure=o.failure is not None and o.failure.code in INFRA_CODES,
        sim_s=o.sim_s,
        wall_s=time.time() - t_wall,
        energy=o.energy,
        trace=o.trace,
        min_tip_z_m=o.final.min_tip_z_m if o.final is not None else None,
        reset_error=o.reset_error,
        mcap_path=o.mcap,
        software=software,
    )
