"""``execute_skill()``: bind the arguments, check the preconditions, run the body in a nested
sandbox over the same robot, check the postconditions, and account for every primitive the body
used so the episode trace stays complete."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict

from armbench.primitives import (
    Observation,
    Robot,
    SkillFailed,
    SkillNotAvailable,
    SkillPostconditionFailed,
    SkillPreconditionFailed,
    SkillResult,
)
from armbench.sandbox import ProgramRun, SandboxLimits, run_program
from armbench.skills.library import SkillLibrary
from armbench.skills.spec import Condition, Skill, resolve

SKILL_LIMITS = SandboxLimits(cpu_s=5.0, memory_mb=256, wall_s=120.0, sim_s=60.0, max_calls=60)
"""Default budget of one skill call (inside the caller's own limits)."""


class ConditionCheck(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    condition: str
    ok: bool
    reason: str = ""
    deferred: bool = False
    """Needs the world: not evaluated here, only at validation time."""


def check_robot_condition(cond: Condition, obs: Observation) -> ConditionCheck:
    """Evaluate a robot-only condition on an observation (``cube_near`` is deferred)."""
    name = cond.describe()
    if cond.kind == "not_holding":
        return ConditionCheck(condition=name, ok=not obs.holding,
                              reason="" if not obs.holding else "gripper is holding")  # fmt: skip
    if cond.kind == "holding":
        return ConditionCheck(condition=name, ok=obs.holding,
                              reason="" if obs.holding else "gripper is not holding")  # fmt: skip
    if cond.kind == "tcp_above":
        z_min = cond.z if cond.z is not None else 0.0
        ok = obs.tcp.z >= z_min
        reason = "" if ok else f"TCP at z={obs.tcp.z:.3f} < {z_min}"
        return ConditionCheck(condition=name, ok=ok, reason=reason)
    return ConditionCheck(condition=name, ok=True, deferred=True)


def check_world_condition(
    cond: Condition, args: Mapping[str, str | float], cubes: Mapping[str, tuple[float, float]]
) -> ConditionCheck:
    """Evaluate ``cube_near`` against known cube positions (ground truth or a detection pass)."""
    name = cond.describe()
    if cond.kind != "cube_near":
        return ConditionCheck(condition=name, ok=True)
    color, x, y = resolve(cond, args)
    if color is None or color not in cubes:
        return ConditionCheck(condition=name, ok=False, reason=f"no {color} cube in the world")
    cx, cy = cubes[color]
    d = math.hypot(cx - x, cy - y)
    ok = d <= cond.tol_m
    reason = "" if ok else f"{color} cube is {d * 1000:.0f} mm away"
    return ConditionCheck(condition=name, ok=ok, reason=reason)


class SkillRunner:
    """:class:`armbench.primitives.SkillExecutor` over a library. ``last_run`` keeps the nested
    :class:`ProgramRun` of the most recent call for the validator and for tests."""

    def __init__(self, library: SkillLibrary, limits: SandboxLimits = SKILL_LIMITS) -> None:
        self.library = library
        self.limits = limits
        self.last_run: ProgramRun | None = None
        self.last_checks: tuple[ConditionCheck, ...] = ()

    def lookup(self, name: str, kwargs: Mapping[str, object]) -> Skill:
        skill = self.library.get(name)
        if skill is None:
            raise SkillNotAvailable(
                f"skill {name!r} is not in the library; available: {', '.join(self.library.names)}",
                name=name, available=list(self.library.names),
            )  # fmt: skip
        return skill

    def execute(self, robot: Robot, name: str, kwargs: Mapping[str, object]) -> SkillResult:
        skill = self.lookup(name, kwargs)
        program = skill.bind(kwargs)
        pre = tuple(check_robot_condition(c, robot.state()) for c in skill.preconditions)
        failed = [c for c in pre if not c.ok]
        if failed:
            raise SkillPreconditionFailed(
                f"{name}: {failed[0].condition}: {failed[0].reason}",
                skill=name, condition=failed[0].condition, reason=failed[0].reason,
            )  # fmt: skip
        t_start = robot.backend.sim_time()
        t_wall = time.time()
        run = run_program(program, robot, self.limits)
        self.last_run = run
        if run.outcome != "completed":
            err = run.primitive_error()
            if err is not None:
                raise err
            raise SkillFailed(
                f"{name}: body stopped: {run.describe()}",
                skill=name, outcome=run.outcome, error=run.error_message or "",
            )  # fmt: skip
        post = tuple(check_robot_condition(c, robot.state()) for c in skill.postconditions)
        self.last_checks = pre + post
        failed = [c for c in post if not c.ok and not c.deferred]
        if failed:
            raise SkillPostconditionFailed(
                f"{name}: {failed[0].condition}: {failed[0].reason}",
                skill=name, condition=failed[0].condition, reason=failed[0].reason,
                calls=list(run.primitives),
            )  # fmt: skip
        return SkillResult(
            primitive="execute_skill",
            t_start_sim=t_start,
            t_end_sim=robot.backend.sim_time(),
            wall_s=time.time() - t_wall,
            skill=name,
            skill_sha256=skill.sha256(),
            calls=run.primitives,
            speed_scales=run.move_speed_scales,
            postconditions_checked=tuple(c.condition for c in post if not c.deferred),
            postconditions_deferred=tuple(c.condition for c in post if c.deferred),
        )
