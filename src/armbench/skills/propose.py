"""Skill candidates from agent B's completed programs (development seeds).

A program that solved an instance is turned into a skill by *parametrising the goal*: every
constant the task wording gave the agent (colours, target coordinates) is replaced, where it
appears literally in the program, by a named parameter. Programs that do not contain a goal
constant verbatim are not parametrisable and are skipped (they are reported as such). The same
signature from several seeds collapses to one candidate (first seed wins; all contributing
episodes stay in the origin).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from armbench.llm.types import sha256_text
from armbench.runner import EpisodeRecord
from armbench.runner.report import read_records
from armbench.sandbox import check_program
from armbench.skills.spec import Condition, Origin, Param, Skill
from armbench.tasks import TaskInstance, get_task
from armbench.tasks.spec import PlaceGoal, SortGoal, StackGoal

SKILL_NAMES: dict[str, str] = {
    "pick_place": "pick_place_cube",
    "place_obstacle": "place_cube_over_wall",
    "stack2": "stack_cube",
    "sort3": "sort_cubes",
}
"""Task name -> skill name; a task outside this table gets ``<task>_skill``."""


class GoalConstant(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    param: str
    kind: str
    literal: str
    """How the constant appears in the task wording (``"red"`` as a string, ``-0.360`` as text);
    numbers are also matched as the shortest ``repr`` of the same value (``-0.36``)."""
    value: str | float


def goal_constants(instance: TaskInstance) -> tuple[GoalConstant, ...]:
    """The parameters a skill learned on this instance takes, and their values here."""
    goal = instance.goal
    out: list[GoalConstant] = []
    if isinstance(goal, PlaceGoal):
        out.append(_color("color", goal.target_color))
        out += _point("x", "y", goal.at.x, goal.at.y)
    elif isinstance(goal, StackGoal):
        out.append(_color("top", goal.top_color))
        out.append(_color("base", goal.base_color))
    elif isinstance(goal, SortGoal):
        for i, (color, at) in enumerate(goal.bins.items(), start=1):
            out.append(_color(f"color_{i}", color))
            out += _point(f"x_{i}", f"y_{i}", at.x, at.y)
    return tuple(out)


def _color(param: str, color: str) -> GoalConstant:
    return GoalConstant(param=param, kind="color", literal=color, value=color)


def _point(px: str, py: str, x: float, y: float) -> list[GoalConstant]:
    return [
        GoalConstant(param=px, kind="float", literal=f"{x:.3f}", value=x),
        GoalConstant(param=py, kind="float", literal=f"{y:.3f}", value=y),
    ]


def goal_arguments(instance: TaskInstance) -> dict[str, str | float]:
    """Keyword arguments that call the task's skill on this instance."""
    return {c.param: c.value for c in goal_constants(instance)}


def conditions_for(instance: TaskInstance) -> tuple[tuple[Condition, ...], tuple[Condition, ...]]:
    """(preconditions, postconditions) a skill learned on this task must satisfy."""
    goal = instance.goal
    pre = (Condition(kind="not_holding"),)
    post: list[Condition] = [Condition(kind="not_holding")]
    if isinstance(goal, PlaceGoal):
        post.append(Condition(kind="cube_near", color="color", x="x", y="y"))
    elif isinstance(goal, SortGoal):
        for i in range(1, len(goal.bins) + 1):
            post.append(Condition(kind="cube_near", color=f"color_{i}", x=f"x_{i}", y=f"y_{i}"))
    return pre, tuple(post)


class Parametrised(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    body: str
    params: tuple[Param, ...]
    replaced: dict[str, int]
    """param -> occurrences replaced."""
    missing: tuple[str, ...]
    """Goal constants that never appear literally in the program (then not parametrisable)."""


def parametrise(program: str, constants: Iterable[GoalConstant]) -> Parametrised:
    body = program
    params: list[Param] = []
    replaced: dict[str, int] = {}
    missing: list[str] = []
    for c in constants:
        if c.kind == "color":
            pattern = re.compile(rf"""(["']){re.escape(c.literal)}\1""")
        else:
            texts = sorted({c.literal, repr(round(float(c.value), 3))}, key=len, reverse=True)
            alts = "|".join(re.escape(t) for t in texts)
            pattern = re.compile(rf"(?<![\w.])(?:{alts})(?![\d])")
        body, n = pattern.subn(c.param, body)
        replaced[c.param] = n
        if n == 0:
            missing.append(c.param)
        kind = "color" if c.kind == "color" else "float"
        params.append(Param(name=c.param, kind=kind, description=_describe_param(c)))
    return Parametrised(body=body, params=tuple(params), replaced=replaced, missing=tuple(missing))


def _describe_param(c: GoalConstant) -> str:
    if c.kind == "color":
        return "colour name of the cube"
    axis = c.param.split("_")[0]
    return f"target {axis} of the cube centre in base_link (m)"


class Rejected(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    episode_id: str
    task: str
    seed: int
    reason: str


class Proposal(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    candidates: tuple[Skill, ...]
    rejected: tuple[Rejected, ...]
    n_episodes: int = Field(ge=0)
    n_completed: int = Field(ge=0)


def _program_path(programs: Path, rec: EpisodeRecord) -> Path | None:
    attempt = rec.trace.attempts if rec.trace is not None else 1
    name, version = rec.task.split("@")
    p = programs / f"{name}_v{version}_s{rec.seed}_a{attempt}.py"
    return p if p.exists() else None


def _reject(rec: EpisodeRecord, reason: str) -> Rejected:
    return Rejected(episode_id=rec.episode_id, task=rec.task, seed=rec.seed, reason=reason)


def _skill_name(task_name: str) -> str:
    return SKILL_NAMES.get(task_name, f"{task_name}_skill")


def propose_from_records(
    records: Iterable[EpisodeRecord], programs: Path, *, agent: str = "B"
) -> Proposal:
    """One candidate per signature from the successful episodes of ``agent``."""
    scene_cfg = None
    by_signature: dict[str, Skill] = {}
    rejected: list[Rejected] = []
    n_episodes = n_completed = 0
    for rec in records:
        if rec.agent != agent:
            continue
        n_episodes += 1
        if not rec.ok or rec.trace is None or rec.trace.program_outcome != "completed":
            continue
        n_completed += 1
        path = _program_path(programs, rec)
        if path is None:
            rejected.append(_reject(rec, "program file not found"))
            continue
        program = path.read_text(encoding="utf-8")
        if (
            rec.trace.program_sha256 is not None
            and sha256_text(program) != rec.trace.program_sha256
        ):
            rejected.append(_reject(rec, "program hash does not match the episode trace"))
            continue
        if scene_cfg is None:
            from armbench.scene import load_scene_config  # noqa: PLC0415 - optional dependency path

            scene_cfg = load_scene_config()
        task = get_task(rec.task)
        instance = task.instance(rec.seed, scene_cfg)
        par = parametrise(program, goal_constants(instance))
        if par.missing:
            rejected.append(_reject(rec, f"goal constants not in program: {par.missing}"))
            continue
        pre, post = conditions_for(instance)
        skill = Skill(
            name=_skill_name(task.id.name),
            description=f"{task.description} Learned from agent {agent} on {task.id}.",
            params=par.params,
            body=par.body,
            preconditions=pre,
            postconditions=post,
            origin=Origin(
                task=str(task.id),
                run_id=rec.run_id,
                episode_ids=(rec.episode_id,),
                seeds=(rec.seed,),
                program_sha256=sha256_text(program),
                agent=agent,
            ),
        )
        check = check_program(skill.bind(skill.placeholder_args()))
        if not check.ok:
            rejected.append(_reject(rec, f"parametrised body rejected: {check.summary()}"))
            continue
        sig = skill.signature()
        if sig in by_signature:
            first = by_signature[sig]
            if first.body != skill.body:
                rejected.append(_reject(rec, f"same signature as {first.name} with another body"))
                continue
            origin = first.origin.model_copy(
                update={
                    "episode_ids": (*first.origin.episode_ids, rec.episode_id),
                    "seeds": (*first.origin.seeds, rec.seed),
                }
            )
            by_signature[sig] = first.model_copy(update={"origin": origin})
        else:
            by_signature[sig] = skill
    return Proposal(
        candidates=tuple(by_signature.values()),
        rejected=tuple(rejected),
        n_episodes=n_episodes,
        n_completed=n_completed,
    )


def propose_from_run(run_dir: Path, *, agent: str = "B") -> Proposal:
    """``run_dir`` is an ``armbench run`` output: ``episodes.jsonl`` + ``programs/``."""
    records = read_records([run_dir / "episodes.jsonl"])
    return propose_from_records(records, run_dir / "programs", agent=agent)


def bind_arguments(skill: Skill, instance: TaskInstance) -> Mapping[str, str | float]:
    """Arguments of ``skill`` for ``instance`` (both must come from the same task kind)."""
    args = goal_arguments(instance)
    expected = {p.name for p in skill.params}
    if set(args) != expected:
        msg = f"{skill.signature()} cannot be bound from a {instance.task} instance"
        raise ValueError(msg)
    return args
