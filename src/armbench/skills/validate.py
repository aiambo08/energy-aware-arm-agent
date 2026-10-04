"""Skill validation (plan F7): a candidate enters the library only if, on the
``skill_validation`` seeds, its body completes inside the sandbox, every pre- and postcondition
holds (robot-side ones on the robot, ``cube_near`` against the world) **and** the task it was
learned on judges the episode a success, on at least ``min_pass_rate`` of the seeds."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from armbench import __version__
from armbench.perception import Camera, load_perception_params
from armbench.primitives import (
    KinematicBackend,
    PrimitiveError,
    PrimitiveParams,
    Robot,
    load_primitive_params,
)
from armbench.runner import FakeWorld
from armbench.sandbox import SandboxLimits
from armbench.scene import SceneConfig, load_scene_config
from armbench.seeds import DEFAULT_SEEDS_FILE, load_seed_split
from armbench.skills.execute import SKILL_LIMITS, ConditionCheck, SkillRunner, check_world_condition
from armbench.skills.library import SkillLibrary
from armbench.skills.propose import bind_arguments
from armbench.skills.spec import Skill, Validation
from armbench.tasks import get_task

VALIDATION_SPLIT = "skill_validation"
MIN_PASS_RATE = 0.9
"""Plan F7: ">= 9/10 seeds"; applied to the whole 20-seed validation range (>= 18/20)."""


class SeedResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: int
    ok: bool
    reason: str = ""
    outcome: str | None = None
    n_calls: int = Field(ge=0, default=0)
    sim_s: float = 0.0
    checks: tuple[ConditionCheck, ...] = ()
    verdict: str = ""


class ValidationReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    skill: str
    signature: str
    sha256: str
    task: str
    split: str
    results: tuple[SeedResult, ...]
    n_ok: int = Field(ge=0)
    min_pass_rate: float
    accepted: bool

    @property
    def pass_rate(self) -> float:
        return self.n_ok / len(self.results) if self.results else 0.0


class Validator:
    """Runs candidates on the kinematic backend (fast, deterministic, no Gazebo)."""

    def __init__(  # noqa: PLR0913 - keyword-only knobs with defaults
        self,
        *,
        seeds: Sequence[int] | None = None,
        scene_cfg: SceneConfig | None = None,
        params: PrimitiveParams | None = None,
        limits: SandboxLimits = SKILL_LIMITS,
        min_pass_rate: float = MIN_PASS_RATE,
        seeds_file: Path = DEFAULT_SEEDS_FILE,
    ) -> None:
        split = load_seed_split(seeds_file)
        self.seeds = (
            list(seeds) if seeds is not None else list(split.splits[VALIDATION_SPLIT].seeds)
        )
        self.scene_cfg = scene_cfg or load_scene_config()
        self.params = params or load_primitive_params()
        self.camera = Camera.from_spec(load_perception_params().camera)
        self.limits = limits
        self.min_pass_rate = min_pass_rate

    def run_seed(self, skill: Skill, seed: int) -> SeedResult:
        task = get_task(skill.origin.task)
        instance = task.instance(seed, self.scene_cfg)
        backend = KinematicBackend(self.params, self.camera)
        runner = SkillRunner(SkillLibrary([skill]), self.limits)
        robot = Robot(backend, params=self.params, skills=runner)
        robot.reset()
        world = FakeWorld(backend, robot)
        world.place(instance)
        world.begin(uuid.uuid4().hex[:12])
        args = bind_arguments(skill, instance)
        t0 = backend.sim_time()
        error = ""
        try:
            robot.execute_skill(skill.name, **args)
        except PrimitiveError as exc:
            error = f"{exc.code}: {exc.message}"
        run = runner.last_run
        checks = list(runner.last_checks)
        cubes = {c.color: (c.x, c.y) for c in backend.cubes}
        checks += [
            check_world_condition(c, args, cubes) for c in skill.postconditions if c.needs_world
        ]
        sim_s = backend.sim_time() - t0
        verdict = task.check(instance, world.final_state(instance, sim_s))
        failed = [c for c in checks if not c.ok]
        reason = error or (f"{failed[0].condition}: {failed[0].reason}" if failed else "")
        if not reason and not verdict.ok:
            reason = f"task verdict: {verdict.reason}"
        return SeedResult(
            seed=seed,
            ok=not reason,
            reason=reason,
            outcome=run.outcome if run is not None else None,
            n_calls=run.n_calls if run is not None else 0,
            sim_s=sim_s,
            checks=tuple(checks),
            verdict=verdict.reason,
        )

    def validate(self, skill: Skill) -> tuple[Skill, ValidationReport]:
        results = tuple(self.run_seed(skill, s) for s in self.seeds)
        n_ok = sum(r.ok for r in results)
        accepted = len(results) > 0 and n_ok / len(results) >= self.min_pass_rate
        record = Validation(
            split=VALIDATION_SPLIT,
            seeds=tuple(self.seeds),
            n_ok=n_ok,
            failed={r.seed: r.reason for r in results if not r.ok},
            min_pass_rate=self.min_pass_rate,
            accepted=accepted,
            backend="fake",
            armbench=__version__,
        )
        report = ValidationReport(
            skill=skill.name,
            signature=skill.signature(),
            sha256=skill.sha256(),
            task=skill.origin.task,
            split=VALIDATION_SPLIT,
            results=results,
            n_ok=n_ok,
            min_pass_rate=self.min_pass_rate,
            accepted=accepted,
        )
        return skill.model_copy(update={"validation": record}), report

    def build_library(
        self, candidates: Sequence[Skill], library: SkillLibrary | None = None
    ) -> tuple[SkillLibrary, list[ValidationReport]]:
        """Validate every candidate and add the accepted ones (deduplicated by signature)."""
        lib = library if library is not None else SkillLibrary()
        reports: list[ValidationReport] = []
        for cand in candidates:
            if lib.by_signature(cand.signature()) is not None:
                continue
            skill, report = self.validate(cand)
            reports.append(report)
            if report.accepted:
                lib.add(skill)
        return lib, reports
