"""Energy reference for the energy-aware agents C and C+S (phase F8, decision D8).

C receives, in its prompt, the electrical energy baseline A needed for the *same task* on the
development seeds — a budget to beat, not a measurement of its own previous attempt (the plan's
"Wh of the previous attempt" is undefined with ``max_attempts = 1`` and would break the pairing
with B). The reference is built once from a finished baseline-A simulation run, stored as JSON
with the provenance (run, seeds, agent, backend, variant, eta) and identified by the SHA-256 of
its content; every C/C+S run records that hash, so the number the agent saw is auditable.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from armbench.energy.model import Variant

REFERENCE_FILE: Final = "reference.json"
SCHEMA_VERSION: Final = 1


class EnergyReferenceError(Exception):
    """The reference is missing, malformed, or has no entry for the task."""


class ReferenceSample(BaseModel):
    """One completed baseline episode: what the reference statistics are computed from."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: int
    wh: float = Field(ge=0)
    sim_s: float = Field(ge=0)
    n_primitives: int = Field(ge=0)


class TaskReference(BaseModel):
    """Per-task budget: the distribution of baseline A's Wh over the development seeds."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task: str
    n: int = Field(ge=1)
    seeds: tuple[int, ...]
    wh_median: float = Field(ge=0)
    wh_iqm: float = Field(ge=0)
    wh_min: float = Field(ge=0)
    wh_max: float = Field(ge=0)
    sim_s_median: float = Field(ge=0)
    n_primitives_median: float = Field(ge=0)

    @classmethod
    def of(cls, task: str, samples: Sequence[ReferenceSample]) -> TaskReference:
        if not samples:
            msg = f"no completed episodes for {task}"
            raise EnergyReferenceError(msg)
        whs = sorted(s.wh for s in samples)
        n = len(whs)
        lo, hi = n // 4, n - n // 4
        return cls(
            task=task,
            n=n,
            seeds=tuple(sorted(s.seed for s in samples)),
            wh_median=statistics.median(whs),
            wh_iqm=statistics.fmean(whs[lo:hi] if hi > lo else whs),
            wh_min=whs[0],
            wh_max=whs[-1],
            sim_s_median=statistics.median(s.sim_s for s in samples),
            n_primitives_median=statistics.median(float(s.n_primitives) for s in samples),
        )


class ReferenceSource(BaseModel):
    """Where the numbers come from (the run they were aggregated from)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    agent: str
    backend: str
    seeds: tuple[int, ...]
    split: str | None = None
    git_sha: str | None = None
    armbench: str | None = None


class EnergyReference(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = SCHEMA_VERSION
    variant: Variant
    eta: float = Field(gt=0, le=1)
    source: ReferenceSource
    tasks: dict[str, TaskReference]
    created_at: str

    @classmethod
    def build(
        cls,
        samples: Mapping[str, Sequence[ReferenceSample]],
        *,
        source: ReferenceSource,
        variant: Variant,
        eta: float,
        created_at: str | None = None,
    ) -> EnergyReference:
        if source.agent != "A":
            msg = f"the reference must come from baseline A, not agent {source.agent!r} (D8)"
            raise EnergyReferenceError(msg)
        if source.backend != "sim":
            msg = f"the reference needs measured energy (backend sim), got {source.backend!r}"
            raise EnergyReferenceError(msg)
        if not samples:
            msg = "no tasks to build the reference from"
            raise EnergyReferenceError(msg)
        tasks = {t: TaskReference.of(t, s) for t, s in sorted(samples.items())}
        return cls(
            variant=variant,
            eta=eta,
            source=source,
            tasks=tasks,
            created_at=created_at or datetime.now(UTC).isoformat(timespec="seconds"),
        )

    def canonical(self) -> str:
        """Content identity: everything but the build timestamp, so rebuilding the reference
        from the same run gives the same hash."""
        payload = self.model_dump(mode="json", exclude={"created_at"})
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical().encode()).hexdigest()

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(self.tasks)

    def for_task(self, task_id: str) -> TaskReference:
        try:
            return self.tasks[task_id]
        except KeyError:
            msg = f"no energy reference for task {task_id!r}; known: {', '.join(self.tasks)}"
            raise EnergyReferenceError(msg) from None

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / REFERENCE_FILE
        path.write_text(json.dumps(self.model_dump(mode="json"), indent=1) + "\n")
        return path

    @classmethod
    def load(cls, directory: Path) -> EnergyReference:
        path = directory / REFERENCE_FILE
        if not path.is_file():
            msg = f"no energy reference at {path} (run `armbench energy-ref build` first)"
            raise EnergyReferenceError(msg)
        try:
            return cls.model_validate_json(path.read_text())
        except ValueError as exc:
            msg = f"malformed energy reference {path}: {exc}"
            raise EnergyReferenceError(msg) from exc
