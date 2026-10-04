"""Episode log schema (JSONL, one record per line, ``schema_version`` 1).

A record is everything a reader needs to reproduce, audit or re-score the episode without the
simulator: the versioned task and seed (the instance is a pure function of both), the agent's
bookkeeping, the ground-truth verdict, the energy table for every protocol variant and where
the raw torque samples live so Wh can be recomputed with other parameters.
"""

from __future__ import annotations

import hashlib
import json
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from armbench.agents import AgentTrace
from armbench.energy import EnergyBreakdown, Variant
from armbench.tasks import TaskInstance

SCHEMA_VERSION: Final = 1
BackendName = Literal["fake", "sim"]
Stage = Literal["infra", "agent", "robot", "judge"]
INFRA_CODES: frozenset[str] = frozenset({"scene", "sim_not_ready", "camera_timeout"})
"""Failure codes that blame the harness (spawn/settle, bridge), not the agent or the robot."""


class EnergyRow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    variant: Variant
    eta: float
    duration_s: float
    n_samples: int
    mechanical_j: float
    copper_j: float
    base_j: float
    total_j: float
    total_wh: float

    @classmethod
    def from_breakdown(cls, b: EnergyBreakdown) -> EnergyRow:
        return cls(**b.model_dump(), total_wh=b.total_wh)


class EnergyRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source: Literal["gazebo_effort"]
    topic: str
    n_samples: int = Field(ge=0)
    rate_hz: float = Field(ge=0)
    rows: tuple[EnergyRow, ...]
    samples_path: str | None = None
    """Raw ``{t, q, qd, tau}`` samples (gzipped JSONL) relative to the run directory."""

    def wh(self, variant: Variant, eta: float) -> float | None:
        for r in self.rows:
            if r.variant == variant and abs(r.eta - eta) < 1e-9:
                return r.total_wh
        return None

    @property
    def wh_a(self) -> float:
        return self.rows[0].total_wh


class Failure(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    stage: Stage
    code: str
    message: str
    details: dict[str, object] = Field(default_factory=dict)


class Software(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    armbench: str
    git_sha: str | None = None
    image: str | None = None


class EpisodeRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = SCHEMA_VERSION
    run_id: str
    episode_id: str
    task: str
    task_version: int
    agent: str
    seed: int = Field(ge=0)
    repeat: int = Field(ge=0)
    backend: BackendName
    instance_sha256: str
    started_at: str
    ok: bool
    reason: str
    metrics: dict[str, float] = Field(default_factory=dict)
    failure: Failure | None = None
    infra_failure: bool = False
    sim_s: float | None = None
    wall_s: float = Field(ge=0)
    energy: EnergyRecord | None = None
    trace: AgentTrace | None = None
    min_tip_z_m: float | None = None
    reset_error: str | None = None
    mcap_path: str | None = None
    software: Software

    def line(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


def instance_sha256(instance: TaskInstance) -> str:
    canonical = json.dumps(instance.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
