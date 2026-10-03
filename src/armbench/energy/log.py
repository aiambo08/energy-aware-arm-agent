"""Episode sample log: one JSON object per line ``{"t", "q", "qd", "tau"}``.

Written by the ROS-side meter/recorder and read back on the host for the energy
report, so every Wh in the benchmark can be recomputed offline from the raw log.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, ConfigDict

from armbench.energy.model import EnergyBreakdown, Variant, episode_energy, sensitivity
from armbench.energy.params import EnergyParams

if TYPE_CHECKING:
    from collections.abc import Iterable


class Sample(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    t: float
    q: tuple[float, ...]
    qd: tuple[float, ...]
    tau: tuple[float, ...]


class Samples(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    t: np.ndarray
    q: np.ndarray
    qd: np.ndarray
    tau: np.ndarray

    @property
    def n(self) -> int:
        return int(self.t.shape[0])


def write_samples_jsonl(path: Path, samples: Iterable[Sample]) -> int:
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for s in samples:
            fh.write(s.model_dump_json() + "\n")
            n += 1
    return n


def read_samples_jsonl(path: Path, n_joints: int) -> Samples:
    rows = [
        Sample.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for i, r in enumerate(rows):
        if not len(r.q) == len(r.qd) == len(r.tau) == n_joints:
            msg = f"line {i}: expected {n_joints} joints"
            raise ValueError(msg)
    shape = (len(rows), n_joints)
    return Samples(
        t=np.array([r.t for r in rows], dtype=float),
        q=np.array([r.q for r in rows], dtype=float).reshape(shape),
        qd=np.array([r.qd for r in rows], dtype=float).reshape(shape),
        tau=np.array([r.tau for r in rows], dtype=float).reshape(shape),
    )


def episode_from_jsonl(
    path: Path,
    params: EnergyParams,
    *,
    variant: Variant = Variant.A,
    eta: float | None = None,
    full_sensitivity: bool = False,
) -> list[EnergyBreakdown]:
    s = read_samples_jsonl(path, params.n_joints)
    if full_sensitivity:
        return sensitivity(s.t, s.qd, s.tau, params)
    return [episode_energy(s.t, s.qd, s.tau, params, variant=variant, eta=eta)]
