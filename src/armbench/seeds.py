"""Seed split loading and the final-evaluation lock.

The split lives in ``configs/seeds.yaml`` so that it is visible, versioned and
test-checked. The runner (phase F5) must call :func:`check_seeds_allowed`
before every episode.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

REPO_ROOT: Final = Path(__file__).resolve().parents[2]
DEFAULT_SEEDS_FILE: Final = REPO_ROOT / "configs" / "seeds.yaml"


class LockedSeedError(PermissionError):
    """Raised when a locked seed is requested without the final-eval unlock."""


class SeedRange(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    description: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    locked: bool = False

    @model_validator(mode="after")
    def _ordered(self) -> SeedRange:
        if self.end < self.start:
            msg = f"end ({self.end}) < start ({self.start})"
            raise ValueError(msg)
        return self

    @property
    def seeds(self) -> range:
        return range(self.start, self.end + 1)

    def __contains__(self, seed: object) -> bool:
        return isinstance(seed, int) and self.start <= seed <= self.end


class SeedSplit(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int
    splits: dict[str, SeedRange]

    @model_validator(mode="after")
    def _disjoint(self) -> SeedSplit:
        names = list(self.splits)
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                ra, rb = self.splits[a], self.splits[b]
                if ra.start <= rb.end and rb.start <= ra.end:
                    msg = f"seed ranges overlap: {a} and {b}"
                    raise ValueError(msg)
        return self

    def split_of(self, seed: int) -> str | None:
        for name, rng in self.splits.items():
            if seed in rng:
                return name
        return None

    def locked_seeds(self) -> frozenset[int]:
        return frozenset(s for rng in self.splits.values() if rng.locked for s in rng.seeds)


def load_seed_split(path: Path = DEFAULT_SEEDS_FILE) -> SeedSplit:
    with path.open("rb") as fh:
        raw = yaml.safe_load(fh)
    return SeedSplit.model_validate(raw)


def check_seeds_allowed(
    seeds: list[int],
    split: SeedSplit,
    *,
    final_eval: bool = False,
    protocol_hash: str | None = None,
) -> None:
    """Raise :class:`LockedSeedError` if any locked seed is used without unlock.

    Unlocking requires both ``final_eval=True`` and a non-empty
    ``protocol_hash`` (the git hash of the pre-registered protocol).
    """
    locked = split.locked_seeds()
    offending = sorted(s for s in seeds if s in locked)
    if not offending:
        return
    if final_eval and protocol_hash:
        return
    msg = (
        f"seeds {offending} belong to a locked split; "
        "pass final_eval=True and a protocol_hash to use them"
    )
    raise LockedSeedError(msg)
