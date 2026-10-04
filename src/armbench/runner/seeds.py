"""``--seeds`` parsing: a split name (``dev``), a range (``400-449``) or a list (``1,2,3``)."""

from __future__ import annotations

from armbench.seeds import SeedSplit, check_seeds_allowed


def parse_seeds(spec: str, split: SeedSplit) -> list[int]:
    if spec in split.splits:
        return list(split.splits[spec].seeds)
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in spec.split(",") if s.strip()]


def authorise_seeds(
    spec: str, split: SeedSplit, *, final_eval: bool, protocol_hash: str | None
) -> list[int]:
    """Parse and refuse locked seeds unless ``final_eval`` and a protocol hash are given."""
    seeds = parse_seeds(spec, split)
    check_seeds_allowed(seeds, split, final_eval=final_eval, protocol_hash=protocol_hash)
    return seeds
