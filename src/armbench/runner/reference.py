"""Build the F8 energy reference from a finished baseline-A run directory (D8)."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from armbench.energy import (
    EnergyReference,
    EnergyReferenceError,
    ReferenceSample,
    ReferenceSource,
    Variant,
)
from armbench.runner.report import read_records
from armbench.seeds import load_seed_split


def reference_from_run(run_dir: Path, *, variant: Variant, eta: float) -> EnergyReference:
    """Aggregate the completed episodes of ``run_dir`` (``run.json`` + ``episodes.jsonl``) into
    a per-task reference for ``(variant, eta)``. Only successful episodes with a complete energy
    table count; the run must be baseline A on the simulation backend."""
    meta_path, log = run_dir / "run.json", run_dir / "episodes.jsonl"
    if not meta_path.is_file() or not log.is_file():
        msg = f"{run_dir} is not a run directory (needs run.json and episodes.jsonl)"
        raise EnergyReferenceError(msg)
    meta = json.loads(meta_path.read_text())
    records = read_records([log])
    samples: dict[str, list[ReferenceSample]] = defaultdict(list)
    skipped = 0
    for r in records:
        wh = None if r.energy is None else r.energy.wh(variant, eta)
        if not r.ok or wh is None or r.sim_s is None:
            skipped += 1
            continue
        samples[r.task].append(
            ReferenceSample(
                seed=r.seed,
                wh=wh,
                sim_s=r.sim_s,
                n_primitives=r.trace.n_primitives if r.trace is not None else 0,
            )
        )
    if not samples:
        msg = f"no completed episode with a ({variant}, eta={eta}) energy row in {log}"
        raise EnergyReferenceError(msg)
    seeds = sorted({r.seed for r in records})
    splits = load_seed_split()
    names = {splits.split_of(s) for s in seeds}
    source = ReferenceSource(
        run_id=str(meta.get("run_id", run_dir.name)),
        agent=str(meta.get("agent", records[0].agent)),
        backend=str(meta.get("backend", records[0].backend)),
        seeds=tuple(seeds),
        split=next(iter(names)) if len(names) == 1 and None not in names else None,
        git_sha=meta.get("git_sha"),
        armbench=meta.get("armbench"),
    )
    return EnergyReference.build(samples, source=source, variant=variant, eta=eta)
