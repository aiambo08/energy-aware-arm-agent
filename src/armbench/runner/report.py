"""Aggregate episode logs into per-task tables and judge them against the phase thresholds.

Success rates carry Wilson 95 % intervals; Wh and durations are summarised with medians,
interquartile means and percentiles (no normality assumed); repeatability is the coefficient
of variation of Wh across repeats of the same (task, seed).
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from armbench.energy import Variant
from armbench.runner.schema import EpisodeRecord
from armbench.seeds import SeedSplit, load_seed_split

Z95 = 1.959964


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for ``k`` successes in ``n`` trials (0, 0) when ``n == 0``."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile, ``q`` in [0, 100]."""
    if not values:
        return math.nan
    xs = sorted(values)
    pos = (len(xs) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def iqm(values: Sequence[float]) -> float:
    """Interquartile mean: mean of the middle 50 % (robust to a few outliers)."""
    if not values:
        return math.nan
    xs = sorted(values)
    n = len(xs)
    lo, hi = n // 4, n - n // 4
    return statistics.fmean(xs[lo:hi]) if hi > lo else statistics.fmean(xs)


def cv(values: Sequence[float]) -> float:
    """Coefficient of variation (sample std / mean); NaN with fewer than two values."""
    if len(values) < 2:
        return math.nan
    mean = statistics.fmean(values)
    return statistics.stdev(values) / mean if mean else math.nan


class Dist(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    n: int
    median: float
    iqm: float
    p05: float
    p95: float
    min: float
    max: float

    @classmethod
    def of(cls, values: Sequence[float]) -> Dist:
        if not values:
            return cls(n=0, median=math.nan, iqm=math.nan, p05=math.nan, p95=math.nan,
                       min=math.nan, max=math.nan)  # fmt: skip
        return cls(
            n=len(values),
            median=statistics.median(values),
            iqm=iqm(values),
            p05=percentile(values, 5),
            p95=percentile(values, 95),
            min=min(values),
            max=max(values),
        )


class EnergyStats(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    variant: Variant
    eta: float
    wh: Dist


class RepeatStats(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: int
    n: int
    wh_mean: float
    wh_cv: float
    sim_s_cv: float
    all_ok: bool


class LLMStats(BaseModel):
    """Cost and behaviour of an LLM agent over a task's episodes (from the agent traces)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str | None
    models: dict[str, int]
    n_traced: int
    prompt_tokens: int
    completion_tokens: int
    tokens_per_episode: Dist
    cost_usd: float
    cost_per_episode_usd: float
    cost_per_400_episodes_usd: float
    """What a full pre-registered evaluation (5 configs x 4 tasks x 20 seeds) would cost at
    this rate, if nothing were cached."""
    cached_rate: float
    """Share of episodes answered entirely from the cache (replay)."""
    latency_s: Dist
    """Provider latency of live (non-cached) episodes."""
    attempts: Dist
    program_outcomes: dict[str, int]
    isolation: dict[str, int]
    """How many episodes ran under each sandbox layer set (e.g. without ``unshare_net``)."""
    skills_used: dict[str, int] = {}
    """Episodes in which each library skill completed (agent B+S; empty for B)."""
    skill_episode_rate: float = 0.0
    """Share of episodes that completed at least one skill call."""


class TaskSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    task: str
    agent: str
    backend: str
    split: str
    """Seed split of ``configs/seeds.yaml`` the episodes belong to (``custom`` if none)."""
    n: int
    n_ok: int
    success: float
    success_ci95: tuple[float, float]
    n_infra: int
    infra_rate: float
    failures: dict[str, int]
    failed_seeds: list[int]
    energy_complete: int
    """Episodes whose energy table has every (variant, eta) row the protocol requires."""
    energy_complete_rate: float
    n_energy_rows_expected: int
    energy: list[EnergyStats]
    sim_s: Dist
    wall_s: Dist
    n_primitives: Dist
    observe_moves: Dist
    repeats: list[RepeatStats]
    repeat_cv_max: float | None
    llm: LLMStats | None = None


class Thresholds(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    success_min: float = 0.95
    energy_complete_min: float = 1.0
    repeat_cv_max: float = 0.03
    infra_rate_max: float = 0.01
    wall_s_max: float = 120.0


class Report(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = 1
    sources: list[str]
    n_records: int
    tasks: list[TaskSummary]
    thresholds: Thresholds
    checks: dict[str, bool]
    passed: bool


def read_records(paths: Iterable[Path]) -> list[EpisodeRecord]:
    out: list[EpisodeRecord] = []
    for p in paths:
        with p.open() as fh:
            for line in fh:
                if line.strip():
                    out.append(EpisodeRecord.model_validate_json(line))
    return out


def _repeats(records: list[EpisodeRecord]) -> list[RepeatStats]:
    by_seed: dict[int, list[EpisodeRecord]] = defaultdict(list)
    for r in records:
        by_seed[r.seed].append(r)
    out = []
    for seed, group in sorted(by_seed.items()):
        if len(group) < 2:
            continue
        whs = [r.energy.wh_a for r in group if r.energy is not None]
        sims = [r.sim_s for r in group if r.sim_s is not None]
        out.append(
            RepeatStats(
                seed=seed,
                n=len(group),
                wh_mean=statistics.fmean(whs) if whs else math.nan,
                wh_cv=cv(whs),
                sim_s_cv=cv(sims),
                all_ok=all(r.ok for r in group),
            )
        )
    return out


def _llm_stats(records: list[EpisodeRecord]) -> LLMStats | None:
    traces = [r.trace for r in records if r.trace is not None and r.trace.llm_provider]
    if not traces:
        return None
    n = len(traces)
    cost = sum(t.cost_usd or 0.0 for t in traces)
    tokens = [float(t.prompt_tokens + t.completion_tokens) for t in traces]
    live = [t.llm_latency_s for t in traces if not t.llm_cached and t.llm_latency_s is not None]
    return LLMStats(
        provider=traces[0].llm_provider,
        models=dict(sorted(Counter(t.llm_model or "?" for t in traces).items())),
        n_traced=n,
        prompt_tokens=sum(t.prompt_tokens for t in traces),
        completion_tokens=sum(t.completion_tokens for t in traces),
        tokens_per_episode=Dist.of(tokens),
        cost_usd=cost,
        cost_per_episode_usd=cost / n,
        cost_per_400_episodes_usd=400 * cost / n,
        cached_rate=sum(1 for t in traces if t.llm_cached) / n,
        latency_s=Dist.of(live),
        attempts=Dist.of([float(t.attempts) for t in traces]),
        program_outcomes=dict(sorted(Counter(t.program_outcome or "?" for t in traces).items())),
        isolation=dict(sorted(Counter(",".join(t.sandbox_isolation) for t in traces).items())),
        skills_used=dict(sorted(Counter(n for t in traces for n in t.skills_used).items())),
        skill_episode_rate=sum(1 for t in traces if t.skills_used) / n,
    )


def summarise(records: list[EpisodeRecord], n_rows_expected: int, split: str) -> TaskSummary:
    first = records[0]
    n, n_ok = len(records), sum(r.ok for r in records)
    infra = [r for r in records if r.infra_failure]
    failures = Counter(f"{r.failure.stage}:{r.failure.code}" for r in records if r.failure)
    complete = [
        r for r in records if r.energy is not None and len(r.energy.rows) == n_rows_expected
    ]
    energy: list[EnergyStats] = []
    if complete:
        for row in complete[0].energy.rows if complete[0].energy else ():
            whs = [
                w
                for r in complete
                if r.energy is not None and (w := r.energy.wh(row.variant, row.eta)) is not None
            ]
            energy.append(EnergyStats(variant=row.variant, eta=row.eta, wh=Dist.of(whs)))
    repeats = _repeats(records)
    cvs = [s.wh_cv for s in repeats if not math.isnan(s.wh_cv)]
    return TaskSummary(
        task=first.task,
        agent=first.agent,
        backend=first.backend,
        split=split,
        n=n,
        n_ok=n_ok,
        success=n_ok / n,
        success_ci95=wilson(n_ok, n),
        n_infra=len(infra),
        infra_rate=len(infra) / n,
        failures=dict(sorted(failures.items())),
        failed_seeds=sorted({r.seed for r in records if not r.ok}),
        energy_complete=len(complete),
        energy_complete_rate=len(complete) / n,
        n_energy_rows_expected=n_rows_expected,
        energy=energy,
        sim_s=Dist.of([r.sim_s for r in records if r.sim_s is not None]),
        wall_s=Dist.of([r.wall_s for r in records]),
        n_primitives=Dist.of([float(r.trace.n_primitives) for r in records if r.trace]),
        observe_moves=Dist.of([float(r.trace.n_observe_moves) for r in records if r.trace]),
        repeats=repeats,
        repeat_cv_max=max(cvs) if cvs else None,
        llm=_llm_stats(records),
    )


def judge(tasks: list[TaskSummary], th: Thresholds, *, require_energy: bool) -> dict[str, bool]:
    """One boolean per DoD row; energy rows only bind for simulation runs."""
    checks: dict[str, bool] = {}
    for t in tasks:
        key = f"{t.task}/{t.agent} [{t.split}]"
        checks[f"{key}: success >= {th.success_min:.0%}"] = t.success >= th.success_min
        checks[f"{key}: infra failures < {th.infra_rate_max:.0%}"] = (
            t.infra_rate < th.infra_rate_max
        )
        checks[f"{key}: wall per episode < {th.wall_s_max:.0f} s"] = t.wall_s.max < th.wall_s_max
        if require_energy:
            checks[f"{key}: energy table complete in {th.energy_complete_min:.0%}"] = (
                t.energy_complete_rate >= th.energy_complete_min
            )
            if t.repeats:
                checks[f"{key}: repeat Wh CV < {th.repeat_cv_max:.0%}"] = (
                    t.repeat_cv_max is not None and t.repeat_cv_max < th.repeat_cv_max
                )
    return checks


def build_report(
    paths: Sequence[Path],
    *,
    n_rows_expected: int,
    thresholds: Thresholds | None = None,
    require_repeats: bool = False,
    seed_split: SeedSplit | None = None,
) -> Report:
    """One summary per (task, agent, backend, seed split) so dev and extended seeds never mix."""
    th = thresholds or Thresholds()
    splits = seed_split or load_seed_split()
    records = read_records(paths)
    groups: dict[tuple[str, str, str, str], list[EpisodeRecord]] = defaultdict(list)
    for r in records:
        groups[(r.task, r.agent, r.backend, splits.split_of(r.seed) or "custom")].append(r)
    tasks = [summarise(g, n_rows_expected, k[3]) for k, g in sorted(groups.items())]
    sim = any(t.backend == "sim" for t in tasks)
    checks = judge(tasks, th, require_energy=sim)
    if require_repeats:
        checks["repeatability measured (>= 1 seed repeated)"] = any(t.repeats for t in tasks)
    return Report(
        sources=[str(p) for p in paths],
        n_records=len(records),
        tasks=tasks,
        thresholds=th,
        checks=checks,
        passed=bool(tasks) and all(checks.values()),
    )


def table(report: Report) -> str:
    """Plain-text table, one row per task."""
    head = (
        f"{'task':18s} {'agent':5s} {'split':12s} {'n':>4s} {'ok':>4s} {'success':>8s} "
        f"{'CI95':>16s} {'infra':>5s} {'Wh A med':>9s} {'wall p95':>8s} {'CV max':>7s}"
    )
    lines = [head]
    for t in report.tasks:
        lo, hi = t.success_ci95
        wh = t.energy[0].wh.median if t.energy else math.nan
        cvm = f"{t.repeat_cv_max:.1%}" if t.repeat_cv_max is not None else "-"
        lines.append(
            f"{t.task:18s} {t.agent:5s} {t.split:12s} {t.n:4d} {t.n_ok:4d} {t.success:8.1%} "
            f"[{lo:5.1%}, {hi:5.1%}] {t.n_infra:5d} {wh:9.4f} {t.wall_s.p95:8.1f} {cvm:>7s}"
        )
        if t.llm is not None:
            lat = f"{t.llm.latency_s.p95:.2f} s" if t.llm.latency_s.n else "-"
            outcomes = ", ".join(f"{k}={v}" for k, v in t.llm.program_outcomes.items())
            lines.append(
                f"{'':18s} llm {t.llm.provider}/{'+'.join(t.llm.models)}: "
                f"tokens/ep med {t.llm.tokens_per_episode.median:.0f}, "
                f"cost {t.llm.cost_usd:.4f} USD ({t.llm.cost_per_400_episodes_usd:.2f}/400 ep), "
                f"cached {t.llm.cached_rate:.0%}, latency p95 {lat}, programs: {outcomes}"
            )
            if t.llm.skills_used:
                used = ", ".join(f"{k}={v}" for k, v in t.llm.skills_used.items())
                lines.append(
                    f"{'':18s} skills: {used} ({t.llm.skill_episode_rate:.0%} of episodes); "
                    f"primitives/ep med {t.n_primitives.median:.0f}"
                )
    lines.append("")
    for name, ok in report.checks.items():
        lines.append(f"[{'PASS' if ok else 'FAIL'}] {name}")
    lines.append(f"\nGATE {'PASSED' if report.passed else 'FAILED'}")
    return "\n".join(lines)


def write_report(report: Report, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.model_dump(mode="json"), indent=1, sort_keys=True) + "\n")
