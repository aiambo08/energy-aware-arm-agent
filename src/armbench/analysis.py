"""Pre-registered F9 analysis (``docs/protocol.md`` §6): ``armbench analyze``.

Inputs are run directories (``run.json`` + ``episodes.jsonl``). Per (agent, task, seed) the
latest episode without an infrastructure failure is kept; infrastructure failures are counted
and reported, never scored. Agent failures (rejected programs, exceptions, wrong final state)
are failures.

Statistics: Wilson 95 % intervals per cell; Wh over successful episodes summarised by median
and IQM; paired comparisons per hypothesis over the (task, seed) pairs both agents ran, with
percentile bootstrap intervals stratified by task (resample seeds within each task) from a
fixed seed, so the same logs give the same report bit for bit.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from armbench.protocol import Analysis, Manifest
from armbench.runner.report import iqm, read_records, wilson
from armbench.runner.schema import EpisodeRecord

Key = tuple[str, int]
"""(task, seed)."""
Stat = Callable[[np.ndarray], np.ndarray]


class AnalysisError(Exception):
    """The run directories cannot be analysed under the protocol."""


def _median(x: np.ndarray) -> np.ndarray:
    return np.asarray(np.median(x, axis=-1))


def _mean(x: np.ndarray) -> np.ndarray:
    return np.asarray(np.mean(x, axis=-1))


def load_runs(dirs: Sequence[Path]) -> tuple[list[dict[str, object]], list[EpisodeRecord]]:
    metas: list[dict[str, object]] = []
    records: list[EpisodeRecord] = []
    for d in dirs:
        meta_path, log = d / "run.json", d / "episodes.jsonl"
        if not (meta_path.is_file() and log.is_file()):
            msg = f"{d} is not a run directory (needs run.json and episodes.jsonl)"
            raise AnalysisError(msg)
        meta = json.loads(meta_path.read_text())
        meta["dir"] = d.name
        metas.append(meta)
        records.extend(read_records([log]))
    return metas, records


def select(
    records: Sequence[EpisodeRecord],
) -> tuple[dict[str, dict[Key, EpisodeRecord]], dict[str, list[str]]]:
    """Scored episode per (agent, task, seed) and the infrastructure failures per agent."""
    scored: dict[str, dict[Key, EpisodeRecord]] = {}
    infra: dict[str, list[str]] = {}
    for r in sorted(records, key=lambda r: r.started_at):
        if r.repeat != 0:
            continue
        if r.infra_failure:
            infra.setdefault(r.agent, []).append(f"{r.task}/{r.seed}:{r.episode_id}")
            continue
        scored.setdefault(r.agent, {})[(r.task, r.seed)] = r
    return scored, infra


def wh(r: EpisodeRecord, variant: str, eta: float) -> float | None:
    if r.energy is None:
        return None
    for row in r.energy.rows:
        if str(row.variant) == variant and math.isclose(row.eta, eta):
            return row.total_wh
    return None


def bootstrap(
    strata: Sequence[np.ndarray], stat: Stat, cfg: Analysis, rng: np.random.Generator
) -> tuple[float, float, float]:
    """(point, lo, hi): ``stat`` over the pooled strata, percentile CI resampling within each."""
    strata = [s for s in strata if s.size]
    if not strata:
        return (math.nan, math.nan, math.nan)
    point = float(stat(np.concatenate(strata)))
    parts = [s[rng.integers(0, s.size, size=(cfg.bootstrap_resamples, s.size))] for s in strata]
    dist = stat(np.concatenate(parts, axis=1))
    alpha = (1.0 - cfg.ci_level) / 2.0
    lo, hi = np.quantile(dist, [alpha, 1.0 - alpha])
    return (point, float(lo), float(hi))


def _cells(
    scored: dict[str, dict[Key, EpisodeRecord]], tasks: Sequence[str], variant: str, eta: float
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for agent, by_key in scored.items():
        for task in tasks:
            recs = [r for (t, _), r in sorted(by_key.items()) if t == task]
            if not recs:
                continue
            k = sum(1 for r in recs if r.ok)
            whs = [w for r in recs if r.ok and (w := wh(r, variant, eta)) is not None]
            sims = [r.sim_s for r in recs if r.ok and r.sim_s is not None]
            tokens = [r.trace.prompt_tokens + r.trace.completion_tokens for r in recs if r.trace]
            out.append({
                "agent": agent, "task": task, "n": len(recs), "k": k,
                "success": k / len(recs), "wilson95": list(wilson(k, len(recs))),
                "wh_success_median": float(np.median(whs)) if whs else None,
                "wh_success_iqm": iqm(whs) if whs else None,
                "sim_s_success_median": float(np.median(sims)) if sims else None,
                "wall_s_median": float(np.median([r.wall_s for r in recs])),
                "tokens_median": float(np.median(tokens)) if tokens else None,
                "cost_usd": sum((r.trace.cost_usd or 0.0) for r in recs if r.trace),
                "sandbox_outcomes": _counts(
                    r.trace.program_outcome or "none" for r in recs if r.trace
                ),
                "skills_used_episodes": sum(1 for r in recs if r.trace and r.trace.skills_used),
                "slow_move_episodes": sum(1 for r in recs if r.trace and r.trace.n_slow_moves),
            })  # fmt: skip
    return out


def _counts(items: object) -> dict[str, int]:
    out: dict[str, int] = {}
    for i in items:  # type: ignore[attr-defined]
        out[str(i)] = out.get(str(i), 0) + 1
    return dict(sorted(out.items()))


def compare(  # noqa: PLR0913
    x: dict[Key, EpisodeRecord],
    y: dict[Key, EpisodeRecord],
    tasks: Sequence[str],
    *,
    rows: Sequence[tuple[str, float]],
    cfg: Analysis,
    rng: np.random.Generator,
) -> dict[str, object]:
    """Treatment ``x`` vs control ``y`` over the shared (task, seed) pairs."""
    keys = sorted(set(x) & set(y))
    succ = [np.array([float(x[k].ok) - float(y[k].ok) for k in keys if k[0] == t]) for t in tasks]
    s_point, s_lo, s_hi = bootstrap(succ, _mean, cfg, rng)
    energy: list[dict[str, object]] = []
    for variant, eta in rows:
        strata = []
        for t in tasks:
            d = []
            for k in keys:
                if k[0] != t or not (x[k].ok and y[k].ok):
                    continue
                wx, wy = wh(x[k], variant, eta), wh(y[k], variant, eta)
                if wx is not None and wy is not None and wy > 0:
                    d.append(wx / wy - 1.0)
            strata.append(np.array(d))
        point, lo, hi = bootstrap(strata, _median, cfg, rng)
        energy.append({
            "variant": variant, "eta": eta, "n_pairs": int(sum(s.size for s in strata)),
            "median_rel_diff": point, "ci": [lo, hi], "saving": bool(hi < 0.0),
        })  # fmt: skip
    noninferior = bool(s_lo > -cfg.noninferiority_margin)
    return {
        "n_pairs": len(keys),
        "success_diff": s_point,
        "success_diff_ci": [s_lo, s_hi],
        "success_superior": bool(s_lo > 0.0),
        "success_noninferior": noninferior,
        "energy": energy,
        "saving_all_rows": bool(energy) and all(bool(e["saving"]) for e in energy),
        "saving_any_row": any(bool(e["saving"]) for e in energy),
    }


def verdicts(comparisons: dict[str, dict[str, object]]) -> dict[str, str]:
    """H1: success superiority. H2/H3: saving in every sensitivity row with non-inferior
    success; 'mixed' when some but not all rows save or success is inferior."""
    out: dict[str, str] = {}
    for h, c in comparisons.items():
        if h == "H1":
            out[h] = "supported" if c["success_superior"] else "not supported"
        elif c["saving_all_rows"] and c["success_noninferior"]:
            out[h] = "supported"
        elif c["saving_any_row"]:
            out[h] = "mixed"
        else:
            out[h] = "not supported"
    energy = [out[h] for h in ("H2", "H3") if h in out]
    if energy and all(v == "supported" for v in energy):
        out["scenario"] = "favourable"
    elif any(v != "not supported" for v in energy):
        out["scenario"] = "mixed"
    else:
        out["scenario"] = "null"
    return out


def analyse(
    dirs: Sequence[Path],
    manifest: Manifest,
    rows: Sequence[tuple[str, float]],
    protocol_sha256: str | None,
) -> dict[str, object]:
    metas, records = load_runs(dirs)
    scored, infra = select(records)
    tasks = list(manifest.tasks)
    cfg = manifest.analysis
    rng = np.random.default_rng(cfg.bootstrap_seed)
    n_scored = sum(len(v) for v in scored.values())
    n_infra = sum(len(v) for v in infra.values())
    expected = len(manifest.agents) * len(tasks)
    requested = {int(s) for m in metas for s in m.get("seeds", [])}  # type: ignore[attr-defined]
    seeds = sorted(requested | {s for v in scored.values() for (_, s) in v})
    missing = [
        f"{a}/{t}/{s}" for a in manifest.agents for t in tasks for s in seeds
        if (t, s) not in scored.get(a, {})
    ]  # fmt: skip
    comparisons = {
        h: compare(scored.get(tx, {}), scored.get(ty, {}), tasks, rows=rows, cfg=cfg, rng=rng)
        | {"treatment": tx, "control": ty}
        for h, (tx, ty) in manifest.hypotheses.items()
    }
    hashes = sorted({str(m.get("protocol_sha256")) for m in metas})
    report: dict[str, object] = {
        "protocol_id": manifest.protocol_id,
        "protocol_sha256": protocol_sha256,
        "runs_protocol_sha256": hashes,
        "runs_match_protocol": protocol_sha256 is not None and hashes == [protocol_sha256],
        "runs": [
            {k: m.get(k) for k in ("dir", "run_id", "agent", "git_sha", "llm")} for m in metas
        ],
        "seeds": seeds,
        "n_scored": n_scored,
        "n_expected": expected * len(seeds),
        "missing": missing,
        "infra_failures": {a: sorted(v) for a, v in sorted(infra.items())},
        "valid_fraction": n_scored / (n_scored + n_infra) if n_scored + n_infra else 0.0,
        "sensitivity_rows": [list(r) for r in rows],
        "cells": _cells(scored, tasks, *rows[0]) if rows else [],
        "comparisons": comparisons,
        "verdicts": verdicts(comparisons),
        "technical_failure": [],
        "analysis": cfg.model_dump(),
    }
    technical = []
    if float(report["valid_fraction"]) < cfg.min_valid_fraction:  # type: ignore[arg-type]
        technical.append(f"valid fraction below {cfg.min_valid_fraction}")
    if missing:
        technical.append(f"{len(missing)} (agent, task, seed) cells without a scored episode")
    if technical:
        report["technical_failure"] = technical
        report["verdicts"] = {"scenario": "technical failure"}
    blob = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=True)
    report["report_sha256"] = hashlib.sha256(blob.encode()).hexdigest()
    return report


def markdown(report: dict[str, object]) -> str:
    lines = [f"# armbench analysis — {report['protocol_id']}", ""]
    valid = report["valid_fraction"]
    lines += [f"- protocol sha256: `{report['protocol_sha256']}` "
              f"(runs match: {report['runs_match_protocol']})",
              f"- scored episodes: {report['n_scored']}/{report['n_expected']}, "
              f"valid fraction {valid:.3f}",
              f"- report sha256: `{report['report_sha256']}`", ""]  # fmt: skip
    lines += ["| agent | task | success | Wilson 95 % | Wh median (ok) | Wh IQM (ok) |",
              "|---|---|---|---|---|---|"]  # fmt: skip
    for c in report["cells"]:  # type: ignore[attr-defined]
        lo, hi = c["wilson95"]
        whm, whi = c["wh_success_median"], c["wh_success_iqm"]
        lines.append(
            f"| {c['agent']} | {c['task']} | {c['k']}/{c['n']} | [{lo:.2f}, {hi:.2f}] | "
            f"{'—' if whm is None else f'{whm:.4f}'} | {'—' if whi is None else f'{whi:.4f}'} |"
        )
    lines += ["", "| hypothesis | pair | Δ success [CI] | Wh rel. diff (variant, η): median [CI] |",
              "|---|---|---|---|"]  # fmt: skip
    for h, c in report["comparisons"].items():  # type: ignore[attr-defined]
        lo, hi = c["success_diff_ci"]
        en = "; ".join(
            f"{e['variant']},{e['eta']}: {e['median_rel_diff']:+.3f} "
            f"[{e['ci'][0]:+.3f}, {e['ci'][1]:+.3f}]"
            for e in c["energy"]
        )
        lines.append(
            f"| {h} | {c['treatment']} vs {c['control']} | "
            f"{c['success_diff']:+.3f} [{lo:+.3f}, {hi:+.3f}] | {en} |"
        )
    v = report["verdicts"]
    lines += ["", "**Verdicts:** " + ", ".join(f"{k}: {val}" for k, val in v.items()), ""]  # type: ignore[attr-defined]
    return "\n".join(lines)
