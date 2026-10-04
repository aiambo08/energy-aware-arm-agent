"""F8 gate: the energy reference (D8) and the B vs C / B+S vs C+S pilot (Gazebo, dev seeds).

Inputs are four `armbench run` directories produced on the simulation backend with the same
tasks and seeds: --b, --c, --bs, --cs. The energy reference C and C+S read is checked against
`--energy-ref` (frozen file, hash registered in the C/C+S runs, rebuilt bit for bit from the
baseline-A run it names) and the F8 tests are run for real.

Thresholds (docs/plan.es.md, F8): the reference is baseline A on the simulation backend and its
hash is in every C/C+S run and trace; every C/C+S prompt carried the energy block and no B/B+S
prompt did; the fraction of C/C+S episodes that used the energy lever (speed_scale < 1) is
*reported*; the C vs B and C+S vs B+S Wh comparison is *reported* without an improvement
threshold (hypothesis H2 of the protocol, evaluated in F9 with a real model: the template
provider is a fixed recipe, not a language model); no cost abort; no infrastructure failures;
complete energy tables.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from armbench import __version__
from armbench.energy import EnergyReference, Variant, load_energy_params
from armbench.llm import load_llm_params
from armbench.runner import EpisodeRecord, build_report
from armbench.runner.reference import reference_from_run
from armbench.runner.report import Report, read_records

ROOT = Path(__file__).resolve().parents[1]
PAIRS = (("B", "C"), ("B+S", "C+S"))


def reference_check(ref_dir: Path, runs: dict[str, dict[str, object]]) -> dict[str, object]:
    """Baseline-A provenance, hash registered in the energy-aware runs, reproducible rebuild."""
    ref = EnergyReference.load(ref_dir)
    source_dir = ROOT / "runs" / ref.source.run_id
    candidates = [p for p in (ROOT / "runs").glob("*") if (p / "run.json").is_file()]
    for p in candidates:
        meta = json.loads((p / "run.json").read_text())
        if meta.get("run_id") == ref.source.run_id:
            source_dir = p
            break
    rebuilt_sha = None
    if (source_dir / "episodes.jsonl").is_file():
        rebuilt_sha = reference_from_run(source_dir, variant=ref.variant, eta=ref.eta).sha256()
    registered: dict[str, bool] = {}
    for a, m in runs.items():
        if a not in {"C", "C+S"}:
            continue
        meta = m.get("energy_ref")
        registered[a] = isinstance(meta, dict) and meta.get("sha256") == ref.sha256()
    return {
        "dir": str(ref_dir),
        "sha256": ref.sha256(),
        "variant": str(ref.variant),
        "eta": ref.eta,
        "source": ref.source.model_dump(mode="json"),
        "source_is_baseline_a_sim": ref.source.agent == "A" and ref.source.backend == "sim",
        "rebuilt_sha256": rebuilt_sha,
        "rebuild_matches": rebuilt_sha == ref.sha256(),
        "hash_registered_in_runs": registered,
        "tasks": {t: r.model_dump(mode="json") for t, r in ref.tasks.items()},
    }


def f8_tests() -> dict[str, object]:
    run = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         "tests/unit/test_agent_c.py"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )  # fmt: skip
    m = re.search(r"(\d+) passed", run.stdout)
    return {
        "tests_passed": int(m.group(1)) if m else 0,
        "tests_failed": run.returncode != 0,
        "summary": run.stdout.strip().splitlines()[-1] if run.stdout.strip() else run.stderr[-500:],
    }


def prompt_and_lever(records: dict[str, list[EpisodeRecord]], ref_sha: str) -> dict[str, object]:
    """Per agent: energy block presence, reference hash in traces, speed_scale < 1 usage."""
    out: dict[str, object] = {}
    for agent, recs in records.items():
        traces = [r.trace for r in recs if r.trace is not None]
        with_block = [t for t in traces if t.energy_reference_wh is not None]
        per_task: dict[str, dict[str, float]] = {}
        for r in recs:
            if r.trace is None:
                continue
            d = per_task.setdefault(r.task, {"n": 0, "slow": 0, "moves": 0})
            d["n"] += 1
            d["slow"] += 1 if r.trace.n_slow_moves else 0
            d["moves"] += r.trace.n_moves
        mins = [t.speed_scale_min for t in traces if t.speed_scale_min is not None]
        out[agent] = {
            "n_episodes": len(recs),
            "n_with_energy_block": len(with_block),
            "reference_hash_in_every_trace": bool(with_block)
            and all(t.energy_reference_sha256 == ref_sha for t in with_block),
            "n_with_speed_scale_below_1": sum(1 for t in traces if t.n_slow_moves),
            "lever_episode_rate": (sum(1 for t in traces if t.n_slow_moves) / len(traces))
            if traces
            else 0.0,
            "speed_scale_min": min(mins) if mins else None,
            "per_task": per_task,
        }
    return out


def pilot(reports: dict[str, Report]) -> list[dict[str, object]]:
    """Paired per-task comparison B vs C and B+S vs C+S; no threshold (H2)."""
    rows: list[dict[str, object]] = []
    for base, aware in PAIRS:
        for tb in reports[base].tasks:
            ta = next((t for t in reports[aware].tasks if t.task == tb.task), None)
            if ta is None:
                continue
            wh_b = tb.energy[0].wh.median if tb.energy else None
            wh_a = ta.energy[0].wh.median if ta.energy else None
            rows.append(
                {"task": tb.task, "pair": f"{aware} vs {base}",
                 base: {"n": tb.n, "success": tb.success, "ci95": tb.success_ci95,
                        "wh_A_eta_median": wh_b, "sim_s_median": tb.sim_s.median,
                        "primitives_median": tb.n_primitives.median,
                        "moves_median": tb.llm.moves_per_episode.median if tb.llm else None,
                        "lever_episode_rate": tb.llm.slow_move_episode_rate if tb.llm else None,
                        "tokens_per_episode_median": tb.llm.tokens_per_episode.median
                        if tb.llm else None},
                 aware: {"n": ta.n, "success": ta.success, "ci95": ta.success_ci95,
                         "wh_A_eta_median": wh_a, "sim_s_median": ta.sim_s.median,
                         "primitives_median": ta.n_primitives.median,
                         "moves_median": ta.llm.moves_per_episode.median if ta.llm else None,
                         "lever_episode_rate": ta.llm.slow_move_episode_rate if ta.llm else None,
                         "energy_reference_wh": ta.llm.energy_reference_wh if ta.llm else None,
                         "wh_over_reference_median": ta.llm.wh_over_reference.median
                         if ta.llm and ta.llm.wh_over_reference.n else None,
                         "tokens_per_episode_median": ta.llm.tokens_per_episode.median
                         if ta.llm else None},
                 "wh_ratio_aware_over_base": (wh_a / wh_b) if wh_a and wh_b else None}
            )  # fmt: skip
    return rows


def cost_aborts(records: dict[str, list[EpisodeRecord]]) -> dict[str, object]:
    """Episodes the provider or agent stopped for money/tokens, and the spend against the cap."""
    aborted = [
        f"{a}:{r.task}:{r.seed}"
        for a, recs in records.items()
        for r in recs
        if r.failure is not None and "budget" in r.failure.code
    ]
    spent = {
        a: sum(r.trace.cost_usd or 0.0 for r in recs if r.trace is not None)
        for a, recs in records.items()
    }
    cap = load_llm_params().max_usd_per_run
    return {"n_aborts": len(aborted), "aborted": aborted, "spent_usd": spent, "cap_usd": cap,
            "under_cap": all(v <= cap for v in spent.values())}  # fmt: skip


def judge(
    ref: dict[str, object],
    tests: dict[str, object],
    lever: dict[str, object],
    cost: dict[str, object],
    reports: dict[str, Report],
) -> dict[str, bool]:
    def stat(agent: str, key: str) -> object:
        block = lever[agent]
        return block[key] if isinstance(block, dict) else None

    registered = ref["hash_registered_in_runs"]
    all_tasks = [t for r in reports.values() for t in r.tasks]
    return {
        "f8_tests_pass": int(str(tests["tests_passed"])) > 0 and not bool(tests["tests_failed"]),
        "reference_is_baseline_a_rebuilt_bit_for_bit": bool(ref["source_is_baseline_a_sim"])
        and bool(ref["rebuild_matches"]),
        "reference_hash_registered_in_c_runs_and_traces": isinstance(registered, dict)
        and all(registered.values())
        and all(bool(stat(a, "reference_hash_in_every_trace")) for a in ("C", "C+S")),
        "energy_block_in_every_c_prompt_and_no_b_prompt": all(
            stat(a, "n_with_energy_block") == stat(a, "n_episodes") for a in ("C", "C+S")
        )
        and all(stat(a, "n_with_energy_block") == 0 for a in ("B", "B+S")),
        "c_lever_usage_reported": all(
            isinstance(stat(a, "lever_episode_rate"), float) for a in ("C", "C+S")
        ),
        "pilot_reported_for_both_pairs": all(
            reports[a].tasks and all(t.llm is not None for t in reports[a].tasks)
            for pair in PAIRS
            for a in pair
        ),
        "no_cost_abort": cost["n_aborts"] == 0 and bool(cost["under_cap"]),
        "no_infra_failures": all(t.n_infra == 0 for t in all_tasks),
        "energy_tables_complete": all(t.energy_complete == t.n for t in all_tasks),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--b", type=Path, required=True, help="agent B run (sim)")
    parser.add_argument("--c", type=Path, required=True, help="agent C run (sim)")
    parser.add_argument("--bs", type=Path, required=True, help="agent B+S run (sim)")
    parser.add_argument("--cs", type=Path, required=True, help="agent C+S run (sim)")
    parser.add_argument("--energy-ref", type=Path, default=ROOT / "energy_ref")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "f8_energy_aware.json")
    args = parser.parse_args(argv)

    dirs = {"B": args.b, "C": args.c, "B+S": args.bs, "C+S": args.cs}
    n_rows = len(Variant) * len(load_energy_params().etas())
    reports = {a: build_report([d / "episodes.jsonl"], n_rows_expected=n_rows)
               for a, d in dirs.items()}  # fmt: skip
    records = {a: read_records([d / "episodes.jsonl"]) for a, d in dirs.items()}
    runs = {a: json.loads((d / "run.json").read_text()) for a, d in dirs.items()}
    ref = reference_check(args.energy_ref, runs)
    tests = f8_tests()
    lever = prompt_and_lever(records, str(ref["sha256"]))
    cost = cost_aborts(records)
    checks = judge(ref, tests, lever, cost, reports)
    out = {
        "phase": "F8",
        "armbench": __version__,
        "git_sha": runs["C"].get("git_sha"),
        "decision_d8": "C and C+S receive baseline A's Wh for the same task (median over the "
        "dev seeds, variant A, nominal eta) as the reference budget; never their own previous "
        "attempt (max_attempts = 1 keeps the pairing with B).",
        "runs": runs,
        "energy_reference": ref,
        "f8_tests": tests,
        "prompt_and_lever": lever,
        "cost": cost,
        "pilot": pilot(reports),
        **{a: r.model_dump(mode="json") for a, r in reports.items()},
        "thresholds": {
            "lever_usage": "reported",
            "pilot_improvement": None,
            "cost_aborts_max": 0,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    for name, ok in checks.items():
        print(f"{'PASS' if ok else 'FAIL'} {name}")
    print(f"-> {args.out}")
    return 0 if out["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
