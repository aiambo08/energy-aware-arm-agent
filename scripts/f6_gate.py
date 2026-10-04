"""F6 gate: agent B (template provider) in Gazebo, replay exactness, cost and the attack battery.

Inputs are two `armbench run` directories produced on the simulation backend:
  --live    agent B with the provider named in configs/llm.yaml (programs fetched on the host,
            replayed inside the containers from the run's own bundle);
  --replay  the same tasks/seeds run again with --provider replay (no provider call at all).

Thresholds (docs/plan.es.md, F6): >= 40 attacks stopped (tests in CI), malformed programs never
crash the runner (tests), replay reproduces the primitive calls of every episode, cost per
episode measured with a < 30 EUR extrapolation to the final evaluation, B >= 50 % success on
pick_place dev seeds, LLM latency p95 reported.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from armbench import __version__
from armbench.llm import Ledger
from armbench.runner import EpisodeRecord, build_report
from armbench.runner.report import Report

ROOT = Path(__file__).resolve().parents[1]
MAX_FINAL_EVAL_USD = 30.0
MIN_PICK_PLACE_SUCCESS = 0.5


def read_records(run_dir: Path) -> list[EpisodeRecord]:
    lines = (run_dir / "episodes.jsonl").read_text().splitlines()
    return [EpisodeRecord.model_validate_json(line) for line in lines if line.strip()]


def key(r: EpisodeRecord) -> tuple[str, int, int]:
    return (r.task, r.seed, r.repeat)


def replay_check(live: list[EpisodeRecord], replay: list[EpisodeRecord]) -> dict[str, object]:
    by_key = {key(r): r for r in live}
    compared = exact = 0
    mismatches: list[dict[str, object]] = []
    for r in replay:
        ref = by_key.get(key(r))
        if ref is None or ref.trace is None or r.trace is None:
            continue
        compared += 1
        same = (
            ref.trace.program_calls == r.trace.program_calls
            and ref.trace.program_sha256 == r.trace.program_sha256
            and ref.trace.response_sha256 == r.trace.response_sha256
            and ref.ok == r.ok
        )
        if same:
            exact += 1
        else:
            mismatches.append(
                {"task": r.task, "seed": r.seed, "live": ref.trace.program_calls,
                 "replay": r.trace.program_calls, "live_ok": ref.ok, "replay_ok": r.ok}
            )  # fmt: skip
    cached = sum(1 for r in replay if r.trace is not None and r.trace.llm_cached)
    return {
        "compared": compared,
        "exact": exact,
        "replay_cached": cached,
        "replay_n": len(replay),
        "mismatches": mismatches[:20],
    }


def attack_battery() -> dict[str, object]:
    """Count the sandbox attacks and run the sandbox tests, as CI does."""
    collected = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/unit/test_sandbox.py", "-q", "--co",
         "-k", "test_attack_battery"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )  # fmt: skip
    n_attacks = sum(1 for line in collected.stdout.splitlines() if "::test_attack_battery[" in line)
    run = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/unit/test_sandbox.py", "tests/unit/test_llm.py",
         "tests/unit/test_agent_b.py", "-q"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )  # fmt: skip
    m = re.search(r"(\d+) passed", run.stdout)
    return {
        "attack_cases": n_attacks,
        "tests_passed": int(m.group(1)) if m else 0,
        "tests_failed": run.returncode != 0,
        "summary": run.stdout.strip().splitlines()[-1] if run.stdout.strip() else run.stderr[-500:],
    }


def percentile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    return ys[min(len(ys) - 1, round(q * (len(ys) - 1)))]


def host_llm(live_dir: Path, n_episodes: int) -> dict[str, object]:
    """What the host-side prefetch cost: the containers only replay, so the ledger is the source."""
    ledger = Ledger.load(live_dir / "llm" / "ledger.json")
    per_episode = ledger.usd / n_episodes if n_episodes else 0.0
    return {
        "n_calls": ledger.n_calls,
        "n_cached": ledger.n_cached,
        "prompt_tokens": ledger.prompt_tokens,
        "completion_tokens": ledger.completion_tokens,
        "models": ledger.models,
        "usd": ledger.usd,
        "usd_per_episode": per_episode,
        "usd_per_400_episodes": per_episode * 400,
        "latency_s_p50": percentile(ledger.latencies_s, 0.5),
        "latency_s_p95": percentile(ledger.latencies_s, 0.95),
    }


def judge(
    report: Report,
    replay: dict[str, object],
    battery: dict[str, object],
    host: dict[str, object],
) -> dict[str, bool]:
    pick = [t for t in report.tasks if t.task.startswith("pick_place")]
    llm = [t.llm for t in report.tasks if t.llm is not None]
    final_eval_usd = float(str(host["usd_per_400_episodes"]))
    return {
        "attacks_ge_40_and_tests_green": int(str(battery["attack_cases"])) >= 40
        and not bool(battery["tests_failed"]),
        "replay_exact": replay["compared"] == replay["exact"]
        and int(str(replay["compared"])) > 0
        and replay["replay_cached"] == replay["replay_n"],
        "cost_measured_and_under_budget": bool(llm)
        and (int(str(host["n_calls"])) + int(str(host["n_cached"]))) > 0
        and final_eval_usd < MAX_FINAL_EVAL_USD,
        "pick_place_success_ge_50pct": bool(pick)
        and all(t.success >= MIN_PICK_PLACE_SUCCESS for t in pick),
        "latency_reported": int(str(host["n_calls"])) > 0,
        "no_infra_failures": all(t.n_infra == 0 for t in report.tasks),
        "energy_tables_complete": all(t.energy_complete == t.n for t in report.tasks),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--live", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--n-rows", type=int, required=True, help="expected rows per task table")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "f6_agent_b.json")
    args = parser.parse_args(argv)

    live, replay = read_records(args.live), read_records(args.replay)
    report = build_report([args.live / "episodes.jsonl"], n_rows_expected=args.n_rows)
    replay_report = build_report([args.replay / "episodes.jsonl"], n_rows_expected=args.n_rows)
    rep_check = replay_check(live, replay)
    battery = attack_battery()
    host = host_llm(args.live, len(live))
    checks = judge(report, rep_check, battery, host)
    run_meta = json.loads((args.live / "run.json").read_text())
    out = {
        "phase": "F6",
        "armbench": __version__,
        "git_sha": run_meta.get("git_sha"),
        "live_run": run_meta,
        "live": report.model_dump(mode="json"),
        "replay": replay_report.model_dump(mode="json"),
        "host_llm": host,
        "replay_check": rep_check,
        "attack_battery": battery,
        "thresholds": {
            "attacks_min": 40,
            "final_eval_budget_usd": MAX_FINAL_EVAL_USD,
            "pick_place_success_min": MIN_PICK_PLACE_SUCCESS,
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
