"""F7 gate: the frozen skill library and the B vs B+S pilot (Gazebo, dev seeds).

Inputs are two `armbench run` directories produced on the simulation backend with the same
tasks and seeds: --b (agent B, no library) and --bs (agent B+S, the frozen library in
`--skills`). Optional --fake-b/--fake-bs are the same pilot on the kinematic backend (host
latency and token counts come from there when the Gazebo bundles were served from the cache).

Thresholds (docs/plan.es.md, F7): post-conditions detect every injected failure (tests in CI),
the library is frozen with its hash registered in the B+S run, every skill passed >= 90 % of the
skill_validation seeds (20-39) in the sandbox, the B+S pilot actually exercised the library in
Gazebo, no infrastructure failures. The B vs B+S comparison (success, tokens, latency,
primitive calls, Wh) is *reported* without an improvement threshold: it is hypothesis H1 of the
protocol, not a requirement, and the template provider is not a language model.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from armbench import __version__
from armbench.energy import Variant, load_energy_params
from armbench.runner import EpisodeRecord, build_report
from armbench.runner.report import Report, read_records
from armbench.skills import FROZEN_FILE, SkillLibrary
from armbench.skills.validate import MIN_PASS_RATE, VALIDATION_SPLIT

ROOT = Path(__file__).resolve().parents[1]


def key(r: EpisodeRecord) -> tuple[str, int, int]:
    return (r.task, r.seed, r.repeat)


def library_check(skills_dir: Path, bs_run: dict[str, object]) -> dict[str, object]:
    """Frozen, hash registered in the run, every skill validated on the right split."""
    lib = SkillLibrary.load(skills_dir, require_frozen=True)
    manifest = json.loads((skills_dir / FROZEN_FILE).read_text())
    run_skills = bs_run.get("skills")
    registered = isinstance(run_skills, dict) and run_skills.get("sha256") == lib.sha256()
    skills = []
    for s in lib:
        v = s.validation
        skills.append(
            {"name": s.name, "signature": s.signature(), "sha256": s.sha256(),
             "origin_task": s.origin.task, "origin_run": s.origin.run_id,
             "origin_seeds": list(s.origin.seeds),
             "validation": None if v is None else {
                 "split": v.split, "n_seeds": len(v.seeds), "n_ok": v.n_ok,
                 "accepted": v.accepted, "failed": v.failed, "backend": v.backend}}
        )  # fmt: skip
    validated = all(
        s.validation is not None
        and s.validation.accepted
        and s.validation.split == VALIDATION_SPLIT
        and s.validation.n_ok >= MIN_PASS_RATE * len(s.validation.seeds)
        for s in lib
    )
    return {
        "dir": str(skills_dir),
        "frozen": lib.frozen,
        "sha256": lib.sha256(),
        "manifest_sha256": manifest["sha256"],
        "frozen_at": manifest["frozen_at"],
        "n_skills": len(lib),
        "hash_registered_in_run": registered,
        "all_validated": validated,
        "skills": skills,
    }


def fault_injection_tests() -> dict[str, object]:
    """The F7 tests that inject failures a post-condition must catch (and the sandbox battery
    cases for skills); their count and verdict, from a real pytest run."""
    run = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         "tests/unit/test_skills.py", "-k",
         "injected or postcondition or precondition or tampered or frozen or unknown"
         " or robot_failure"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )  # fmt: skip
    m = re.search(r"(\d+) passed", run.stdout)
    passed = int(m.group(1)) if m else 0
    return {
        "tests_passed": passed,
        "tests_failed": run.returncode != 0,
        "summary": run.stdout.strip().splitlines()[-1] if run.stdout.strip() else run.stderr[-500:],
    }


def skill_usage(bs: list[EpisodeRecord]) -> dict[str, object]:
    traces = [r.trace for r in bs if r.trace is not None]
    per_task: dict[str, dict[str, int]] = {}
    for r in bs:
        if r.trace is None:
            continue
        for name in r.trace.skills_used:
            per_task.setdefault(r.task, {})
            per_task[r.task][name] = per_task[r.task].get(name, 0) + 1
    return {
        "n_episodes": len(bs),
        "n_with_skill": sum(1 for t in traces if t.skills_used),
        "per_task": per_task,
        "inner_calls_counted": all(t.n_primitives >= len(t.program_calls) for t in traces),
    }


def _median(xs: list[float]) -> float | None:
    if not xs:
        return None
    ys = sorted(xs)
    mid = len(ys) // 2
    return ys[mid] if len(ys) % 2 else 0.5 * (ys[mid - 1] + ys[mid])


def pilot(
    b: Report, bs: Report, fake_b: Report | None, fake_bs: Report | None
) -> list[dict[str, object]]:
    """Paired per-task comparison; no threshold (H1)."""
    rows: list[dict[str, object]] = []
    for tb in b.tasks:
        tbs = next((t for t in bs.tasks if t.task == tb.task), None)
        if tbs is None:
            continue
        fb = next((t for t in (fake_b.tasks if fake_b else []) if t.task == tb.task), None)
        fbs = next((t for t in (fake_bs.tasks if fake_bs else []) if t.task == tb.task), None)
        wh_b = tb.energy[0].wh.median if tb.energy else None
        wh_bs = tbs.energy[0].wh.median if tbs.energy else None
        tok_b = tb.llm.tokens_per_episode.median if tb.llm else None
        tok_bs = tbs.llm.tokens_per_episode.median if tbs.llm else None
        rows.append(
            {"task": tb.task,
             "B": {"n": tb.n, "success": tb.success, "ci95": tb.success_ci95,
                   "tokens_per_episode_median": tok_b,
                   "primitives_median": tb.n_primitives.median,
                   "wh_A_eta_median": wh_b, "sim_s_median": tb.sim_s.median,
                   "latency_s_p95_host": fb.llm.latency_s.p95 if fb and fb.llm else None},
             "B+S": {"n": tbs.n, "success": tbs.success, "ci95": tbs.success_ci95,
                     "tokens_per_episode_median": tok_bs,
                     "primitives_median": tbs.n_primitives.median,
                     "skills_used": tbs.llm.skills_used if tbs.llm else {},
                     "wh_A_eta_median": wh_bs, "sim_s_median": tbs.sim_s.median,
                     "latency_s_p95_host": fbs.llm.latency_s.p95 if fbs and fbs.llm else None},
             "wh_ratio_BS_over_B": (wh_bs / wh_b) if wh_b and wh_bs else None}
        )  # fmt: skip
    return rows


def judge(
    lib: dict[str, object],
    tests: dict[str, object],
    usage: dict[str, object],
    b: Report,
    bs: Report,
) -> dict[str, bool]:
    return {
        "postconditions_catch_injected_failures": int(str(tests["tests_passed"])) > 0
        and not bool(tests["tests_failed"]),
        "library_frozen_hash_registered": bool(lib["frozen"])
        and lib["sha256"] == lib["manifest_sha256"]
        and bool(lib["hash_registered_in_run"]),
        "every_skill_validated_on_split_20_39": bool(lib["all_validated"]),
        "bs_exercised_library_in_gazebo": int(str(usage["n_with_skill"])) > 0
        and bool(usage["inner_calls_counted"]),
        "pilot_reported_for_b_and_bs": bool(b.tasks)
        and bool(bs.tasks)
        and all(t.llm is not None for t in b.tasks + bs.tasks),
        "no_infra_failures": all(t.n_infra == 0 for t in b.tasks + bs.tasks),
        "energy_tables_complete": all(t.energy_complete == t.n for t in b.tasks + bs.tasks),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--b", type=Path, required=True, help="agent B run (sim)")
    parser.add_argument("--bs", type=Path, required=True, help="agent B+S run (sim)")
    parser.add_argument("--fake-b", type=Path, default=None)
    parser.add_argument("--fake-bs", type=Path, default=None)
    parser.add_argument("--skills", type=Path, default=ROOT / "skills")
    parser.add_argument(
        "--validation", type=Path, default=ROOT / "reports" / "f7_skill_validation.json"
    )
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "f7_skills.json")
    args = parser.parse_args(argv)

    n_rows = len(Variant) * len(load_energy_params().etas())
    b = build_report([args.b / "episodes.jsonl"], n_rows_expected=n_rows)
    bs = build_report([args.bs / "episodes.jsonl"], n_rows_expected=n_rows)
    fake_b = (
        build_report([args.fake_b / "episodes.jsonl"], n_rows_expected=0) if args.fake_b else None
    )
    fake_bs = (
        build_report([args.fake_bs / "episodes.jsonl"], n_rows_expected=0) if args.fake_bs else None
    )
    bs_run = json.loads((args.bs / "run.json").read_text())
    b_run = json.loads((args.b / "run.json").read_text())
    lib = library_check(args.skills, bs_run)
    tests = fault_injection_tests()
    usage = skill_usage(read_records([args.bs / "episodes.jsonl"]))
    checks = judge(lib, tests, usage, b, bs)
    validation = json.loads(args.validation.read_text()) if args.validation.exists() else None
    out = {
        "phase": "F7",
        "armbench": __version__,
        "git_sha": bs_run.get("git_sha"),
        "runs": {"B": b_run, "B+S": bs_run},
        "library": lib,
        "validation": validation,
        "fault_injection_tests": tests,
        "skill_usage": usage,
        "pilot": pilot(b, bs, fake_b, fake_bs),
        "B": b.model_dump(mode="json"),
        "B+S": bs.model_dump(mode="json"),
        "fake_B": fake_b.model_dump(mode="json") if fake_b else None,
        "fake_B+S": fake_bs.model_dump(mode="json") if fake_bs else None,
        "thresholds": {
            "skill_min_pass_rate": MIN_PASS_RATE,
            "validation_split": VALIDATION_SPLIT,
            "pilot_improvement": None,
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
