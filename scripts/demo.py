"""Replay one F9 episode without a GPU, an API key or Docker (phase F10 quickstart).

The committed dataset ``data/f9-v1.tar.gz`` carries every LLM answer of the pre-registered
evaluation. This script extracts it, seeds ``cache/llm`` with those answers, and re-runs the
chosen ``(task, agent, seed)`` on the kinematic backend with ``--provider replay`` (cache only,
never a network call). It then checks that the replayed prompt and program are identical to the
Gazebo episode of the dataset and prints both verdicts and the logged Wh. The primitive-call
sequence is reported, not required: a program that branches on its detections can take another
path on the kinematic backend, whose synthetic camera and exact motions are not Gazebo's.
The kinematic backend has no energy model, so the Wh shown is the one Gazebo measured in F9.

    uv run python scripts/demo.py --task pick_place --agent C --seed 100
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

from armbench.protocol import protocol_sha256

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "data" / "f9-v1.tar.gz"
WORK = ROOT / ".armbench"
RUN_DIRS = {"A": "f9_A", "B": "f9_B", "B+S": "f9_BSS", "C": "f9_C", "C+S": "f9_CSS"}
TASKS = ("pick_place@1", "stack2@1", "sort3@1", "place_obstacle@1")
SAME = ("prompt_sha256", "program_sha256")
INFO = ("program_calls", "skills_used")


def extract(bundle: Path, work: Path) -> Path:
    data = work / "f9-v1"
    if not (data / "MANIFEST.json").is_file():
        work.mkdir(parents=True, exist_ok=True)
        with tarfile.open(bundle) as tar:
            tar.extractall(work, filter="data")
    return data


def seed_cache(run: Path, cache: Path) -> int:
    added = 0
    for src in sorted((run / "llm" / "cache").rglob("*.json")):
        dst = cache / src.relative_to(run / "llm" / "cache")
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            added += 1
    return added


def find(records: Path, task: str, seed: int) -> dict[str, object]:
    for line in records.read_text().splitlines():
        r = json.loads(line)
        if r["task"] == task and r["seed"] == seed:
            return dict(r)
    msg = f"no episode {task} seed {seed} in {records}"
    raise SystemExit(msg)


def wh(record: dict[str, object]) -> float | None:
    energy = record.get("energy")
    if not isinstance(energy, dict):
        return None
    for row in energy["rows"]:
        if row["variant"] == "A" and row["eta"] == 0.7:
            return float(row["total_wh"])
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", default="pick_place", help="Task name, with or without @1.")
    ap.add_argument("--agent", default="C", choices=sorted(RUN_DIRS))
    ap.add_argument("--seed", type=int, default=100, help="One of the F9 seeds 100-119.")
    ap.add_argument("--bundle", type=Path, default=BUNDLE)
    args = ap.parse_args(argv)
    task = args.task if "@" in args.task else f"{args.task}@1"
    if task not in TASKS:
        ap.error(f"task must be one of {', '.join(TASKS)}")

    data = extract(args.bundle, WORK)
    run = data / RUN_DIRS[args.agent]
    added = seed_cache(run, ROOT / "cache" / "llm")
    logged = find(run / "episodes.jsonl", task, args.seed)
    out = WORK / "demo" / f"{RUN_DIRS[args.agent]}_{task.split('@')[0]}_{args.seed}"
    shutil.rmtree(out, ignore_errors=True)
    cmd = [
        "armbench", "run", "--task", task, "--agent", args.agent, "--seeds", str(args.seed),
        "--backend", "fake", "--provider", "replay", "--final-eval",
        "--protocol-hash", protocol_sha256(), "--llm-config", "configs/llm.nebius.yaml",
        "--out", str(out),
    ]  # fmt: skip
    print(f"dataset {data} ({added} cached answers added to cache/llm)", flush=True)
    print("$ " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=ROOT, check=False)
    if proc.returncode != 0:
        return proc.returncode
    replayed = find(out / "episodes.jsonl", task, args.seed)

    lt, rt = logged["trace"], replayed["trace"]
    if not isinstance(lt, dict) or not isinstance(rt, dict):
        msg = "episode records without an agent trace"
        raise SystemExit(msg)
    print(f"\n{args.agent} {task} seed {args.seed}")
    print(f"  Gazebo (F9 log):     ok={logged['ok']}  {logged['reason']}  Wh={wh(logged)}")
    print(f"  kinematic (replay):  ok={replayed['ok']}  {replayed['reason']}")
    mismatches = [k for k in SAME if lt.get(k) != rt.get(k)]
    for k in SAME:
        print(f"  {k:<16} {'same' if k not in mismatches else 'DIFFERENT'}")
    for k in INFO:
        state = "same" if lt.get(k) == rt.get(k) else "differs (program branches on what it sees)"
        print(f"  {k:<16} {state}")
    if mismatches:
        print("replay differs from the logged episode", file=sys.stderr)
        return 1
    print("replay served the logged LLM answer: same prompt and same program, no network call")
    return 0


if __name__ == "__main__":
    sys.exit(main())
