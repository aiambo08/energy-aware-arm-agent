"""Build the static results page ``site/index.html`` (phase F10, GitHub Pages).

One self-contained HTML file (no CDN, no fetch, opens from ``file://``): the pre-registered
verdicts, per-cell success with Wilson 95 % intervals and Wh, the paired comparisons with their
bootstrap CIs, an interactive success-vs-Wh (Pareto) chart, and side-by-side 3D replays of B and
C on seed 100 of every task. The replays are forward kinematics of the 100 Hz joint samples
Gazebo logged during F9 (cubes and obstacle drawn at their seeded initial pose).

    uv run python scripts/build_site.py          # needs runs/f9_* (or the full dataset)
"""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
import sys
from pathlib import Path

from armbench.energy import Variant
from armbench.kinematics.ur5e import UR5eModel
from armbench.runner import EpisodeRecord
from armbench.runner.report import read_records
from armbench.scene.generator import load_scene_config
from armbench.tasks.registry import get_task

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "site" / "template.html"
AGENTS = ("A", "B", "B+S", "C", "C+S")
TASKS = ("pick_place@1", "stack2@1", "sort3@1", "place_obstacle@1")
REPLAY_AGENTS = ("B", "C")
REPLAY_SEED = 100
FRAME_STRIDE = 5


Runs = dict[str, tuple[Path, list[EpisodeRecord]]]


def wh(record: EpisodeRecord) -> float | None:
    if record.energy is None:
        return None
    for row in record.energy.rows:
        if row.variant == Variant.A and row.eta == 0.7:
            return round(row.total_wh, 5)
    return None


def load_runs(runs_root: Path, report: dict[str, object]) -> Runs:
    entries = report["runs"]
    if not isinstance(entries, list):
        msg = "report without a runs list"
        raise SystemExit(msg)
    runs: Runs = {}
    for r in entries:
        d = runs_root / str(r["dir"])
        runs[str(r["agent"])] = (d, read_records([d / "episodes.jsonl"]))
    return runs


def pareto(runs: Runs) -> list[dict[str, object]]:
    points: list[dict[str, object]] = []
    for task in (*TASKS, "all"):
        for agent in AGENTS:
            recs = [r for r in runs[agent][1] if task in ("all", r.task)]
            whs = [w for r in recs if r.ok and (w := wh(r)) is not None]
            points.append({
                "task": task,
                "agent": agent,
                "success": sum(r.ok for r in recs) / len(recs),
                "n": len(recs),
                "wh_median": round(statistics.median(whs), 5) if whs else None,
            })  # fmt: skip
    return points


def scene(task: str, seed: int) -> dict[str, object]:
    inst = get_task(task).instance(seed, load_scene_config())
    cubes = [
        {"color": c.color, "xyz": [round(c.x, 4), round(c.y, 4), round(c.z, 4)], "size": c.size}
        for c in inst.scene.cubes
    ]
    boxes = [{"xyz": [b.x, b.y, b.z], "yaw": b.yaw, "size": list(b.size)} for b in inst.obstacles]
    return {"cubes": cubes, "obstacles": boxes, "prompt": inst.prompt}


def replay(run_dir: Path, record: EpisodeRecord, model: UR5eModel) -> dict[str, object]:
    path = run_dir / "samples" / f"{record.episode_id}.jsonl.gz"
    rows = [json.loads(x) for x in gzip.decompress(path.read_bytes()).splitlines()]
    t0 = rows[0]["t"]
    frames = []
    for row in rows[::FRAME_STRIDE]:
        pts = [[0.0, 0.0, 0.0]] + [f[:3, 3].round(4).tolist() for f in model.fk_all(row["q"])]
        frames.append({"t": round(row["t"] - t0, 3), "p": pts})
    return {
        "ok": record.ok,
        "reason": record.reason,
        "wh": wh(record),
        "duration_s": round(rows[-1]["t"] - t0, 3),
        "frames": frames,
    }


def build(runs_root: Path, report_path: Path, out: Path) -> int:
    report = json.loads(report_path.read_text())
    runs = load_runs(runs_root, report)
    model = UR5eModel()
    replays = []
    for task in TASKS:
        item: dict[str, object] = {"task": task, "seed": REPLAY_SEED, **scene(task, REPLAY_SEED)}
        for agent in REPLAY_AGENTS:
            run_dir, recs = runs[agent]
            rec = next(r for r in recs if r.task == task and r.seed == REPLAY_SEED)
            item[agent] = replay(run_dir, rec, model)
        replays.append(item)
    data = {
        "protocol_sha256": report["protocol_sha256"],
        "report_sha256": report["report_sha256"],
        "n_scored": report["n_scored"],
        "valid_fraction": report["valid_fraction"],
        "verdicts": report["verdicts"],
        "cells": report["cells"],
        "comparisons": report["comparisons"],
        "pareto": pareto(runs),
        "replays": replays,
    }
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    html = TEMPLATE.read_text().replace("/*__DATA__*/null", blob)
    out.write_text(html)
    print(f"{out} {out.stat().st_size} bytes")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-root", type=Path, default=ROOT / "runs")
    ap.add_argument("--report", type=Path, default=ROOT / "reports" / "f9_eval.json")
    ap.add_argument("--out", type=Path, default=ROOT / "site" / "index.html")
    args = ap.parse_args(argv)
    return build(args.runs_root, args.report, args.out)


if __name__ == "__main__":
    sys.exit(main())
