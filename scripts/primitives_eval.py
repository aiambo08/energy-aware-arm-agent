"""F4 gate: the robot primitives against the live simulation (docs/plan.es.md, F4 DoD).

1. ``contracts`` container: 200 random reachable ``move_to`` (position/orientation error,
   clearance monitor), 100 unreachable poses (``OutOfReach`` and the arm does not move),
   ``speed_scale`` sweep (monotone duration), 100 ``reset()`` from random poses.
2. ``pick`` containers (``--chunk`` seeds each, fresh simulation per chunk): scripted
   pick-and-place on seeded scenes judged with Gazebo cube poses.
3. ``reports/f4_primitives.json`` with the metrics and the pass/fail verdict (``THRESHOLDS``).

Usage: ``uv run python scripts/primitives_eval.py [--seeds 400-499] [--skip-contracts]``
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from armbench.primitives import DEFAULT_PRIMITIVES_FILE, load_primitive_params
from armbench.scene import load_scene_config

REPO = Path(__file__).resolve().parents[1]
CONTAINER = "armbench_f4_gate"
WORLD = "/work/ros_ws/install/armbench_description/share/armbench_description/worlds/tabletop.sdf"
LAUNCH_CMD = f"ros2 launch armbench_bringup sim.launch.py headless:=true world:={WORLD}"
IN_CONTAINER_OUT = "/tmp/f4"  # noqa: S108 - inside the throwaway container
TABLE_PENETRATION_M = 0.002
"""Finger tips (or any arm frame) this far below the table top count as a collision."""
THRESHOLDS: dict[str, float] = {
    "moves_min": 200,
    "pos_p95_mm_max": 5.0,
    "yaw_p95_deg_max": 2.0,
    "collisions_max": 0,
    "unreachable_min": 100,
    "unreachable_detected_min": 1.0,
    "unreachable_max_dq_rad_max": 1e-3,
    "resets_min": 100,
    "reset_sim_s_max": 5.0,
    "reset_q_err_rad_max": 0.02,
    "pick_seeds_min": 100,
    "pick_success_min": 0.95,
}


def run(cmd: list[str], timeout: float = 600) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)


@dataclass(frozen=True)
class GateOpts:
    image: str
    timeout_s: float
    chunk: int
    rng_seed: int


def run_section(opts: GateOpts, section: str, extra: list[str], out: Path) -> dict[str, object]:
    """Boot a fresh simulation, run ``primitives_check --section`` and copy its JSON to ``out``."""
    run(["docker", "rm", "-f", CONTAINER], timeout=60)
    t0 = time.time()
    started = run(
        ["docker", "run", "-d", "--network", "none", "--name", CONTAINER, opts.image, "bash",
         "-c", LAUNCH_CMD]
    )  # fmt: skip
    if started.returncode != 0:
        msg = f"docker run failed: {started.stderr.strip()}"
        raise RuntimeError(msg)
    in_out = f"{IN_CONTAINER_OUT}/{out.name}"
    try:
        proc = run(
            ["docker", "exec", CONTAINER, "/usr/local/bin/entrypoint.sh", "ros2", "run",
             "armbench_bringup", "primitives_check", "--section", section, "--out", in_out,
             "--timeout", str(opts.timeout_s), "--rng-seed", str(opts.rng_seed), *extra],
            timeout=3 * 3600,
        )  # fmt: skip
        out.parent.mkdir(parents=True, exist_ok=True)
        copied = run(["docker", "cp", f"{CONTAINER}:{in_out}", str(out)])
        if copied.returncode != 0:
            msg = (
                f"docker cp failed: {copied.stderr.strip()} (rc={proc.returncode}, "
                f"stderr tail: {proc.stderr[-2000:]})"
            )
            raise RuntimeError(msg)
    finally:
        run(["docker", "rm", "-f", CONTAINER], timeout=60)
    data = json.loads(out.read_text())
    data["rc"] = proc.returncode
    data["stderr_tail"] = proc.stderr[-3000:]
    data["wall_s_container"] = round(time.time() - t0, 1)
    out.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    print(f"{section} {extra}: rc={proc.returncode}, {data['wall_s_container']} s", flush=True)
    return dict(data)


def parse_seeds(spec: str) -> list[int]:
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in spec.split(",") if s]


def pct(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else math.nan


def section(report: dict[str, object], key: str) -> dict[str, object]:
    value = report.get(key)
    if not isinstance(value, dict):
        raise KeyError(key)
    return {str(k): v for k, v in value.items()}


def rows(report: dict[str, object], key: str) -> list[dict[str, object]]:
    value = report.get(key, [])
    if not isinstance(value, list):
        raise KeyError(key)
    return [{str(k): v for k, v in item.items()} for item in value if isinstance(item, dict)]


def num(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return math.nan
    return float(value)


def stats(values: list[float]) -> dict[str, float]:
    return {"median": pct(values, 50), "p95": pct(values, 95), "max": max(values, default=math.nan)}


def summarize_contracts(c: dict[str, object]) -> dict[str, object]:
    try:
        acc, unr, spd, rst = (section(c, k) for k in ("accuracy", "unreachable", "speed", "reset"))
    except KeyError:
        return {"error": c.get("error", "incomplete contracts section")}
    mv = rows(acc, "moves")
    pos = [num(m["pos_err_mm"]) for m in mv]
    yaw = [num(m["yaw_err_deg"]) for m in mv]
    collisions = [
        m
        for m in mv
        if num(m["min_tip_z_m"]) < -TABLE_PENETRATION_M
        or num(m["min_frame_z_m"]) < -TABLE_PENETRATION_M
    ]
    ratio = [num(m["sim_s"]) / num(m["planned_s"]) for m in mv]
    cases = rows(unr, "cases")
    detected = sum(1 for k in cases if k["raised"] == "out_of_reach")
    dq = [num(k.get("max_dq_rad")) for k in cases]
    goals = sum(int(num(k["goals_sent"])) for k in cases)
    sweep = rows(spd, "moves")
    durations = [num(m["sim_s"]) for m in sweep]
    scales = [num(m["speed_scale"]) for m in sweep]
    resets = rows(rst, "resets")
    r_sim = [num(r["sim_s"]) for r in resets]
    r_wall = [num(r["wall_s"]) for r in resets]
    r_q = [num(r["q_err_rad"]) for r in resets]
    return {
        "moves": {
            "n": len(mv),
            "rejected_during_sampling": len(rows(acc, "rejected")),
            "pos_mm": stats(pos),
            "yaw_deg": stats(yaw),
            "min_tip_z_m": min((num(m["min_tip_z_m"]) for m in mv), default=math.nan),
            "min_frame_z_m": min((num(m["min_frame_z_m"]) for m in mv), default=math.nan),
            "collisions": len(collisions),
            "sim_s_over_planned": {"median": pct(ratio, 50), "max": max(ratio, default=math.nan)},
        },
        "unreachable": {
            "n": len(cases),
            "detected_out_of_reach": detected,
            "detected_fraction": detected / len(cases) if cases else math.nan,
            "max_dq_rad": max(dq, default=math.nan),
            "goals_sent": goals,
            "other_codes": sorted(
                {str(k["raised"]) for k in cases if k["raised"] != "out_of_reach"}
            ),
        },
        "speed": {
            "scales": scales,
            "sim_s": durations,
            "strictly_decreasing": all(a > b for a, b in itertools.pairwise(durations)),
        },
        "reset": {
            "n": len(resets),
            "sim_s_max": max(r_sim, default=math.nan),
            "sim_s_median": pct(r_sim, 50),
            "wall_s_max": max(r_wall, default=math.nan),
            "q_err_rad_max": max(r_q, default=math.nan),
            "all_opened": all(num(r["opening_m"]) > 0.08 for r in resets),
        },
    }


def failure_code(episode: dict[str, object]) -> str:
    error = episode.get("error")
    return str(error.get("code", "judge")) if isinstance(error, dict) else "judge"


def summarize_pick(chunks: list[dict[str, object]]) -> dict[str, object]:
    episodes = [e for c in chunks for e in rows(c, "episodes")]
    ok = [e for e in episodes if e["ok"] is True]
    failed = [e for e in episodes if e["ok"] is not True]
    errors: dict[str, int] = {}
    for e in failed:
        errors[failure_code(e)] = errors.get(failure_code(e), 0) + 1
    task_s = [num(e["task_sim_s"]) for e in episodes if "task_sim_s" in e]
    tips = [num(e["min_tip_z_m"]) for e in episodes if "min_tip_z_m" in e]
    return {
        "n": len(episodes),
        "n_ok": len(ok),
        "success": len(ok) / len(episodes) if episodes else math.nan,
        "failures_by_code": errors,
        "failed_seeds": [int(num(e["seed"])) for e in failed],
        "place_err_mm": stats([num(e["place_err_mm"]) for e in ok]),
        "task_sim_s": {"median": pct(task_s, 50), "max": max(task_s, default=math.nan)},
        "min_tip_z_m": min(tips, default=math.nan),
        "detections_match_cubes": sum(1 for e in episodes if e.get("n_detections") == e["n_cubes"]),
        "scene_retries": sum(1 for e in episodes if e.get("retried") is True),
    }


def judge(contracts: dict[str, object], pick: dict[str, object]) -> dict[str, bool]:
    t = THRESHOLDS
    checks: dict[str, bool] = {}
    try:
        mv, unr, spd, rst = (
            section(contracts, k) for k in ("moves", "unreachable", "speed", "reset")
        )
    except KeyError:
        checks["contracts_complete"] = False
    else:
        checks["moves_n"] = num(mv["n"]) >= t["moves_min"]
        checks["pos_p95"] = num(section(mv, "pos_mm")["p95"]) < t["pos_p95_mm_max"]
        checks["yaw_p95"] = num(section(mv, "yaw_deg")["p95"]) < t["yaw_p95_deg_max"]
        checks["collisions"] = num(mv["collisions"]) <= t["collisions_max"]
        checks["unreachable_n"] = num(unr["n"]) >= t["unreachable_min"]
        checks["unreachable_detected"] = (
            num(unr["detected_fraction"]) >= t["unreachable_detected_min"]
        )
        checks["unreachable_no_motion"] = (
            num(unr["max_dq_rad"]) <= t["unreachable_max_dq_rad_max"]
            and num(unr["goals_sent"]) == 0
        )
        checks["speed_monotone"] = spd["strictly_decreasing"] is True
        checks["resets_n"] = num(rst["n"]) >= t["resets_min"]
        checks["reset_time"] = num(rst["sim_s_max"]) < t["reset_sim_s_max"]
        checks["reset_pose"] = (
            num(rst["q_err_rad_max"]) <= t["reset_q_err_rad_max"] and rst["all_opened"] is True
        )
    checks["pick_n"] = num(pick["n"]) >= t["pick_seeds_min"]
    checks["pick_success"] = num(pick["success"]) >= t["pick_success_min"]
    checks["passed"] = all(checks.values())
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="armbench-sim:dev")
    parser.add_argument(
        "--seeds", default="400-499", help="outside every range of configs/seeds.yaml"
    )
    parser.add_argument("--chunk", type=int, default=25, help="pick seeds per fresh container")
    parser.add_argument("--n-moves", type=int, default=200)
    parser.add_argument("--n-unreachable", type=int, default=100)
    parser.add_argument("--n-reset", type=int, default=100)
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=60.0, help="per wait inside the node")
    parser.add_argument("--raw", type=Path, default=REPO / "reports" / "_f4_raw")
    parser.add_argument("--out", type=Path, default=REPO / "reports" / "f4_primitives.json")
    parser.add_argument("--skip-contracts", action="store_true", help="reuse --raw/contracts.json")
    parser.add_argument("--skip-pick", action="store_true", help="reuse --raw/pick_*.json")
    args = parser.parse_args()
    opts = GateOpts(args.image, args.timeout, args.chunk, args.rng_seed)
    args.raw.mkdir(parents=True, exist_ok=True)
    contracts_path = args.raw / "contracts.json"
    if not args.skip_contracts:
        run_section(
            opts, "contracts",
            ["--n-moves", str(args.n_moves), "--n-unreachable", str(args.n_unreachable),
             "--n-reset", str(args.n_reset)],
            contracts_path,
        )  # fmt: skip
    seeds = parse_seeds(args.seeds)
    chunk_paths = []
    for i in range(0, len(seeds), args.chunk):
        part = seeds[i : i + args.chunk]
        path = args.raw / f"pick_{part[0]:04d}_{part[-1]:04d}.json"
        chunk_paths.append(path)
        if not args.skip_pick:
            run_section(opts, "pick", ["--seeds", f"{part[0]}-{part[-1]}"], path)
    contracts_raw = json.loads(contracts_path.read_text()) if contracts_path.exists() else {}
    pick_raw = [json.loads(p.read_text()) for p in chunk_paths if p.exists()]
    contracts = summarize_contracts(contracts_raw)
    pick = summarize_pick(pick_raw)
    checks = judge(contracts, pick)
    report: dict[str, object] = {
        "phase": "F4",
        "image": args.image,
        "seeds": args.seeds,
        "rng_seed": args.rng_seed,
        "primitives_config": str(DEFAULT_PRIMITIVES_FILE.relative_to(REPO)),
        "primitives_config_hash": load_primitive_params().model_dump_json(),
        "scene_config_hash": load_scene_config().config_hash(),
        "thresholds": THRESHOLDS,
        "table_penetration_m": TABLE_PENETRATION_M,
        "contracts": contracts,
        "pick_and_place": pick,
        "containers": {
            "contracts_wall_s": contracts_raw.get("wall_s_container"),
            "pick_wall_s": [c.get("wall_s_container") for c in pick_raw],
        },
        "checks": checks,
        "passed": checks["passed"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"checks": checks, "contracts": contracts, "pick": pick}, indent=1))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
