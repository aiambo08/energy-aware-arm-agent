"""F3 gate: ``detect()`` against Gazebo ground truth on 200 seeded scenes.

1. Boots the simulation image (``--network none``), runs the ``scene_capture`` node for
   ``--seeds`` (one RGB-D frame + PosePublisher poses per seed) and copies the dataset to
   ``--dataset`` (git-ignored). ``--skip-capture`` reuses an existing dataset.
2. Runs ``armbench.perception.detect`` on every frame on the host CPU and matches detections to
   cubes (same colour, nearest centre within ``MATCH_RADIUS_M``).
3. Writes ``reports/f3_perception.json`` with recall, false positives, XY/Z/yaw errors, latency
   and the pass/fail verdict against ``THRESHOLDS`` (docs/plan.es.md, F3 DoD).

Usage: ``uv run python scripts/perception_eval.py [--seeds 200-399] [--skip-capture]``
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from armbench.perception import Camera, Detection, detect, load_perception_params, yaw_difference
from armbench.perception.params import DEFAULT_PERCEPTION_FILE, PerceptionParams
from armbench.scene import load_scene_config

REPO = Path(__file__).resolve().parents[1]
CONTAINER = "armbench_f3_capture"
WORLD = "/work/ros_ws/install/armbench_description/share/armbench_description/worlds/tabletop.sdf"
LAUNCH_CMD = f"ros2 launch armbench_bringup sim.launch.py headless:=true world:={WORLD}"
IN_CONTAINER_DATASET = "/tmp/f3_dataset"  # noqa: S108 - inside the throwaway container
MATCH_RADIUS_M = 0.03
EDGE_MARGIN_PX = 20.0
THRESHOLDS: dict[str, float] = {
    "min_scenes": 200,
    "recall_min": 0.99,
    "false_positive_rate_max": 0.01,  # unmatched detections / ground-truth cubes
    "xy_median_mm_max": 5.0,
    "xy_p95_mm_max": 10.0,
    "z_p95_mm_max": 10.0,
    "latency_p95_ms_max": 50.0,
}


def run(cmd: list[str], timeout: float = 600.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)


@dataclass(frozen=True)
class CaptureOpts:
    image: str
    settle_s: float
    timeout_s: float
    chunk: int


def capture_chunk(opts: CaptureOpts, seeds: list[int], chunk_dir: Path) -> dict[str, object]:
    """Boot a fresh sim, run scene_capture for ``seeds`` and copy its dataset into ``chunk_dir``."""
    run(["docker", "rm", "-f", CONTAINER], timeout=60)
    t0 = time.time()
    started = run(
        [
            "docker",
            "run",
            "-d",
            "--network",
            "none",
            "--name",
            CONTAINER,
            opts.image,
            "bash",
            "-c",
            LAUNCH_CMD,
        ]
    )
    if started.returncode != 0:
        msg = f"docker run failed: {started.stderr.strip()}"
        raise RuntimeError(msg)
    try:
        proc = run(
            [
                "docker",
                "exec",
                CONTAINER,
                "/usr/local/bin/entrypoint.sh",
                "ros2",
                "run",
                "armbench_bringup",
                "scene_capture",
                "--seeds",
                ",".join(str(s) for s in seeds),
                "--out",
                IN_CONTAINER_DATASET,
                "--settle-s",
                str(opts.settle_s),
                "--timeout",
                str(opts.timeout_s),
            ],
            timeout=opts.timeout_s * 4 * len(seeds) + 300,
        )
        copied = run(["docker", "cp", f"{CONTAINER}:{IN_CONTAINER_DATASET}", str(chunk_dir)])
        if copied.returncode != 0:
            msg = f"docker cp failed: {copied.stderr.strip()} (capture rc={proc.returncode})"
            raise RuntimeError(msg)
    finally:
        run(["docker", "rm", "-f", CONTAINER], timeout=60)
    summary = json.loads((chunk_dir / "capture_summary.json").read_text())
    summary["capture_rc"] = proc.returncode
    summary["capture_stderr_tail"] = proc.stderr[-3000:]
    summary["wall_s_total"] = round(time.time() - t0, 1)
    return dict(summary)


def capture(opts: CaptureOpts, seeds: str, dataset: Path) -> dict[str, object]:
    """Capture all seeds, ``chunk`` per fresh container (long-lived worlds degrade: after a few
    hundred spawn/remove cycles gz-transport service calls start timing out). Returns the merged
    capture summary."""
    if dataset.exists():
        shutil.rmtree(dataset)
    dataset.mkdir(parents=True)
    all_seeds = parse_seeds(seeds)
    chunks: list[dict[str, object]] = []
    scenes: list[dict[str, object]] = []
    chunk = opts.chunk
    for i in range(0, len(all_seeds), chunk):
        chunk_dir = dataset / f"_chunk_{i // chunk:03d}"
        summary = capture_chunk(opts, all_seeds[i : i + chunk], chunk_dir)
        for d in sorted(chunk_dir.glob("seed_*")):
            shutil.move(str(d), dataset / d.name)
        chunk_scenes = summary.pop("scenes")
        if not isinstance(chunk_scenes, list):
            msg = "capture_summary.json without scenes list"
            raise RuntimeError(msg)
        scenes.extend(chunk_scenes)
        summary["seeds"] = all_seeds[i : i + chunk]
        chunks.append(summary)
        print(
            f"chunk {i // chunk}: {summary['n_ok']}/{summary['n']} ok, "
            f"rc={summary['capture_rc']}, {summary['wall_s_total']} s",
            flush=True,
        )
    return {
        "n": len(scenes),
        "n_ok": sum(1 for s in scenes if s["ok"]),
        "n_retried": sum(1 for s in scenes if s.get("retried")),
        "chunk_size": chunk,
        "wall_s_total": round(sum(float(str(c["wall_s_total"])) for c in chunks), 1),
        "chunks": chunks,
        "scenes": scenes,
    }


def parse_seeds(spec: str) -> list[int]:
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in spec.split(",") if s]


@dataclass
class Match:
    seed: int
    name: str
    color: str
    n_cubes: int
    xy_mm: float
    z_mm: float
    yaw_deg: float
    pixel_r: float  # distance of the GT top centre from the principal point [px]


@dataclass
class SceneResult:
    seed: int
    n_cubes: int
    n_visible: int
    n_detected: int
    n_matched: int
    n_false_positive: int
    latency_ms: float
    missed: list[str]


def load_frame(d: Path) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    bgr = cv2.imread(str(d / "rgb.png"), cv2.IMREAD_COLOR)
    if bgr is None:
        msg = f"cannot read {d / 'rgb.png'}"
        raise FileNotFoundError(msg)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    depth = np.load(d / "depth.npy")
    meta: dict[str, object] = json.loads((d / "meta.json").read_text())
    return rgb, depth, meta


def evaluate_scene(d: Path, params: PerceptionParams) -> tuple[SceneResult, list[Match], float]:
    rgb, depth, meta = load_frame(d)
    cam_meta = meta["camera"]
    if not isinstance(cam_meta, dict):
        msg = "meta.json: camera must be an object"
        raise TypeError(msg)
    k = tuple(float(v) for v in cam_meta["k"])
    camera = Camera.from_spec(params.camera, k)
    seed = int(str(meta["seed"]))
    cubes = meta["cubes"]
    if not isinstance(cubes, list):
        msg = "meta.json: cubes must be a list"
        raise TypeError(msg)
    detect(rgb, depth, params=params, camera=camera)  # warm-up (first call pays allocations)
    res = detect(rgb, depth, params=params, camera=camera)
    unmatched: list[Detection] = list(res.detections)
    matches: list[Match] = []
    missed: list[str] = []
    n_visible = 0
    gz_vs_req_max = 0.0
    for c in cubes:
        gz, req = c["gazebo"], c["requested"]
        gz_vs_req_max = max(
            gz_vs_req_max,
            math.hypot(gz["x"] - req["x"], gz["y"] - req["y"]),
            abs(gz["z"] - req["z"]),
        )
        top = np.array([[gz["x"], gz["y"], gz["z"] + params.cube_size_m / 2.0]])
        px, _ = camera.project_base(top)
        if not camera.intrinsics.in_image(px, EDGE_MARGIN_PX)[0]:
            continue  # outside the image: not a recall case
        n_visible += 1
        same = [
            dd
            for dd in unmatched
            if dd.color == c["color"]
            and math.hypot(dd.position[0] - gz["x"], dd.position[1] - gz["y"]) <= MATCH_RADIUS_M
        ]
        if not same:
            missed.append(str(c["name"]))
            continue
        best = min(
            same, key=lambda dd: math.hypot(dd.position[0] - gz["x"], dd.position[1] - gz["y"])
        )
        unmatched.remove(best)
        matches.append(
            Match(
                seed=seed,
                name=str(c["name"]),
                color=str(c["color"]),
                n_cubes=len(cubes),
                xy_mm=1e3 * math.hypot(best.position[0] - gz["x"], best.position[1] - gz["y"]),
                z_mm=1e3 * (best.position[2] - gz["z"]),
                yaw_deg=math.degrees(yaw_difference(best.yaw_rad, gz["yaw"])),
                pixel_r=float(
                    math.hypot(px[0, 0] - camera.intrinsics.cx, px[0, 1] - camera.intrinsics.cy)
                ),
            )
        )
    scene = SceneResult(
        seed=seed,
        n_cubes=len(cubes),
        n_visible=n_visible,
        n_detected=len(res.detections),
        n_matched=len(matches),
        n_false_positive=len(unmatched),
        latency_ms=res.latency_ms,
        missed=missed,
    )
    return scene, matches, gz_vs_req_max


def percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0}
    arr = np.asarray(values, dtype=float)
    return {
        "n": len(values),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


def summarise(
    scenes: list[SceneResult], matches: list[Match], gz_vs_req_max_m: float
) -> dict[str, object]:
    n_visible = sum(s.n_visible for s in scenes)
    n_matched = sum(s.n_matched for s in scenes)
    n_fp = sum(s.n_false_positive for s in scenes)
    n_gt = sum(s.n_cubes for s in scenes)
    xy = [m.xy_mm for m in matches]
    z = [abs(m.z_mm) for m in matches]
    yaw = [abs(m.yaw_deg) for m in matches]
    lat = [s.latency_ms for s in scenes]
    colours = sorted({m.color for m in matches})
    by_colour = {
        c: {
            "xy_mm": percentiles([m.xy_mm for m in matches if m.color == c]),
            "z_abs_mm": percentiles([abs(m.z_mm) for m in matches if m.color == c]),
        }
        for c in colours
    }
    worst = sorted(matches, key=lambda m: -m.xy_mm)[:5]
    metrics: dict[str, object] = {
        "n_scenes": len(scenes),
        "n_cubes_gt": n_gt,
        "n_cubes_visible": n_visible,
        "n_matched": n_matched,
        "n_missed": n_visible - n_matched,
        "n_false_positive": n_fp,
        "recall": n_matched / n_visible if n_visible else 0.0,
        "false_positive_rate": n_fp / n_gt if n_gt else 0.0,
        "xy_mm": percentiles(xy),
        "z_abs_mm": percentiles(z),
        "z_signed_mm_mean": float(np.mean([m.z_mm for m in matches])) if matches else 0.0,
        "yaw_abs_deg": percentiles(yaw),
        "latency_ms": percentiles(lat),
        "by_colour": by_colour,
        "scenes_with_misses": [s.seed for s in scenes if s.missed],
        "scenes_with_false_positives": [s.seed for s in scenes if s.n_false_positive],
        "worst_xy": [asdict(m) for m in worst],
        "gazebo_vs_requested_pose_max_mm": 1e3 * gz_vs_req_max_m,
    }
    return metrics


def verdict(metrics: dict[str, object]) -> dict[str, bool]:
    xy = metrics["xy_mm"]
    z = metrics["z_abs_mm"]
    lat = metrics["latency_ms"]
    if not (isinstance(xy, dict) and isinstance(z, dict) and isinstance(lat, dict)):
        msg = "metrics must contain percentile tables"
        raise TypeError(msg)
    n_scenes = metrics["n_scenes"]
    recall = metrics["recall"]
    fpr = metrics["false_positive_rate"]
    if not (isinstance(n_scenes, int) and isinstance(recall, float) and isinstance(fpr, float)):
        msg = "metrics has unexpected types"
        raise TypeError(msg)
    checks = {
        "scenes_ok": n_scenes >= THRESHOLDS["min_scenes"],
        "recall_ok": recall >= THRESHOLDS["recall_min"],
        "false_positive_ok": fpr <= THRESHOLDS["false_positive_rate_max"],
        "xy_median_ok": bool(xy.get("median", math.inf) < THRESHOLDS["xy_median_mm_max"]),
        "xy_p95_ok": bool(xy.get("p95", math.inf) < THRESHOLDS["xy_p95_mm_max"]),
        "z_p95_ok": bool(z.get("p95", math.inf) < THRESHOLDS["z_p95_mm_max"]),
        "latency_p95_ok": bool(lat.get("p95", math.inf) < THRESHOLDS["latency_p95_ms_max"]),
    }
    checks["passed"] = all(checks.values())
    return checks


def evaluate(dataset: Path, params: PerceptionParams) -> dict[str, object]:
    scenes: list[SceneResult] = []
    matches: list[Match] = []
    gz_vs_req = 0.0
    for d in sorted(p for p in dataset.iterdir() if p.is_dir() and p.name.startswith("seed_")):
        scene, m, g = evaluate_scene(d, params)
        scenes.append(scene)
        matches.extend(m)
        gz_vs_req = max(gz_vs_req, g)
    metrics = summarise(scenes, matches, gz_vs_req)
    return {"metrics": metrics, "checks": verdict(metrics), "scenes": [asdict(s) for s in scenes]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="armbench-sim:dev")
    parser.add_argument(
        "--seeds", default="200-399", help="outside every range of configs/seeds.yaml"
    )
    parser.add_argument("--dataset", type=Path, default=REPO / "reports" / "_f3_dataset")
    parser.add_argument("--out", type=Path, default=REPO / "reports" / "f3_perception.json")
    parser.add_argument("--settle-s", type=float, default=1.0)
    parser.add_argument(
        "--timeout", type=float, default=30.0, help="per wait inside the capture node"
    )
    parser.add_argument(
        "--chunk", type=int, default=25, help="seeds per fresh simulation container"
    )
    parser.add_argument(
        "--skip-capture", action="store_true", help="evaluate an existing --dataset"
    )
    args = parser.parse_args()
    params = load_perception_params()
    report: dict[str, object] = {
        "phase": "F3",
        "image": args.image,
        "seeds": args.seeds,
        "perception_config": str(DEFAULT_PERCEPTION_FILE.relative_to(REPO)),
        "scene_config_hash": load_scene_config().config_hash(),
        "thresholds": THRESHOLDS,
        "match_radius_m": MATCH_RADIUS_M,
    }
    if not args.skip_capture:
        report["capture"] = capture(
            CaptureOpts(args.image, args.settle_s, args.timeout, args.chunk),
            args.seeds,
            args.dataset,
        )
    report.update(evaluate(args.dataset, params))
    checks = report["checks"]
    if not isinstance(checks, dict):
        msg = "checks missing"
        raise TypeError(msg)
    report["passed"] = bool(checks["passed"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    metrics = report["metrics"]
    print(json.dumps({"passed": report["passed"], "checks": checks}, indent=1))
    if isinstance(metrics, dict):
        keep = (
            "n_scenes",
            "recall",
            "false_positive_rate",
            "xy_mm",
            "z_abs_mm",
            "yaw_abs_deg",
            "latency_ms",
        )
        print(json.dumps({k: metrics[k] for k in keep}, indent=1))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
