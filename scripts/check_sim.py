"""F1 gate: boot the full armbench simulation and run the in-container self-check.

Runs on the host (needs Docker). For each repetition it starts a container running
``ros2 launch armbench_bringup sim.launch.py headless:=true world:=check_scene.sdf``, then
executes ``ros2 run armbench_bringup sim_check`` inside it, which waits for the controllers,
measures camera FPS and real-time factor, sweeps every arm joint, opens/closes the gripper
and performs a fixed-waypoint pick-and-place of ``cube_0`` (ground-truth slip from Gazebo).
Aggregates the per-run JSON into ``reports/f1_sim.json``; exits non-zero if any threshold
fails.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CONTAINER = "armbench-f1-check"
IN_CONTAINER_REPORT = "/tmp/sim_check.json"  # noqa: S108 - path inside the throwaway container
WORLD = (
    "/work/ros_ws/install/armbench_description/share/armbench_description/worlds/check_scene.sdf"
)
LAUNCH_CMD = f"ros2 launch armbench_bringup sim.launch.py headless:=true world:={WORLD}"

THRESHOLDS: dict[str, float] = {
    "ready_s": 60.0,  # controllers active, measured from `docker run`
    "rtf_min": 0.5,  # real-time factor while the arm moves (headless, camera on)
    "camera_fps_min": 10.0,  # RGB frames per simulated second
    "grasp_success_rate_min": 0.96,  # 48/50
    "grasp_slip_max_m": 0.005,
}


def run(cmd: list[str], timeout: float | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout)


def stop_container() -> None:
    run(["docker", "rm", "-f", CONTAINER], timeout=60)


def check_once(image: str, ready_timeout_s: float, no_grasp: bool) -> dict[str, Any]:
    stop_container()
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
            image,
            "bash",
            "-c",
            LAUNCH_CMD,
        ]
    )
    if started.returncode != 0:
        return {"ok": False, "error": f"docker run failed: {started.stderr.strip()}"}
    cmd = [
        "docker",
        "exec",
        CONTAINER,
        "/usr/local/bin/entrypoint.sh",
        "ros2",
        "run",
        "armbench_bringup",
        "sim_check",
        "--launch-start",
        repr(t0),
        "--timeout",
        str(ready_timeout_s),
        "--out",
        IN_CONTAINER_REPORT,
    ]
    if no_grasp:
        cmd.append("--no-grasp")
    try:
        proc = run(cmd, timeout=ready_timeout_s + 300)
    except subprocess.TimeoutExpired:
        logs = run(["docker", "logs", "--tail", "60", CONTAINER], timeout=30)
        stop_container()
        return {
            "ok": False,
            "error": "sim_check timed out",
            "container_log_tail": logs.stdout[-3000:],
        }
    report = run(["docker", "exec", CONTAINER, "cat", IN_CONTAINER_REPORT], timeout=30)
    logs = run(["docker", "logs", "--tail", "60", CONTAINER], timeout=30)
    stop_container()
    result: dict[str, Any] = {"ok": proc.returncode == 0, "wall_s": round(time.time() - t0, 2)}
    if report.returncode == 0 and report.stdout.strip():
        result["check"] = json.loads(report.stdout)
    else:
        result["ok"] = False
        result["error"] = (proc.stderr.strip() or proc.stdout.strip())[-2000:]
        result["container_log_tail"] = (logs.stdout + logs.stderr)[-3000:]
    return result


def _vals(runs: list[dict[str, Any]], *keys: str) -> list[float]:
    out: list[float] = []
    for r in runs:
        node: Any = r.get("check")
        for k in keys:
            node = node.get(k) if isinstance(node, dict) else None
        if isinstance(node, int | float):
            out.append(float(node))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="armbench-sim:dev")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, default=Path("reports/f1_sim.json"))
    parser.add_argument("--ready-timeout-s", type=float, default=THRESHOLDS["ready_s"])
    parser.add_argument("--no-grasp", action="store_true", help="skip the pick-and-place")
    args = parser.parse_args(argv)

    runs = [
        check_once(args.image, args.ready_timeout_s, args.no_grasp) for _ in range(args.repeats)
    ]
    n = len(runs)
    booted = [r for r in runs if "check" in r and r["check"].get("controllers_active")]
    ready = _vals(runs, "ready_s_from_launch")
    rtf = _vals(runs, "rtf_moving")
    fps = _vals(runs, "camera", "rgb_fps_sim")
    grasps = [r["check"]["grasp"] for r in runs if "check" in r and "grasp" in r["check"]]
    grasp_ok = sum(1 for g in grasps if g.get("ok"))
    slips = [g["transport_slip_m"] for g in grasps if g.get("transport_slip_m") is not None]
    effort_ok = all(
        r["check"].get("joint_states", {}).get("effort_for_all_joints")
        for r in runs
        if "check" in r
    )
    sweeps_ok = all(r["check"].get("joint_sweep", {}).get("ok") for r in runs if "check" in r)
    gripper_ok = all(r["check"].get("gripper", {}).get("ok") for r in runs if "check" in r)

    summary = {
        "runs": n,
        "boots_ok": len(booted),
        "ready_max_s": max(ready) if ready else None,
        "ready_mean_s": round(sum(ready) / len(ready), 2) if ready else None,
        "rtf_moving_min": min(rtf) if rtf else None,
        "camera_fps_sim_min": round(min(fps), 2) if fps else None,
        "effort_published_all_runs": effort_ok and len(booted) == n,
        "joint_sweep_all_runs": sweeps_ok and len(booted) == n,
        "gripper_all_runs": gripper_ok and len(booted) == n,
        "grasp_attempts": len(grasps),
        "grasp_ok": grasp_ok,
        "grasp_success_rate": round(grasp_ok / len(grasps), 4) if grasps else None,
        "grasp_slip_max_m": max(slips) if slips else None,
    }
    passed = (
        summary["boots_ok"] == n
        and ready
        and max(ready) < THRESHOLDS["ready_s"]
        and rtf
        and min(rtf) >= THRESHOLDS["rtf_min"]
        and fps
        and min(fps) >= THRESHOLDS["camera_fps_min"]
        and summary["effort_published_all_runs"]
        and summary["joint_sweep_all_runs"]
        and summary["gripper_all_runs"]
    )
    if not args.no_grasp:
        passed = bool(
            passed
            and grasps
            and grasp_ok / len(grasps) >= THRESHOLDS["grasp_success_rate_min"]
            and (not slips or max(slips) <= THRESHOLDS["grasp_slip_max_m"])
        )

    report = {
        "phase": "F1",
        "check": "ur5e_gripper_camera_headless",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "image": args.image,
        "world": WORLD,
        "host": {
            "cpus": os.cpu_count(),
            "platform": platform.platform(),
            "ci": os.environ.get("GITHUB_ACTIONS") == "true",
        },
        "thresholds": THRESHOLDS,
        "summary": summary,
        "runs": runs,
        "passed": bool(passed),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"summary": summary, "passed": report["passed"]}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
