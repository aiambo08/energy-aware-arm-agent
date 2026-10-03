"""F0 gate: boot an empty headless Gazebo world inside the simulation image.

Runs on the host (needs Docker). For each repetition it starts a container
running ``gz sim -s -r empty.sdf`` plus a ``ros_gz_bridge`` for ``/clock``,
then measures the wall time until ``ros2 topic echo /clock --once`` returns.
Writes a JSON report; exits non-zero if any threshold fails.
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

BOOT_THRESHOLD_S = 30.0
IMAGE_THRESHOLD_GB = 5.0
CONTAINER = "armbench-f0-check"

SIM_CMD = (
    "gz sim -s -r -v 1 empty.sdf & "
    "ros2 run ros_gz_bridge parameter_bridge "
    "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock & "
    "wait"
)


def run(cmd: list[str], timeout: float | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout)


_UNITS = {"B": 1, "kB": 1e3, "MB": 1e6, "GB": 1e9, "KiB": 2**10, "MiB": 2**20, "GiB": 2**30}


def parse_human_size(text: str) -> float:
    text = text.strip()
    for unit in sorted(_UNITS, key=len, reverse=True):
        if text.endswith(unit):
            return float(text[: -len(unit)]) * _UNITS[unit]
    return float(text)


def image_size_bytes(image: str) -> int:
    """Uncompressed size: sum of layer sizes from ``docker history``.

    ``docker image inspect .Size`` reports the compressed content size when the
    containerd image store is enabled, so it is not comparable across hosts.
    """
    out = run(["docker", "history", "--format", "{{.Size}}", image])
    if out.returncode != 0:
        msg = f"docker history failed: {out.stderr.strip()}"
        raise RuntimeError(msg)
    return int(sum(parse_human_size(line) for line in out.stdout.splitlines() if line.strip()))


def image_versions(image: str) -> dict[str, str]:
    gz = run(["docker", "run", "--rm", image, "gz", "sim", "--version"], timeout=120)
    ros = run(["docker", "run", "--rm", image, "bash", "-c", "echo $ROS_DISTRO"], timeout=120)
    return {
        "gz_sim": gz.stdout.strip().splitlines()[0] if gz.stdout.strip() else gz.stderr.strip(),
        "ros_distro": ros.stdout.strip(),
    }


def stop_container() -> None:
    run(["docker", "rm", "-f", CONTAINER], timeout=60)


def boot_once(image: str, timeout_s: float) -> dict[str, Any]:
    stop_container()
    t0 = time.monotonic()
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
            SIM_CMD,
        ],
        timeout=120,
    )
    if started.returncode != 0:
        return {"ok": False, "error": f"docker run failed: {started.stderr.strip()}"}
    echo = run(
        [
            "docker",
            "exec",
            CONTAINER,
            "/usr/local/bin/entrypoint.sh",
            "timeout",
            str(int(timeout_s) + 5),
            "ros2",
            "topic",
            "echo",
            "--once",
            "/clock",
        ],
        timeout=timeout_s + 30,
    )
    elapsed = time.monotonic() - t0
    logs = run(["docker", "logs", "--tail", "40", CONTAINER], timeout=30)
    stop_container()
    ok = echo.returncode == 0 and "clock" in echo.stdout
    result: dict[str, Any] = {"ok": ok, "boot_s": round(elapsed, 2)}
    if not ok:
        result["error"] = (echo.stderr.strip() or echo.stdout.strip())[-2000:]
        result["container_log_tail"] = (logs.stdout + logs.stderr)[-3000:]
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="armbench-sim:dev")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, default=Path("reports/f0_sim.json"))
    parser.add_argument("--boot-threshold-s", type=float, default=BOOT_THRESHOLD_S)
    parser.add_argument("--image-threshold-gb", type=float, default=IMAGE_THRESHOLD_GB)
    args = parser.parse_args(argv)

    size_b = image_size_bytes(args.image)
    size_gb = size_b / 1e9
    versions = image_versions(args.image)
    boots = [boot_once(args.image, args.boot_threshold_s) for _ in range(args.repeats)]
    times = [b["boot_s"] for b in boots if b["ok"]]
    all_ok = len(times) == len(boots)
    boot_max = max(times) if times else None
    passed = all_ok and boot_max is not None and boot_max < args.boot_threshold_s
    passed = passed and size_gb < args.image_threshold_gb

    report = {
        "phase": "F0",
        "check": "empty_world_headless_boot",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "image": args.image,
        "image_size_bytes": size_b,
        "image_size_gb": round(size_gb, 3),
        "versions": versions,
        "host": {
            "cpus": os.cpu_count(),
            "platform": platform.platform(),
            "ci": os.environ.get("GITHUB_ACTIONS") == "true",
        },
        "thresholds": {
            "boot_s": args.boot_threshold_s,
            "image_gb": args.image_threshold_gb,
        },
        "boots": boots,
        "boot_max_s": boot_max,
        "boot_mean_s": round(sum(times) / len(times), 2) if times else None,
        "passed": passed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
