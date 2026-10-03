"""Run the host quality gates (ruff, ruff format, mypy, pytest) and write a report.

Used by the F0 PR to produce ``reports/f0_quality.json``; later phases reuse it
with a different ``--phase``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

GATES: dict[str, list[str]] = {
    "ruff_check": ["uv", "run", "ruff", "check", "."],
    "ruff_format": ["uv", "run", "ruff", "format", "--check", "."],
    "mypy_strict": ["uv", "run", "mypy"],
    "pytest": ["uv", "run", "pytest", "-q", "-m", "not sim"],
}


def run_gate(cmd: list[str]) -> dict[str, Any]:
    t0 = time.monotonic()
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    tail = (proc.stdout + proc.stderr).strip().splitlines()[-3:]
    return {
        "cmd": " ".join(cmd),
        "returncode": proc.returncode,
        "seconds": round(time.monotonic() - t0, 2),
        "tail": tail,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", default="F0")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--max-total-s", type=float, default=180.0)
    args = parser.parse_args(argv)
    out = args.out or Path(f"reports/{args.phase.lower()}_quality.json")

    gates = {name: run_gate(cmd) for name, cmd in GATES.items()}
    total = round(sum(g["seconds"] for g in gates.values()), 2)
    passed = all(g["returncode"] == 0 for g in gates.values()) and total < args.max_total_s
    report = {
        "phase": args.phase,
        "check": "host_quality_gates",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "gates": gates,
        "total_seconds": total,
        "threshold_total_seconds": args.max_total_s,
        "passed": passed,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
