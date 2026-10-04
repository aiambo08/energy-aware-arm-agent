"""Run ``episode_runner`` inside fresh, network-isolated simulation containers.

A long-lived Gazebo world degrades after a few dozen spawn/remove cycles (F3), so seeds are
chunked and every chunk gets its own container; DDS discovery is disabled with
``--network none`` so parallel containers never see each other's topics.
"""

from __future__ import annotations

import shutil
import subprocess
import time
import uuid
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_IMAGE = "armbench-sim:dev"
ENTRYPOINT = "/usr/local/bin/entrypoint.sh"
CONTAINER_OUT = "/work/armbench_run"
LAUNCH = "ros2 launch armbench_bringup sim.launch.py headless:=true"
DOCKER = shutil.which("docker") or "docker"


class SimRunOptions(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    task: str
    agent: str = "A"
    seeds: list[int]
    repeat: int = Field(ge=1, default=1)
    chunk: int = Field(ge=1, default=25)
    image: str = DEFAULT_IMAGE
    mcap: bool = False
    final_eval: bool = False
    protocol_hash: str | None = None
    git_sha: str | None = None
    timeout_s: float = Field(gt=0, default=3600.0)
    """Wall limit for one chunk (readiness + all its episodes)."""


def chunks(seeds: Sequence[int], size: int) -> list[list[int]]:
    return [list(seeds[i : i + size]) for i in range(0, len(seeds), size)]


def seed_spec(seeds: Sequence[int]) -> str:
    return ",".join(str(s) for s in seeds)


def _run(cmd: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - fixed executable, our own arguments
        cmd, capture_output=True, text=True, timeout=timeout, check=False
    )


def runner_args(opts: SimRunOptions, seeds: Sequence[int], run_id: str) -> list[str]:
    args = [
        "ros2", "run", "armbench_bringup", "episode_runner",
        "--task", opts.task, "--agent", opts.agent, "--seeds", seed_spec(seeds),
        "--repeat", str(opts.repeat), "--run-id", run_id, "--out-dir", CONTAINER_OUT,
        "--image", opts.image,
    ]  # fmt: skip
    if opts.mcap:
        args.append("--mcap")
    if opts.final_eval:
        args.append("--final-eval")
    if opts.protocol_hash:
        args += ["--protocol-hash", opts.protocol_hash]
    if opts.git_sha:
        args += ["--git-sha", opts.git_sha]
    return args


def run_chunk(
    opts: SimRunOptions, seeds: Sequence[int], run_id: str, out_dir: Path
) -> dict[str, object]:
    """One container: launch the sim, run the episodes, copy the results into ``out_dir``."""
    name = f"armbench-run-{uuid.uuid4().hex[:8]}"
    t0 = time.time()
    info: dict[str, object] = {"container": name, "seeds": list(seeds), "ok": False}
    _run([DOCKER, "rm", "-f", name], 60)
    started = _run(
        [
            DOCKER,
            "run",
            "-d",
            "--network",
            "none",
            "--name",
            name,
            opts.image,
            "bash",
            "-c",
            f"{ENTRYPOINT} {LAUNCH} > /work/sim.log 2>&1",
        ],
        120,
    )
    if started.returncode != 0:
        info["error"] = f"docker run failed: {started.stderr.strip()}"
        return info
    try:
        proc = _run(
            [DOCKER, "exec", name, ENTRYPOINT, *runner_args(opts, seeds, run_id)], opts.timeout_s
        )
        info["rc"] = proc.returncode
        info["stderr_tail"] = proc.stderr[-2000:]
        out_dir.mkdir(parents=True, exist_ok=True)
        chunk_dir = out_dir / name
        copied = _run([DOCKER, "cp", f"{name}:{CONTAINER_OUT}", str(chunk_dir)], 600)
        if copied.returncode != 0:
            info["error"] = f"docker cp failed: {copied.stderr.strip()}"
            return info
        log = chunk_dir / "episodes.jsonl"
        had_log = log.exists()
        if had_log:
            with (out_dir / "episodes.jsonl").open("a") as dst, log.open() as src:
                shutil.copyfileobj(src, dst)
            for sub in ("samples", "mcap"):
                src_dir = chunk_dir / sub
                if src_dir.exists():
                    dst_dir = out_dir / sub
                    dst_dir.mkdir(exist_ok=True)
                    for item in src_dir.iterdir():
                        shutil.move(str(item), dst_dir / item.name)
        status = chunk_dir / "status.json"
        if status.exists():
            (out_dir / "chunks").mkdir(exist_ok=True)
            shutil.move(str(status), out_dir / "chunks" / f"{name}.json")
        shutil.rmtree(chunk_dir, ignore_errors=True)
        info["ok"] = proc.returncode == 0 and had_log
    except subprocess.TimeoutExpired:
        info["error"] = f"chunk exceeded {opts.timeout_s:.0f} s"
    finally:
        _run([DOCKER, "rm", "-f", name], 60)
        info["wall_s"] = round(time.time() - t0, 1)
    return info


def run_sim(
    opts: SimRunOptions, out_dir: Path, run_id: str | None = None
) -> list[dict[str, object]]:
    run_id = run_id or uuid.uuid4().hex[:12]
    return [run_chunk(opts, c, run_id, out_dir) for c in chunks(opts.seeds, opts.chunk)]
