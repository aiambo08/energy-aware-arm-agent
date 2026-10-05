"""Record the simulated RGB camera while one episode runs in Gazebo, then encode MP4 + GIF.

    uv run python scripts/media/record_episode.py --task sort3@1 --agent A --seed 0 \
        --out docs/media/gazebo_sort3

Needs Docker, the ``armbench-sim:dev`` image and ffmpeg. Only rule-based agent A or a cached LLM
agent (``--llm-dir``) make sense here; the frames are illustration, not evaluation data.
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
import uuid
from pathlib import Path

import cv2

from armbench.runner.docker import CONTAINER_OUT, DOCKER, ENTRYPOINT, LAUNCH

HERE = Path(__file__).resolve().parent


def sh(cmd: list[str], timeout: float = 600) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)


def drop_blank(frames: Path) -> int:
    """Remove the black frames the camera publishes before the scene renders; renumber the rest."""
    kept = []
    for f in sorted(frames.glob("*.jpg")):
        img = cv2.imread(str(f))
        if img is not None and img.mean() > 10:
            kept.append(f)
    for f in frames.glob("*.jpg"):
        if f not in kept:
            f.unlink()
    for i, f in enumerate(kept):
        f.rename(frames / f"k{i:05d}.jpg")
    for f in frames.glob("k*.jpg"):
        f.rename(frames / f.name[1:])
    return len(kept)


def encode(frames: Path, out: Path, fps: int, speed: float, width: int) -> None:
    ffmpeg = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps)]
    src = [*ffmpeg, "-i", str(frames / "%05d.jpg")]
    pts = f"setpts=PTS/{speed}"
    sh([*src, "-vf", f"{pts},scale={width}:-2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(out.with_suffix(".mp4"))])  # fmt: skip
    pal = (
        f"{pts},fps=8,scale={width}:-1:flags=lanczos,split[a][b];"
        "[a]palettegen=max_colors=48[p];[b][p]paletteuse=dither=none"
    )
    sh([*src, "-vf", pal, str(out.with_suffix(".gif"))])


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--task", default="sort3@1")
    ap.add_argument("--agent", default="A")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--image", default="armbench-sim:dev")
    ap.add_argument("--out", type=Path, required=True, help="output path without extension")
    ap.add_argument("--fps", type=int, default=15, help="camera rate in the world file")
    ap.add_argument("--speed", type=float, default=2.0, help="playback speed-up")
    ap.add_argument("--width", type=int, default=480)
    a = ap.parse_args()
    name = f"armbench-media-{uuid.uuid4().hex[:8]}"
    started = sh([DOCKER, "run", "-d", "--network", "none", "--name", name, a.image, "bash", "-c",
                  f"{ENTRYPOINT} {LAUNCH} > /work/sim.log 2>&1"], 120)  # fmt: skip
    if started.returncode != 0:
        print(started.stderr)
        return 1
    try:
        sh([DOCKER, "cp", str(HERE / "camera_recorder.py"), f"{name}:/work/rec.py"])
        sh([DOCKER, "exec", "-d", name, ENTRYPOINT, "python3", "/work/rec.py",
            "--out", "/work/frames", "--stop", "/work/stop"])  # fmt: skip
        runner = ["ros2", "run", "armbench_bringup", "episode_runner", "--task", a.task]
        runner += ["--agent", a.agent, "--seeds", str(a.seed), "--run-id", "media"]
        runner += ["--out-dir", CONTAINER_OUT, "--image", a.image]
        run = sh([DOCKER, "exec", name, ENTRYPOINT, *runner], 900)
        print(run.stdout[-1500:], run.stderr[-1500:])
        sh([DOCKER, "exec", name, "touch", "/work/stop"])
        sh(["sleep", "2"])
        with tempfile.TemporaryDirectory() as tmp:
            frames = Path(tmp) / "frames"
            sh([DOCKER, "cp", f"{name}:/work/frames", str(frames)])
            n = drop_blank(frames)
            print(f"{n} frames")
            if n == 0:
                return 1
            a.out.parent.mkdir(parents=True, exist_ok=True)
            encode(frames, a.out, a.fps, a.speed, a.width)
    finally:
        sh([DOCKER, "rm", "-f", name], 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
