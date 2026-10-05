"""Package the F9 evaluation runs as a reproducible, checksummed dataset (phase F10).

Two bundles are built from the run directories named in ``reports/f9_eval.json``:

- ``core`` (committed as ``data/f9-v1.tar.gz``): per run ``run.json``, ``episodes.jsonl``, the
  generated programs and their sandbox traces, and the LLM bundle (response cache, ledger,
  profile, frozen skills, energy reference). Enough to re-run ``armbench analyze`` and to replay
  every LLM answer without a key.
- ``full`` (``--full``, for Zenodo): ``core`` plus the 100 Hz joint samples
  (``samples/*.jsonl.gz``) and the per-container chunk logs.

The archive is deterministic: sorted members, mtime 0, uid/gid 0, gzip mtime 0, so the same runs
always give the same sha256. ``MANIFEST.json`` inside lists the sha256 of every file.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "f9-v1"
CORE_PARTS = ("run.json", "episodes.jsonl", "programs", "llm")
FULL_PARTS = (*CORE_PARTS, "samples", "chunks", "chunks.json")

README = """# armbench F9 evaluation dataset ({kind})

Pre-registered evaluation `f9-v1` of https://github.com/aiambo08/energy-aware-arm-agent
(protocol sha256 `{protocol}`): agents A, B, B+S, C, C+S x tasks pick_place@1, stack2@1,
sort3@1, place_obstacle@1 x locked seeds 100-119 in Gazebo Harmonic, LLM
`Qwen/Qwen3-235B-A22B-Instruct-2507` (Nebius Token Factory, temperature 0).

- `<run>/run.json`: run metadata (agent, git sha, LLM profile, protocol hash).
- `<run>/episodes.jsonl`: one EpisodeRecord (schema v1) per episode: verdict, failure stage,
  Wh for energy variants A/B x eta 0.6/0.7/0.8, agent trace (tokens, cost, calls).
- `<run>/programs/`: the program the agent wrote per episode and its sandbox trace.
- `<run>/llm/cache/`: content-addressed LLM responses; `armbench run --provider replay` serves
  them without a network call.
{full_note}- `MANIFEST.json`: sha256 of every file in this bundle.

Run directories `f9_BSS` and `f9_CSS` hold agents B+S and C+S (the agent id is in `run.json`).
Regenerate the analysis: `uv run armbench analyze f9-v1/f9_A f9-v1/f9_B f9-v1/f9_BSS
f9-v1/f9_C f9-v1/f9_CSS --out f9_eval.json` (report sha256 `{report}`).

License: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). Cite as in CITATION.cff of
the repository.
"""
FULL_NOTE = (
    "- `<run>/samples/*.jsonl.gz`: 100 Hz joint samples `{t, q, qd, tau}` (Gazebo effort) the\n"
    "  energy of each episode was integrated from.\n"
    "- `<run>/chunks/`: per-container logs (25 seeds per fresh simulation container).\n"
)


def collect(run_dirs: list[Path], parts: tuple[str, ...]) -> list[tuple[str, Path]]:
    """``(archive name, source path)`` for every regular file, sorted by archive name."""
    files: list[tuple[str, Path]] = []
    for run in run_dirs:
        for part in parts:
            src = run / part
            if src.is_file():
                files.append((f"{DATASET_ID}/{run.name}/{part}", src))
            elif src.is_dir():
                for f in sorted(p for p in src.rglob("*") if p.is_file()):
                    rel = f.relative_to(run).as_posix()
                    files.append((f"{DATASET_ID}/{run.name}/{rel}", f))
    return sorted(files)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def add(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mtime = 0
    info.mode = 0o644
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    tar.addfile(info, io.BytesIO(data))


def build(run_dirs: list[Path], out: Path, *, full: bool, report: dict[str, object]) -> str:
    files = collect(run_dirs, FULL_PARTS if full else CORE_PARTS)
    contents = {name: src.read_bytes() for name, src in files}
    kind = "full" if full else "core"
    readme = README.format(
        kind=kind,
        protocol=report["protocol_sha256"],
        report=report["report_sha256"],
        full_note=FULL_NOTE if full else "",
    ).encode()
    contents[f"{DATASET_ID}/README.md"] = readme
    manifest = {
        "dataset_id": DATASET_ID,
        "kind": kind,
        "protocol_sha256": report["protocol_sha256"],
        "report_sha256": report["report_sha256"],
        "license": "CC-BY-4.0",
        "n_files": len(contents),
        "files": {name: sha256(data) for name, data in sorted(contents.items())},
    }
    contents[f"{DATASET_ID}/MANIFEST.json"] = (json.dumps(manifest, indent=1) + "\n").encode()
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = io.BytesIO()
    with (
        gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, compresslevel=9) as gz,
        tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar,
    ):
        for name in sorted(contents):
            add(tar, name, contents[name])
    out.write_bytes(raw.getvalue())
    return sha256(raw.getvalue())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--report", type=Path, default=ROOT / "reports" / "f9_eval.json")
    ap.add_argument("--runs-root", type=Path, default=ROOT / "runs")
    ap.add_argument("--full", action="store_true", help="Add joint samples and chunk logs.")
    ap.add_argument("--out", type=Path, help="Archive path (default data/ or dist/).")
    args = ap.parse_args(argv)
    report = json.loads(args.report.read_text())
    run_dirs = [args.runs_root / str(r["dir"]) for r in report["runs"]]
    missing = [str(d) for d in run_dirs if not (d / "episodes.jsonl").is_file()]
    if missing:
        print(f"missing run directories: {missing}", file=sys.stderr)
        return 2
    default = (
        ROOT / "dist" / f"armbench-{DATASET_ID}-full.tar.gz"
        if args.full
        else ROOT / "data" / f"{DATASET_ID}.tar.gz"
    )
    out = args.out or default
    digest = build(run_dirs, out, full=args.full, report=report)
    print(f"{out} {out.stat().st_size} bytes sha256 {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
