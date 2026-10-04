"""``armbench`` command-line interface (grows with each phase)."""

from __future__ import annotations

import json
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from pydantic import BaseModel, ConfigDict

from armbench import __version__
from armbench.energy import DEFAULT_ENERGY_FILE, Variant, episode_from_jsonl, load_energy_params
from armbench.kinematics import N_JOINTS, UR5eModel, rotation_vector
from armbench.scene import (
    DEFAULT_SCENE_FILE,
    generate_scene,
    load_scene_config,
    scene_to_json,
    scene_to_sdf_models,
    scene_to_sdf_world,
)
from armbench.seeds import DEFAULT_SEEDS_FILE, LockedSeedError, load_seed_split
from armbench.tasks import TASK_IDS, get_task

app = typer.Typer(no_args_is_help=True, add_completion=False, help="armbench tools.")


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command()
def seeds(
    path: Annotated[Path, typer.Option(help="Seed split YAML.")] = DEFAULT_SEEDS_FILE,
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """Show the seed split and which ranges are locked."""
    split = load_seed_split(path)
    if as_json:
        typer.echo(json.dumps(split.model_dump(), indent=2))
        return
    for name, rng in split.splits.items():
        lock = " [LOCKED]" if rng.locked else ""
        typer.echo(f"{name:17s} {rng.start:4d}-{rng.end:<4d}{lock}  {rng.description}")


@app.command()
def scene(  # noqa: PLR0913
    *,
    seed: Annotated[int, typer.Option(min=0, help="Scene seed.")],
    n_cubes: Annotated[
        int | None, typer.Option(min=1, help="Number of cubes (default: drawn from the seed).")
    ] = None,
    out: Annotated[
        Path | None, typer.Option(help="Write the SDF here instead of printing it.")
    ] = None,
    template: Annotated[
        Path | None,
        typer.Option(help="World template with a <!-- ARMBENCH_SCENE --> marker line."),
    ] = None,
    config: Annotated[Path, typer.Option(help="Scene config YAML.")] = DEFAULT_SCENE_FILE,
    as_json: Annotated[
        bool, typer.Option("--json", help="Print the scene as JSON instead of SDF.")
    ] = False,
) -> None:
    """Generate a seeded tabletop scene as Gazebo SDF models (or a full world)."""
    cfg = load_scene_config(config)
    generated = generate_scene(seed, cfg, n_cubes=n_cubes)
    if as_json:
        text = scene_to_json(generated)
    elif template is not None:
        text = scene_to_sdf_world(generated, cfg, template)
    else:
        text = scene_to_sdf_models(generated, cfg)
    if out is None:
        typer.echo(text)
        return
    out.write_text(text + "\n", encoding="utf-8")
    typer.echo(f"wrote {out} ({len(generated.cubes)} cubes, seed {generated.seed})")


@app.command(context_settings={"ignore_unknown_options": True})
def fk(
    q: Annotated[list[float], typer.Argument(help="Six joint angles in radians.")],
) -> None:
    """Print the tool0 position and rotation vector (base_link frame) for joint angles q1..q6.

    Negative angles are accepted as-is (``armbench fk 0 -1.57 0 -1.57 0 0``).
    """
    if len(q) != N_JOINTS:
        msg = f"expected {N_JOINTS} joint angles, got {len(q)}"
        raise typer.BadParameter(msg)
    pose = UR5eModel().fk(np.asarray(q))
    xyz = pose[:3, 3]
    rotvec = rotation_vector(pose[:3, :3])
    typer.echo(f"xyz_m:      {xyz[0]:+.6f} {xyz[1]:+.6f} {xyz[2]:+.6f}")
    typer.echo(f"rotvec_rad: {rotvec[0]:+.6f} {rotvec[1]:+.6f} {rotvec[2]:+.6f}")


@app.command()
def energy(  # noqa: PLR0913
    log: Annotated[Path, typer.Argument(help="Episode JSONL log ({t, q, qd, tau} per line).")],
    *,
    config: Annotated[Path, typer.Option(help="Energy model YAML.")] = DEFAULT_ENERGY_FILE,
    variant: Annotated[Variant, typer.Option(help="Mechanical-power variant.")] = Variant.A,
    eta: Annotated[float | None, typer.Option(min=0.0, max=1.0, help="Override eta.")] = None,
    full: Annotated[bool, typer.Option("--full", help="All variant x eta combinations.")] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """Integrate the energy model over a recorded episode log and print Wh."""
    params = load_energy_params(config)
    rows = episode_from_jsonl(log, params, variant=variant, eta=eta, full_sensitivity=full)
    if as_json:
        payload = [{**r.model_dump(mode="json"), "total_wh": r.total_wh} for r in rows]
        typer.echo(json.dumps(payload, indent=2))
        return
    header = ("variant", "eta", "T[s]", "mech[J]", "cu[J]", "base[J]", "Wh")
    typer.echo(
        f"{header[0]:7s} {header[1]:>5s} {header[2]:>8s} {header[3]:>10s} "
        f"{header[4]:>10s} {header[5]:>10s} {header[6]:>9s}"
    )
    for r in rows:
        typer.echo(
            f"{r.variant.value:7s} {r.eta:5.2f} {r.duration_s:8.3f} {r.mechanical_j:10.3f} "
            f"{r.copper_j:10.3f} {r.base_j:10.3f} {r.total_wh:9.5f}"
        )


if __name__ == "__main__":  # pragma: no cover
    app()


def _git_sha() -> str | None:
    try:
        proc = subprocess.run(  # noqa: S603 - fixed git invocation
            [shutil.which("git") or "git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    return proc.stdout.strip() or None


def _expand_tasks(tasks: list[str]) -> list[str]:
    out: list[str] = []
    for t in tasks:
        out += list(TASK_IDS) if t == "all" else [str(get_task(t).id)]
    return out


class _RunSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    tasks: list[str]
    agent: str
    seeds: list[int]
    repeat: int
    run_id: str


def _run_fake(spec: _RunSpec, out: Path) -> int:
    from armbench.agents import get_agent  # noqa: PLC0415 - heavy imports only for `run`
    from armbench.perception import Camera, load_perception_params  # noqa: PLC0415
    from armbench.primitives import KinematicBackend, Robot, load_primitive_params  # noqa: PLC0415
    from armbench.runner import FakeWorld, Software, run_episode  # noqa: PLC0415

    params = load_primitive_params()
    camera = Camera.from_spec(load_perception_params().camera)
    backend = KinematicBackend(params, camera)
    robot = Robot(backend, params=params)
    robot.reset()
    world = FakeWorld(backend, robot)
    agent = get_agent(spec.agent)
    cfg = load_scene_config()
    software = Software(armbench=__version__, git_sha=_git_sha())
    n_ok = 0
    with (out / "episodes.jsonl").open("a") as fh:
        for task_id in spec.tasks:
            task = get_task(task_id)
            for seed in spec.seeds:
                instance = task.instance(seed, cfg)
                for rep in range(spec.repeat):
                    rec = run_episode(
                        robot, agent, task, instance, world,
                        run_id=spec.run_id, repeat=rep, backend="fake", software=software,
                    )  # fmt: skip
                    fh.write(rec.line() + "\n")
                    n_ok += rec.ok
                    typer.echo(f"{rec.task} seed {seed} rep {rep}: ok={rec.ok} {rec.reason}")
    return n_ok


@app.command()
def run(  # noqa: PLR0913
    *,
    task: Annotated[list[str], typer.Option(help="Task id (name@version) or 'all'; repeatable.")],
    agent: Annotated[str, typer.Option(help="Agent id (A = scripted baseline).")] = "A",
    seeds: Annotated[str, typer.Option(help="Split name (dev), range (400-449) or list.")] = "dev",
    backend: Annotated[str, typer.Option(help="'sim' (Gazebo in Docker) or 'fake'.")] = "sim",
    repeat: Annotated[int, typer.Option(min=1, help="Episodes per seed.")] = 1,
    out: Annotated[Path | None, typer.Option(help="Run directory (default runs/<run_id>).")] = None,
    final_eval: Annotated[bool, typer.Option("--final-eval", help="Unlock locked seeds.")] = False,
    protocol_hash: Annotated[str | None, typer.Option(help="Pre-registered protocol hash.")] = None,
    image: Annotated[str, typer.Option(help="Simulation image.")] = "armbench-sim:dev",
    chunk: Annotated[int, typer.Option(min=1, help="Seeds per fresh container.")] = 25,
    mcap: Annotated[bool, typer.Option("--mcap", help="Record an MCAP bag per episode.")] = False,
    seeds_file: Annotated[Path, typer.Option(help="Seed split YAML.")] = DEFAULT_SEEDS_FILE,
) -> None:
    """Run episodes and append JSONL records to <out>/episodes.jsonl."""
    from armbench.runner import SimRunOptions, authorise_seeds, run_sim  # noqa: PLC0415

    if backend not in ("sim", "fake"):
        raise typer.BadParameter("backend must be 'sim' or 'fake'")
    try:
        seed_list = authorise_seeds(
            seeds, load_seed_split(seeds_file), final_eval=final_eval, protocol_hash=protocol_hash
        )
    except LockedSeedError as exc:
        typer.echo(f"refused: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    tasks = _expand_tasks(task)
    run_id = uuid.uuid4().hex[:12]
    out_dir = out or Path("runs") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run.json").write_text(
        json.dumps(
            {"run_id": run_id, "tasks": tasks, "agent": agent, "seeds": seed_list,
             "repeat": repeat, "backend": backend, "image": image if backend == "sim" else None,
             "git_sha": _git_sha(), "armbench": __version__},
            indent=1,
        )
        + "\n"
    )  # fmt: skip
    if backend == "fake":
        spec = _RunSpec(tasks=tasks, agent=agent, seeds=seed_list, repeat=repeat, run_id=run_id)
        n_ok = _run_fake(spec, out_dir)
        typer.echo(f"{n_ok}/{len(tasks) * len(seed_list) * repeat} ok -> {out_dir}")
        return
    infos = []
    for task_id in tasks:
        opts = SimRunOptions(
            task=task_id, agent=agent, seeds=seed_list, repeat=repeat, chunk=chunk, image=image,
            mcap=mcap, final_eval=final_eval, protocol_hash=protocol_hash, git_sha=_git_sha(),
        )  # fmt: skip
        for info in run_sim(opts, out_dir, run_id):
            info["task"] = task_id
            infos.append(info)
            typer.echo(json.dumps(info))
    (out_dir / "chunks.json").write_text(json.dumps(infos, indent=1) + "\n")
    if not all(i["ok"] for i in infos):
        typer.echo("some chunks failed; see chunks.json", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"done -> {out_dir}")


@app.command()
def report(
    sources: Annotated[list[Path], typer.Argument(help="Run directories or episodes.jsonl files.")],
    out: Annotated[Path | None, typer.Option(help="Write the JSON report here.")] = None,
    require_repeats: Annotated[
        bool, typer.Option("--require-repeats", help="Fail unless some seed was repeated.")
    ] = False,
    strict: Annotated[bool, typer.Option("--strict", help="Exit 1 when the gate fails.")] = False,
    energy_file: Annotated[Path, typer.Option(help="Energy YAML.")] = DEFAULT_ENERGY_FILE,
) -> None:
    """Aggregate episode logs: success (Wilson CI), Wh, repeatability, infra, durations."""
    from armbench.runner import build_report, table, write_report  # noqa: PLC0415

    paths = [p / "episodes.jsonl" if p.is_dir() else p for p in sources]
    n_rows = len(Variant) * len(load_energy_params(energy_file).etas())
    rep = build_report(paths, n_rows_expected=n_rows, require_repeats=require_repeats)
    typer.echo(table(rep))
    if out is not None:
        write_report(rep, out)
        typer.echo(f"wrote {out}")
    if strict and not rep.passed:
        raise typer.Exit(code=1)
