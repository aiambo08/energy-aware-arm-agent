"""``armbench`` command-line interface (grows with each phase)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

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
from armbench.seeds import DEFAULT_SEEDS_FILE, load_seed_split

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
