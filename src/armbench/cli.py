"""``armbench`` command-line interface (grows with each phase)."""

from __future__ import annotations

import json
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, cast, get_args

import numpy as np
import typer
from pydantic import BaseModel, ConfigDict

from armbench import __version__
from armbench.agents.base import Agent
from armbench.energy import DEFAULT_ENERGY_FILE, Variant, episode_from_jsonl, load_energy_params
from armbench.kinematics import N_JOINTS, UR5eModel, rotation_vector
from armbench.llm import (
    DEFAULT_LLM_FILE,
    ProviderKind,
    ResponseCache,
    load_llm_params,
    make_provider,
)
from armbench.paths import SKILLS_DIR
from armbench.scene import (
    DEFAULT_SCENE_FILE,
    generate_scene,
    load_scene_config,
    scene_to_json,
    scene_to_sdf_models,
    scene_to_sdf_world,
)
from armbench.seeds import DEFAULT_SEEDS_FILE, LockedSeedError, load_seed_split
from armbench.skills.library import LibraryError
from armbench.tasks import TASK_IDS, get_task

if TYPE_CHECKING:
    from armbench.skills import SkillRunner

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
    llm_config: Path = DEFAULT_LLM_FILE
    provider: ProviderKind | None = None
    """Overrides the provider named in ``llm_config`` (e.g. ``replay`` to reproduce a run)."""
    skills_dir: Path = SKILLS_DIR
    """Frozen skill library for agent B+S."""

    @property
    def uses_llm(self) -> bool:
        from armbench.agents import LLM_AGENT_IDS  # noqa: PLC0415 - heavy imports only for `run`

        return self.agent in LLM_AGENT_IDS

    @property
    def uses_skills(self) -> bool:
        return self.agent == "B+S"


def _make_agent(spec: _RunSpec, out: Path) -> Agent:
    """A needs nothing; B gets a provider stack whose ledger and programs land in ``out``;
    B+S additionally the frozen library."""
    from armbench.agents import get_agent  # noqa: PLC0415 - heavy imports only for `run`

    if not spec.uses_llm:
        return get_agent(spec.agent)
    llm = load_llm_params(spec.llm_config)
    provider = make_provider(llm, kind=spec.provider, ledger_path=out / "llm" / "ledger.json")
    return get_agent(
        spec.agent, llm=llm, provider=provider, artifacts_dir=out, skills_dir=spec.skills_dir
    )


def _skill_runner(spec: _RunSpec) -> SkillRunner | None:
    from armbench.skills import SkillLibrary, SkillRunner  # noqa: PLC0415

    if not spec.uses_skills:
        return None
    return SkillRunner(SkillLibrary.load(spec.skills_dir, require_frozen=True))


def _run_fake(spec: _RunSpec, out: Path) -> int:
    from armbench.perception import Camera, load_perception_params  # noqa: PLC0415
    from armbench.primitives import KinematicBackend, Robot, load_primitive_params  # noqa: PLC0415
    from armbench.runner import FakeWorld, Software, run_episode  # noqa: PLC0415

    params = load_primitive_params()
    camera = Camera.from_spec(load_perception_params().camera)
    backend = KinematicBackend(params, camera)
    robot = Robot(backend, params=params, skills=_skill_runner(spec))
    robot.reset()
    world = FakeWorld(backend, robot)
    agent = _make_agent(spec, out)
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


def _llm_bundle(spec: _RunSpec, out: Path) -> Path:
    """Pre-fetch every first-turn program on the host and write ``out/llm`` = ``llm.yaml`` +
    a cache holding exactly this run's requests, for replay inside the containers."""
    from armbench.agents import LLMAgent, get_agent, prefetch  # noqa: PLC0415
    from armbench.primitives import load_primitive_params  # noqa: PLC0415
    from armbench.skills import FROZEN_FILE, LIBRARY_FILE  # noqa: PLC0415

    bundle = out / "llm"
    bundle.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(spec.llm_config, bundle / "llm.yaml")
    if spec.uses_skills:
        (bundle / "skills").mkdir(exist_ok=True)
        for f in (LIBRARY_FILE, FROZEN_FILE):
            shutil.copyfile(spec.skills_dir / f, bundle / "skills" / f)
    llm = load_llm_params(spec.llm_config)
    provider = make_provider(llm, kind=spec.provider, ledger_path=bundle / "ledger.json")
    agent = get_agent(spec.agent, llm=llm, provider=provider, skills_dir=spec.skills_dir)
    if not isinstance(agent, LLMAgent):  # pragma: no cover - guarded by uses_llm
        msg = f"agent {spec.agent} does not use an LLM"
        raise typer.BadParameter(msg)
    cfg = load_scene_config()
    instances = [get_task(t).instance(s, cfg) for t in spec.tasks for s in spec.seeds]
    run_cache = ResponseCache(bundle / "cache")
    pairs = prefetch(agent, load_primitive_params(), instances)
    for req, resp in pairs:
        run_cache.put(req, resp)
    tokens = sum(r.usage.total for _, r in pairs)
    cost = sum(r.cost_usd for _, r in pairs)
    cached = sum(1 for _, r in pairs if r.cached)
    typer.echo(
        f"llm prefetch: {len(pairs)} programs ({cached} cached), {tokens} tokens, "
        f"{cost:.4f} USD -> {bundle}"
    )
    return bundle


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
    llm_config: Annotated[Path, typer.Option(help="LLM YAML for agent B.")] = DEFAULT_LLM_FILE,
    provider: Annotated[
        str | None, typer.Option(help="Override the provider: template, openai or replay.")
    ] = None,
    skills_dir: Annotated[
        Path, typer.Option(help="Frozen skill library for agent B+S.")
    ] = SKILLS_DIR,
) -> None:
    """Run episodes and append JSONL records to <out>/episodes.jsonl."""
    from armbench.runner import SimRunOptions, authorise_seeds, run_sim  # noqa: PLC0415

    if backend not in ("sim", "fake"):
        raise typer.BadParameter("backend must be 'sim' or 'fake'")
    if provider is not None and provider not in get_args(ProviderKind):
        raise typer.BadParameter("provider must be template, openai or replay")
    provider_kind = cast("ProviderKind | None", provider)
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
    spec = _RunSpec(
        tasks=tasks, agent=agent, seeds=seed_list, repeat=repeat, run_id=run_id,
        llm_config=llm_config, provider=provider_kind, skills_dir=skills_dir,
    )  # fmt: skip
    llm_meta = None
    if spec.uses_llm:
        llm = load_llm_params(llm_config)
        llm_meta = {"config": str(llm_config), "provider": provider_kind or llm.provider,
                    "model": llm.model, "temperature": llm.temperature, "seed": llm.seed,
                    "max_attempts": llm.max_attempts}  # fmt: skip
    skills_meta = None
    if spec.uses_skills:
        from armbench.skills import SkillLibrary  # noqa: PLC0415

        try:
            frozen = SkillLibrary.load(skills_dir, require_frozen=True)
        except LibraryError as exc:
            typer.echo(f"refused: {exc}", err=True)
            raise typer.Exit(code=2) from exc
        skills_meta = {"dir": str(skills_dir), "sha256": frozen.sha256(), "n_skills": len(frozen),
                       "names": list(frozen.names)}  # fmt: skip
    (out_dir / "run.json").write_text(
        json.dumps(
            {"run_id": run_id, "tasks": tasks, "agent": agent, "seeds": seed_list,
             "repeat": repeat, "backend": backend, "image": image if backend == "sim" else None,
             "git_sha": _git_sha(), "armbench": __version__, "llm": llm_meta,
             "skills": skills_meta},
            indent=1,
        )
        + "\n"
    )  # fmt: skip
    if backend == "fake":
        n_ok = _run_fake(spec, out_dir)
        typer.echo(f"{n_ok}/{len(tasks) * len(seed_list) * repeat} ok -> {out_dir}")
        return
    llm_dir = _llm_bundle(spec, out_dir) if spec.uses_llm else None
    infos = []
    for task_id in tasks:
        opts = SimRunOptions(
            task=task_id, agent=agent, seeds=seed_list, repeat=repeat, chunk=chunk, image=image,
            mcap=mcap, final_eval=final_eval, protocol_hash=protocol_hash, git_sha=_git_sha(),
            llm_dir=llm_dir,
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
    missing = [p for p in paths if not p.is_file()]
    if missing:
        typer.echo(f"no episode log at: {', '.join(str(p) for p in missing)}", err=True)
        raise typer.Exit(code=2)
    n_rows = len(Variant) * len(load_energy_params(energy_file).etas())
    rep = build_report(paths, n_rows_expected=n_rows, require_repeats=require_repeats)
    typer.echo(table(rep))
    if out is not None:
        write_report(rep, out)
        typer.echo(f"wrote {out}")
    if strict and not rep.passed:
        raise typer.Exit(code=1)


skills_app = typer.Typer(no_args_is_help=True, help="Skill library (phase F7).")
app.add_typer(skills_app, name="skills")


@skills_app.command("propose")
def skills_propose(
    run_dir: Annotated[Path, typer.Option("--run", help="An `armbench run` directory.")],
    out: Annotated[Path, typer.Option(help="Candidates JSON.")] = Path("skills/candidates.json"),
    agent: Annotated[str, typer.Option(help="Whose programs to learn from.")] = "B",
) -> None:
    """Turn the agent's successful programs into parametrised skill candidates."""
    from armbench.skills.propose import propose_from_run  # noqa: PLC0415

    proposal = propose_from_run(run_dir, agent=agent)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(proposal.model_dump_json(indent=1) + "\n")
    for c in proposal.candidates:
        typer.echo(f"candidate {c.signature()}  seeds={list(c.origin.seeds)}")
    for r in proposal.rejected:
        typer.echo(f"rejected  {r.task} seed {r.seed}: {r.reason}")
    typer.echo(
        f"{len(proposal.candidates)} candidate(s) from {proposal.n_completed}/"
        f"{proposal.n_episodes} completed episodes -> {out}"
    )


@skills_app.command("validate")
def skills_validate(
    candidates: Annotated[Path, typer.Option(help="Candidates JSON.")] = Path(
        "skills/candidates.json"
    ),
    out: Annotated[Path, typer.Option(help="Library directory.")] = SKILLS_DIR,
    report: Annotated[Path | None, typer.Option(help="Validation report JSON.")] = None,
    seeds: Annotated[
        str | None, typer.Option(help="Validation seeds (default: the skill_validation split).")
    ] = None,
) -> None:
    """Validate candidates on the skill_validation seeds and write the accepted ones (unfrozen)."""
    from armbench.runner import authorise_seeds  # noqa: PLC0415
    from armbench.skills import SkillLibrary  # noqa: PLC0415
    from armbench.skills.propose import Proposal  # noqa: PLC0415
    from armbench.skills.validate import Validator  # noqa: PLC0415

    if (out / "FROZEN.json").exists():
        typer.echo(f"refused: {out} is frozen; a frozen library never changes", err=True)
        raise typer.Exit(code=2)
    proposal = Proposal.model_validate_json(candidates.read_text())
    seed_list = None
    if seeds is not None:
        seed_list = authorise_seeds(seeds, load_seed_split(), final_eval=False, protocol_hash=None)
    validator = Validator(seeds=seed_list)
    library = SkillLibrary.load(out)
    library, reports = validator.build_library(proposal.candidates, library)
    for r in reports:
        verdict = "accepted" if r.accepted else "REJECTED"
        typer.echo(f"{verdict} {r.signature}: {r.n_ok}/{len(r.results)} on {r.split}")
        for s in r.results:
            if not s.ok:
                typer.echo(f"    seed {s.seed}: {s.reason}")
    library.save(out)
    if report is not None:
        report.parent.mkdir(parents=True, exist_ok=True)
        payload = {"seeds": validator.seeds, "min_pass_rate": validator.min_pass_rate,
                   "reports": [r.model_dump(mode="json") for r in reports]}  # fmt: skip
        report.write_text(json.dumps(payload, indent=1) + "\n")
    typer.echo(f"{len(library)} skill(s) in {out} (unfrozen; run `armbench skills freeze`)")


@skills_app.command("freeze")
def skills_freeze(
    directory: Annotated[Path, typer.Option("--dir", help="Library directory.")] = SKILLS_DIR,
) -> None:
    """Freeze the library: write FROZEN.json with its hash (D7). Refuses an already frozen one."""
    from armbench.skills import FROZEN_FILE, SkillLibrary  # noqa: PLC0415

    if (directory / FROZEN_FILE).exists():
        typer.echo(f"refused: {directory / FROZEN_FILE} already exists", err=True)
        raise typer.Exit(code=2)
    library = SkillLibrary.load(directory)
    if len(library) == 0:
        typer.echo("refused: the library is empty", err=True)
        raise typer.Exit(code=2)
    manifest = library.freeze()
    library.save(directory, manifest=manifest)
    typer.echo(f"frozen {len(library)} skill(s) sha256={manifest.sha256} -> {directory}")


@skills_app.command("show")
def skills_show(
    directory: Annotated[Path, typer.Option("--dir", help="Library directory.")] = SKILLS_DIR,
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """List the skills, their contracts, validation results and the library hash."""
    from armbench.skills import SkillLibrary  # noqa: PLC0415

    library = SkillLibrary.load(directory)
    if as_json:
        payload = {"frozen": library.frozen, "sha256": library.sha256(),
                   "skills": [s.model_dump(mode="json") for s in library]}  # fmt: skip
        typer.echo(json.dumps(payload, indent=1))
        return
    state = "frozen" if library.frozen else "unfrozen"
    typer.echo(f"{len(library)} skill(s), {state}, sha256={library.sha256()}")
    for s in library:
        v = s.validation
        val = f"{v.n_ok}/{len(v.seeds)} on {v.split}" if v is not None else "not validated"
        typer.echo(f"- {s.signature()}  [{s.sha256()[:12]}]  {val}")
        typer.echo(f"    {s.description}")
        typer.echo(f"    requires: {', '.join(c.describe() for c in s.preconditions) or '-'}")
        typer.echo(f"    ensures:  {', '.join(c.describe() for c in s.postconditions) or '-'}")
        typer.echo(
            f"    origin: {s.origin.task} run {s.origin.run_id} seeds {list(s.origin.seeds)}"
        )
