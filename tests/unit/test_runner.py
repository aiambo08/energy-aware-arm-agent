"""Episode runner (F5): record schema, failure mapping, seed guard, report statistics, CLI."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from typer.testing import CliRunner

from armbench.agents import AgentTrace, get_agent
from armbench.cli import app
from armbench.energy import Variant
from armbench.perception import Camera, load_perception_params
from armbench.primitives import (
    CameraTimeout,
    KinematicBackend,
    OutOfReach,
    Robot,
    load_primitive_params,
)
from armbench.runner import (
    EnergyRecord,
    EnergyRow,
    EpisodeRecord,
    FakeWorld,
    SimRunOptions,
    Software,
    Thresholds,
    authorise_seeds,
    build_report,
    parse_seeds,
    run_episode,
    table,
)
from armbench.runner.docker import chunks, runner_args
from armbench.runner.report import cv, iqm, percentile, wilson
from armbench.scene import load_scene_config
from armbench.seeds import LockedSeedError, load_seed_split
from armbench.tasks import TaskInstance, get_task

SOFTWARE = Software(armbench="test")
ETAS = (0.70, 0.60, 0.80)


@pytest.fixture
def rig() -> tuple[Robot, FakeWorld]:
    params = load_primitive_params()
    backend = KinematicBackend(params, Camera.from_spec(load_perception_params().camera))
    robot = Robot(backend, params=params)
    robot.reset()
    return robot, FakeWorld(backend, robot)


def _episode(
    rig: tuple[Robot, FakeWorld], agent: object, task_id: str = "pick_place@1", seed: int = 0
) -> EpisodeRecord:
    robot, world = rig
    task = get_task(task_id)
    inst = task.instance(seed, load_scene_config())
    return run_episode(
        robot, agent, task, inst, world,  # type: ignore[arg-type]
        run_id="t", repeat=0, backend="fake", software=SOFTWARE,
    )  # fmt: skip


class _Raises:
    id = "raiser"

    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    def solve(self, robot: Robot, instance: TaskInstance) -> AgentTrace:
        raise self.exc


class _Idle:
    id = "idle"

    def solve(self, robot: Robot, instance: TaskInstance) -> AgentTrace:
        return AgentTrace()


# -- records -----------------------------------------------------------------------------------
def test_successful_episode_record_roundtrips(rig: tuple[Robot, FakeWorld]) -> None:
    rec = _episode(rig, get_agent("A"))
    assert rec.ok
    assert rec.failure is None
    assert not rec.infra_failure
    assert rec.schema_version == 1
    assert rec.task == "pick_place@1"
    assert rec.task_version == 1
    assert rec.trace is not None
    assert rec.trace.n_primitives > 0
    assert rec.sim_s is not None
    assert 0 < rec.sim_s < 60
    assert rec.energy is None  # the kinematic fake has no torques
    again = EpisodeRecord.model_validate_json(rec.line())
    assert again == rec
    assert json.loads(rec.line())["instance_sha256"] == rec.instance_sha256


def test_instance_hash_is_stable_across_runs(rig: tuple[Robot, FakeWorld]) -> None:
    a = _episode(rig, get_agent("A"), seed=3)
    b = _episode(rig, get_agent("A"), seed=3)
    c = _episode(rig, get_agent("A"), seed=4)
    assert a.instance_sha256 == b.instance_sha256 != c.instance_sha256
    assert a.episode_id != b.episode_id


def test_robot_error_is_not_infra(rig: tuple[Robot, FakeWorld]) -> None:
    rec = _episode(rig, _Raises(OutOfReach("too far", x=2.0)))
    assert not rec.ok
    assert rec.failure is not None
    assert rec.failure.stage == "robot"
    assert rec.failure.code == "out_of_reach"
    assert rec.failure.details == {"x": 2.0}
    assert not rec.infra_failure
    assert rec.reason.startswith("robot:")


def test_camera_timeout_counts_as_infra(rig: tuple[Robot, FakeWorld]) -> None:
    rec = _episode(rig, _Raises(CameraTimeout("no frame")))
    assert rec.failure is not None
    assert rec.failure.stage == "infra"
    assert rec.infra_failure


def test_lookup_error_blames_the_agent(rig: tuple[Robot, FakeWorld]) -> None:
    rec = _episode(rig, _Raises(KeyError("green")))
    assert rec.failure is not None
    assert rec.failure.stage == "agent"


def test_idle_agent_fails_the_judge_with_metrics(rig: tuple[Robot, FakeWorld]) -> None:
    rec = _episode(rig, _Idle())
    assert not rec.ok
    assert rec.failure is not None
    assert rec.failure.stage == "judge"
    assert rec.failure.code == "task_failed"
    assert "place_err_m" in rec.metrics
    assert rec.trace is not None


def test_scene_failure_is_infra_and_skips_the_agent(rig: tuple[Robot, FakeWorld]) -> None:
    robot, world = rig

    class Broken(FakeWorld):
        def place(self, instance: TaskInstance) -> str | None:
            return "spawn"

    rec = _episode((robot, Broken(world.backend, robot)), _Raises(RuntimeError("never")))
    assert rec.infra_failure
    assert rec.failure is not None
    assert rec.failure.code == "scene"
    assert rec.trace is None
    assert rec.sim_s is None


def test_robot_is_reset_after_every_episode(rig: tuple[Robot, FakeWorld]) -> None:
    robot, _ = rig
    _episode(rig, get_agent("A"), task_id="sort3@1")
    snap = robot.backend.snapshot(1.0)
    assert snap is not None
    assert max(abs(a - b) for a, b in zip(snap.q, robot.params.ready_q, strict=True)) < 1e-6


# -- seeds -------------------------------------------------------------------------------------
def test_parse_seeds_forms() -> None:
    split = load_seed_split()
    assert parse_seeds("dev", split) == list(range(10))
    assert parse_seeds("400-404", split) == [400, 401, 402, 403, 404]
    assert parse_seeds("3,1,2", split) == [3, 1, 2]


def test_authorise_refuses_locked_seeds_without_protocol() -> None:
    split = load_seed_split()
    with pytest.raises(LockedSeedError):
        authorise_seeds("final_eval", split, final_eval=False, protocol_hash=None)
    with pytest.raises(LockedSeedError):
        authorise_seeds("100-101", split, final_eval=True, protocol_hash=None)
    assert authorise_seeds("final_eval", split, final_eval=True, protocol_hash="abc") == list(
        range(100, 120)
    )


# -- statistics --------------------------------------------------------------------------------
def test_wilson_matches_reference_values() -> None:
    lo, hi = wilson(14, 20)
    assert lo == pytest.approx(0.4810, abs=1e-3)
    assert hi == pytest.approx(0.8545, abs=1e-3)
    assert wilson(20, 20)[1] == 1.0
    assert wilson(0, 20)[0] == 0.0
    assert wilson(0, 0) == (0.0, 0.0)


def test_percentile_iqm_cv() -> None:
    xs = [float(i) for i in range(1, 11)]
    assert percentile(xs, 0) == 1.0
    assert percentile(xs, 100) == 10.0
    assert percentile(xs, 50) == 5.5
    assert iqm(xs) == pytest.approx(5.5)
    assert iqm([1.0, 2.0, 3.0, 100.0]) == pytest.approx(2.5)
    assert cv([2.0, 2.0, 2.0]) == 0.0
    assert math.isnan(cv([1.0]))
    assert math.isnan(percentile([], 50))


def _energy(wh: float) -> EnergyRecord:
    rows = tuple(
        EnergyRow(
            variant=v,
            eta=e,
            duration_s=5.0,
            n_samples=500,
            mechanical_j=1.0,
            copper_j=1.0,
            base_j=500.0,
            total_j=wh * 3600 * (1.0 if v is Variant.A else 1.1),
            total_wh=wh * (1.0 if v is Variant.A else 1.1),
        )
        for v in Variant
        for e in ETAS
    )
    return EnergyRecord(source="gazebo_effort", topic="/e", n_samples=500, rate_hz=100.0, rows=rows)


def _rec(
    seed: int, rep: int, *, ok: bool = True, wh: float | None = 0.2, infra: bool = False
) -> EpisodeRecord:
    return EpisodeRecord(
        run_id="r", episode_id=f"e{seed}{rep}", task="pick_place@1", task_version=1, agent="A",
        seed=seed, repeat=rep, backend="sim", instance_sha256="x", started_at="now", ok=ok,
        reason="" if ok else "fail", infra_failure=infra, sim_s=6.0, wall_s=20.0,
        energy=_energy(wh) if wh is not None else None, trace=AgentTrace(n_primitives=9),
        software=SOFTWARE,
    )  # fmt: skip


def _write(path: Path, recs: list[EpisodeRecord]) -> Path:
    path.write_text("".join(r.line() + "\n" for r in recs))
    return path


def test_report_passes_on_clean_sim_log(tmp_path: Path) -> None:
    recs = [_rec(s, 0) for s in range(400, 420)]
    recs += [_rec(400, r, wh=0.2 + 0.001 * r) for r in range(1, 10)]
    rep = build_report(
        [_write(tmp_path / "e.jsonl", recs)], n_rows_expected=6, require_repeats=True
    )
    t = rep.tasks[0]
    assert t.n == 29
    assert t.success == 1.0
    assert t.energy_complete_rate == 1.0
    assert len(t.energy) == 6
    assert t.energy[0].variant is Variant.A
    assert t.energy[0].wh.median == pytest.approx(0.2, abs=1e-3)
    assert t.repeats[0].seed == 400
    assert t.repeats[0].n == 10
    assert t.repeat_cv_max is not None
    assert t.repeat_cv_max < 0.03
    assert rep.passed, rep.checks
    out = table(rep)
    assert "GATE PASSED" in out
    assert "pick_place@1" in out


def test_report_fails_on_low_success_missing_energy_or_infra(tmp_path: Path) -> None:
    recs = [_rec(s, 0, ok=s % 5 != 0) for s in range(400, 420)]
    recs[1] = _rec(401, 0, wh=None)
    recs[2] = _rec(402, 0, ok=False, infra=True)
    rep = build_report([_write(tmp_path / "e.jsonl", recs)], n_rows_expected=6)
    t = rep.tasks[0]
    assert t.success == pytest.approx(15 / 20)
    assert t.energy_complete == 19
    assert t.n_infra == 1
    assert not rep.passed
    failed = {k for k, v in rep.checks.items() if not v}
    assert any("success" in k for k in failed)
    assert any("energy table" in k for k in failed)
    assert any("infra" in k for k in failed)
    assert t.failed_seeds == [400, 402, 405, 410, 415]


def test_report_repeat_cv_threshold(tmp_path: Path) -> None:
    recs = [_rec(0, r, wh=0.2 * (1 + 0.1 * r)) for r in range(10)]
    rep = build_report(
        [_write(tmp_path / "e.jsonl", recs)], n_rows_expected=6, thresholds=Thresholds()
    )
    assert rep.tasks[0].repeat_cv_max is not None
    assert rep.tasks[0].repeat_cv_max > 0.03
    assert not rep.checks["pick_place@1/A [dev]: repeat Wh CV < 3%"]


def test_report_keeps_seed_splits_apart(tmp_path: Path) -> None:
    recs = [_rec(s, 0) for s in range(10)] + [_rec(s, 0, ok=s != 418) for s in range(400, 450)]
    recs.append(_rec(777, 0))
    rep = build_report([_write(tmp_path / "e.jsonl", recs)], n_rows_expected=6)
    by_split = {t.split: t for t in rep.tasks}
    assert set(by_split) == {"dev", "dev_extended", "custom"}
    assert by_split["dev"].n == 10
    assert by_split["dev"].success == 1.0
    assert by_split["dev_extended"].failed_seeds == [418]
    assert by_split["custom"].n == 1
    assert rep.checks["pick_place@1/A [dev_extended]: success >= 95%"]


def test_energy_rows_incomplete_when_an_eta_is_missing(tmp_path: Path) -> None:
    rep = build_report([_write(tmp_path / "e.jsonl", [_rec(0, 0)])], n_rows_expected=8)
    assert rep.tasks[0].energy_complete == 0
    assert rep.tasks[0].energy == []


# -- docker orchestration (no docker needed) ---------------------------------------------------
def test_chunks_and_runner_args() -> None:
    assert chunks(list(range(7)), 3) == [[0, 1, 2], [3, 4, 5], [6]]
    opts = SimRunOptions(
        task="stack2@1", seeds=[1, 2], repeat=3, mcap=True, final_eval=True, protocol_hash="h"
    )
    args = runner_args(opts, [1, 2], "rid")
    assert args[:4] == ["ros2", "run", "armbench_bringup", "episode_runner"]
    assert "--mcap" in args
    assert "--final-eval" in args
    assert args[args.index("--protocol-hash") + 1] == "h"
    assert args[args.index("--seeds") + 1] == "1,2"
    assert args[args.index("--repeat") + 1] == "3"


# -- CLI ---------------------------------------------------------------------------------------
def test_cli_run_fake_and_report(tmp_path: Path) -> None:
    runner = CliRunner()
    out = tmp_path / "run"
    res = runner.invoke(
        app, ["run", "--task", "pick_place@1", "--task", "stack2@1", "--backend", "fake",
              "--seeds", "0-2", "--repeat", "2", "--out", str(out)],
    )  # fmt: skip
    assert res.exit_code == 0, res.output
    lines = (out / "episodes.jsonl").read_text().splitlines()
    assert len(lines) == 12
    assert json.loads(lines[0])["schema_version"] == 1
    assert json.loads((out / "run.json").read_text())["tasks"] == ["pick_place@1", "stack2@1"]
    report_path = tmp_path / "report.json"
    res = runner.invoke(app, ["report", str(out), "--out", str(report_path), "--strict"])
    assert res.exit_code == 0, res.output
    data = json.loads(report_path.read_text())
    assert data["passed"]
    assert {t["task"] for t in data["tasks"]} == {"pick_place@1", "stack2@1"}
    assert all(t["n"] == 6 and t["n_ok"] == 6 for t in data["tasks"])


def test_cli_run_refuses_locked_seeds(tmp_path: Path) -> None:
    res = CliRunner().invoke(
        app, ["run", "--task", "pick_place@1", "--backend", "fake", "--seeds", "final_eval",
              "--out", str(tmp_path)],
    )  # fmt: skip
    assert res.exit_code == 2
    assert not (tmp_path / "episodes.jsonl").exists()
    res = CliRunner().invoke(
        app, ["run", "--task", "pick_place@1", "--backend", "fake", "--seeds", "100",
              "--final-eval", "--out", str(tmp_path)],
    )  # fmt: skip
    assert res.exit_code == 2


def test_report_without_log_exits_cleanly(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["report", str(tmp_path)])
    assert result.exit_code == 2
    assert "no episode log" in result.output
