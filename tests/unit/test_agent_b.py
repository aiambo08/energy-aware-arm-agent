"""Agent B end to end on the kinematic backend: success, trace, failure mapping, replay."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from armbench.agents import AGENT_IDS, LLMAgent, get_agent, prefetch
from armbench.agents.base import Agent
from armbench.cli import app
from armbench.llm import LLMParams, ResponseCache, StaticProvider, load_llm_params, make_provider
from armbench.perception import Camera, load_perception_params
from armbench.primitives import KinematicBackend, Robot, load_primitive_params
from armbench.runner import EpisodeRecord, FakeWorld, Software, build_report, run_episode, table
from armbench.sandbox import SandboxLimits
from armbench.scene import load_scene_config
from armbench.tasks import TASK_IDS, get_task

SW = Software(armbench="test")
CFG = load_scene_config()


@pytest.fixture(scope="module")
def rig() -> tuple[Robot, FakeWorld]:
    params = load_primitive_params()
    backend = KinematicBackend(params, Camera.from_spec(load_perception_params().camera))
    robot = Robot(backend, params=params)
    robot.reset()
    return robot, FakeWorld(backend, robot)


@pytest.fixture
def llm() -> LLMParams:
    return load_llm_params()


def episode(rig: tuple[Robot, FakeWorld], agent: Agent, tid: str, seed: int) -> EpisodeRecord:
    robot, world = rig
    task = get_task(tid)
    inst = task.instance(seed, CFG)
    return run_episode(
        robot, agent, task, inst, world, run_id="t", repeat=0, backend="fake", software=SW
    )


def test_registry_keeps_a_and_adds_b(llm: LLMParams, tmp_path: Path) -> None:
    assert AGENT_IDS == ("A", "B", "B+S")
    assert get_agent("A").id == "A"
    b = get_agent("B", llm=llm, provider=make_provider(llm, cache_dir=tmp_path))
    assert isinstance(b, LLMAgent) and b.id == "B" and b.library is None
    with pytest.raises(KeyError, match=r"known: A, B, B\+S"):
        get_agent("Z")


@pytest.mark.parametrize("tid", TASK_IDS)
def test_b_template_solves_dev_seeds(
    rig: tuple[Robot, FakeWorld], llm: LLMParams, tmp_path: Path, tid: str
) -> None:
    agent = get_agent(
        "B", llm=llm, provider=make_provider(llm, cache_dir=tmp_path / "c"), artifacts_dir=tmp_path
    )
    for seed in range(5):
        rec = episode(rig, agent, tid, seed)
        assert rec.ok, (tid, seed, rec.reason, rec.trace and rec.trace.program_error)
        t = rec.trace
        assert (
            t is not None
            and t.program_outcome == "completed"
            and t.n_primitives == len(t.program_calls)
        )
        assert t.llm_provider == "template" and t.llm_model == "template-v1" and t.cost_usd == 0.0
        assert t.prompt_sha256 and t.response_sha256 and t.program_sha256
        assert "ast" in t.sandbox_isolation and "builtins" in t.sandbox_isolation
        assert t.program_calls[0] == "detect" and "grasp" in t.program_calls
    stem = f"{tid.replace('@', '_v')}_s0_a1"
    assert (tmp_path / "programs" / f"{stem}.py").is_file()
    assert (
        json.loads((tmp_path / "programs" / f"{stem}.run.json").read_text())["outcome"]
        == "completed"
    )


def test_replay_reproduces_the_same_primitive_calls(
    rig: tuple[Robot, FakeWorld], llm: LLMParams, tmp_path: Path
) -> None:
    live = get_agent("B", llm=llm, provider=make_provider(llm, cache_dir=tmp_path))
    rec1 = episode(rig, live, "sort3@1", 2)
    rec2 = episode(rig, live, "sort3@1", 2)
    assert rec1.ok and rec2.ok
    assert rec1.trace and rec2.trace
    assert not rec1.trace.llm_cached and rec2.trace.llm_cached
    assert rec1.trace.program_calls == rec2.trace.program_calls
    assert rec1.trace.program_sha256 == rec2.trace.program_sha256
    replay = get_agent("B", llm=llm, provider=make_provider(llm, kind="replay", cache_dir=tmp_path))
    rec3 = episode(rig, replay, "sort3@1", 2)
    assert rec3.ok and rec3.trace and rec3.trace.llm_cached
    assert rec3.trace.program_calls == rec1.trace.program_calls
    assert rec3.trace.response_sha256 == rec1.trace.response_sha256
    miss = episode(rig, replay, "sort3@1", 7)
    assert (
        not miss.ok
        and miss.failure
        and miss.failure.stage == "agent"
        and miss.failure.code == "cache_miss"
    )


def test_prefetch_fills_a_cache_the_replay_agent_can_use(llm: LLMParams, tmp_path: Path) -> None:
    params = load_primitive_params()
    agent = LLMAgent(make_provider(llm, cache_dir=tmp_path / "global"), llm)
    insts = [get_task(t).instance(0, CFG) for t in TASK_IDS]
    pairs = prefetch(agent, params, insts)
    assert len(pairs) == 4 and all(not r.cached for _, r in pairs)
    run_cache = ResponseCache(tmp_path / "run")
    for rq, rs in pairs:
        run_cache.put(rq, rs)
    assert len(run_cache) == 4
    again = prefetch(agent, params, insts)
    assert all(r.cached for _, r in again)


def failure_of(
    rig: tuple[Robot, FakeWorld], llm: LLMParams, text: str, **kw: object
) -> EpisodeRecord:
    params = llm.model_copy(update=kw) if kw else llm
    agent = LLMAgent(
        StaticProvider(text), params, limits=SandboxLimits(cpu_s=2.0, max_calls=20, sim_s=20.0)
    )
    return episode(rig, agent, "pick_place@1", 0)


def test_rejected_program_is_an_agent_failure_with_trace(
    rig: tuple[Robot, FakeWorld], llm: LLMParams
) -> None:
    rec = failure_of(rig, llm, "```python\nimport os\n```")
    assert not rec.ok and rec.failure
    assert rec.failure.stage == "agent" and rec.failure.code == "program_rejected"
    assert rec.trace and rec.trace.program_outcome == "rejected" and rec.trace.n_primitives == 0
    assert "Import" in (rec.trace.program_error or "")


def test_primitive_error_in_program_is_a_robot_failure(
    rig: tuple[Robot, FakeWorld], llm: LLMParams
) -> None:
    rec = failure_of(rig, llm, "robot.move_to(Pose(2.0, 2.0, 2.0))\n")
    assert not rec.ok and rec.failure
    assert rec.failure.stage == "robot" and rec.failure.code == "out_of_reach"


def test_program_exception_and_limits_are_agent_failures(
    rig: tuple[Robot, FakeWorld], llm: LLMParams
) -> None:
    rec = failure_of(rig, llm, "x = 1 / 0\n")
    assert rec.failure and rec.failure.stage == "agent" and rec.failure.code == "program_exception"
    rec = failure_of(rig, llm, "for i in range(100):\n    robot.observe()\n")
    assert rec.failure and rec.failure.code == "program_call_limit"
    assert rec.trace and rec.trace.n_primitives == 20


def test_token_budget_is_enforced(rig: tuple[Robot, FakeWorld], llm: LLMParams) -> None:
    rec = failure_of(rig, llm, "robot.observe()\n", max_tokens_per_episode=1)
    assert rec.failure and rec.failure.stage == "agent" and rec.failure.code == "token_budget"


def test_retry_turn_feeds_the_error_back(rig: tuple[Robot, FakeWorld], llm: LLMParams) -> None:
    seen: list[int] = []

    def answer(request: object) -> str:
        msgs = request.messages  # type: ignore[attr-defined]
        seen.append(len(msgs))
        if len(msgs) == 2:
            return "x = 1 / 0\n"
        assert "ZeroDivisionError" in msgs[-1].content
        return "robot.observe()\n"

    params = llm.model_copy(update={"max_attempts": 2})
    agent = LLMAgent(StaticProvider(answer), params)
    rec = episode(rig, agent, "pick_place@1", 0)
    assert seen == [2, 4]
    assert rec.trace and rec.trace.attempts == 2 and rec.trace.program_outcome == "completed"


def test_guest_package_stays_light() -> None:
    code = (
        "import sys; import armbench.guest.worker; "
        "print(sorted(m for m in ('numpy', 'cv2', 'yaml') if m in sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]"


def test_cli_run_b_fake_then_replay_and_report(tmp_path: Path) -> None:
    runner = CliRunner()
    out = tmp_path / "run"
    res = runner.invoke(
        app,
        [
            "run",
            "--backend",
            "fake",
            "--agent",
            "B",
            "--task",
            "pick_place@1",
            "--seeds",
            "0-1",
            "--out",
            str(out),
        ],
    )
    assert res.exit_code == 0, res.output
    recs = [
        EpisodeRecord.model_validate_json(line)
        for line in (out / "episodes.jsonl").read_text().splitlines()
    ]
    assert len(recs) == 2 and all(r.ok for r in recs)
    assert json.loads((out / "run.json").read_text())["llm"]["provider"] == "template"
    assert (out / "llm" / "ledger.json").is_file()
    assert len(list((out / "programs").glob("*.py"))) == 2
    replay_dir = tmp_path / "replay"
    res = runner.invoke(
        app,
        [
            "run",
            "--backend",
            "fake",
            "--agent",
            "B",
            "--task",
            "pick_place@1",
            "--seeds",
            "0-1",
            "--out",
            str(replay_dir),
            "--provider",
            "replay",
        ],
    )
    assert res.exit_code == 0, res.output
    again = [
        EpisodeRecord.model_validate_json(line)
        for line in (replay_dir / "episodes.jsonl").read_text().splitlines()
    ]
    assert [r.trace.program_sha256 for r in again if r.trace] == [
        r.trace.program_sha256 for r in recs if r.trace
    ]
    assert all(r.trace and r.trace.llm_cached for r in again)
    rep = build_report([out / "episodes.jsonl"], n_rows_expected=6)
    llm_stats = rep.tasks[0].llm
    assert llm_stats is not None and llm_stats.provider == "template" and llm_stats.cost_usd == 0.0
    assert llm_stats.program_outcomes == {"completed": 2}
    assert "llm template/template-v1" in table(rep)
    bad = tmp_path / "bad"
    res = runner.invoke(
        app,
        [
            "run",
            "--backend",
            "fake",
            "--agent",
            "B",
            "--task",
            "pick_place@1",
            "--seeds",
            "0",
            "--out",
            str(bad),
            "--provider",
            "nope",
        ],
    )
    assert res.exit_code != 0
