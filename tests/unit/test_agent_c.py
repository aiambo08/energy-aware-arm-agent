"""F8: energy-aware agents C and C+S (D8: baseline A's Wh as the reference budget)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from typer.testing import CliRunner

from armbench.agents import (
    AGENT_IDS,
    ENERGY_AGENT_IDS,
    LLM_AGENT_IDS,
    SKILL_AGENT_IDS,
    LLMAgent,
    get_agent,
)
from armbench.agents.prompt import ENERGY_HEADER, build_messages, energy_section
from armbench.cli import app
from armbench.energy import (
    REFERENCE_FILE,
    EnergyReference,
    EnergyReferenceError,
    ReferenceSample,
    ReferenceSource,
    TaskReference,
    Variant,
    load_energy_params,
)
from armbench.llm import LLMParams, load_llm_params, make_provider
from armbench.llm.fake import ENERGY_EDITS, TemplateProvider, energy_program, template_program
from armbench.llm.types import LLMRequest, Message
from armbench.paths import ENERGY_REF_DIR, REPO_ROOT, SKILLS_DIR
from armbench.perception import Camera, load_perception_params
from armbench.primitives import KinematicBackend, Robot, load_primitive_params
from armbench.runner import EpisodeRecord, FakeWorld, Software, run_episode
from armbench.runner.reference import reference_from_run
from armbench.runner.report import build_report, read_records
from armbench.scene import load_scene_config
from armbench.skills import SkillLibrary, SkillRunner
from armbench.tasks import TASK_IDS, TaskInstance, get_task

CFG = load_scene_config()
PARAMS = load_primitive_params()
SW = Software(armbench="test")
F5_RUN = REPO_ROOT / "runs" / "f5_dev"

pytestmark = pytest.mark.skipif(
    not (ENERGY_REF_DIR / REFERENCE_FILE).is_file(), reason="energy_ref/reference.json missing"
)


@pytest.fixture(scope="module")
def llm() -> LLMParams:
    return load_llm_params()


@pytest.fixture(scope="module")
def reference() -> EnergyReference:
    return EnergyReference.load(ENERGY_REF_DIR)


def instance(task: str = "pick_place@1", seed: int = 0) -> TaskInstance:
    return get_task(task).instance(seed, CFG)


def samples(task: str, whs: list[float]) -> dict[str, list[ReferenceSample]]:
    return {task: [ReferenceSample(seed=i, wh=w, sim_s=7.0, n_primitives=9)
                   for i, w in enumerate(whs)]}  # fmt: skip


SOURCE = ReferenceSource(run_id="r", agent="A", backend="sim", seeds=(0, 1, 2), split="dev")


def episode(agent_id: str, task: str, seed: int, out: Path, llm: LLMParams) -> EpisodeRecord:
    agent = get_agent(agent_id, llm=llm, artifacts_dir=out)
    skills = SkillRunner(SkillLibrary.load(SKILLS_DIR)) if agent_id in SKILL_AGENT_IDS else None
    backend = KinematicBackend(PARAMS, Camera.from_spec(load_perception_params().camera))
    robot = Robot(backend, params=PARAMS, skills=skills)
    robot.reset()
    task_obj = get_task(task)
    return run_episode(
        robot, agent, task_obj, task_obj.instance(seed, CFG), FakeWorld(backend, robot),
        run_id="f8", repeat=0, backend="fake", software=SW,
    )  # fmt: skip


# -- registry ----------------------------------------------------------------------------------


def test_registry_adds_c_and_cs_without_touching_b(llm: LLMParams, tmp_path: Path) -> None:
    assert AGENT_IDS == ("A", "B", "B+S", "C", "C+S")
    assert {"B", "B+S", "C", "C+S"} == LLM_AGENT_IDS
    assert {"B+S", "C+S"} == SKILL_AGENT_IDS and {"C", "C+S"} == ENERGY_AGENT_IDS
    b = get_agent("B", llm=llm)
    c = get_agent("C", llm=llm)
    cs = get_agent("C+S", llm=llm)
    assert isinstance(b, LLMAgent) and b.energy_ref is None and b.library is None
    assert isinstance(c, LLMAgent) and c.energy_ref is not None and c.library is None
    assert isinstance(cs, LLMAgent) and cs.energy_ref is not None and cs.library is not None
    assert cs.library.frozen and cs.energy_ref.sha256() == c.energy_ref.sha256()
    with pytest.raises(EnergyReferenceError, match="no energy reference"):
        get_agent("C", llm=llm, energy_ref_dir=tmp_path)
    assert get_agent("B", llm=llm, energy_ref_dir=tmp_path).id == "B", "B never reads it"


# -- prompt ------------------------------------------------------------------------------------


def test_energy_block_only_in_c_and_cs(llm: LLMParams, reference: EnergyReference) -> None:
    inst = instance()
    b, bs, c, cs = (get_agent(a, llm=llm) for a in ("B", "B+S", "C", "C+S"))
    assert isinstance(b, LLMAgent) and isinstance(bs, LLMAgent)
    assert isinstance(c, LLMAgent) and isinstance(cs, LLMAgent)
    rb, rbs, rc, rcs = (a.request(PARAMS, inst) for a in (b, bs, c, cs))
    assert rb.messages[0] == rc.messages[0] == rcs.messages[0], "system prompt is shared"
    assert ENERGY_HEADER not in rb.messages[1].content
    assert ENERGY_HEADER not in rbs.messages[1].content
    assert ENERGY_HEADER in rc.messages[1].content and ENERGY_HEADER in rcs.messages[1].content
    assert rc.messages[1].content.endswith(rb.messages[1].content), "C = B + energy block"
    assert "# Skills" in rcs.messages[1].content, "C+S also offers the skills"
    assert rcs.messages[1].content.index("# Skills") < rcs.messages[1].content.index("# Energy")
    ref = reference.for_task("pick_place@1")
    params = load_energy_params()
    block = rc.messages[1].content
    assert f"median of {ref.wh_median:.3f} Wh" in block
    assert f"eta = {params.eta:.2f}" in block and f"P0 = {params.p0_w:.0f} W" in block
    assert "speed_scale" in block and "fewer moves" in block
    assert rb.messages[1].content == build_messages(inst, "x")[1].content.replace("x", "x")


def test_energy_block_quotes_the_task_not_another(reference: EnergyReference) -> None:
    params = load_energy_params()
    blocks = {t: energy_section(reference, reference.for_task(t), params) for t in TASK_IDS}
    assert len(set(blocks.values())) == len(TASK_IDS), "every task gets its own budget"
    for t in TASK_IDS:
        assert f"{reference.for_task(t).wh_median:.3f}" in blocks[t]


def test_unknown_task_is_refused_before_any_llm_call(llm: LLMParams, tmp_path: Path) -> None:
    ref = EnergyReference.build(
        samples("stack2@1", [0.2, 0.3]), source=SOURCE, variant=Variant.A, eta=0.7
    )
    ref.save(tmp_path)
    provider = make_provider(llm)
    agent = get_agent("C", llm=llm, provider=provider, energy_ref_dir=tmp_path)
    assert isinstance(agent, LLMAgent)
    with pytest.raises(EnergyReferenceError, match="no energy reference for task 'pick_place@1'"):
        agent.request(PARAMS, instance("pick_place@1"))


# -- template provider -------------------------------------------------------------------------


def test_template_energy_variant_is_a_fixed_edit_of_the_b_program() -> None:
    base = template_program(instance().prompt)
    assert base is not None
    energy = energy_program(base)
    assert energy != base
    for old, new in ENERGY_EDITS:
        assert old in base and new in energy and old not in energy
    assert energy_program(energy) == energy, "idempotent"


def test_template_answers_energy_prompt_with_the_energy_variant(llm: LLMParams) -> None:
    inst = instance()
    provider = TemplateProvider()
    b, c = (get_agent(a, llm=llm, provider=provider) for a in ("B", "C"))
    assert isinstance(b, LLMAgent) and isinstance(c, LLMAgent)
    rb, rc = b.request(PARAMS, inst), c.request(PARAMS, inst)
    pb, pc = provider.complete(rb).text, provider.complete(rc).text
    assert pb != pc and "APPROACH = 0.06" in pc and "APPROACH = 0.10" in pb
    base = template_program(inst.prompt)
    assert base is not None and base in pb, "B's answer is unchanged by F8"
    assert energy_program(base) in pc


def test_template_ignores_energy_words_inside_the_task_sentence(llm: LLMParams) -> None:
    text = "# Energy\nbudget 1 Wh\n# Task\n" + instance().prompt + "\n\nThere are 3 cubes."
    req = LLMRequest(model="m", messages=(Message(role="user", content=text),), max_tokens=10)
    answer = TemplateProvider().complete(req).text
    assert "APPROACH = 0.06" in answer


# -- episodes on the fake backend --------------------------------------------------------------


@pytest.mark.parametrize("task", TASK_IDS)
def test_c_solves_dev_seeds_and_traces_the_reference(
    task: str, llm: LLMParams, reference: EnergyReference, tmp_path: Path
) -> None:
    for seed in (0, 3, 7):
        rec = episode("C", task, seed, tmp_path, llm)
        assert rec.ok, rec.failure
        t = rec.trace
        assert t is not None and t.program_outcome == "completed"
        assert t.energy_reference_wh == reference.for_task(task).wh_median
        assert t.energy_reference_sha256 == reference.sha256()
        assert t.n_moves > 0 and t.n_slow_moves > 0 and t.n_slow_moves <= t.n_moves
        assert t.speed_scale_min == pytest.approx(0.7)


def test_b_trace_has_no_reference_but_counts_its_moves(llm: LLMParams, tmp_path: Path) -> None:
    rec = episode("B", "pick_place@1", 0, tmp_path, llm)
    assert rec.ok
    t = rec.trace
    assert t is not None
    assert t.energy_reference_wh is None and t.energy_reference_sha256 is None
    assert t.n_moves == sum(1 for c in t.program_calls if c == "move_to")
    assert t.n_slow_moves > 0 and t.speed_scale_min == pytest.approx(0.5)


def test_cs_uses_the_skill_and_counts_inner_moves(
    llm: LLMParams, reference: EnergyReference, tmp_path: Path
) -> None:
    rec = episode("C+S", "stack2@1", 1, tmp_path, llm)
    assert rec.ok, rec.failure
    t = rec.trace
    assert t is not None and t.skills_used == ("stack_cube",)
    assert t.program_calls == ("execute_skill",), "the top-level program is one skill call"
    assert t.n_moves > 0 and t.n_slow_moves > 0, "moves inside the skill body are counted"
    assert t.energy_reference_wh == reference.for_task("stack2@1").wh_median
    assert t.n_primitives > t.n_moves


def test_c_and_b_run_the_same_primitives_in_a_different_order_of_speeds(
    llm: LLMParams, tmp_path: Path
) -> None:
    b = episode("B", "pick_place@1", 2, tmp_path / "b", llm)
    c = episode("C", "pick_place@1", 2, tmp_path / "c", llm)
    assert b.trace is not None and c.trace is not None
    assert b.trace.program_calls == c.trace.program_calls, "same call sequence (D8 pairing)"
    assert b.trace.program_sha256 != c.trace.program_sha256
    assert b.trace.n_moves == c.trace.n_moves
    assert c.trace.n_slow_moves < b.trace.n_slow_moves, "the energy variant lifts at full speed"
    assert (c.trace.speed_scale_min or 0) > (b.trace.speed_scale_min or 0)


# -- report ------------------------------------------------------------------------------------


def test_report_counts_energy_prompts_and_slow_moves(llm: LLMParams, tmp_path: Path) -> None:
    log = tmp_path / "episodes.jsonl"
    with log.open("w") as fh:
        for a in ("B", "C"):
            for seed in (0, 1):
                fh.write(episode(a, "pick_place@1", seed, tmp_path / a, llm).line() + "\n")
    report = build_report([log], n_rows_expected=0)
    by_agent = {t.agent: t for t in report.tasks}
    assert by_agent["B"].llm is not None and by_agent["C"].llm is not None
    assert by_agent["B"].llm.energy_prompt_rate == 0.0
    assert by_agent["B"].llm.energy_reference_wh is None
    assert by_agent["C"].llm.energy_prompt_rate == 1.0
    assert by_agent["C"].llm.energy_reference_wh == pytest.approx(0.2105, abs=1e-3)
    assert by_agent["C"].llm.slow_move_episode_rate == 1.0
    assert by_agent["C"].llm.speed_scale_min == pytest.approx(0.7)
    assert by_agent["C"].llm.moves_per_episode.median == 6
    assert by_agent["C"].llm.wh_over_reference.n == 0, "fake backend measures no energy"


# -- reference model ---------------------------------------------------------------------------


def test_reference_rejects_wrong_provenance_and_empty_input() -> None:
    s = samples("pick_place@1", [0.2, 0.21, 0.19])
    with pytest.raises(EnergyReferenceError, match="baseline A"):
        EnergyReference.build(s, source=SOURCE.model_copy(update={"agent": "B"}),
                              variant=Variant.A, eta=0.7)  # fmt: skip
    with pytest.raises(EnergyReferenceError, match="backend sim"):
        EnergyReference.build(s, source=SOURCE.model_copy(update={"backend": "fake"}),
                              variant=Variant.A, eta=0.7)  # fmt: skip
    with pytest.raises(EnergyReferenceError, match="no tasks"):
        EnergyReference.build({}, source=SOURCE, variant=Variant.A, eta=0.7)
    with pytest.raises(EnergyReferenceError, match="no completed episodes"):
        TaskReference.of("x", [])
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        ReferenceSample(seed=0, wh=-0.1, sim_s=1.0, n_primitives=1)


def test_reference_hash_is_content_only_and_roundtrips(tmp_path: Path) -> None:
    s = samples("pick_place@1", [0.2, 0.21, 0.19])
    r1 = EnergyReference.build(s, source=SOURCE, variant=Variant.A, eta=0.7, created_at="t1")
    r2 = EnergyReference.build(s, source=SOURCE, variant=Variant.A, eta=0.7, created_at="t2")
    r3 = EnergyReference.build(samples("pick_place@1", [0.2, 0.21, 0.191]), source=SOURCE,
                               variant=Variant.A, eta=0.7)  # fmt: skip
    assert r1.sha256() == r2.sha256() != r3.sha256()
    assert (
        r1.sha256() != EnergyReference.build(s, source=SOURCE, variant=Variant.B, eta=0.7).sha256()
    )
    path = r1.save(tmp_path)
    assert path.name == REFERENCE_FILE
    assert EnergyReference.load(tmp_path) == r1
    with pytest.raises(EnergyReferenceError, match="unknown"):
        r1.for_task("unknown@1")
    path.write_text("{not json")
    with pytest.raises(EnergyReferenceError, match="malformed"):
        EnergyReference.load(tmp_path)


@settings(max_examples=60, deadline=None)
@given(st.lists(st.floats(min_value=0.0, max_value=5.0, allow_nan=False), min_size=1, max_size=40))
def test_task_reference_statistics_are_order_statistics(whs: list[float]) -> None:
    t = TaskReference.of("x", samples("x", whs)["x"])
    assert t.n == len(whs) and t.wh_min == min(whs) and t.wh_max == max(whs)
    assert t.wh_min <= t.wh_iqm <= t.wh_max and t.wh_min <= t.wh_median <= t.wh_max
    shuffled = TaskReference.of("x", list(reversed(samples("x", whs)["x"])))
    assert shuffled == t, "independent of sample order"


@pytest.mark.skipif(not (F5_RUN / "episodes.jsonl").is_file(), reason="runs/f5_dev missing")
def test_committed_reference_is_rebuilt_bit_for_bit_from_the_f5_run(
    reference: EnergyReference,
) -> None:
    rebuilt = reference_from_run(F5_RUN, variant=Variant.A, eta=0.7)
    assert rebuilt.sha256() == reference.sha256()
    assert set(rebuilt.tasks) == set(TASK_IDS) and all(t.n == 10 for t in rebuilt.tasks.values())
    assert rebuilt.source.agent == "A" and rebuilt.source.split == "dev"
    assert rebuilt.tasks["pick_place@1"].wh_median == pytest.approx(0.2105, abs=5e-4)
    with pytest.raises(EnergyReferenceError, match="not a run directory"):
        reference_from_run(F5_RUN / "nope", variant=Variant.A, eta=0.7)
    with pytest.raises(EnergyReferenceError, match="no completed episode"):
        reference_from_run(F5_RUN, variant=Variant.A, eta=0.123)


# -- CLI ---------------------------------------------------------------------------------------


@pytest.mark.skipif(not (F5_RUN / "episodes.jsonl").is_file(), reason="runs/f5_dev missing")
def test_cli_energy_ref_build_show_and_refusals(tmp_path: Path) -> None:
    runner = CliRunner()
    out = tmp_path / "ref"
    res = runner.invoke(app, ["energy-ref", "build", "--run", str(F5_RUN), "--out", str(out)])
    assert res.exit_code == 0, res.output
    assert "4 task(s)" in res.output and "sha256=" in res.output
    res = runner.invoke(app, ["energy-ref", "build", "--run", str(F5_RUN), "--out", str(out)])
    assert res.exit_code == 2 and "exists" in res.output
    res = runner.invoke(app, ["energy-ref", "show", "--dir", str(out), "--json"])
    assert res.exit_code == 0
    payload = json.loads(res.output)
    assert payload["sha256"] == EnergyReference.load(out).sha256()
    assert payload["source"]["agent"] == "A" and set(payload["tasks"]) == set(TASK_IDS)
    res = runner.invoke(app, ["energy-ref", "show", "--dir", str(tmp_path / "none")])
    assert res.exit_code == 2 and "no energy reference" in res.output
    res = runner.invoke(app, ["energy-ref", "build", "--run", str(tmp_path), "--out",
                              str(tmp_path / "x")])  # fmt: skip
    assert res.exit_code == 2 and "not a run directory" in res.output


def test_cli_run_c_records_the_reference_and_refuses_a_missing_task(tmp_path: Path) -> None:
    runner = CliRunner()
    out = tmp_path / "run"
    res = runner.invoke(app, ["run", "--task", "pick_place@1", "--agent", "C", "--backend",
                              "fake", "--seeds", "0-1", "--out", str(out)])  # fmt: skip
    assert res.exit_code == 0, res.output
    meta = json.loads((out / "run.json").read_text())
    ref = EnergyReference.load(ENERGY_REF_DIR)
    assert meta["energy_ref"]["sha256"] == ref.sha256()
    assert meta["energy_ref"]["reference_wh"] == {
        "pick_place@1": ref.tasks["pick_place@1"].wh_median
    }
    assert meta["energy_ref"]["source_agent"] == "A" and meta["skills"] is None
    recs = read_records([out / "episodes.jsonl"])
    assert len(recs) == 2 and all(r.ok for r in recs)
    assert all(
        r.trace is not None and r.trace.energy_reference_sha256 == ref.sha256() for r in recs
    )
    partial = tmp_path / "partial"
    EnergyReference.build(samples("stack2@1", [0.2]), source=SOURCE, variant=Variant.A,
                          eta=0.7).save(partial)  # fmt: skip
    res = runner.invoke(app, ["run", "--task", "pick_place@1", "--agent", "C", "--backend",
                              "fake", "--seeds", "0", "--out", str(tmp_path / "r2"),
                              "--energy-ref-dir", str(partial)])  # fmt: skip
    assert res.exit_code == 2 and "no energy reference for task" in res.output
    res = runner.invoke(app, ["run", "--task", "pick_place@1", "--agent", "B", "--backend",
                              "fake", "--seeds", "0", "--out", str(tmp_path / "r3"),
                              "--energy-ref-dir", str(tmp_path / "absent")])  # fmt: skip
    assert res.exit_code == 0, "B does not need the reference"
    assert json.loads((tmp_path / "r3" / "run.json").read_text())["energy_ref"] is None
