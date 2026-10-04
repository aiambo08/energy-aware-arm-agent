"""Phase F7: skill format, library (dedup, retrieval, freeze), execution contracts, proposal,
validation (9/10 rule, injected failures) and agent B+S end to end on the kinematic backend."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from armbench.agents import LLMAgent, get_agent
from armbench.agents.prompt import build_messages, skills_section
from armbench.llm import StaticProvider, load_llm_params, make_provider
from armbench.llm.fake import TemplateProvider, offered_skills, template_skill_program
from armbench.paths import SKILLS_DIR
from armbench.perception import Camera, load_perception_params
from armbench.primitives import (
    KinematicBackend,
    OutOfReach,
    Robot,
    SkillFailed,
    SkillNotAvailable,
    SkillPostconditionFailed,
    SkillPreconditionFailed,
    load_primitive_params,
)
from armbench.runner import EpisodeRecord, FakeWorld, Software, run_episode
from armbench.sandbox import SandboxLimits, check_program, run_program
from armbench.scene import load_scene_config
from armbench.skills import (
    Condition,
    FrozenLibraryError,
    LibraryError,
    Origin,
    Param,
    Skill,
    SkillLibrary,
    SkillRunner,
    check_robot_condition,
)
from armbench.skills.propose import (
    conditions_for,
    goal_arguments,
    goal_constants,
    parametrise,
    propose_from_run,
)
from armbench.skills.validate import Validator
from armbench.tasks import TaskInstance, get_task

CFG = load_scene_config()
PARAMS = load_primitive_params()
SW = Software(armbench="test")
FAST = SandboxLimits(cpu_s=5.0, memory_mb=256, wall_s=60.0, sim_s=60.0, max_calls=60)

PLACE_BODY = """\
def best(dets, c):
    hits = [d for d in dets if d.color == c]
    full = [d for d in hits if d.complete]
    return (full or hits)[0] if hits else None

d = best(robot.detect(), color)
if d is None or not d.complete:
    robot.move_to(Pose(-0.30, 0.0, 0.15))
    d = best(robot.detect(), color) or d
pick = Pose(d.position[0], d.position[1], d.position[2], d.yaw_rad)
place = Pose(x, y, pick.z)
robot.move_to(pick.above(0.10))
robot.move_to(pick, 0.5)
robot.grasp()
robot.move_to(pick.above(0.10), 0.5)
robot.move_to(place.above(0.10))
robot.move_to(place.above(0.005), 0.5)
robot.release()
robot.move_to(place.above(0.10))
"""

ORIGIN = Origin(
    task="pick_place@1", run_id="r", episode_ids=("e",), seeds=(0,), program_sha256="0" * 64,
    agent="B",
)  # fmt: skip


def place_skill(body: str = PLACE_BODY, name: str = "pick_place_cube", **extra: object) -> Skill:
    inst = get_task("pick_place@1").instance(0, CFG)
    pre, post = conditions_for(inst)
    fields: dict[str, object] = {
        "name": name,
        "description": "Move the cube of a colour to a point on the table.",
        "params": (
            Param(name="color", kind="color", description="cube colour"),
            Param(name="x", kind="float", description="target x"),
            Param(name="y", kind="float", description="target y"),
        ),
        "body": body,
        "preconditions": pre,
        "postconditions": post,
        "origin": ORIGIN,
    }
    fields.update(extra)
    return Skill.model_validate(fields)


def rig(skills: SkillRunner | None) -> tuple[Robot, KinematicBackend, FakeWorld]:
    backend = KinematicBackend(PARAMS, Camera.from_spec(load_perception_params().camera))
    robot = Robot(backend, params=PARAMS, skills=skills)
    robot.reset()
    return robot, backend, FakeWorld(backend, robot)


def place_instance(seed: int = 0) -> TaskInstance:
    return get_task("pick_place@1").instance(seed, CFG)


# -- format ------------------------------------------------------------------------------------


def test_signature_hash_and_call_syntax() -> None:
    s = place_skill()
    assert s.signature() == "pick_place_cube(color: color, x: float, y: float)"
    assert (
        s.call_syntax()
        == 'robot.execute_skill("pick_place_cube", color=<color>, x=<float>, y=<float>)'
    )
    same = s.model_copy(update={"origin": ORIGIN.model_copy(update={"run_id": "other"})})
    assert same.sha256() == s.sha256(), "origin and validation are not part of the behaviour hash"
    assert place_skill(body=PLACE_BODY + "\n").sha256() != s.sha256()
    assert s.model_copy(update={"version": 2}).sha256() != s.sha256()


def test_check_args_rejects_bad_arguments() -> None:
    s = place_skill()
    assert s.check_args({"color": "red", "x": -0.4, "y": 0.1}) == {
        "color": "red",
        "x": -0.4,
        "y": 0.1,
    }
    with pytest.raises(TypeError, match="missing"):
        s.check_args({"color": "red", "x": -0.4})
    with pytest.raises(TypeError, match="unexpected"):
        s.check_args({"color": "red", "x": -0.4, "y": 0.1, "z": 0.0})
    with pytest.raises(TypeError, match="finite"):
        s.check_args({"color": "red", "x": float("nan"), "y": 0.1})
    with pytest.raises(TypeError, match="colour"):
        s.check_args({"color": 3, "x": -0.4, "y": 0.1})
    with pytest.raises(TypeError, match="number"):
        s.check_args({"color": "red", "x": "far", "y": 0.1})


def test_bound_program_passes_the_sandbox_checker() -> None:
    s = place_skill()
    assert check_program(s.bind(s.placeholder_args())).ok
    bound = s.bind({"color": "red", "x": -0.4, "y": 0.1})
    assert bound.startswith("color = 'red'\nx = -0.4\ny = 0.1\n")


# -- library -----------------------------------------------------------------------------------


def test_library_dedups_by_signature_and_rejects_name_clash() -> None:
    lib = SkillLibrary()
    a = place_skill()
    b = place_skill(body=PLACE_BODY + "\n")
    assert lib.add(a) is a
    assert lib.add(b) is a, "same signature: the first validated skill wins"
    assert lib.skills[0].body == PLACE_BODY
    other = a.model_copy(update={"params": a.params[:2]})
    with pytest.raises(LibraryError, match="already used with signature"):
        lib.add(other)
    assert len(lib) == 1 and "pick_place_cube" in lib


def test_retrieval_is_lexical_and_deterministic() -> None:
    lib = SkillLibrary()
    lib.add(place_skill())
    lib.add(place_skill(name="stack_cube", description="Put one cube on top of another cube."))
    hits = lib.retrieve("Move the red cube so that its centre is at x=-0.4, y=0.1", k=3)
    assert hits[0].skill.name == "pick_place_cube"
    assert all(h.score > 0 and h.overlap for h in hits)
    assert lib.retrieve("completely unrelated words", k=3) == []
    assert lib.retrieve("stack the cubes on top of each other", k=1)[0].skill.name == "stack_cube"


def test_library_hash_is_order_independent_and_freeze_is_read_only(tmp_path: Path) -> None:
    a = place_skill()
    b = place_skill(name="stack_cube", description="Put one cube on top of another cube.")
    lib1 = SkillLibrary([a, b])
    lib2 = SkillLibrary([b, a])
    assert lib1.sha256() == lib2.sha256()
    manifest = lib1.freeze()
    assert lib1.frozen and manifest.sha256 == lib1.sha256() and manifest.n_skills == 2
    with pytest.raises(FrozenLibraryError):
        lib1.add(place_skill(name="third"))
    with pytest.raises(FrozenLibraryError):
        lib1.remove("stack_cube")
    lib1.save(tmp_path, manifest=manifest)
    loaded = SkillLibrary.load(tmp_path, require_frozen=True)
    assert loaded.frozen and loaded.sha256() == manifest.sha256 and loaded.names == lib1.names
    assert SkillLibrary.frozen_sha256(tmp_path) == manifest.sha256
    assert SkillLibrary.load(tmp_path).to_json() == lib1.to_json()


def test_tampered_frozen_library_is_refused(tmp_path: Path) -> None:
    lib = SkillLibrary([place_skill()])
    lib.save(tmp_path, manifest=lib.freeze())
    raw = json.loads((tmp_path / "library.json").read_text())
    raw["skills"][0]["body"] += "\nrobot.reset()\n"
    (tmp_path / "library.json").write_text(json.dumps(raw))
    with pytest.raises(LibraryError, match="does not match"):
        SkillLibrary.load(tmp_path)
    with pytest.raises(LibraryError, match="no skill library"):
        SkillLibrary.load(tmp_path / "missing", require_frozen=True)
    (tmp_path / "FROZEN.json").unlink()
    with pytest.raises(LibraryError, match="not frozen"):
        SkillLibrary.load(tmp_path, require_frozen=True)


# -- execution ---------------------------------------------------------------------------------


def test_robot_without_library_raises_skill_not_available() -> None:
    robot, _, _ = rig(None)
    with pytest.raises(SkillNotAvailable, match="no skill library"):
        robot.execute_skill("pick_place_cube", color="red", x=-0.4, y=0.1)


def test_unknown_skill_lists_the_available_ones() -> None:
    runner = SkillRunner(SkillLibrary([place_skill()]), FAST)
    robot, _, _ = rig(runner)
    with pytest.raises(SkillNotAvailable) as exc:
        robot.execute_skill("fly", color="red", x=-0.4, y=0.1)
    assert exc.value.details["available"] == ["pick_place_cube"]


def test_skill_runs_and_reports_inner_calls() -> None:
    skill = place_skill()
    runner = SkillRunner(SkillLibrary([skill]), FAST)
    robot, backend, world = rig(runner)
    inst = place_instance(0)
    world.place(inst)
    args = goal_arguments(inst)
    res = robot.execute_skill(skill.name, **args)
    assert res.primitive == "execute_skill" and res.skill == skill.name
    assert res.skill_sha256 == skill.sha256()
    assert res.calls and "grasp" in res.calls and "release" in res.calls
    assert res.postconditions_checked == ("not holding",)
    assert res.postconditions_deferred == ("cube color within 2 cm of (x, y)",)
    assert res.t_end_sim > res.t_start_sim
    cube = next(c for c in backend.cubes if c.color == args["color"])
    assert abs(cube.x - float(args["x"])) < 0.02 and abs(cube.y - float(args["y"])) < 0.02


def test_precondition_failure_moves_nothing() -> None:
    skill = place_skill()
    runner = SkillRunner(SkillLibrary([skill]), FAST)
    robot, backend, world = rig(runner)
    inst = place_instance(1)
    world.place(inst)
    args = goal_arguments(inst)
    cube = next(c for c in backend.cubes if c.color == args["color"])
    robot.move_to(
        robot.tcp_pose(backend.q).model_copy(update={"x": cube.x, "y": cube.y, "z": cube.z})
    )
    robot.grasp()
    assert backend.held is not None
    n_before = len(backend.follow_log)
    with pytest.raises(SkillPreconditionFailed, match="not holding"):
        robot.execute_skill(skill.name, **args)
    assert len(backend.follow_log) == n_before, "a failed precondition never moves the arm"
    assert runner.last_run is None


def test_injected_postcondition_failure_is_detected() -> None:
    """Fault injection: a body that keeps holding the cube violates ``not_holding``."""
    body = PLACE_BODY.replace("robot.release()\n", "")
    skill = place_skill(body=body)
    runner = SkillRunner(SkillLibrary([skill]), FAST)
    robot, backend, world = rig(runner)
    inst = place_instance(2)
    world.place(inst)
    with pytest.raises(SkillPostconditionFailed, match="not holding") as exc:
        robot.execute_skill(skill.name, **goal_arguments(inst))
    assert exc.value.details["calls"]
    assert backend.held is not None


def test_body_errors_map_to_typed_errors() -> None:
    boom = place_skill(body="raise RuntimeError('cube not found: ' + color)\n")
    far = place_skill(name="far", body="robot.move_to(Pose(x, y, 0.9))\n")
    runner = SkillRunner(SkillLibrary([boom, far]), FAST)
    robot, _, world = rig(runner)
    inst = place_instance(3)
    world.place(inst)
    args = goal_arguments(inst)
    with pytest.raises(SkillFailed, match="cube not found"):
        robot.execute_skill(boom.name, **args)
    with pytest.raises(OutOfReach):
        robot.execute_skill("far", **args)


def test_robot_conditions() -> None:
    robot, _, _ = rig(None)
    obs = robot.state()
    assert check_robot_condition(Condition(kind="not_holding"), obs).ok
    assert not check_robot_condition(Condition(kind="holding"), obs).ok
    assert check_robot_condition(Condition(kind="tcp_above", z=0.05), obs).ok
    assert not check_robot_condition(Condition(kind="tcp_above", z=5.0), obs).ok
    deferred = check_robot_condition(Condition(kind="cube_near", color="color", x="x", y="y"), obs)
    assert deferred.deferred and deferred.ok


def test_sandboxed_program_can_call_a_skill_and_the_run_accounts_for_it() -> None:
    skill = place_skill()
    runner = SkillRunner(SkillLibrary([skill]), FAST)
    robot, _, world = rig(runner)
    inst = place_instance(4)
    world.place(inst)
    a = goal_arguments(inst)
    program = (
        f'robot.execute_skill("pick_place_cube", color="{a["color"]}", x={a["x"]}, y={a["y"]})\n'
    )
    run = run_program(program, robot, FAST)
    assert run.outcome == "completed", run.describe()
    assert run.primitives == ("execute_skill",)
    assert run.skills_used == ("pick_place_cube",)
    assert run.n_calls == 1 and run.n_primitive_calls == len(run.calls[0].inner_calls) >= 8
    bad = run_program('robot.execute_skill("nope")\n', robot, FAST)
    assert bad.outcome == "primitive_error" and bad.error_code == "skill_not_available"


# -- proposal ----------------------------------------------------------------------------------


def test_goal_constants_cover_every_task_kind() -> None:
    names = {t: [c.param for c in goal_constants(get_task(t).instance(0, CFG))] for t in (
        "pick_place@1", "stack2@1", "sort3@1", "place_obstacle@1"
    )}  # fmt: skip
    assert names["pick_place@1"] == ["color", "x", "y"] == names["place_obstacle@1"]
    assert names["stack2@1"] == ["top", "base"]
    assert names["sort3@1"] == [f"{k}_{i}" for i in (1, 2, 3) for k in ("color", "x", "y")]


def test_parametrise_replaces_literals_and_reports_missing() -> None:
    inst = place_instance(0)
    consts = goal_constants(inst)
    color, x, y = (c.literal for c in consts)
    program = f'd = best(robot.detect(), "{color}")\nplace = Pose({x}, {y}, 0.02)\nk = {x}5\n'
    par = parametrise(program, consts)
    assert par.missing == ()
    assert par.body == f"d = best(robot.detect(), color)\nplace = Pose(x, y, 0.02)\nk = {x}5\n"
    short = repr(round(float(x), 3))
    par2 = parametrise(f"place = Pose({short}, {y}, 0.02)\n", consts)
    assert par2.missing == ("color",) and par2.body == "place = Pose(x, y, 0.02)\n"


def test_propose_from_run_dir(tmp_path: Path) -> None:
    """Agent B's template programs on dev seeds become one candidate per task."""
    llm = load_llm_params()
    agent = get_agent("B", llm=llm, provider=make_provider(llm, kind="template"),
                      artifacts_dir=tmp_path)  # fmt: skip
    robot, _, world = rig(None)
    with (tmp_path / "episodes.jsonl").open("w") as fh:
        for tid in ("pick_place@1", "stack2@1"):
            for seed in (0, 1):
                task = get_task(tid)
                rec = run_episode(robot, agent, task, task.instance(seed, CFG), world,
                                  run_id="r1", repeat=0, backend="fake", software=SW)  # fmt: skip
                assert rec.ok, rec.reason
                fh.write(rec.line() + "\n")
    proposal = propose_from_run(tmp_path)
    assert proposal.n_episodes == 4 and proposal.n_completed == 4 and not proposal.rejected
    sigs = sorted(c.signature() for c in proposal.candidates)
    assert sigs == [
        "pick_place_cube(color: color, x: float, y: float)",
        "stack_cube(top: color, base: color)",
    ]
    pp = next(c for c in proposal.candidates if c.name == "pick_place_cube")
    assert pp.origin.seeds == (0, 1) and len(pp.origin.episode_ids) == 2
    assert pp.origin.task == "pick_place@1" and pp.origin.agent == "B"
    assert (
        "color" in pp.body
        and '"' + goal_constants(place_instance(0))[0].literal + '"' not in pp.body
    )
    # a tampered program file is rejected by hash
    name, version = pp.origin.task.split("@")
    path = tmp_path / "programs" / f"{name}_v{version}_s0_a1.py"
    path.write_text(path.read_text() + "# edited\n")
    tampered = propose_from_run(tmp_path)
    assert any("hash" in r.reason for r in tampered.rejected)
    records = [
        EpisodeRecord.model_validate_json(line)
        for line in (tmp_path / "episodes.jsonl").read_text().splitlines()
    ]
    assert all(r.trace is not None and r.trace.program_outcome == "completed" for r in records)


# -- validation --------------------------------------------------------------------------------


def test_validator_accepts_a_good_skill_and_records_it() -> None:
    v = Validator(seeds=range(20, 30), min_pass_rate=0.9)
    skill, report = v.validate(place_skill())
    assert report.accepted and report.n_ok == 10 and report.pass_rate == 1.0
    assert skill.validation is not None and skill.validation.accepted
    assert (
        skill.validation.seeds == tuple(range(20, 30))
        and skill.validation.split == "skill_validation"
    )
    assert skill.validation.backend == "fake"
    assert skill.sha256() == place_skill().sha256(), "validation is not part of the hash"


def test_validator_rejects_below_nine_of_ten() -> None:
    """Fault injection: a body that places the cube 5 cm off violates ``cube_near`` everywhere."""
    off = place_skill(
        body=PLACE_BODY.replace("place = Pose(x, y, pick.z)", "place = Pose(x + 0.05, y, pick.z)")
    )
    v = Validator(seeds=range(20, 30))
    skill, report = v.validate(off)
    assert not report.accepted and report.n_ok == 0
    assert all("cube color" in r.reason or "task verdict" in r.reason for r in report.results)
    assert skill.validation is not None and not skill.validation.accepted
    assert set(skill.validation.failed) == set(range(20, 30))
    lib, reports = v.build_library([off, place_skill()])
    assert [r.accepted for r in reports] == [False, True] and lib.names == ("pick_place_cube",)


def test_default_validation_seeds_are_the_skill_validation_split() -> None:
    v = Validator()
    assert v.seeds == list(range(20, 40)) and v.min_pass_rate == 0.9


# -- agent B+S ---------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def frozen_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("skills")
    lib = SkillLibrary([place_skill()])
    lib.save(d, manifest=lib.freeze())
    return d


def test_prompt_section_lists_contract_and_b_prompt_is_unchanged(frozen_dir: Path) -> None:
    lib = SkillLibrary.load(frozen_dir)
    inst = place_instance(0)
    section = skills_section(list(lib))
    assert 'robot.execute_skill("pick_place_cube", color=<color>, x=<float>, y=<float>)' in section
    assert (
        "requires: not holding; ensures: not holding, cube color within 2 cm of (x, y)" in section
    )
    plain = build_messages(inst, "SYS")
    with_skills = build_messages(inst, "SYS", list(lib))
    assert plain[0] == with_skills[0], "the system prompt is identical for B and B+S"
    assert "# Skills" not in plain[1].content and "# Skills" in with_skills[1].content
    assert with_skills[1].content.endswith(plain[1].content)


def test_template_provider_answers_with_the_offered_skill(frozen_dir: Path) -> None:
    lib = SkillLibrary.load(frozen_dir)
    inst = place_instance(0)
    user = build_messages(inst, "SYS", list(lib))[1].content
    assert offered_skills(user) == [
        ("pick_place_cube", ("color", "x", "y"), lib.skills[0].description)
    ]
    program = template_skill_program(user)
    assert program is not None and program.startswith(
        'robot.execute_skill("pick_place_cube", color='
    )
    assert template_skill_program(build_messages(inst, "SYS")[1].content) is None
    stack = get_task("stack2@1").instance(0, CFG)
    assert template_skill_program(build_messages(stack, "SYS", list(lib))[1].content) is None


def test_agent_bs_requires_a_frozen_library(tmp_path: Path) -> None:
    llm = load_llm_params()
    with pytest.raises(LibraryError):
        get_agent("B+S", llm=llm, provider=TemplateProvider(), skills_dir=tmp_path)
    SkillLibrary([place_skill()]).save(tmp_path)
    with pytest.raises(LibraryError, match="not frozen"):
        get_agent("B+S", llm=llm, provider=TemplateProvider(), skills_dir=tmp_path)


def test_agent_bs_uses_the_skill_and_traces_it(frozen_dir: Path) -> None:
    llm = load_llm_params()
    agent = get_agent("B+S", llm=llm, provider=TemplateProvider(), skills_dir=frozen_dir)
    assert isinstance(agent, LLMAgent) and agent.id == "B+S" and agent.library is not None
    runner = SkillRunner(agent.library, FAST)
    robot, _, world = rig(runner)
    task = get_task("pick_place@1")
    inst = task.instance(5, CFG)
    assert [s.name for s in agent.skills_for(inst)] == ["pick_place_cube"]
    rec = run_episode(
        robot, agent, task, inst, world, run_id="bs", repeat=0, backend="fake", software=SW
    )
    assert rec.ok, rec.reason
    assert rec.agent == "B+S" and rec.trace is not None
    assert rec.trace.skills_used == ("pick_place_cube",)
    assert rec.trace.program_calls == ("execute_skill",)
    assert rec.trace.n_primitives >= 8, "inner primitive calls are accounted"
    # the plain B agent on the same robot never touches the library
    b = get_agent("B", llm=llm, provider=TemplateProvider())
    rec_b = run_episode(
        robot, b, task, inst, world, run_id="b", repeat=0, backend="fake", software=SW
    )
    assert rec_b.ok and rec_b.trace is not None and rec_b.trace.skills_used == ()
    assert rec_b.trace.prompt_tokens < rec.trace.prompt_tokens


def test_agent_bs_falls_back_to_a_program_when_no_skill_fits(frozen_dir: Path) -> None:
    llm = load_llm_params()
    agent = get_agent("B+S", llm=llm, provider=TemplateProvider(), skills_dir=frozen_dir)
    assert isinstance(agent, LLMAgent) and agent.library is not None
    robot, _, world = rig(SkillRunner(agent.library, FAST))
    task = get_task("stack2@1")
    rec = run_episode(robot, agent, task, task.instance(0, CFG), world, run_id="bs", repeat=0,
                      backend="fake", software=SW)  # fmt: skip
    assert rec.ok and rec.trace is not None and rec.trace.skills_used == ()
    assert "execute_skill" not in rec.trace.program_calls


def test_primitive_error_inside_a_skill_is_a_robot_failure(frozen_dir: Path) -> None:
    lib = SkillLibrary.load(frozen_dir)
    robot, _, world = rig(SkillRunner(lib, FAST))
    task = get_task("pick_place@1")
    inst = task.instance(6, CFG)
    a = goal_arguments(inst)
    answer = (
        "```python\n"
        f'robot.execute_skill("pick_place_cube", color="{a["color"]}", x=5.0, y={a["y"]})\n'
        "```"
    )
    agent = LLMAgent(StaticProvider(answer), load_llm_params(), agent_id="B+S", library=lib)
    rec = run_episode(
        robot, agent, task, inst, world, run_id="e", repeat=0, backend="fake", software=SW
    )
    assert not rec.ok and rec.failure is not None
    assert rec.failure.stage == "robot" and rec.failure.code == OutOfReach.code


def test_committed_library_is_frozen_and_validated() -> None:
    """``skills/`` in the repository: frozen, hash matches the manifest, every skill accepted
    on the skill_validation split at the protocol's pass rate."""
    lib = SkillLibrary.load(SKILLS_DIR, require_frozen=True)
    manifest = json.loads((SKILLS_DIR / "FROZEN.json").read_text())
    assert lib.frozen and manifest["sha256"] == lib.sha256()
    assert manifest["skills"] == {s.name: s.sha256() for s in lib}
    assert set(lib.names) >= {"pick_place_cube", "stack_cube", "sort_cubes", "place_cube_over_wall"}
    for s in lib:
        v = s.validation
        assert v is not None and v.accepted and v.split == "skill_validation"
        assert v.seeds == tuple(range(20, 40)) and v.n_ok >= 0.9 * len(v.seeds)
        assert check_program(s.bind(s.placeholder_args())).ok
