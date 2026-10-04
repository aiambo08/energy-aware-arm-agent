"""Static whitelist, worker containment and the attack battery (every attack must be refused
or stopped: never a completed run that reached outside the API, never an escaped exception)."""

from __future__ import annotations

import random

import pytest

from armbench.perception import Camera, load_perception_params
from armbench.primitives import KinematicBackend, Robot, load_primitive_params
from armbench.sandbox import (
    ALLOWED_BUILTINS,
    ProgramLimits,
    SandboxLimits,
    check_program,
    run_program,
    worker_command,
)
from armbench.scene import load_scene_config
from armbench.tasks import get_task

FAST = SandboxLimits(cpu_s=2.0, memory_mb=256, wall_s=20.0, sim_s=30.0, max_calls=12, stdout_kb=1)


@pytest.fixture(scope="module")
def robot() -> Robot:
    params = load_primitive_params()
    inst = get_task("pick_place@1").instance(0, load_scene_config())
    backend = KinematicBackend(
        params, Camera.from_spec(load_perception_params().camera), cubes=list(inst.scene.cubes)
    )
    r = Robot(backend, params=params)
    r.reset()
    return r


def test_worker_command_uses_isolated_interpreter() -> None:
    cmd, layers = worker_command()
    assert "-I" in cmd and cmd[-1] == "armbench.guest.worker"
    assert all(isinstance(x, str) for x in layers)


def test_static_check_accepts_the_allowed_subset() -> None:
    src = (
        "def f(a, b=1, *rest, **kw):\n"
        "    return [x for x in rest if x > a] + sorted(kw.values())\n"
        "d = {'a': 1}\n"
        "s = {1, 2}\n"
        "t = (1, 2)\n"
        "ok = all(i > 0 for i in t) and any(x for x in s)\n"
        "try:\n"
        "    v = 1 / 0\n"
        "except ZeroDivisionError as e:\n"
        "    v = str(e)\n"
        "p = Pose(-0.5, 0.0, 0.1).above(0.1).with_yaw(math.pi / 4)\n"
        "print(f'{p.x:.2f} {v} {ok}', max(1, 2), min([3, 4]), abs(-1), round(1.26, 1))\n"
        "for i, x in enumerate(t):\n"
        "    while x > 10:\n"
        "        x -= 1\n"
        "    if x is None or x is not None:\n"
        "        continue\n"
        "lam = lambda q: q * 2\n"
        "z = lam(2) ** 2 % 3 // 1\n"
    )
    chk = check_program(src)
    assert chk.ok, chk.summary()
    assert "print" in ALLOWED_BUILTINS


def test_run_program_completes_prints_and_records_calls(robot: Robot) -> None:
    src = (
        "dets = robot.detect()\n"
        "print(len(dets), dets[0].color)\n"
        "o = robot.observe()\n"
        "robot.move_to(Pose(-0.45, 0.0, 0.20), 0.5)\n"
        "print(o.holding, robot.observe().tcp.z > 0.1)\n"
    )
    run = run_program(src, robot, FAST)
    assert run.outcome == "completed", run.describe()
    assert run.primitives == ("detect", "observe", "move_to", "observe")
    assert all(c.ok for c in run.calls) and run.calls[2].sim_s > 0
    assert run.stdout.splitlines()[1] == "False True"
    assert run.exit_code == 0 and run.error_code is None
    assert run.isolation[:2] == ("ast", "builtins")
    assert run.primitive_error() is None


def test_next_over_a_generator_is_allowed(robot: Robot) -> None:
    src = (
        "dets = robot.detect()\n"
        "first = next((d for d in dets if d.color == dets[0].color), None)\n"
        "none = next((d for d in dets if d.color == 'none'), None)\n"
        "print(first is not None, none)\n"
        "try:\n"
        "    next(iter([]))\n"
        "except StopIteration:\n"
        "    print('stop')\n"
    )
    chk = check_program(src)
    assert not chk.ok and "iter" in chk.summary()
    src = src.replace("next(iter([]))", "next(x for x in [])")
    assert check_program(src).ok
    run = run_program(src, robot, FAST)
    assert run.outcome == "completed", run.describe()
    assert run.stdout.splitlines() == ["True None", "stop"]


def test_primitive_error_is_rebuilt_on_the_parent_side(robot: Robot) -> None:
    src = (
        "try:\n"
        "    robot.move_to(Pose(2.0, 2.0, 2.0))\n"
        "except (OutOfReach, Collision) as e:\n"
        "    print('caught', e.code)\n"
        "robot.move_to(Pose(2.0, 2.0, 2.0))\n"
    )
    run = run_program(src, robot, FAST)
    assert run.outcome == "primitive_error" and run.error_code in ("out_of_reach", "collision")
    assert run.stdout.strip() == f"caught {run.error_code}"
    assert run.calls[0].ok is False and run.calls[0].error_code == run.error_code
    err = run.primitive_error()
    assert err is not None and err.code == run.error_code
    assert run.error_line == 5


STOP = (
    "rejected",
    "exception",
    "primitive_error",
    "timeout",
    "call_limit",
    "sim_limit",
    "killed",
    "protocol",
)

ATTACKS: list[tuple[str, str, tuple[str, ...]]] = [
    ("import", "import os\nos.system('id')\n", ("rejected",)),
    ("from-import", "from subprocess import run\n", ("rejected",)),
    ("import in try", "try:\n    import socket\nexcept Exception:\n    pass\n", ("rejected",)),
    ("dunder import", "__import__('os')\n", ("rejected",)),
    ("open", "open('/etc/passwd').read()\n", ("rejected",)),
    ("eval", "eval('1+1')\n", ("rejected",)),
    ("exec", "exec('x = 1')\n", ("rejected",)),
    ("compile", "compile('1', '', 'eval')\n", ("rejected",)),
    ("getattr", "getattr(robot, '_backend')\n", ("rejected",)),
    ("setattr", "setattr(robot, 'x', 1)\n", ("rejected",)),
    ("type", "type(robot)\n", ("rejected",)),
    ("globals", "globals()['robot']\n", ("rejected",)),
    ("vars", "vars(robot)\n", ("rejected",)),
    ("builtins name", "__builtins__['open']\n", ("rejected",)),
    ("breakpoint", "breakpoint()\n", ("rejected",)),
    ("input", "input()\n", ("rejected",)),
    ("exit", "exit()\n", ("rejected",)),
    ("memoryview", "memoryview(b'x')\n", ("rejected",)),
    ("object", "isinstance(robot, object)\n", ("rejected",)),
    ("class def", "class A:\n    pass\n", ("rejected",)),
    ("with", "with robot as r:\n    pass\n", ("rejected",)),
    ("yield", "def g():\n    yield 1\nlist(g())\n", ("rejected",)),
    ("async", "async def f():\n    pass\n", ("rejected",)),
    ("decorator", "@print\ndef f():\n    pass\n", ("rejected",)),
    ("global", "global x\nx = 1\n", ("rejected",)),
    ("walrus", "(y := 1)\n", ("rejected",)),
    ("match", "match 1:\n    case 1:\n        pass\n", ("rejected",)),
    ("subclasses walk", "().__class__.__bases__[0].__subclasses__()\n", ("rejected",)),
    ("robot dict", "robot.__dict__\n", ("rejected",)),
    ("math loader", "math.__loader__\n", ("rejected",)),
    ("lambda code", "(lambda: 0).__code__\n", ("rejected",)),
    ("record class", "robot.observe().__class__\n", ("rejected",)),
    ("method self", "[].append.__self__\n", ("rejected",)),
    ("generator frame", "g = (i for i in range(3))\ng.gi_frame\n", ("rejected",)),
    (
        "traceback frame",
        "try:\n    1 / 0\nexcept Exception as e:\n    e.__traceback__.tb_frame\n",
        ("rejected",),
    ),
    ("str.format", "'{0.__class__}'.format(1)\n", ("rejected",)),
    ("format_map", "'{x}'.format_map({'x': 1})\n", ("rejected",)),
    ("underscore name", "_hidden = 1\n", ("rejected",)),
    ("unknown name", "os.system('id')\n", ("rejected",)),
    ("syntax error", "def (:\n", ("rejected",)),
    ("null byte", "x = 1\x00\n", ("rejected",)),
    ("too long", "x = 1\n" * 10_000, ("rejected",)),
    ("deep nesting", "x = " + "(" * 400 + "1" + ")" * 400 + "\n", ("rejected", "exception")),
    ("cpu spin", "while True:\n    pass\n", ("killed", "timeout")),
    ("memory bomb", "x = [0] * (10 ** 9)\n", ("exception", "killed")),
    ("string doubling", "s = 'a'\nwhile True:\n    s += s\n", ("exception", "killed")),
    ("huge power", "x = 10 ** (10 ** 8)\n", ("exception", "killed", "timeout")),
    ("recursion", "def r():\n    r()\nr()\n", ("exception",)),
    ("call flood", "for i in range(1000):\n    robot.observe()\n", ("call_limit",)),
    (
        "call flood swallowing errors",
        "while True:\n    try:\n        robot.observe()\n    except Exception:\n        pass\n",
        ("call_limit",),
    ),
    ("bad pose type", "robot.move_to('home')\n", ("exception",)),
    ("bad speed type", "robot.move_to(Pose(-0.5, 0.0, 0.2), 'fast')\n", ("exception",)),
    ("bad detect arg", "robot.detect(['red'])\n", ("exception",)),
    ("extra arg", "robot.grasp(1)\n", ("exception",)),
    ("non-api attribute", "robot.home()\n", ("exception",)),
    ("backend attribute", "robot.backend.sim_time()\n", ("exception",)),
    ("nan pose", "Pose(float('nan'), 0.0, 0.0)\n", ("exception",)),
    ("inf pose", "Pose(1e308 * 10, 0.0, 0.0)\n", ("exception",)),
    ("out of reach", "robot.move_to(Pose(9.0, 9.0, 9.0))\n", ("primitive_error",)),
    ("skill", "robot.execute_skill('fly')\n", ("primitive_error",)),
    ("rebind robot", "robot = 5\nrobot.observe()\n", ("rejected", "exception")),
    ("delete robot", "del robot\nrobot.observe()\n", ("rejected", "exception")),
    ("raise base", "raise BaseException('x')\n", ("rejected", "exception")),
    ("raise SystemExit", "raise SystemExit(0)\n", ("rejected",)),
    ("raise KeyboardInterrupt", "raise KeyboardInterrupt()\n", ("rejected",)),
    ("bytearray", "bytearray(10 ** 9)\n", ("rejected",)),
    ("robot private attr", "robot._backend\n", ("rejected",)),
    ("math dunder", "math.__name__\n", ("rejected",)),
    ("Pose dunder", "Pose.__init__\n", ("rejected",)),
    (
        "nonlocal",
        "def f():\n    x = 1\n    def g():\n        nonlocal x\n    g()\nf()\n",
        ("rejected",),
    ),
    (
        "nested dict pose",
        "d = {'x': 0.3}\nfor i in range(10000):\n    d = {'d': d}\nrobot.move_to(d)\n",
        ("exception",),
    ),
    ("nan speed", "robot.move_to(Pose(-0.5, 0.0, 0.2), float('nan'))\n", ("exception",)),
    ("huge speed", "robot.move_to(Pose(-0.5, 0.0, 0.2), 1e308)\n", ("exception",)),
    ("extra kwarg", "robot.move_to(Pose(-0.5, 0.0, 0.2), 1.0, foo=1)\n", ("exception",)),
    ("star None", "a = [None, 1.0]\nrobot.move_to(*a)\n", ("exception",)),
    ("bool pose", "robot.move_to(Pose(True, False, True), True)\n", ("exception",)),
    ("big int pose", "robot.move_to(Pose(10 ** 400, 0, 0))\n", ("exception",)),
    ("nan above", "robot.move_to(Pose(-0.5, 0.0, 0.2).above(float('nan')))\n", ("exception",)),
    ("detect None", "robot.detect(None)\n", ("exception",)),
    ("mutate result", "d = robot.detect('any')\nd[0].x = 5\n", ("exception",)),
    ("assign robot attr", "robot.x = 1\n", ("exception",)),
    ("rpc spoof done", 'print(\'{"op":"done","ok":true}\')\nrobot.observe()\n', ("completed",)),
    (
        "rpc spoof call",
        'print(\'{"op":"call","name":"reset"}\')\nrobot.observe()\n',
        ("completed",),
    ),
    ("binary stdout", "print('\\x00\\xff' * 100)\nrobot.observe()\n", ("completed",)),
    ("newline in arg", 'robot.detect(\'red\\n{"op":"done"}\')\n', ("exception",)),
    ("print flood", "print('x' * 100000)\n", ("completed",)),
]


@pytest.mark.parametrize(("name", "src", "expected"), ATTACKS, ids=[a[0] for a in ATTACKS])
def test_attack_battery(robot: Robot, name: str, src: str, expected: tuple[str, ...]) -> None:
    run = run_program(src, robot, FAST)
    assert run.outcome in expected, (name, run.describe(), run.stderr_tail)
    if run.outcome == "rejected":
        assert not run.check.ok and run.n_calls == 0
    if name.startswith("rpc spoof") or name == "binary stdout":
        assert run.n_calls == 1 and run.stdout.startswith(("{", "\x00"))
    if name == "print flood":
        assert run.stdout_truncated and len(run.stdout) <= 1024 + 64
    if name == "call flood swallowing errors":
        assert run.n_calls == FAST.max_calls


def test_at_least_forty_attacks_are_stopped() -> None:
    stopped = [a for a in ATTACKS if "completed" not in a[2]]
    assert len(stopped) >= 40


def test_sim_limit_stops_a_program_that_keeps_moving(robot: Robot) -> None:
    src = (
        "while True:\n"
        "    robot.move_to(Pose(-0.45, 0.10, 0.25))\n"
        "    robot.move_to(Pose(-0.45, -0.10, 0.25))\n"
    )
    lim = FAST.model_copy(update={"sim_s": 3.0, "max_calls": 1000})
    run = run_program(src, robot, lim)
    assert run.outcome == "sim_limit", run.describe()
    assert 0 < run.n_calls < 1000


def test_wall_timeout_kills_a_sleeping_loop(robot: Robot) -> None:
    src = "x = 0\nwhile True:\n    x = (x + 1) % 7\n"
    lim = SandboxLimits(cpu_s=30.0, wall_s=1.0, max_calls=5)
    run = run_program(src, robot, lim)
    assert run.outcome in ("timeout", "killed"), run.describe()
    assert run.wall_s < 10


def test_program_limits_reject_oversize_before_parsing() -> None:
    chk = check_program("x = 1\n" * 50, ProgramLimits(max_source_chars=100))
    assert not chk.ok and "chars" in chk.summary()
    chk = check_program("x = " + " + ".join(["1"] * 300) + "\n", ProgramLimits(max_nodes=100))
    assert not chk.ok and "nodes" in chk.summary()


VALID = (
    "dets = robot.detect()\n"
    "best = None\n"
    "for d in dets:\n"
    "    if d.color == 'green' and (best is None or d.complete):\n"
    "        best = d\n"
    "if best is not None:\n"
    "    p = Pose(best.position[0], best.position[1], best.position[2], best.yaw_rad)\n"
    "    robot.move_to(p.above(0.1))\n"
    "    robot.move_to(p, 0.5)\n"
    "    robot.grasp()\n"
    "    robot.move_to(p.above(0.1), 0.5)\n"
    "    robot.release()\n"
    "print('done', len(dets))\n"
)


def _mutate(rng: random.Random, src: str) -> str:
    alphabet = "()[]{}.:_'\"= \n#\\abcxyz0123456789+-*/%<>!,"
    chars = list(src)
    for _ in range(rng.randint(1, 6)):
        op = rng.random()
        pos = rng.randrange(len(chars) + 1)
        if op < 0.4 and chars:
            del chars[min(pos, len(chars) - 1)]
        elif op < 0.8:
            chars.insert(pos, rng.choice(alphabet))
        else:
            a, b = sorted((pos, rng.randrange(len(chars) + 1)))
            del chars[a:b]
    return "".join(chars)


def test_random_mutations_never_escape(robot: Robot) -> None:
    rng = random.Random(1234)
    base = run_program(VALID, robot, FAST)
    assert base.outcome == "completed", base.describe()
    outcomes: dict[str, int] = {}
    for _ in range(60):
        src = _mutate(rng, VALID)
        run = run_program(src, robot, FAST)
        assert run.outcome in (*STOP, "completed"), run.describe()
        assert run.n_calls <= FAST.max_calls
        outcomes[run.outcome] = outcomes.get(run.outcome, 0) + 1
    assert outcomes.get("rejected", 0) > 0
