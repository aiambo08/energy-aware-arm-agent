"""Run one agent program against a :class:`Robot` inside the sandbox (parent side).

Defence in depth, from the plan (F6):

1. :func:`check_program` — AST whitelist; rejected programs never reach a process.
2. A child process with no network (``unshare -rn`` where the kernel permits unprivileged
   user namespaces; the simulation containers additionally run with ``--network none``),
   an empty read-only working directory and ``RLIMIT_FSIZE = 0`` (no file it could write).
3. Address-space, CPU-second, fork and wall-clock limits, plus caps on the number of primitive
   calls, on simulated time and on captured stdout.
4. The program only reaches the robot through a typed JSON line protocol: the parent accepts
   the seven primitive names, validates every argument with the Pydantic contracts of
   :mod:`armbench.primitives` and ships results back as JSON.

Nothing the child does can raise in the parent except through ``PrimitiveError`` instances the
parent itself produced while serving a call; those are reported in the :class:`ProgramRun` and
re-raised by the agent so the episode log blames the right stage.
"""

from __future__ import annotations

import contextlib
import functools
import json
import os
import selectors
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from armbench.guest.api import PRIMITIVES
from armbench.guest.ast_check import ProgramCheck, ProgramLimits, check_program
from armbench.primitives import ERRORS, Pose, PrimitiveError, Robot

Outcome = Literal[
    "completed",
    "rejected",
    "exception",
    "primitive_error",
    "timeout",
    "call_limit",
    "sim_limit",
    "killed",
    "protocol",
]
WORKER_MODULE: Final = "armbench.guest.worker"
_REPORTED: Final[dict[str, Outcome]] = {
    "completed": "completed",
    "exception": "exception",
    "primitive_error": "primitive_error",
    "timeout": "timeout",
    "call_limit": "call_limit",
    "sim_limit": "sim_limit",
    "protocol": "protocol",
}
_ERRORS_BY_CODE: Final[dict[str, type[PrimitiveError]]] = {e.code: e for e in ERRORS}
_SIG_KILLED: Final = frozenset({-signal.SIGXCPU, -signal.SIGKILL, -signal.SIGSEGV, 137, 152})


class SandboxLimits(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    cpu_s: float = Field(gt=0, default=10.0)
    """CPU seconds of the child (program logic only; time inside primitives is the parent's)."""
    memory_mb: int = Field(gt=0, default=512)
    wall_s: float = Field(gt=0, default=300.0)
    """Wall seconds for the whole program, primitives included."""
    sim_s: float = Field(gt=0, default=60.0)
    """Simulated seconds the program may consume before it is stopped (task limit)."""
    max_calls: int = Field(gt=0, default=200)
    stdout_kb: int = Field(gt=0, default=16)
    program: ProgramLimits = ProgramLimits()


class CallRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    primitive: str
    ok: bool
    error_code: str | None = None
    sim_s: float = 0.0
    wall_s: float = Field(ge=0, default=0.0)
    skill: str | None = None
    """Skill name when ``primitive == "execute_skill"``."""
    inner_calls: tuple[str, ...] = ()
    """Primitives the skill body ran (a skill call is one call to the program, several to the
    robot; ``ProgramRun.n_primitive_calls`` counts the latter)."""


class ProgramRun(BaseModel):
    """What happened to one program; JSON-serialisable for the episode log."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    outcome: Outcome
    check: ProgramCheck
    calls: tuple[CallRecord, ...] = ()
    error_code: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    error_details: dict[str, object] = Field(default_factory=dict)
    error_line: int | None = None
    stdout: str = ""
    stdout_truncated: bool = False
    wall_s: float = Field(ge=0, default=0.0)
    sim_s: float = 0.0
    isolation: tuple[str, ...] = ()
    """Containment layers that were active (``ast``, ``builtins``, ``rlimits:...``,
    ``unshare_net``, ``container``)."""
    exit_code: int | None = None
    stderr_tail: str = ""

    @property
    def n_calls(self) -> int:
        return len(self.calls)

    @property
    def primitives(self) -> tuple[str, ...]:
        return tuple(c.primitive for c in self.calls)

    @property
    def n_primitive_calls(self) -> int:
        """Robot primitives actually run: top-level calls plus those inside skills."""
        return sum(len(c.inner_calls) if c.inner_calls else 1 for c in self.calls)

    @property
    def skills_used(self) -> tuple[str, ...]:
        return tuple(c.skill for c in self.calls if c.skill is not None and c.ok)

    def primitive_error(self) -> PrimitiveError | None:
        """The unhandled primitive error, rebuilt as the parent-side class (``None`` otherwise)."""
        if self.outcome != "primitive_error" or self.error_code is None:
            return None
        cls = _ERRORS_BY_CODE.get(self.error_code, PrimitiveError)
        return cls(self.error_message or self.error_code, **self.error_details)

    def describe(self) -> str:
        """One line for logs and for the retry feedback given to an LLM."""
        if self.outcome == "completed":
            return f"completed after {self.n_calls} primitive calls"
        if self.outcome == "rejected":
            return f"rejected by the static check: {self.check.summary()}"
        where = f" (line {self.error_line})" if self.error_line else ""
        what = (
            f"{self.error_type or self.outcome}: {self.error_message}" if self.error_message else ""
        )
        return f"{self.outcome}{where} after {self.n_calls} primitive calls {what}".rstrip()


class _DeadlineError(Exception):
    pass


@functools.cache
def unshare_available() -> bool:
    """Can this host put a child in a fresh user + network namespace (``unshare -rn``)?"""
    if os.environ.get("ARMBENCH_SANDBOX_UNSHARE", "1") in ("0", "no", "false"):
        return False
    exe = shutil.which("unshare")
    if exe is None:
        return False
    try:
        proc = subprocess.run(  # noqa: S603 - fixed probe
            [exe, "-rn", "true"], capture_output=True, timeout=10, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


def worker_command(python: str | None = None) -> tuple[list[str], tuple[str, ...]]:
    """Command line of the worker and the isolation layers it adds."""
    cmd: list[str] = []
    layers: list[str] = []
    if unshare_available():
        cmd += [shutil.which("unshare") or "unshare", "-rn"]
        layers.append("unshare_net")
    cmd += [python or sys.executable, "-I", "-m", WORKER_MODULE]
    if Path("/.dockerenv").exists():
        layers.append("container")
    return cmd, tuple(layers)


class _LineReader:
    """Non-blocking line reader over the child's stdout with an absolute deadline."""

    def __init__(self, fd: int, deadline: float) -> None:
        self.fd = fd
        self.deadline = deadline
        self.buf = b""
        self.eof = False
        self.sel = selectors.DefaultSelector()
        self.sel.register(fd, selectors.EVENT_READ)

    def line(self) -> bytes | None:
        while True:
            if b"\n" in self.buf:
                head, self.buf = self.buf.split(b"\n", 1)
                return head
            if self.eof:
                return None
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                raise _DeadlineError
            if not self.sel.select(timeout=min(remaining, 1.0)):
                continue
            chunk = os.read(self.fd, 65536)
            if not chunk:
                self.eof = True
            self.buf += chunk

    def close(self) -> None:
        self.sel.close()


class _Dispatcher:
    """Serves primitive calls on the real robot and records each one."""

    def __init__(self, robot: Robot) -> None:
        self.robot = robot
        self.calls: list[CallRecord] = []
        self.last_error: PrimitiveError | None = None

    def serve(self, msg: dict[str, object]) -> dict[str, object]:
        name = msg.get("call")
        args = msg.get("args", [])
        kwargs = msg.get("kwargs", {})
        if not isinstance(name, str) or name not in PRIMITIVES:
            return {"ok": False, "code": "bad_arguments", "message": f"unknown primitive {name!r}"}
        if not isinstance(args, list) or not isinstance(kwargs, dict):
            return {"ok": False, "code": "bad_arguments", "message": "malformed call"}
        t_wall = time.time()
        t_sim0 = self.robot.backend.sim_time()
        ok, code = True, None
        skill: str | None = None
        inner: tuple[str, ...] = ()
        try:
            result = self._invoke(name, args, kwargs)
            reply: dict[str, object] = {"ok": True, "result": result}
            if name == "execute_skill" and isinstance(result, dict):
                skill = str(result.get("skill"))
                inner = tuple(str(c) for c in result.get("calls", ()))
        except PrimitiveError as exc:
            ok, code = False, exc.code
            self.last_error = exc
            reply = {"ok": False, "code": exc.code, "message": exc.message,
                     "details": _jsonable(exc.details)}  # fmt: skip
        except (TypeError, ValueError, ValidationError) as exc:
            ok, code = False, "bad_arguments"
            reply = {"ok": False, "code": "bad_arguments", "message": f"{name}: {exc}"[:2000]}
        t_sim1 = self.robot.backend.sim_time()
        self.calls.append(
            CallRecord(
                primitive=name,
                ok=ok,
                error_code=code,
                sim_s=_finite(t_sim1 - t_sim0),
                wall_s=time.time() - t_wall,
                skill=skill,
                inner_calls=inner,
            )
        )
        return reply

    def _invoke(  # noqa: PLR0911 - one return per primitive
        self, name: str, args: list[object], kwargs: dict[str, object]
    ) -> object:
        r = self.robot
        if name == "observe":
            _arity(name, args, kwargs, 0)
            return r.observe().model_dump(mode="json")
        if name == "detect":
            target = _one_arg(name, args, kwargs, "target", "any")
            if not isinstance(target, str):
                msg = "detect(target) takes a colour name or 'any'"
                raise TypeError(msg)
            return [d.model_dump(mode="json") for d in r.detect(target)]
        if name == "move_to":
            pose_raw, speed = _move_args(args, kwargs)
            pose = Pose.model_validate(pose_raw)
            return r.move_to(pose, speed).model_dump(mode="json")
        if name == "grasp":
            _arity(name, args, kwargs, 0)
            return r.grasp().model_dump(mode="json")
        if name == "release":
            _arity(name, args, kwargs, 0)
            return r.release().model_dump(mode="json")
        if name == "reset":
            _arity(name, args, kwargs, 0)
            return r.reset().model_dump(mode="json")
        skill = _one_arg(name, args, kwargs, "name", None)
        if not isinstance(skill, str):
            msg = "execute_skill(name, **kwargs) needs the skill name"
            raise TypeError(msg)
        rest = {k: v for k, v in kwargs.items() if k != "name"}
        return r.execute_skill(skill, **rest).model_dump(mode="json")


def _arity(name: str, args: list[object], kwargs: dict[str, object], n: int) -> None:
    if len(args) != n or kwargs:
        msg = f"{name}() takes {n} positional arguments and no keywords"
        raise TypeError(msg)


def _one_arg(
    name: str, args: list[object], kwargs: dict[str, object], key: str, default: object
) -> object:
    if len(args) > 1 or (args and key in kwargs):
        msg = f"{name}() takes one argument ({key})"
        raise TypeError(msg)
    if args:
        return args[0]
    if key in kwargs:
        return kwargs[key]
    if name == "detect" and kwargs:
        msg = f"detect() got unexpected keywords {sorted(kwargs)}"
        raise TypeError(msg)
    return default


def _move_args(args: list[object], kwargs: dict[str, object]) -> tuple[object, float]:
    allowed = {"pose", "speed_scale"}
    if not allowed >= kwargs.keys() or len(args) > 2:
        msg = "move_to(pose, speed_scale=1.0)"
        raise TypeError(msg)
    pose = args[0] if args else kwargs.get("pose")
    if pose is None:
        msg = "move_to() needs a Pose"
        raise TypeError(msg)
    speed = args[1] if len(args) > 1 else kwargs.get("speed_scale", 1.0)
    if isinstance(speed, bool) or not isinstance(speed, int | float):
        msg = "speed_scale must be a number"
        raise TypeError(msg)
    return pose, float(speed)


def _finite(x: float) -> float:
    return x if x == x and abs(x) != float("inf") else 0.0  # noqa: PLR0124 - NaN check


def _jsonable(value: object) -> dict[str, object]:
    try:
        return json.loads(json.dumps(value, default=str))  # type: ignore[no-any-return]
    except (TypeError, ValueError):
        return {"repr": repr(value)[:500]}


def _kill(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def run_program(  # noqa: PLR0912, PLR0915
    source: str,
    robot: Robot,
    limits: SandboxLimits | None = None,
    *,
    python: str | None = None,
) -> ProgramRun:
    """Check, execute and account for one program; never raises for anything the program does."""
    lim = limits or SandboxLimits()
    t_wall = time.time()
    check = check_program(source, lim.program)
    layers: list[str] = ["ast", "builtins"]
    if not check.ok:
        return ProgramRun(
            outcome="rejected", check=check, wall_s=time.time() - t_wall,
            error_code="program_rejected", error_message=check.summary(), isolation=tuple(layers),
        )  # fmt: skip
    cmd, extra = worker_command(python)
    layers += extra
    workdir = Path(tempfile.mkdtemp(prefix="armbench_sbx_"))
    os.chmod(workdir, stat.S_IRUSR | stat.S_IXUSR)
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LANG": "C.UTF-8"}
    stderr_file = tempfile.TemporaryFile()  # noqa: SIM115 - closed on every path below
    dispatcher = _Dispatcher(robot)
    outcome: Outcome = "protocol"
    status: dict[str, object] = {}
    exit_code: int | None = None
    t_sim0 = robot.backend.sim_time()
    try:
        proc = subprocess.Popen(  # noqa: S603 - fixed interpreter, our own worker module
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr_file,
            cwd=workdir,
            env=env,
            start_new_session=True,
            bufsize=0,
        )
    except OSError as exc:
        stderr_file.close()
        shutil.rmtree(workdir, ignore_errors=True)
        return ProgramRun(
            outcome="protocol", check=check, wall_s=time.time() - t_wall,
            error_message=f"could not start the worker: {exc}", isolation=tuple(layers),
        )  # fmt: skip
    assert proc.stdin is not None  # noqa: S101 - PIPE above
    assert proc.stdout is not None  # noqa: S101 - PIPE above
    deadline = time.monotonic() + lim.wall_s
    reader = _LineReader(proc.stdout.fileno(), deadline)
    header = {
        "source": source,
        "limits": {"cpu_s": lim.cpu_s, "memory_mb": lim.memory_mb},
        "stdout_limit": lim.stdout_kb * 1024,
    }

    def send(msg: dict[str, object]) -> bool:
        assert proc.stdin is not None  # noqa: S101
        try:
            proc.stdin.write((json.dumps(msg, separators=(",", ":")) + "\n").encode("utf-8"))
            proc.stdin.flush()
        except (BrokenPipeError, OSError):
            return False
        return True

    try:
        if not send(header):
            raise _DeadlineError
        while True:
            raw = reader.line()
            if raw is None:
                break
            try:
                msg = json.loads(raw)
            except ValueError:
                status = {"status": "protocol", "message": "unparseable message from the worker"}
                break
            if not isinstance(msg, dict):
                status = {"status": "protocol", "message": "non-object message from the worker"}
                break
            if msg.get("ready"):
                applied = msg.get("limits")
                if isinstance(applied, list) and applied:
                    layers.append("rlimits:" + ",".join(str(a) for a in applied))
                continue
            if msg.get("done"):
                status = msg
                break
            if "call" in msg:
                if len(dispatcher.calls) >= lim.max_calls:
                    outcome = "call_limit"
                    status = {
                        "status": "call_limit",
                        "message": f"more than {lim.max_calls} primitive calls",
                    }
                    send({"ok": False, "code": "stop", "message": status["message"]})
                    break
                reply = dispatcher.serve(msg)
                if robot.backend.sim_time() - t_sim0 > lim.sim_s:
                    outcome = "sim_limit"
                    status = {
                        "status": "sim_limit",
                        "message": f"more than {lim.sim_s:.0f} simulated seconds",
                    }
                    send({"ok": False, "code": "stop", "message": status["message"]})
                    break
                if not send(reply):
                    break
                continue
            status = {"status": "protocol", "message": f"unexpected message keys {sorted(msg)}"}
            break
    except _DeadlineError:
        outcome = "timeout"
        status = {"status": "timeout", "message": f"program exceeded {lim.wall_s:.0f} wall seconds"}
    finally:
        reader.close()
        if status.get("done"):
            with contextlib.suppress(OSError):
                proc.stdin.close()
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=5)
        _kill(proc)
        exit_code = proc.returncode
        try:
            proc.stdin.close()
            proc.stdout.close()
        except OSError:
            pass
        stderr_file.seek(0)
        stderr_tail = stderr_file.read()[-2000:].decode("utf-8", "replace")
        stderr_file.close()
        shutil.rmtree(workdir, ignore_errors=True)

    st = str(status.get("status", ""))
    if st in _REPORTED:
        outcome = _REPORTED[st]
    elif exit_code in _SIG_KILLED:
        outcome = "killed"
        status = {"message": f"worker killed (exit {exit_code}): CPU or memory limit"}
    else:
        outcome = "protocol"
        status = {"message": f"worker ended without a report (exit {exit_code})"}
    code: str | None
    if outcome == "primitive_error":
        code = str(status.get("code") or "primitive_error")
    elif outcome == "completed":
        code = None
    else:
        code = {"exception": "program_exception", "timeout": "program_timeout",
                "call_limit": "program_call_limit", "sim_limit": "program_sim_limit",
                "killed": "program_killed", "protocol": "program_protocol"}[outcome]  # fmt: skip
    details = status.get("details")
    line = status.get("line")
    return ProgramRun(
        outcome=outcome,
        check=check,
        calls=tuple(dispatcher.calls),
        error_code=code,
        error_type=str(status["exc_type"]) if status.get("exc_type") else None,
        error_message=str(status["message"])[:2000] if status.get("message") is not None else None,
        error_details=details if isinstance(details, dict) else {},
        error_line=int(line) if isinstance(line, int) else None,
        stdout=str(status.get("stdout", "")),
        stdout_truncated=bool(status.get("stdout_truncated", False)),
        wall_s=time.time() - t_wall,
        sim_s=_finite(robot.backend.sim_time() - t_sim0),
        isolation=tuple(layers),
        exit_code=exit_code,
        stderr_tail=stderr_tail,
    )
