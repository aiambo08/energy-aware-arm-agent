"""Sandbox worker: the child process that executes one agent program (layers 2-4).

Started by :func:`armbench.sandbox.run_program` with ``python -I -m armbench.guest.worker``
(and under ``unshare -rn`` where the kernel allows it). Protocol, one JSON object per line:

* parent -> child: a header ``{"source", "limits", "stdout_limit"}``; later, one reply per call
  ``{"ok": true, "result": ...}`` or ``{"ok": false, "code", "message", "details"}``.
* child -> parent: ``{"ready": true}`` once the resource limits are in place; one
  ``{"call", "args", "kwargs"}`` per primitive the program invokes; finally
  ``{"done": true, "status", ...}`` with the captured stdout.

Before the program runs the child lowers its own hard limits (address space, CPU seconds, no
file creation, no forks, no core dumps) and replaces ``__builtins__`` with the whitelist of
:mod:`armbench.guest.ast_check`; ``print`` writes into a capped buffer that is returned to
the parent, never to the protocol channel.
"""

from __future__ import annotations

import builtins
import json
import math
import os
import resource
import sys
import traceback
from collections.abc import Callable, Mapping
from typing import TextIO

from armbench.guest import api
from armbench.guest.ast_check import ALLOWED_BUILTINS, ALLOWED_EXCEPTIONS

EXIT_LOST_PARENT = 3
EXIT_STOPPED = 4

API_ERRORS: tuple[type[api.PrimitiveError], ...] = (api.PrimitiveError, *api.ERRORS)


class Capture:
    """``print`` target with a byte budget; keeps the head and notes the truncation."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.parts: list[str] = []
        self.size = 0
        self.truncated = False

    def write(self, s: str) -> int:
        if not s:
            return 0
        room = self.limit - self.size
        if len(s) > room:
            self.truncated = True
        if room > 0:
            self.parts.append(s[:room])
            self.size += min(len(s), room)
        return len(s)

    def flush(self) -> None:
        return None

    def text(self) -> str:
        return "".join(self.parts)


def apply_limits(limits: Mapping[str, object]) -> list[str]:
    """Lower the process limits; returns the names of the limits that took effect."""
    applied: list[str] = []
    mem = int(limits.get("memory_mb", 512)) * 1024 * 1024  # type: ignore[call-overload]
    cpu = math.ceil(float(limits.get("cpu_s", 10.0)))  # type: ignore[arg-type]
    wanted: list[tuple[str, int, tuple[int, int]]] = [
        ("as", resource.RLIMIT_AS, (mem, mem)),
        ("cpu", resource.RLIMIT_CPU, (cpu, cpu + 1)),
        ("fsize", resource.RLIMIT_FSIZE, (0, 0)),
        ("nproc", resource.RLIMIT_NPROC, (0, 0)),
        ("core", resource.RLIMIT_CORE, (0, 0)),
    ]
    for name, res, lim in wanted:
        try:
            resource.setrlimit(res, lim)
            applied.append(name)
        except (ValueError, OSError):
            continue
    return applied


def safe_builtins(printer: Callable[..., None]) -> dict[str, object]:
    real = vars(builtins)
    out: dict[str, object] = {name: real[name] for name in ALLOWED_BUILTINS}
    for name in ALLOWED_EXCEPTIONS:
        if name in real:
            out[name] = real[name]
    for err in API_ERRORS:
        out[err.__name__] = err
    out["print"] = printer
    return out


class Channel:
    def __init__(self, inp: TextIO, out: TextIO) -> None:
        self.inp = inp
        self.out = out

    def send(self, msg: Mapping[str, object]) -> None:
        self.out.write(json.dumps(msg, separators=(",", ":")) + "\n")
        self.out.flush()

    def recv(self) -> dict[str, object]:
        line = self.inp.readline()
        if not line:
            raise SystemExit(EXIT_LOST_PARENT)
        data = json.loads(line)
        if not isinstance(data, dict):
            raise SystemExit(EXIT_LOST_PARENT)
        return data


class RobotProxy:
    """``robot`` as the program sees it: every primitive is a round trip to the parent."""

    __slots__ = ("_chan",)

    def __init__(self, chan: Channel) -> None:
        object.__setattr__(self, "_chan", chan)

    def __getattr__(self, name: str) -> Callable[..., object]:
        if name not in api.PRIMITIVES:
            msg = f"robot has no primitive {name!r}; available: {', '.join(api.PRIMITIVES)}"
            raise AttributeError(msg)
        chan: Channel = object.__getattribute__(self, "_chan")

        def call(*args: object, **kwargs: object) -> object:
            chan.send(
                {
                    "call": name,
                    "args": [api.unwrap(a) for a in args],
                    "kwargs": {str(k): api.unwrap(v) for k, v in kwargs.items()},
                }
            )
            reply = chan.recv()
            if reply.get("ok"):
                return api.wrap(reply.get("result"))
            code = str(reply.get("code", "primitive_error"))
            message = str(reply.get("message", ""))
            if code == "bad_arguments":
                raise TypeError(message)
            if code == "stop":
                raise SystemExit(EXIT_STOPPED)
            details = reply.get("details")
            raise api.error_from_code(code, message, details if isinstance(details, dict) else {})

        return call

    def __setattr__(self, name: str, value: object) -> None:
        msg = "robot is read-only"
        raise AttributeError(msg)

    def __repr__(self) -> str:
        return "<robot: " + ", ".join(f"{p}()" for p in api.PRIMITIVES) + ">"


def _program_line(exc: BaseException) -> int | None:
    frames = [f for f in traceback.extract_tb(exc.__traceback__) if f.filename == "<program>"]
    return frames[-1].lineno if frames else None


def execute(source: str, chan: Channel, capture: Capture) -> dict[str, object]:
    def printer(*values: object, sep: str = " ", end: str = "\n") -> None:
        capture.write(sep.join(str(v) for v in values) + end)

    namespace: dict[str, object] = {
        "__builtins__": safe_builtins(printer),
        "robot": RobotProxy(chan),
        "Pose": api.Pose,
        "math": math,
    }
    for err in API_ERRORS:
        namespace[err.__name__] = err
    status: dict[str, object] = {"done": True, "status": "completed"}
    try:
        code = compile(source, "<program>", "exec")
        exec(code, namespace)  # noqa: S102 - the whole point, behind the AST gate and rlimits
    except api.PrimitiveError as exc:
        status = {
            "done": True,
            "status": "primitive_error",
            "code": exc.code,
            "exc_type": type(exc).__name__,
            "message": exc.message,
            "details": exc.details,
            "line": _program_line(exc),
        }
    except SystemExit:
        raise
    except BaseException as exc:
        status = {
            "done": True,
            "status": "exception",
            "exc_type": type(exc).__name__,
            "message": str(exc)[:2000],
            "line": _program_line(exc),
        }
    status["stdout"] = capture.text()
    status["stdout_truncated"] = capture.truncated
    return status


def main() -> None:
    header = json.loads(sys.stdin.readline())
    channel_out = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    chan = Channel(sys.stdin, channel_out)
    capture = Capture(int(header.get("stdout_limit", 16_384)))
    applied = apply_limits(header.get("limits", {}))
    sys.stdout = capture
    chan.send({"ready": True, "limits": applied, "pid": os.getpid()})
    status = execute(str(header["source"]), chan, capture)
    chan.send(status)


if __name__ == "__main__":
    main()
