"""The skill format (phase F7, docs/plan.es.md): name, description, typed signature, executable
pre- and postconditions, origin (episode and program hash), version and validation record.

A skill *body* is an agent program in the sandbox dialect (``robot.<primitive>``, ``Pose``,
``math``) whose free names are the skill's parameters. ``execute_skill(name, **kwargs)`` binds
the parameters as plain assignments on top of the body and runs the result in the same sandbox
as any agent program, so a skill can never do more than the program that produced it.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ParamKind = Literal["color", "float"]
ConditionKind = Literal["not_holding", "holding", "tcp_above", "cube_near"]

NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,47}$")
COLOR_RE = re.compile(r"^[a-z]{1,16}$")


class Param(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(pattern=NAME_RE.pattern)
    kind: ParamKind
    description: str = ""


class Condition(BaseModel):
    """One executable check. ``color``/``x``/``y`` name *parameters* of the skill (they are
    resolved against the call's arguments), ``z`` and ``tol_m`` are literal metres.

    ``not_holding``/``holding``/``tcp_above`` only read the robot (checked on every execution);
    ``cube_near`` needs the position of a cube, which the robot cannot see from where a skill
    usually ends, so it is verified against the world at validation time and reported as
    *deferred* at execution time.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: ConditionKind
    color: str | None = None
    x: str | None = None
    y: str | None = None
    z: float | None = None
    tol_m: float = Field(gt=0, default=0.02)

    @property
    def needs_world(self) -> bool:
        return self.kind == "cube_near"

    def describe(self) -> str:
        if self.kind == "cube_near":
            return f"cube {self.color} within {self.tol_m * 100:.0f} cm of ({self.x}, {self.y})"
        if self.kind == "tcp_above":
            return f"TCP at least {self.z} m above the table"
        return self.kind.replace("_", " ")


class Origin(BaseModel):
    """Where the body came from: the agent episode(s) whose program was parametrised."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task: str
    run_id: str
    episode_ids: tuple[str, ...]
    seeds: tuple[int, ...]
    program_sha256: str
    """Hash of the *original* (unparametrised) program of the first episode."""
    agent: str = "B"


class Validation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    split: str
    seeds: tuple[int, ...]
    n_ok: int = Field(ge=0)
    failed: dict[int, str] = {}
    """seed -> first reason it failed (sandbox outcome, condition, verdict)."""
    min_pass_rate: float = Field(ge=0, le=1)
    accepted: bool
    backend: str
    armbench: str

    @property
    def pass_rate(self) -> float:
        return self.n_ok / len(self.seeds) if self.seeds else 0.0


class Skill(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(pattern=NAME_RE.pattern)
    version: int = Field(ge=1, default=1)
    description: str = Field(min_length=1)
    params: tuple[Param, ...]
    body: str = Field(min_length=1)
    preconditions: tuple[Condition, ...] = ()
    postconditions: tuple[Condition, ...] = ()
    origin: Origin
    validation: Validation | None = None

    @field_validator("params")
    @classmethod
    def _unique_params(cls, params: tuple[Param, ...]) -> tuple[Param, ...]:
        names = [p.name for p in params]
        if len(set(names)) != len(names):
            msg = f"duplicate parameter names: {names}"
            raise ValueError(msg)
        return params

    # -- identity --------------------------------------------------------------------------
    def signature(self) -> str:
        """``name(color: color, x: float, y: float)``: what deduplication compares."""
        inner = ", ".join(f"{p.name}: {p.kind}" for p in self.params)
        return f"{self.name}({inner})"

    def sha256(self) -> str:
        """Hash of everything that changes behaviour: name, version, signature, body and
        conditions (not the origin or the validation record)."""
        payload = {
            "name": self.name,
            "version": self.version,
            "params": [p.model_dump(mode="json") for p in self.params],
            "body": self.body,
            "pre": [c.model_dump(mode="json") for c in self.preconditions],
            "post": [c.model_dump(mode="json") for c in self.postconditions],
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    def call_syntax(self) -> str:
        args = "".join(f", {p.name}=<{p.kind}>" for p in self.params)
        return f'robot.execute_skill("{self.name}"{args})'

    # -- binding ---------------------------------------------------------------------------
    def check_args(self, kwargs: Mapping[str, object]) -> dict[str, str | float]:
        """Validate call arguments against the signature; ``TypeError`` on any mismatch."""
        expected = {p.name: p for p in self.params}
        missing = sorted(set(expected) - set(kwargs))
        extra = sorted(set(kwargs) - set(expected))
        if missing or extra:
            msg = f"{self.signature()}: missing {missing}, unexpected {extra}"
            raise TypeError(msg)
        bound: dict[str, str | float] = {}
        for name, param in expected.items():
            value = kwargs[name]
            if param.kind == "color":
                if not isinstance(value, str) or not COLOR_RE.match(value):
                    msg = f"{self.name}: {name} must be a colour name, got {value!r}"
                    raise TypeError(msg)
                bound[name] = value
            else:
                if isinstance(value, bool) or not isinstance(value, int | float):
                    msg = f"{self.name}: {name} must be a number, got {type(value).__name__}"
                    raise TypeError(msg)
                f = float(value)
                if not math.isfinite(f):
                    msg = f"{self.name}: {name} must be finite"
                    raise TypeError(msg)
                bound[name] = f
        return bound

    def bind(self, kwargs: Mapping[str, object]) -> str:
        """The program the sandbox runs: one assignment per parameter, then the body."""
        bound = self.check_args(kwargs)
        header = "".join(f"{k} = {v!r}\n" for k, v in bound.items())
        return f"{header}\n{self.body}"

    def placeholder_args(self) -> dict[str, str | float]:
        """Syntactically valid arguments (for static checks of the body)."""
        return {p.name: "red" if p.kind == "color" else 0.0 for p in self.params}


def resolve(cond: Condition, args: Mapping[str, str | float]) -> tuple[str | None, float, float]:
    """(colour, x, y) a condition refers to, looked up in the bound arguments."""
    color = str(args[cond.color]) if cond.color is not None else None
    x = float(args[cond.x]) if cond.x is not None else math.nan
    y = float(args[cond.y]) if cond.y is not None else math.nan
    return color, x, y
