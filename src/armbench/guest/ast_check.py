"""Static gate for agent programs (sandbox layer 1): a whitelist over the Python AST.

A program may define functions, assign, loop, branch, do arithmetic and comparisons, build
literals and comprehensions, catch exceptions, format strings and call ``robot.<primitive>``,
``Pose``, ``math.*`` and a short list of builtins. Everything else is rejected before the code
is compiled: ``import``, classes, ``with``, generators and coroutines, decorators, any identifier
or attribute that starts with an underscore (so no dunder introspection), the attributes that
reach frames or format-string lookups, and any free name that is not in the allowed set
(``open``, ``eval``, ``exec``, ``getattr``, ``type``, ``globals`` ... are never bound).
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

ALLOWED_BUILTINS: Final[tuple[str, ...]] = (
    "abs",
    "all",
    "any",
    "bool",
    "dict",
    "divmod",
    "enumerate",
    "filter",
    "float",
    "int",
    "isinstance",
    "len",
    "list",
    "map",
    "max",
    "min",
    "next",
    "pow",
    "print",
    "range",
    "reversed",
    "round",
    "set",
    "sorted",
    "str",
    "sum",
    "tuple",
    "zip",
)
ALLOWED_EXCEPTIONS: Final[tuple[str, ...]] = (
    "Exception",
    "ValueError",
    "TypeError",
    "KeyError",
    "IndexError",
    "AttributeError",
    "ZeroDivisionError",
    "RuntimeError",
    "StopIteration",
    "PrimitiveError",
    "OutOfReach",
    "Singularity",
    "Collision",
    "Timeout",
    "NoObjectGrasped",
    "CameraTimeout",
    "SkillNotAvailable",
    "SkillPreconditionFailed",
    "SkillPostconditionFailed",
    "SkillFailed",
)
ALLOWED_GLOBALS: Final[tuple[str, ...]] = ("robot", "Pose", "math")
ALLOWED_NAMES: Final[frozenset[str]] = frozenset(
    ALLOWED_BUILTINS + ALLOWED_EXCEPTIONS + ALLOWED_GLOBALS
)

FORBIDDEN_ATTRS: Final[frozenset[str]] = frozenset(
    {
        "format",
        "format_map",
        "mro",
        "gi_frame",
        "gi_code",
        "gi_yieldfrom",
        "ag_frame",
        "ag_code",
        "cr_frame",
        "cr_code",
        "f_locals",
        "f_globals",
        "f_builtins",
        "f_code",
        "f_back",
        "tb_frame",
        "tb_next",
        "func_globals",
        "func_code",
        "with_traceback",
        "add_note",
    }
)
"""Non-dunder attributes that reach frames, code objects or attribute lookups through strings."""

ALLOWED_NODES: Final[frozenset[type[ast.AST]]] = frozenset(
    {
        ast.Module,
        ast.FunctionDef,
        ast.arguments,
        ast.arg,
        ast.Return,
        ast.Assign,
        ast.AugAssign,
        ast.AnnAssign,
        ast.For,
        ast.While,
        ast.If,
        ast.IfExp,
        ast.Break,
        ast.Continue,
        ast.Pass,
        ast.Expr,
        ast.Try,
        ast.ExceptHandler,
        ast.Raise,
        ast.Assert,
        ast.Delete,
        ast.BoolOp,
        ast.And,
        ast.Or,
        ast.BinOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.UnaryOp,
        ast.UAdd,
        ast.USub,
        ast.Not,
        ast.Compare,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.Is,
        ast.IsNot,
        ast.In,
        ast.NotIn,
        ast.Call,
        ast.keyword,
        ast.Starred,
        ast.Name,
        ast.Load,
        ast.Store,
        ast.Del,
        ast.Constant,
        ast.Attribute,
        ast.Subscript,
        ast.Slice,
        ast.Tuple,
        ast.List,
        ast.Dict,
        ast.Set,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp,
        ast.comprehension,
        ast.JoinedStr,
        ast.FormattedValue,
        ast.Lambda,
    }
)
"""No Import*, ClassDef, With, Global/Nonlocal, Yield*, Await, Async*, bit ops, matrix multiply,
walrus, match or type statements. Generator expressions are allowed; their ``gi_*`` attributes
(the frame and code objects) are not."""


class Violation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    rule: str
    detail: str
    line: int = Field(ge=0)
    col: int = Field(ge=0)

    def __str__(self) -> str:
        return f"line {self.line}: {self.detail} [{self.rule}]"


class ProgramCheck(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    ok: bool
    violations: tuple[Violation, ...] = ()
    n_nodes: int = Field(ge=0, default=0)
    n_lines: int = Field(ge=0, default=0)

    def summary(self, limit: int = 5) -> str:
        shown = [str(v) for v in self.violations[:limit]]
        more = len(self.violations) - len(shown)
        return "; ".join(shown) + (f"; +{more} more" if more > 0 else "")


class ProgramLimits(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    max_source_chars: int = Field(gt=0, default=20_000)
    max_nodes: int = Field(gt=0, default=5_000)


def _bound_names(tree: ast.AST) -> set[str]:
    """Every identifier the program itself binds (targets, functions, parameters, handlers)."""
    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store | ast.Del):
            bound.add(node.id)
        elif isinstance(node, ast.FunctionDef):
            bound.add(node.name)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
    return bound


def _identifiers(node: ast.AST) -> Iterable[tuple[str, str]]:
    """(kind, identifier) pairs a node introduces or dereferences."""
    if isinstance(node, ast.Name):
        yield "name", node.id
    elif isinstance(node, ast.Attribute):
        yield "attribute", node.attr
    elif isinstance(node, ast.FunctionDef):
        yield "function", node.name
    elif isinstance(node, ast.arg):
        yield "parameter", node.arg
    elif isinstance(node, ast.keyword) and node.arg is not None:
        yield "keyword", node.arg
    elif isinstance(node, ast.ExceptHandler) and node.name:
        yield "handler", node.name


def check_program(source: str, limits: ProgramLimits | None = None) -> ProgramCheck:
    """Reject anything outside the whitelist; never raises."""
    lim = limits or ProgramLimits()
    n_lines = source.count("\n") + 1
    if len(source) > lim.max_source_chars:
        v = Violation(rule="size", detail=f"program longer than {lim.max_source_chars} chars",
                      line=0, col=0)  # fmt: skip
        return ProgramCheck(ok=False, violations=(v,), n_lines=n_lines)
    if "\x00" in source:
        v = Violation(rule="syntax", detail="NUL byte in source", line=0, col=0)
        return ProgramCheck(ok=False, violations=(v,), n_lines=n_lines)
    try:
        tree = ast.parse(source, filename="<program>", mode="exec")
    except (SyntaxError, ValueError, RecursionError, MemoryError) as exc:
        line = getattr(exc, "lineno", None) if isinstance(exc, SyntaxError) else None
        v = Violation(rule="syntax", detail=f"{type(exc).__name__}: {exc}",
                      line=line or 0, col=0)  # fmt: skip
        return ProgramCheck(ok=False, violations=(v,), n_lines=n_lines)
    nodes = list(ast.walk(tree))
    if len(nodes) > lim.max_nodes:
        v = Violation(rule="size", detail=f"more than {lim.max_nodes} AST nodes", line=0, col=0)
        return ProgramCheck(ok=False, violations=(v,), n_nodes=len(nodes), n_lines=n_lines)
    bound = _bound_names(tree)
    out: list[Violation] = []

    def add(node: ast.AST, rule: str, detail: str) -> None:
        line = getattr(node, "lineno", 0)
        col = getattr(node, "col_offset", 0)
        out.append(Violation(rule=rule, detail=detail, line=int(line), col=int(col)))

    for node in nodes:
        if type(node) not in ALLOWED_NODES:
            add(node, "node", f"{type(node).__name__} is not allowed")
            continue
        if isinstance(node, ast.FunctionDef) and node.decorator_list:
            add(node, "decorator", f"decorators are not allowed (def {node.name})")
        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRS:
            add(node, "attribute", f"attribute .{node.attr} is not allowed")
        for kind, ident in _identifiers(node):
            if ident.startswith("_"):
                add(node, "underscore", f"{kind} {ident!r} starts with an underscore")
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id not in bound
            and node.id not in ALLOWED_NAMES
        ):
            add(node, "name", f"name {node.id!r} is not available in the sandbox")
    out.sort(key=lambda v: (v.line, v.col, v.rule))
    return ProgramCheck(ok=not out, violations=tuple(out), n_nodes=len(nodes), n_lines=n_lines)
