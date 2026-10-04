"""Providers that need no network: a deterministic *template* model and a fixed-text stub.

``TemplateProvider`` reads the task sentence in the prompt and answers with a program written
over the sandbox API (observe poses, grasp-yaw choice and pick-and-place like baseline A); when
the prompt offers skills it calls one, and when it carries an ``# Energy`` section it writes the
energy-aware variant of the same program (lower lifts, full-speed transfers). It
exists so that the whole F6 pipeline — prompt, cache, replay, sandbox, cost ledger, episode
log — can be built, tested and run end to end before a real model is plugged in. It is not a
language model: its results say nothing about what an LLM would achieve.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from typing import Final

from armbench.llm.types import LLMRequest, LLMResponse, Usage

CHARS_PER_TOKEN: Final = 4.0
PLACE_RE: Final = re.compile(
    r"Move the (?P<color>[a-z]+) cube so that its centre is at "
    r"x=(?P<x>-?\d+(?:\.\d+)?), y=(?P<y>-?\d+(?:\.\d+)?)"
)
STACK_RE: Final = re.compile(r"Place the (?P<top>[a-z]+) cube on top of the (?P<base>[a-z]+) cube")
SORT_RE: Final = re.compile(
    r"(?P<color>[a-z]+) -> \(x=(?P<x>-?\d+(?:\.\d+)?), y=(?P<y>-?\d+(?:\.\d+)?)\)"
)

HELPERS: Final = """\
OBSERVE_POSES = [Pose(-0.30, 0.0, 0.15), Pose(-0.30, 0.25, 0.15), Pose(-0.30, -0.25, 0.15)]
CUBE = 0.045
APPROACH = 0.10
SLOW = 0.5


def merge(found, dets):
    for d in dets:
        have = found.get(d.color)
        if have is None or (d.complete and not have.complete):
            found[d.color] = d


def see(colors):
    found = {}
    merge(found, robot.detect())
    for pose in OBSERVE_POSES:
        if all(c in found and found[c].complete for c in colors):
            break
        robot.move_to(pose)
        merge(found, robot.detect())
    for c in colors:
        if c not in found:
            raise RuntimeError("cube not found: " + c)
    return found


def dist_xy(det, xy):
    return math.hypot(det.position[0] - xy[0], det.position[1] - xy[1])


def clearance(xy, yaw, others):
    ux, uy = -math.sin(yaw), math.cos(yaw)
    best = 0.05
    for o in others:
        rx, ry = o.position[0] - xy[0], o.position[1] - xy[1]
        along = abs(rx * ux + ry * uy)
        perp = abs(rx * uy - ry * ux)
        best = min(best, math.hypot(max(along - 0.06, 0.0), perp))
    return best


def grasp_pose(det, others):
    xy = (det.position[0], det.position[1])
    yaw = det.yaw_rad
    alt = yaw - math.pi / 2 if yaw >= 0 else yaw + math.pi / 2
    rest = [o for o in others if o.color != det.color]
    if clearance(xy, alt, rest) > clearance(xy, yaw, rest) + 0.01:
        yaw = alt
    return Pose(det.position[0], det.position[1], det.position[2], yaw)


def pick_place(pick, place, carry):
    robot.move_to(pick.above(APPROACH))
    robot.move_to(pick, SLOW)
    robot.grasp()
    robot.move_to(pick.above(carry), SLOW)
    robot.move_to(place.above(carry))
    robot.move_to(place.above(0.005), SLOW)
    robot.release()
    robot.move_to(place.above(APPROACH))
"""

PLACE_TAIL: Final = """\

dets = see(["{color}"])
pick = grasp_pose(dets["{color}"], dets.values())
place = Pose({x}, {y}, pick.z)
pick_place(pick, place, {carry})
"""

STACK_TAIL: Final = """\

dets = see(["{top}", "{base}"])
top, base = dets["{top}"], dets["{base}"]
pick = grasp_pose(top, dets.values())
place = Pose(base.position[0], base.position[1], pick.z + CUBE, base.yaw_rad)
pick_place(pick, place, APPROACH)
"""

SORT_TAIL: Final = """\

bins = {bins}
dets = see(list(bins))
pending = dict(dets)
while pending:
    ready = []
    for c in pending:
        bx, by = bins[c]
        blocked = False
        for oc in pending:
            o = pending[oc]
            if oc != c and math.hypot(o.position[0] - bx, o.position[1] - by) < 0.06:
                blocked = True
        if not blocked:
            ready.append(c)
    if not ready:
        ready = [list(pending)[0]]
    color = min(ready, key=lambda c: dist_xy(pending[c], bins[c]))
    d = pending.pop(color)
    pick = grasp_pose(d, dets.values())
    place = Pose(bins[color][0], bins[color][1], pick.z)
    pick_place(pick, place, APPROACH)
"""


ENERGY_EDITS: Final[tuple[tuple[str, str], ...]] = (
    ("APPROACH = 0.10", "APPROACH = 0.06"),
    ("SLOW = 0.5", "SLOW = 0.7"),
    ("robot.move_to(pick.above(carry), SLOW)", "robot.move_to(pick.above(carry))"),
)
"""What the template changes when the prompt has an ``# Energy`` section: the block says P0
dominates, so it shortens the vertical legs (6 cm approach/carry instead of 10, still above a
4.5 cm neighbour), lifts at full speed and keeps a reduced — but faster — speed_scale only for the
final descents. A fixed recipe, not a model's reasoning."""


def energy_program(program: str) -> str:
    """The energy-aware variant of a template program (see ``ENERGY_EDITS``)."""
    for old, new in ENERGY_EDITS:
        program = program.replace(old, new)
    return program


def template_program(task_text: str) -> str | None:
    """The program for a task sentence, or ``None`` when the sentence is not understood."""
    m = STACK_RE.search(task_text)
    if m:
        return HELPERS + STACK_TAIL.format(top=m.group("top"), base=m.group("base"))
    if "Sort the cubes" in task_text:
        bins = {
            b.group("color"): (float(b.group("x")), float(b.group("y")))
            for b in SORT_RE.finditer(task_text)
        }
        if not bins:
            return None
        body = "{" + ", ".join(f'"{c}": ({x}, {y})' for c, (x, y) in bins.items()) + "}"
        return HELPERS + SORT_TAIL.format(bins=body)
    m = PLACE_RE.search(task_text)
    if m:
        carry = "0.18" if "wall" in task_text else "APPROACH"
        return HELPERS + PLACE_TAIL.format(
            color=m.group("color"), x=m.group("x"), y=m.group("y"), carry=carry
        )
    return None


SKILL_CALL_RE: Final = re.compile(
    r'^- robot\.execute_skill\("(?P<name>\w+)"(?P<params>(?:, \w+=<\w+>)*)\)$'
)
"""A skill line of the ``# Skills`` section agent B+S receives (see ``agents.prompt``)."""


def offered_skills(task_text: str) -> list[tuple[str, tuple[str, ...], str]]:
    """``(name, parameter names, description line)`` for every skill the prompt offers."""
    lines = task_text.splitlines()
    out: list[tuple[str, tuple[str, ...], str]] = []
    for i, line in enumerate(lines):
        m = SKILL_CALL_RE.match(line)
        if m is None:
            continue
        names = tuple(p.split("=")[0] for p in m.group("params").split(", ") if p)
        desc = lines[i + 1].strip() if i + 1 < len(lines) else ""
        out.append((m.group("name"), names, desc))
    return out


def _pick_skill(
    offered: list[tuple[str, tuple[str, ...], str]], wanted: set[str], wall: bool
) -> str | None:
    fits = [(n, d) for n, params, d in offered if set(params) == wanted]
    if not fits:
        return None
    walls = [n for n, d in fits if "wall" in d.lower()]
    plain = [n for n, d in fits if "wall" not in d.lower()]
    pref = (walls or plain) if wall else (plain or walls)
    return pref[0]


def template_skill_program(task_text: str) -> str | None:
    """Agent B+S's deterministic answer: one ``execute_skill`` call when an offered skill's
    parameters match the task kind, else ``None`` (the caller falls back to the plain program).
    The task constants are passed through verbatim."""
    offered = offered_skills(task_text)
    if not offered:
        return None
    task_text = task_text.rsplit("# Task", 1)[-1]
    wall = "wall" in task_text
    m = STACK_RE.search(task_text)
    if m:
        name = _pick_skill(offered, {"top", "base"}, wall)
        return (
            None
            if name is None
            else (
                f'robot.execute_skill("{name}", top="{m.group("top")}", base="{m.group("base")}")\n'
            )
        )
    if "Sort the cubes" in task_text:
        bins = list(SORT_RE.finditer(task_text))
        wanted = {f"{k}_{i}" for i in range(1, len(bins) + 1) for k in ("color", "x", "y")}
        name = _pick_skill(offered, wanted, wall)
        if name is None or not bins:
            return None
        args = ", ".join(
            f'color_{i}="{b.group("color")}", x_{i}={b.group("x")}, y_{i}={b.group("y")}'
            for i, b in enumerate(bins, start=1)
        )
        return f'robot.execute_skill("{name}", {args})\n'
    m = PLACE_RE.search(task_text)
    if m:
        name = _pick_skill(offered, {"color", "x", "y"}, wall)
        return (
            None
            if name is None
            else (
                f'robot.execute_skill("{name}", color="{m.group("color")}", '
                f"x={m.group('x')}, y={m.group('y')})\n"
            )
        )
    return None


def _tokens(text: str) -> int:
    return max(1, round(len(text) / CHARS_PER_TOKEN))


class TemplateProvider:
    id = "template"
    MODEL = "template-v1"

    def complete(self, request: LLMRequest) -> LLMResponse:
        t0 = time.monotonic()
        task_text = request.last_user()
        program = template_skill_program(task_text)
        if program is None:
            program = template_program(task_text.rsplit("# Task", 1)[-1])
            if program is not None and "# Energy" in task_text:
                program = energy_program(program)
        text = "I do not understand this task." if program is None else f"```python\n{program}```\n"
        return LLMResponse(
            text=text,
            model=self.MODEL,
            provider=self.id,
            usage=Usage(
                prompt_tokens=_tokens(request.canonical()), completion_tokens=_tokens(text)
            ),
            latency_s=time.monotonic() - t0,
            finish_reason="stop",
            request_key=request.key(),
        )


class StaticProvider:
    """Answers every request with a fixed text (or a function of the request); for tests."""

    id = "static"

    def __init__(self, text: str | Callable[[LLMRequest], str], model: str = "static") -> None:
        self.text = text
        self.model = model
        self.calls = 0

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        text = self.text(request) if callable(self.text) else self.text
        return LLMResponse(
            text=text,
            model=self.model,
            provider=self.id,
            usage=Usage(
                prompt_tokens=_tokens(request.canonical()), completion_tokens=_tokens(text)
            ),
            request_key=request.key(),
        )
