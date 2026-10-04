"""Versioned manipulation tasks: seeded instances, natural-language goals and success checkers.

A task is identified by ``name@version``. Changing anything that affects what counts as
success (goal sampling, tolerances, scene recolouring) bumps the version, so every episode
log says exactly which contract it was judged against.
"""

from __future__ import annotations

import math
import re
import zlib
from collections.abc import Sequence
from typing import Final, Literal, Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from armbench.scene import Box, Cube, Scene, SceneConfig

TASK_ID_RE: Final = re.compile(r"^(?P<name>[a-z][a-z0-9_]*)@(?P<version>\d+)$")

# Version-1 contract shared by every task: a cube is "at" a point when its centre is within
# POS_TOL_M in the table plane; touching anything the task did not ask to move is a collision.
POS_TOL_M: Final = 0.02
STACK_Z_TOL_M: Final = 0.01
DISTURB_TOL_M: Final = 0.02
OBSTACLE_TOL_M: Final = 0.005
TABLE_PENETRATION_M: Final = 0.002
SIM_LIMIT_S: Final = 60.0
OBSTACLE_RGBA: Final = (0.35, 0.35, 0.35, 1.0)
"""Grey: saturation ~0, so ``detect()`` never reports an obstacle (the task text does)."""


class TaskId(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    version: int = Field(ge=1)

    def __str__(self) -> str:
        return f"{self.name}@{self.version}"

    @classmethod
    def parse(cls, text: str) -> TaskId:
        m = TASK_ID_RE.match(text)
        if m is None:
            msg = f"task id must look like name@version, got {text!r}"
            raise ValueError(msg)
        return cls(name=m.group("name"), version=int(m.group("version")))

    def salt(self) -> int:
        """Stable integer mixed into the seed so each task draws its own goals."""
        return zlib.crc32(str(self).encode("utf-8"))


class XY(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    x: float
    y: float

    def dist(self, x: float, y: float) -> float:
        return math.hypot(self.x - x, self.y - y)


class PlaceGoal(BaseModel):
    """Bring the cube of ``target_color`` to ``at`` (table plane)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["place"] = "place"
    target_color: str
    at: XY


class StackGoal(BaseModel):
    """Put the ``top_color`` cube on top of the ``base_color`` cube (wherever the base is)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["stack"] = "stack"
    top_color: str
    base_color: str


class SortGoal(BaseModel):
    """Bring every cube to the bin of its colour."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["sort"] = "sort"
    bins: dict[str, XY]


Goal = PlaceGoal | StackGoal | SortGoal


class TaskInstance(BaseModel):
    """Everything an episode needs: the scene to spawn, the goal and its wording."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task: TaskId
    seed: int = Field(ge=0)
    scene: Scene
    obstacles: tuple[Box, ...] = ()
    goal: Goal = Field(discriminator="kind")
    prompt: str

    def cube(self, color: str) -> Cube:
        hits = [c for c in self.scene.cubes if c.color == color]
        if len(hits) != 1:
            msg = f"expected exactly one {color} cube, found {len(hits)}"
            raise LookupError(msg)
        return hits[0]

    def required_colors(self) -> tuple[str, ...]:
        """Colours the solver must see to act (unique by construction)."""
        g = self.goal
        if isinstance(g, PlaceGoal):
            return (g.target_color,)
        if isinstance(g, StackGoal):
            return (g.top_color, g.base_color)
        return tuple(g.bins)


class ModelPose(BaseModel):
    """Ground-truth pose of a spawned model at the end of an episode."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    x: float
    y: float
    z: float
    yaw: float = 0.0


class FinalState(BaseModel):
    """What the checker sees: simulator ground truth, never the robot's own perception."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    poses: dict[str, ModelPose]
    min_tip_z_m: float
    sim_s: float = Field(ge=0)


class Verdict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    ok: bool
    reason: str
    metrics: dict[str, float] = Field(default_factory=dict)


class Task(Protocol):
    id: TaskId
    description: str

    def instance(self, seed: int, config: SceneConfig) -> TaskInstance: ...

    def check(self, instance: TaskInstance, final: FinalState) -> Verdict: ...


# -- helpers shared by the task modules ----------------------------------------------------


def task_rng(task: TaskId, seed: int) -> np.random.Generator:
    return np.random.default_rng([seed, task.salt()])


def recolour(scene: Scene, rng: np.random.Generator, unique: int | Sequence[int]) -> Scene:
    """Give the ``unique`` cubes (a count = the first ones, or explicit indices) distinct
    colours and the rest colours drawn (with replacement) from the remaining palette, so every
    colour the goal names is unambiguous."""
    idx = list(range(unique)) if isinstance(unique, int) else list(unique)
    palette = list(_palette_of(scene))
    if len(idx) > len(palette):
        msg = f"need {len(idx)} distinct colours, palette has {len(palette)}"
        raise ValueError(msg)
    order = [palette[int(i)] for i in rng.permutation(len(palette))]
    rest = order[len(idx) :] or order
    cubes: list[Cube] = []
    for i, c in enumerate(scene.cubes):
        colour = order[idx.index(i)] if i in idx else rest[int(rng.integers(len(rest)))]
        cubes.append(c.model_copy(update={"color": colour}))
    return scene.model_copy(update={"cubes": cubes})


_PALETTE: tuple[str, ...] = ("red", "green", "blue", "yellow")


def _palette_of(_scene: Scene) -> tuple[str, ...]:
    return _PALETTE


def free_xy(  # noqa: PLR0913
    rng: np.random.Generator,
    config: SceneConfig,
    taken: list[tuple[float, float]],
    *,
    clearance_m: float,
    margin_m: float,
    min_dist_from: tuple[float, float] | None = None,
    min_dist_m: float = 0.0,
    tries: int = 2000,
) -> XY | None:
    """Rejection-sample a table point at least ``clearance_m`` from every taken point."""
    ws = config.workspace
    for _ in range(tries):
        x = float(rng.uniform(ws.x_min + margin_m, ws.x_max - margin_m))
        y = float(rng.uniform(ws.y_min + margin_m, ws.y_max - margin_m))
        if any(math.hypot(x - px, y - py) < clearance_m for px, py in taken):
            continue
        if min_dist_from is not None and math.hypot(x - min_dist_from[0], y - min_dist_from[1]) < (
            min_dist_m
        ):
            continue
        return XY(x=x, y=y)
    return None


def moved_m(initial: Cube | Box, final: ModelPose | None) -> float:
    if final is None:
        return math.inf
    return math.hypot(final.x - initial.x, final.y - initial.y)


def disturbance(
    instance: TaskInstance, final: FinalState, *, allowed: set[str]
) -> tuple[float, float]:
    """Largest displacement of cubes the task did not ask to move, and of obstacles."""
    cubes = max(
        (
            moved_m(c, final.poses.get(c.name))
            for c in instance.scene.cubes
            if c.name not in allowed
        ),
        default=0.0,
    )
    obstacles = max(
        (moved_m(o, final.poses.get(o.name)) for o in instance.obstacles),
        default=0.0,
    )
    return cubes, obstacles


def common_failure(
    instance: TaskInstance, final: FinalState, *, allowed: set[str]
) -> tuple[str | None, dict[str, float]]:
    """Checks every task shares: sim-time budget, table contact, collateral motion."""
    cubes_moved, obstacles_moved = disturbance(instance, final, allowed=allowed)
    metrics = {
        "others_moved_m": cubes_moved,
        "obstacles_moved_m": obstacles_moved,
        "min_tip_z_m": final.min_tip_z_m,
        "sim_s": final.sim_s,
    }
    if final.sim_s > SIM_LIMIT_S:
        return f"sim time {final.sim_s:.1f} s exceeds {SIM_LIMIT_S:.0f} s", metrics
    if final.min_tip_z_m < -TABLE_PENETRATION_M:
        return f"finger tips went {-final.min_tip_z_m * 1e3:.1f} mm into the table", metrics
    if cubes_moved > DISTURB_TOL_M:
        return f"an untouched cube moved {cubes_moved * 1e3:.0f} mm", metrics
    if obstacles_moved > OBSTACLE_TOL_M:
        return f"obstacle moved {obstacles_moved * 1e3:.1f} mm", metrics
    return None, metrics


def on_table(cube: Cube, pose: ModelPose) -> bool:
    return abs(pose.z - cube.size / 2.0) < STACK_Z_TOL_M
