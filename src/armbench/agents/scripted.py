"""Agent A: fixed code per task, written by hand on the primitives only (no LLM, no memory).

It is the floor every learned or generated policy is compared against, and the proof that the
tasks are solvable with the primitives. Everything here is deliberately simple: approach from
above, straight down, grasp, lift, carry at a fixed height, lower, release, retreat.
"""

from __future__ import annotations

import hashlib
import inspect
import math
import sys
from collections.abc import Iterable
from types import ModuleType
from typing import Final

from armbench.agents.base import AgentTrace
from armbench.perception import Detection
from armbench.primitives import Pose, Robot
from armbench.primitives.scripted import APPROACH_DZ_M, grasp_pose, pick_and_place
from armbench.tasks import PlaceGoal, SortGoal, StackGoal, TaskInstance

OBSERVE_POSES: Final = (
    Pose(x=-0.30, y=0.0, z=0.15),
    Pose(x=-0.30, y=0.25, z=0.15),
    Pose(x=-0.30, y=-0.25, z=0.15),
)
"""TCP poses at the near edge of the table that keep every arm link out of the camera cone
(``ready_pose`` shadows ~30 % of the table: ADR-005)."""
SAME_CUBE_M: Final = 0.03
WALL_CARRY_DZ_M: Final = 0.18
"""Carry height over a 10 cm wall: fingers and the held cube clear its top by > 5 cm."""
PICKS_PER_PRIMITIVES: Final = 8


def merge(found: dict[str, Detection], new: Iterable[Detection]) -> None:
    """Keep one detection per colour: the first complete one wins; a partial one (top face
    occluded by the arm or cut by the frame) is a placeholder until a later view completes it."""
    for d in new:
        have = found.get(d.color)
        if have is None or (d.complete and not have.complete):
            found[d.color] = d


def complete(found: dict[str, Detection]) -> set[str]:
    return {c for c, d in found.items() if d.complete}


def detect_all(robot: Robot, colors: Iterable[str]) -> tuple[dict[str, Detection], int]:
    """Detections for every colour in ``colors``, moving to observe poses only while the views so
    far miss some colour or see it only partially. Returns the detections and the moves it took;
    a colour that is partial from every pose keeps its best (last complete-or-first) detection."""
    wanted = set(colors)
    found: dict[str, Detection] = {}
    merge(found, robot.detect())
    moves = 0
    for pose in OBSERVE_POSES:
        if wanted <= complete(found):
            break
        robot.move_to(pose)
        moves += 1
        merge(found, robot.detect())
    missing = wanted - found.keys()
    if missing:
        msg = f"cubes not found: {sorted(missing)}"
        raise LookupError(msg)
    return found, moves


def _hash_source(module: ModuleType) -> str:
    return hashlib.sha256(inspect.getsource(module).encode("utf-8")).hexdigest()


class ScriptedAgent:
    id = "A"

    def solve(self, robot: Robot, instance: TaskInstance) -> AgentTrace:
        goal = instance.goal
        dets, observe_moves = detect_all(robot, instance.required_colors())
        n = 1 + observe_moves * 2  # detect + (move_to, detect) per observe pose
        if isinstance(goal, PlaceGoal):
            pick = grasp_pose(dets[goal.target_color])
            place = Pose(x=goal.at.x, y=goal.at.y, z=pick.z)
            carry = WALL_CARRY_DZ_M if instance.obstacles else APPROACH_DZ_M
            n += len(pick_and_place(robot, pick, place, carry_dz_m=carry))
        elif isinstance(goal, StackGoal):
            top, base = dets[goal.top_color], dets[goal.base_color]
            pick = grasp_pose(top)
            size = instance.cube(goal.top_color).size
            place = Pose(x=base.position[0], y=base.position[1], z=pick.z + size, yaw=base.yaw_rad)
            n += len(pick_and_place(robot, pick, place))
        elif isinstance(goal, SortGoal):
            n += self._sort(robot, goal, dets)
        return AgentTrace(
            n_primitives=n,
            n_observe_moves=observe_moves,
            program_sha256=_hash_source(sys.modules[__name__]),
        )

    @staticmethod
    def _sort(robot: Robot, goal: SortGoal, dets: dict[str, Detection]) -> int:
        """Move the cubes in order of distance to their bins, each to its bin directly; a bin
        that is still occupied by another cube is served after that cube has left."""
        pending = {c: dets[c] for c in goal.bins}
        n = 0
        while pending:
            ready = [
                c
                for c, d in pending.items()
                if not any(
                    goal.bins[c].dist(o.position[0], o.position[1]) < SAME_CUBE_M + 0.03
                    for oc, o in pending.items()
                    if oc != c
                )
            ]
            if not ready:  # mutual blocking cannot happen with 10 cm bin clearance
                ready = [next(iter(pending))]
            color = min(ready, key=lambda c: goal.bins[c].dist(*pending[c].xy))
            d = pending.pop(color)
            pick = grasp_pose(d)
            place = Pose(x=goal.bins[color].x, y=goal.bins[color].y, z=pick.z)
            n += len(pick_and_place(robot, pick, place))
        return n


def wall_clearance_m(tcp_z: float, cube_size: float, wall_h: float) -> float:
    """Gap between the bottom of a held cube and the top of a wall when the TCP is at ``tcp_z``."""
    return tcp_z - cube_size / 2.0 - wall_h


assert math.isclose(wall_clearance_m(0.0225 + WALL_CARRY_DZ_M, 0.045, 0.10), 0.08)  # noqa: S101
