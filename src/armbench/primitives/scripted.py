"""Fixed-code (no LLM) manipulation routines built only from the primitives.

They are the F4 acceptance task and the F5 baseline A: whatever an agent program can do, this
module does without reasoning, so any gap between the two is attributable to the agent.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np

from armbench.perception import Detection
from armbench.primitives.params import WorkspaceBox
from armbench.primitives.robot import Robot
from armbench.primitives.types import MoveResult, Pose, Result

APPROACH_DZ_M = 0.10
"""Height above the grasp/place pose the TCP visits before descending and after lifting."""
PLACE_DROP_M = 0.005
"""Pads release this high above the pick height so the cube settles onto the table."""
DESCENT_SPEED = 0.5
"""``speed_scale`` for the vertical approach and retreat segments."""


def grasp_pose(detection: Detection) -> Pose:
    """TCP pose whose pads are centred on the detected cube."""
    x, y, z = detection.position
    return Pose(x=x, y=y, z=z, yaw=detection.yaw_rad)


def free_spot(
    occupied: Iterable[tuple[float, float]],
    workspace: WorkspaceBox,
    *,
    clearance_m: float,
    step_m: float = 0.05,
    margin_m: float = 0.05,
) -> tuple[float, float] | None:
    """First grid point of the workspace at least ``clearance_m`` (Chebyshev) from every
    occupied point, scanning x then y deterministically; ``None`` when the table is full."""
    pts = list(occupied)
    xs = np.arange(workspace.x_min + margin_m, workspace.x_max - margin_m + 1e-9, step_m)
    ys = np.arange(workspace.y_min + margin_m, workspace.y_max - margin_m + 1e-9, step_m)
    for x in xs:
        for y in ys:
            if all(max(abs(px - x), abs(py - y)) > clearance_m for px, py in pts):
                return float(x), float(y)
    return None


def pick_and_place(robot: Robot, pick: Pose, place: Pose) -> list[Result]:
    """Approach from above, descend, grasp, lift, carry, lower, release, retreat.

    Raises the primitive error of the step that failed; the arm is then wherever that step
    left it (callers recover with ``robot.reset()``).
    """
    log: list[Result] = []
    log.append(robot.move_to(pick.above(APPROACH_DZ_M)))
    log.append(robot.move_to(pick, DESCENT_SPEED))
    log.append(robot.grasp())
    log.append(robot.move_to(pick.above(APPROACH_DZ_M), DESCENT_SPEED))
    log.append(robot.move_to(place.above(APPROACH_DZ_M)))
    log.append(robot.move_to(place.above(PLACE_DROP_M), DESCENT_SPEED))
    log.append(robot.release())
    log.append(robot.move_to(place.above(APPROACH_DZ_M)))
    return log


def moves(log: Sequence[Result]) -> list[MoveResult]:
    return [r for r in log if isinstance(r, MoveResult)]


def total_sim_s(log: Sequence[Result]) -> float:
    return math.fsum(r.sim_s for r in log)
