"""Versioned manipulation tasks with seeded instances and ground-truth success checkers."""

from armbench.tasks.registry import TASK_IDS, TASKS, get_task
from armbench.tasks.spec import (
    DISTURB_TOL_M,
    OBSTACLE_TOL_M,
    POS_TOL_M,
    SIM_LIMIT_S,
    STACK_Z_TOL_M,
    XY,
    FinalState,
    Goal,
    ModelPose,
    PlaceGoal,
    SortGoal,
    StackGoal,
    Task,
    TaskId,
    TaskInstance,
    Verdict,
)

__all__ = [
    "DISTURB_TOL_M",
    "OBSTACLE_TOL_M",
    "POS_TOL_M",
    "SIM_LIMIT_S",
    "STACK_Z_TOL_M",
    "TASKS",
    "TASK_IDS",
    "XY",
    "FinalState",
    "Goal",
    "ModelPose",
    "PlaceGoal",
    "SortGoal",
    "StackGoal",
    "Task",
    "TaskId",
    "TaskInstance",
    "Verdict",
    "get_task",
]
