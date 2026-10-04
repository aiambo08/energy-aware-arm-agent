"""All tasks the benchmark knows, by ``name@version``."""

from __future__ import annotations

from typing import Final

from armbench.tasks.pick_place import PickPlace
from armbench.tasks.place_obstacle import PlaceObstacle
from armbench.tasks.sort3 import Sort3
from armbench.tasks.spec import Task, TaskId
from armbench.tasks.stack2 import Stack2

TASKS: Final[dict[str, Task]] = {
    str(t.id): t for t in (PickPlace(), Stack2(), Sort3(), PlaceObstacle())
}
TASK_IDS: Final[tuple[str, ...]] = tuple(TASKS)


def get_task(task_id: str | TaskId) -> Task:
    key = str(TaskId.parse(task_id) if isinstance(task_id, str) else task_id)
    try:
        return TASKS[key]
    except KeyError:
        msg = f"unknown task {key!r}; known: {', '.join(TASK_IDS)}"
        raise KeyError(msg) from None
