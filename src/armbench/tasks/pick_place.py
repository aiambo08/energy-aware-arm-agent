"""``pick_place@1``: move the (unique-colour) target cube to a seeded free spot."""

from __future__ import annotations

from typing import Final

from armbench.scene import SceneConfig, generate_scene
from armbench.tasks.spec import (
    POS_TOL_M,
    FinalState,
    PlaceGoal,
    TaskId,
    TaskInstance,
    Verdict,
    common_failure,
    free_xy,
    moved_m,
    on_table,
    recolour,
    task_rng,
)

GOAL_CLEARANCE_M: Final = 0.09
GOAL_MARGIN_M: Final = 0.05


class PickPlace:
    id = TaskId(name="pick_place", version=1)
    description = "Move one cube, identified by colour, to a given point on the table."

    def instance(self, seed: int, config: SceneConfig) -> TaskInstance:
        rng = task_rng(self.id, seed)
        scene = recolour(generate_scene(seed, config), rng, unique=1)
        target = scene.cubes[0]
        at = free_xy(
            rng,
            config,
            [(c.x, c.y) for c in scene.cubes],
            clearance_m=GOAL_CLEARANCE_M,
            margin_m=GOAL_MARGIN_M,
        )
        if at is None:
            msg = f"{self.id} seed {seed}: no free goal spot"
            raise RuntimeError(msg)
        goal = PlaceGoal(target_color=target.color, at=at)
        prompt = (
            f"Move the {target.color} cube so that its centre is at x={at.x:.3f}, y={at.y:.3f} "
            f"(base_link, metres) resting on the table. Do not touch the other cubes."
        )
        return TaskInstance(task=self.id, seed=seed, scene=scene, goal=goal, prompt=prompt)

    def check(self, instance: TaskInstance, final: FinalState) -> Verdict:
        goal = instance.goal
        assert isinstance(goal, PlaceGoal)  # noqa: S101 - discriminated by construction
        target = instance.cube(goal.target_color)
        fail, metrics = common_failure(instance, final, allowed={target.name})
        pose = final.poses.get(target.name)
        err = goal.at.dist(pose.x, pose.y) if pose is not None else float("inf")
        metrics["place_err_m"] = err
        if fail is not None:
            return Verdict(ok=False, reason=fail, metrics=metrics)
        if pose is None:
            return Verdict(ok=False, reason="target pose unknown", metrics=metrics)
        if err > POS_TOL_M:
            return Verdict(ok=False, reason=f"target {err * 1e3:.0f} mm from goal", metrics=metrics)
        if not on_table(target, pose):
            return Verdict(ok=False, reason="target not resting on the table", metrics=metrics)
        metrics["target_moved_m"] = moved_m(target, pose)
        return Verdict(ok=True, reason="target at goal", metrics=metrics)
