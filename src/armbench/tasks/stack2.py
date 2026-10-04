"""``stack2@1``: put the top cube on the base cube (both identified by colour)."""

from __future__ import annotations

from armbench.scene import SceneConfig, generate_scene
from armbench.tasks.spec import (
    POS_TOL_M,
    STACK_Z_TOL_M,
    FinalState,
    StackGoal,
    TaskId,
    TaskInstance,
    Verdict,
    common_failure,
    recolour,
    task_rng,
)


class Stack2:
    id = TaskId(name="stack2", version=1)
    description = "Stack one cube on top of another; both are identified by colour."

    def instance(self, seed: int, config: SceneConfig) -> TaskInstance:
        rng = task_rng(self.id, seed)
        scene = recolour(generate_scene(seed, config), rng, unique=2)
        top, base = scene.cubes[0], scene.cubes[1]
        goal = StackGoal(top_color=top.color, base_color=base.color)
        prompt = (
            f"Place the {top.color} cube on top of the {base.color} cube, centred and resting "
            f"on it. Leave the {base.color} cube where it is and do not touch the other cubes."
        )
        return TaskInstance(task=self.id, seed=seed, scene=scene, goal=goal, prompt=prompt)

    def check(self, instance: TaskInstance, final: FinalState) -> Verdict:
        goal = instance.goal
        assert isinstance(goal, StackGoal)  # noqa: S101 - discriminated by construction
        top, base = instance.cube(goal.top_color), instance.cube(goal.base_color)
        fail, metrics = common_failure(instance, final, allowed={top.name})
        pt, pb = final.poses.get(top.name), final.poses.get(base.name)
        if pt is None or pb is None:
            return Verdict(ok=False, reason="pose unknown", metrics=metrics)
        xy_err = ((pt.x - pb.x) ** 2 + (pt.y - pb.y) ** 2) ** 0.5
        z_err = pt.z - (pb.z + base.size / 2.0 + top.size / 2.0)
        metrics["stack_xy_err_m"] = xy_err
        metrics["stack_z_err_m"] = z_err
        if fail is not None:
            return Verdict(ok=False, reason=fail, metrics=metrics)
        if xy_err > POS_TOL_M:
            return Verdict(ok=False, reason=f"top {xy_err * 1e3:.0f} mm off base", metrics=metrics)
        if abs(z_err) > STACK_Z_TOL_M:
            return Verdict(ok=False, reason=f"top z off by {z_err * 1e3:.0f} mm", metrics=metrics)
        return Verdict(ok=True, reason="stacked", metrics=metrics)
