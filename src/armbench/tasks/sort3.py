"""``sort3@1``: three cubes of distinct colours, one seeded bin per colour."""

from __future__ import annotations

from typing import Final

from armbench.scene import SceneConfig, generate_scene
from armbench.tasks.spec import (
    POS_TOL_M,
    XY,
    FinalState,
    SortGoal,
    TaskId,
    TaskInstance,
    Verdict,
    common_failure,
    free_xy,
    on_table,
    recolour,
    task_rng,
)

N_CUBES: Final = 3
BIN_CLEARANCE_M: Final = 0.10
BIN_MARGIN_M: Final = 0.05


class Sort3:
    id = TaskId(name="sort3", version=1)
    description = "Bring each of three differently coloured cubes to the bin of its colour."

    def instance(self, seed: int, config: SceneConfig) -> TaskInstance:
        rng = task_rng(self.id, seed)
        scene = recolour(generate_scene(seed, config, n_cubes=N_CUBES), rng, unique=N_CUBES)
        taken = [(c.x, c.y) for c in scene.cubes]
        bins: dict[str, XY] = {}
        for c in scene.cubes:
            spot = free_xy(rng, config, taken, clearance_m=BIN_CLEARANCE_M, margin_m=BIN_MARGIN_M)
            if spot is None:
                msg = f"{self.id} seed {seed}: no room for the {c.color} bin"
                raise RuntimeError(msg)
            bins[c.color] = spot
            taken.append((spot.x, spot.y))
        goal = SortGoal(bins=bins)
        where = "; ".join(f"{col} -> (x={b.x:.3f}, y={b.y:.3f})" for col, b in bins.items())
        prompt = (
            "Sort the cubes: move each cube so that its centre rests on the table at the "
            f"point of its colour: {where} (base_link, metres)."
        )
        return TaskInstance(task=self.id, seed=seed, scene=scene, goal=goal, prompt=prompt)

    def check(self, instance: TaskInstance, final: FinalState) -> Verdict:
        goal = instance.goal
        assert isinstance(goal, SortGoal)  # noqa: S101 - discriminated by construction
        fail, metrics = common_failure(
            instance, final, allowed={c.name for c in instance.scene.cubes}
        )
        worst = 0.0
        for color, at in goal.bins.items():
            cube = instance.cube(color)
            pose = final.poses.get(cube.name)
            err = at.dist(pose.x, pose.y) if pose is not None else float("inf")
            metrics[f"err_{color}_m"] = err
            worst = max(worst, err)
            if pose is not None and not on_table(cube, pose):
                worst = float("inf")
        metrics["place_err_max_m"] = worst
        if fail is not None:
            return Verdict(ok=False, reason=fail, metrics=metrics)
        if worst > POS_TOL_M:
            return Verdict(ok=False, reason=f"worst cube {worst * 1e3:.0f} mm off", metrics=metrics)
        return Verdict(ok=True, reason="all cubes in their bins", metrics=metrics)
