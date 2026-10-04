"""``place_obstacle@1``: pick-and-place with a 10 cm wall between the cube and its goal.

The wall is grey (invisible to ``detect()``), heavier than a cube and sits perpendicular to
the pick-goal line at its midpoint; a 10 cm approach height clips it, so a solver must carry
higher or go around, and the checker fails anything that nudges the wall.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np

from armbench.scene import Box, Scene, SceneConfig, generate_scene
from armbench.tasks.spec import (
    OBSTACLE_RGBA,
    POS_TOL_M,
    FinalState,
    PlaceGoal,
    TaskId,
    TaskInstance,
    Verdict,
    common_failure,
    free_xy,
    on_table,
    recolour,
    task_rng,
)

N_CUBES: Final = (3, 2, 1)
"""Cubes to spawn; fewer only when no cube of the seed leaves room for the wall."""
OBSTACLE_NAME: Final = "obstacle"
OBSTACLE_SIZE: Final = (0.16, 0.04, 0.10)
OBSTACLE_MASS_KG: Final = 0.5
GOAL_MIN_DIST_M: Final = 0.22
GOAL_CLEARANCE_M: Final = 0.09
GOAL_MARGIN_M: Final = 0.05
OBSTACLE_CLEARANCE_M: Final = 0.12
WALL_MARGIN_M: Final = 0.01
GOAL_TRIES: Final = 300
"""Centre distance between the wall and any cube (half wall length 0.08 + cube + gripper)."""


class PlaceObstacle:
    id = TaskId(name="place_obstacle", version=1)
    description = "Move one cube to a point with a low wall standing in the straight path."

    def instance(self, seed: int, config: SceneConfig) -> TaskInstance:
        rng = task_rng(self.id, seed)
        for n_cubes in N_CUBES:
            for target in range(n_cubes):
                scene = recolour(generate_scene(seed, config, n_cubes=n_cubes), rng, [target])
                inst = self._place_wall(seed, scene, config, rng, target)
                if inst is not None:
                    return inst
        msg = f"{self.id} seed {seed}: no goal/wall placement found"
        raise RuntimeError(msg)

    def _place_wall(
        self, seed: int, scene: Scene, config: SceneConfig, rng: np.random.Generator, idx: int
    ) -> TaskInstance | None:
        target = scene.cubes[idx]
        taken = [(c.x, c.y) for c in scene.cubes]
        ws = config.workspace
        half = OBSTACLE_SIZE[0] / 2.0
        for _ in range(GOAL_TRIES):
            at = free_xy(
                rng,
                config,
                taken,
                clearance_m=GOAL_CLEARANCE_M,
                margin_m=GOAL_MARGIN_M,
                min_dist_from=(target.x, target.y),
                min_dist_m=GOAL_MIN_DIST_M,
            )
            if at is None:
                return None
            mx, my = (target.x + at.x) / 2.0, (target.y + at.y) / 2.0
            yaw = math.atan2(at.y - target.y, at.x - target.x) + math.pi / 2.0  # across the path
            ends = [
                (mx + half * math.cos(yaw), my + half * math.sin(yaw)),
                (mx - half * math.cos(yaw), my - half * math.sin(yaw)),
            ]
            inside = all(
                ws.x_min + WALL_MARGIN_M <= ex <= ws.x_max - WALL_MARGIN_M
                and ws.y_min + WALL_MARGIN_M <= ey <= ws.y_max - WALL_MARGIN_M
                for ex, ey in ends
            )
            clear = all(math.hypot(mx - px, my - py) >= OBSTACLE_CLEARANCE_M for px, py in taken)
            if not (inside and clear):
                continue
            wall = Box(
                name=OBSTACLE_NAME,
                x=mx,
                y=my,
                z=OBSTACLE_SIZE[2] / 2.0,
                yaw=yaw,
                size=OBSTACLE_SIZE,
                mass_kg=OBSTACLE_MASS_KG,
                rgba=OBSTACLE_RGBA,
                friction_mu=config.cube.friction_mu,
            )
            goal = PlaceGoal(target_color=target.color, at=at)
            prompt = (
                f"Move the {target.color} cube so that its centre is at x={at.x:.3f}, "
                f"y={at.y:.3f} (base_link, metres) resting on the table. A grey wall "
                f"{OBSTACLE_SIZE[2] * 100:.0f} cm high and {OBSTACLE_SIZE[0] * 100:.0f} cm "
                f"long stands between the cube and the goal, centred at x={mx:.3f}, "
                f"y={my:.3f}; the camera does not see it. Do not touch the wall or the "
                "other cubes."
            )
            return TaskInstance(
                task=self.id, seed=seed, scene=scene, obstacles=(wall,), goal=goal, prompt=prompt
            )
        return None

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
        return Verdict(ok=True, reason="target at goal, wall untouched", metrics=metrics)
