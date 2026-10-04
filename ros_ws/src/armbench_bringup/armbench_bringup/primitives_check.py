"""F4 gate experiments on the live simulation (runs inside the simulation container).

``--section contracts``: random reachable moves (accuracy, clearance monitor), unreachable poses
(typed error, arm does not move), ``speed_scale`` sweep, ``reset()`` from random poses.
``--section pick``: scripted pick-and-place on seeded scenes with Gazebo ground truth.
Writes one JSON per run; ``scripts/primitives_eval.py`` orchestrates containers and judges.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import rclpy

from armbench.perception import Detection
from armbench.primitives import (
    MoveResult,
    OutOfReach,
    Pose,
    PrimitiveError,
    PrimitiveParams,
    Robot,
    load_primitive_params,
)
from armbench.primitives.scripted import free_spot, grasp_pose, moves, pick_and_place
from armbench.scene import Scene, generate_scene, load_scene_config
from armbench.scene.generator import DEFAULT_SCENE_FILE
from armbench_bringup.ros_backend import CubePose, GzScene, RosBackend

PLACE_CLEARANCE_M = 0.08
SPEED_SCALES = (0.1, 0.2, 0.35, 0.5, 0.75, 1.0)
SPEED_A = Pose(x=-0.60, y=-0.20, z=0.10)
SPEED_B = Pose(x=-0.40, y=0.20, z=0.20, yaw=0.5)
MAX_REJECTS_PER_MOVE = 20
PLACE_TOL_MM = 15.0
Z_TOL_MM = 5.0
DRIFT_TOL_MM = 10.0


def random_pose(rng: np.random.Generator, params: PrimitiveParams) -> Pose:
    w = params.workspace
    return Pose(
        x=float(rng.uniform(w.x_min, w.x_max)),
        y=float(rng.uniform(w.y_min, w.y_max)),
        z=float(rng.uniform(w.z_min, w.z_max)),
        yaw=float(rng.uniform(-math.pi / 4, math.pi / 4)),
    )


def unreachable_pose(rng: np.random.Generator, i: int) -> Pose:
    """Alternates: beyond the arm's reach in the plane / far above the workspace / behind."""
    kind = i % 3
    if kind == 0:
        return Pose(
            x=float(rng.uniform(-1.4, -0.95)),
            y=float(rng.uniform(-0.6, 0.6)),
            z=float(rng.uniform(0.02, 0.5)),
        )
    if kind == 1:
        return Pose(
            x=float(rng.uniform(-0.7, -0.3)),
            y=float(rng.uniform(-0.3, 0.3)),
            z=float(rng.uniform(0.6, 1.2)),
        )
    return Pose(
        x=float(rng.uniform(0.3, 0.9)),
        y=float(rng.uniform(-0.6, 0.6)),
        z=float(rng.uniform(0.02, 0.5)),
    )


def move_record(res: MoveResult, backend: RosBackend) -> dict:
    m = backend.monitor
    return {
        "target": res.plan.target.model_dump(),
        "speed_scale": res.plan.speed_scale,
        "planned_s": res.plan.duration_s,
        "sim_s": res.sim_s,
        "wall_s": res.wall_s,
        "pos_err_mm": res.position_error_m * 1e3,
        "yaw_err_deg": math.degrees(res.yaw_error_rad),
        "joint_err_rad": res.joint_error_rad,
        "ik_iters": res.plan.ik_iters,
        "min_singular_value": res.plan.min_singular_value,
        "planned_clearance_m": res.plan.path_min_clearance_m,
        "min_frame_z_m": m.min_frame_z,
        "min_tip_z_m": m.min_tip_z,
        "monitor_samples": m.n_samples,
    }


def section_accuracy(robot: Robot, backend: RosBackend, rng: np.random.Generator, n: int) -> dict:
    records: list[dict] = []
    rejected: list[dict] = []
    while len(records) < n:
        pose = random_pose(rng, robot.params)
        speed = float(rng.uniform(0.3, 1.0))
        backend.reset_monitor()
        try:
            res = robot.move_to(pose, speed)
        except PrimitiveError as exc:
            rejected.append({"target": pose.model_dump(), **exc.to_dict()})
            if len(rejected) > MAX_REJECTS_PER_MOVE * n:
                break
            continue
        records.append(move_record(res, backend))
        if len(records) % 20 == 0:
            backend.get_logger().info(f"accuracy {len(records)}/{n}")
    return {"n": len(records), "moves": records, "rejected": rejected}


def section_unreachable(
    robot: Robot, backend: RosBackend, rng: np.random.Generator, n: int
) -> dict:
    records: list[dict] = []
    for i in range(n):
        pose = unreachable_pose(rng, i)
        before = backend.snapshot(5.0)
        follows = backend.follow_count
        rec: dict = {"target": pose.model_dump(), "raised": None}
        try:
            robot.move_to(pose)
            rec["raised"] = "none"
        except OutOfReach as exc:
            rec["raised"] = exc.code
            rec["reason"] = exc.details.get("reason")
        except PrimitiveError as exc:
            rec["raised"] = exc.code
        backend.spin_sim(0.3)
        after = backend.snapshot(5.0)
        if before is not None and after is not None:
            rec["max_dq_rad"] = float(np.max(np.abs(np.subtract(after.q, before.q))))
        rec["goals_sent"] = backend.follow_count - follows
        records.append(rec)
    return {"n": len(records), "cases": records}


def section_speed(robot: Robot, backend: RosBackend) -> dict:
    records: list[dict] = []
    robot.move_to(SPEED_A)
    for scale in SPEED_SCALES:
        backend.reset_monitor()
        res = robot.move_to(SPEED_B, scale)
        records.append(move_record(res, backend))
        robot.move_to(SPEED_A)
    return {"a": SPEED_A.model_dump(), "b": SPEED_B.model_dump(), "moves": records}


def section_reset(robot: Robot, backend: RosBackend, rng: np.random.Generator, n: int) -> dict:
    records: list[dict] = []
    ready_q = np.asarray(robot.params.ready_q)
    while len(records) < n:
        pose = random_pose(rng, robot.params)
        try:
            robot.move_to(pose, float(rng.uniform(0.3, 1.0)))
        except PrimitiveError:
            continue
        backend.reset_monitor()
        res = robot.reset()
        snap = backend.snapshot(5.0)
        q_err = float(np.max(np.abs(np.asarray(snap.q) - ready_q))) if snap else math.nan
        records.append(
            {
                "from": pose.model_dump(),
                "sim_s": res.sim_s,
                "wall_s": res.wall_s,
                "q_err_rad": q_err,
                "opening_m": robot.opening_m(snap) if snap else math.nan,
                "min_tip_z_m": backend.monitor.min_tip_z,
            }
        )
    return {"n": len(records), "resets": records}


def nearest_cube(scene: Scene, det: Detection) -> str:
    return min(
        scene.cubes, key=lambda c: math.hypot(c.x - det.position[0], c.y - det.position[1])
    ).name


def judge_episode(
    rec: dict, scene: Scene, name: str, place: Pose, poses: dict[str, CubePose]
) -> None:
    """Success = target cube within 15 mm of the place pose, upright on the table, other cubes
    moved < 10 mm (all from Gazebo poses)."""
    p = poses.get(name)
    if p is None:
        rec["error"] = {"code": "scene", "message": f"no pose for {name}"}
        return
    rec["place_err_mm"] = math.hypot(p.x - place.x, p.y - place.y) * 1e3
    rec["z_err_mm"] = (p.z - scene.cubes[0].size / 2.0) * 1e3
    drift = 0.0
    for c in scene.cubes:
        if c.name == name:
            continue
        q = poses.get(c.name)
        if q is not None:
            drift = max(drift, math.hypot(q.x - c.x, q.y - c.y))
    rec["others_max_move_mm"] = drift * 1e3
    rec["ok"] = (
        rec["place_err_mm"] < PLACE_TOL_MM
        and abs(rec["z_err_mm"]) < Z_TOL_MM
        and rec["others_max_move_mm"] < DRIFT_TOL_MM
    )


def failure_code(rec: dict) -> str | None:
    """Error code of a failed episode (``scene`` = spawn/settle infrastructure, not the robot)."""
    err = rec.get("error")
    return str(err["code"]) if isinstance(err, dict) and "code" in err else None


def run_pick(robot: Robot, backend: RosBackend, gz: GzScene, seed: int, timeout_s: float) -> dict:
    t_wall = time.time()
    scene = generate_scene(seed, gz.config)
    rec: dict = {"seed": seed, "n_cubes": len(scene.cubes), "ok": False}
    t0 = backend.sim_time()
    try:
        if (err := gz.place(scene)) is not None:
            rec["error"] = {"code": "scene", "message": err}
            return rec
        t_spawned = backend.sim_time()
        if not backend.wait_until(lambda: gz.settled(scene, t_spawned), timeout_s):
            rec["error"] = {"code": "scene", "message": "cubes did not settle"}
            return rec
        backend.spin_sim(0.5)
        backend.reset_monitor()
        dets = robot.detect()
        rec["n_detections"] = len(dets)
        if not dets:
            rec["error"] = {"code": "perception", "message": "no detections"}
            return rec
        target = dets[0]
        name = nearest_cube(scene, target)
        rec["target"] = name
        others = [(d.position[0], d.position[1]) for d in dets[1:]]
        spot = free_spot(others, robot.params.workspace, clearance_m=PLACE_CLEARANCE_M)
        if spot is None:
            rec["error"] = {"code": "scene", "message": "no free spot"}
            return rec
        pick = grasp_pose(target)
        place = Pose(x=spot[0], y=spot[1], z=pick.z)
        rec["place"] = place.model_dump()
        log = pick_and_place(robot, pick, place)
        rec["n_moves"] = len(moves(log))
        rec["task_sim_s"] = backend.sim_time() - t0
        backend.spin_sim(1.0)
        judge_episode(rec, scene, name, place, backend.cube_poses())
    except PrimitiveError as exc:
        rec["error"] = exc.to_dict()
    finally:
        rec["min_tip_z_m"] = backend.monitor.min_tip_z
        try:
            robot.reset()
        except PrimitiveError as exc:
            rec["reset_error"] = exc.to_dict()
        rec["removed_all"] = gz.clear(scene)
        rec["wall_s"] = round(time.time() - t_wall, 2)
    return rec


def parse_seeds(spec: str) -> list[int]:
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in spec.split(",") if s]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", choices=("contracts", "pick"), required=True)
    parser.add_argument("--seeds", default="400-424")
    parser.add_argument("--n-moves", type=int, default=200)
    parser.add_argument("--n-unreachable", type=int, default=100)
    parser.add_argument("--n-reset", type=int, default=100)
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument("--world", default="armbench_tabletop")
    parser.add_argument("--scene-config", type=Path, default=DEFAULT_SCENE_FILE)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    params = load_primitive_params()
    rclpy.init()
    backend = RosBackend(params)
    report: dict = {"section": args.section, "rng_seed": args.rng_seed, "ready": False}
    t_wall = time.time()
    try:
        if not backend.ready(args.timeout):
            report["error"] = "simulation not ready (clock, joint states, camera or action)"
            return
        report["ready"] = True
        robot = Robot(backend, params=params)
        robot.reset()
        rng = np.random.default_rng(args.rng_seed)
        if args.section == "contracts":
            report["accuracy"] = section_accuracy(robot, backend, rng, args.n_moves)
            report["unreachable"] = section_unreachable(robot, backend, rng, args.n_unreachable)
            report["speed"] = section_speed(robot, backend)
            report["reset"] = section_reset(robot, backend, rng, args.n_reset)
        else:
            gz = GzScene(backend, args.world, load_scene_config(args.scene_config))
            episodes = []
            for seed in parse_seeds(args.seeds):
                rec = run_pick(robot, backend, gz, seed, args.timeout)
                if failure_code(rec) == "scene":  # transient gz CLI/settle timeout: one retry
                    rec = run_pick(robot, backend, gz, seed, args.timeout)
                    rec["retried"] = True
                backend.get_logger().info(json.dumps(rec))
                episodes.append(rec)
            report["episodes"] = episodes
            report["n_ok"] = sum(1 for e in episodes if e["ok"])
    finally:
        report["wall_s_total"] = round(time.time() - t_wall, 1)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
        backend.get_logger().info(f"wrote {args.out}")
        backend.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
