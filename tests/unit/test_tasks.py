"""Versioned tasks: seeded instances, checkers, and the scripted baseline A on the fake backend."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from armbench.agents import ScriptedAgent, detect_all, get_agent
from armbench.agents.scripted import FINGER_SWEEP_M, OBSERVE_POSES, grasp_yaw, merge
from armbench.perception import Camera, Detection, load_perception_params
from armbench.perception.synthetic import ARM_RGB
from armbench.primitives import KinematicBackend, Robot, load_primitive_params
from armbench.primitives.errors import CameraTimeout
from armbench.primitives.params import PrimitiveParams
from armbench.scene import (
    Box,
    SceneConfig,
    box_inertia,
    box_to_sdf_model,
    cube_inertia,
    generate_scene,
    load_scene_config,
)
from armbench.tasks import (
    DISTURB_TOL_M,
    POS_TOL_M,
    SIM_LIMIT_S,
    TASK_IDS,
    FinalState,
    ModelPose,
    PlaceGoal,
    SortGoal,
    StackGoal,
    TaskId,
    TaskInstance,
    get_task,
)
from armbench.tasks.place_obstacle import OBSTACLE_NAME, OBSTACLE_SIZE
from armbench.tasks.spec import recolour, task_rng

DEV_SEEDS = range(10)
EXTRA_SEEDS = range(400, 450)
CUBE = 0.045


@pytest.fixture(scope="module")
def config() -> SceneConfig:
    return load_scene_config()


@pytest.fixture(scope="module")
def params() -> PrimitiveParams:
    return load_primitive_params()


@pytest.fixture(scope="module")
def camera() -> Camera:
    return Camera.from_spec(load_perception_params().camera)


def final_state(backend: KinematicBackend, inst: TaskInstance, sim_s: float) -> FinalState:
    poses = {c.name: ModelPose(x=c.x, y=c.y, z=c.z, yaw=c.yaw) for c in backend.cubes}
    for o in inst.obstacles:  # the fake world has no physics: obstacles never move
        poses[o.name] = ModelPose(x=o.x, y=o.y, z=o.z, yaw=o.yaw)
    return FinalState(poses=poses, min_tip_z_m=0.01, sim_s=sim_s)


def initial_state(inst: TaskInstance) -> FinalState:
    poses = {c.name: ModelPose(x=c.x, y=c.y, z=c.z, yaw=c.yaw) for c in inst.scene.cubes}
    for o in inst.obstacles:
        poses[o.name] = ModelPose(x=o.x, y=o.y, z=o.z, yaw=o.yaw)
    return FinalState(poses=poses, min_tip_z_m=0.02, sim_s=0.0)


# -- ids, registry, determinism -----------------------------------------------------------------


def test_task_ids_parse_and_registry() -> None:
    assert TASK_IDS == ("pick_place@1", "stack2@1", "sort3@1", "place_obstacle@1")
    assert str(TaskId.parse("sort3@1")) == "sort3@1"
    with pytest.raises(ValueError, match="name@version"):
        TaskId.parse("sort3")
    with pytest.raises(KeyError, match="unknown task"):
        get_task("fly@1")
    assert get_task(TaskId(name="stack2", version=1)).id.name == "stack2"


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_instances_are_deterministic_and_well_formed(task_id: str, config: SceneConfig) -> None:
    task = get_task(task_id)
    for seed in [*DEV_SEEDS, *EXTRA_SEEDS, 100, 119]:
        a, b = task.instance(seed, config), task.instance(seed, config)
        assert a == b
        assert a.task == task.id
        assert a.seed == seed
        colors = [c.color for c in a.scene.cubes]
        for color in a.required_colors():
            assert colors.count(color) == 1  # every colour the goal names is unambiguous
        assert a.prompt
        assert a.scene.cubes == a.scene.cubes  # positions come from the F1 generator
        ws = config.workspace
        if isinstance(a.goal, PlaceGoal):
            assert ws.x_min < a.goal.at.x < ws.x_max
            assert ws.y_min < a.goal.at.y < ws.y_max
            for c in a.scene.cubes:
                assert a.goal.at.dist(c.x, c.y) >= 0.09
        if isinstance(a.goal, SortGoal):
            assert set(a.goal.bins) == set(colors)
            pts = [(b.x, b.y) for b in a.goal.bins.values()]
            for i, p in enumerate(pts):
                for q in pts[i + 1 :]:
                    assert math.dist(p, q) >= 0.10


def test_tasks_differ_from_each_other_on_the_same_seed(config: SceneConfig) -> None:
    pp = get_task("pick_place@1").instance(3, config)
    po = get_task("place_obstacle@1").instance(3, config)
    assert isinstance(pp.goal, PlaceGoal)
    assert isinstance(po.goal, PlaceGoal)
    assert pp.goal.at != po.goal.at  # each task mixes its own salt into the seed


def test_recolour_keeps_positions_and_makes_unique_colours() -> None:
    scene = generate_scene(5, load_scene_config(), n_cubes=6)
    out = recolour(scene, task_rng(TaskId(name="x", version=1), 5), unique=3)
    assert [(c.x, c.y, c.yaw) for c in out.cubes] == [(c.x, c.y, c.yaw) for c in scene.cubes]
    first = [c.color for c in out.cubes[:3]]
    assert len(set(first)) == 3
    assert not set(first) & {c.color for c in out.cubes[3:]}


def test_recolour_rejects_more_unique_than_palette(config: SceneConfig) -> None:
    scene = generate_scene(1, config, n_cubes=5)
    with pytest.raises(ValueError, match="distinct colours"):
        recolour(scene, np.random.default_rng(0), unique=5)


# -- obstacle geometry ----------------------------------------------------------------------------


def test_obstacle_wall_sits_across_the_path_and_renders(config: SceneConfig) -> None:
    for seed in [*DEV_SEEDS, *EXTRA_SEEDS]:
        inst = get_task("place_obstacle@1").instance(seed, config)
        assert isinstance(inst.goal, PlaceGoal)
        (wall,) = inst.obstacles
        assert wall.name == OBSTACLE_NAME
        assert wall.size == OBSTACLE_SIZE
        target = inst.cube(inst.goal.target_color)
        mid = ((target.x + inst.goal.at.x) / 2, (target.y + inst.goal.at.y) / 2)
        assert math.isclose(wall.x, mid[0])
        assert math.isclose(wall.y, mid[1])
        heading = math.atan2(inst.goal.at.y - target.y, inst.goal.at.x - target.x)
        assert math.isclose(math.cos(wall.yaw - heading), 0.0, abs_tol=1e-9)
        for c in inst.scene.cubes:
            assert math.hypot(wall.x - c.x, wall.y - c.y) >= 0.12
        sdf = box_to_sdf_model(wall)
        assert '<model name="obstacle">' in sdf
        assert "0.16 0.04 0.1" in sdf
        assert "0.35 0.35 0.35 1" in sdf


def test_box_inertia_matches_cube_formula() -> None:
    assert box_inertia(0.05, (CUBE, CUBE, CUBE)) == pytest.approx((cube_inertia(0.05, CUBE),) * 3)
    ixx, iyy, izz = box_inertia(0.5, OBSTACLE_SIZE)
    assert ixx < izz  # the long x side dominates rotation about y and z
    assert izz < iyy


# -- checkers on synthetic final states -----------------------------------------------------------


def test_checkers_reject_untouched_initial_state(config: SceneConfig) -> None:
    for task_id in TASK_IDS:
        task = get_task(task_id)
        inst = task.instance(0, config)
        verdict = task.check(inst, initial_state(inst))
        assert not verdict.ok
        assert "mm" in verdict.reason or "off" in verdict.reason


def test_pick_place_checker_tolerances(config: SceneConfig) -> None:
    task = get_task("pick_place@1")
    inst = task.instance(2, config)
    goal = inst.goal
    assert isinstance(goal, PlaceGoal)
    target = inst.cube(goal.target_color)
    base = initial_state(inst).poses

    def with_target(dx: float, dz: float = 0.0, **kw: float) -> FinalState:
        poses = dict(base)
        poses[target.name] = ModelPose(x=goal.at.x + dx, y=goal.at.y, z=CUBE / 2 + dz)
        return FinalState(poses=poses, min_tip_z_m=kw.get("tip", 0.01), sim_s=kw.get("sim", 20.0))

    assert task.check(inst, with_target(POS_TOL_M - 1e-3)).ok
    assert not task.check(inst, with_target(POS_TOL_M + 1e-3)).ok
    assert not task.check(inst, with_target(0.0, dz=0.03)).ok  # sitting on another cube
    assert not task.check(inst, with_target(0.0, sim=SIM_LIMIT_S + 0.1)).ok
    assert not task.check(inst, with_target(0.0, tip=-0.003)).ok
    other = next(c for c in inst.scene.cubes if c.name != target.name)
    poses = dict(with_target(0.0).poses)
    poses[other.name] = ModelPose(x=other.x + DISTURB_TOL_M + 1e-3, y=other.y, z=other.z)
    bumped = FinalState(poses=poses, min_tip_z_m=0.01, sim_s=10.0)
    assert not task.check(inst, bumped).ok
    assert "untouched cube" in task.check(inst, bumped).reason
    missing = FinalState(poses={}, min_tip_z_m=0.01, sim_s=10.0)
    assert not task.check(inst, missing).ok


def test_stack_checker(config: SceneConfig) -> None:
    task = get_task("stack2@1")
    inst = task.instance(4, config)
    assert isinstance(inst.goal, StackGoal)
    top, base_cube = inst.cube(inst.goal.top_color), inst.cube(inst.goal.base_color)
    poses = dict(initial_state(inst).poses)
    poses[top.name] = ModelPose(x=base_cube.x + 0.005, y=base_cube.y, z=CUBE * 1.5)
    assert task.check(inst, FinalState(poses=poses, min_tip_z_m=0.01, sim_s=15.0)).ok
    poses[top.name] = ModelPose(x=base_cube.x, y=base_cube.y, z=CUBE / 2)  # fell off, on table
    assert not task.check(inst, FinalState(poses=poses, min_tip_z_m=0.01, sim_s=15.0)).ok
    poses[top.name] = ModelPose(x=base_cube.x + 0.03, y=base_cube.y, z=CUBE * 1.5)
    assert not task.check(inst, FinalState(poses=poses, min_tip_z_m=0.01, sim_s=15.0)).ok


def test_obstacle_checker_fails_when_wall_moves(config: SceneConfig) -> None:
    task = get_task("place_obstacle@1")
    inst = task.instance(1, config)
    assert isinstance(inst.goal, PlaceGoal)
    target = inst.cube(inst.goal.target_color)
    poses = dict(initial_state(inst).poses)
    poses[target.name] = ModelPose(x=inst.goal.at.x, y=inst.goal.at.y, z=CUBE / 2)
    assert task.check(inst, FinalState(poses=poses, min_tip_z_m=0.01, sim_s=20.0)).ok
    wall = inst.obstacles[0]
    poses[OBSTACLE_NAME] = ModelPose(x=wall.x + 0.006, y=wall.y, z=wall.z, yaw=wall.yaw)
    verdict = task.check(inst, FinalState(poses=poses, min_tip_z_m=0.01, sim_s=20.0))
    assert not verdict.ok
    assert "obstacle moved" in verdict.reason


# -- baseline A on the kinematic backend --------------------------------------------------------


def solve_on_fake(
    task_id: str, seed: int, config: SceneConfig, params: PrimitiveParams, camera: Camera
) -> tuple[TaskInstance, KinematicBackend, Robot, int]:
    task = get_task(task_id)
    inst = task.instance(seed, config)
    backend = KinematicBackend(params, camera, cubes=list(inst.scene.cubes))
    robot = Robot(backend, params=params)
    robot.reset()
    t0 = backend.sim_time()
    trace = ScriptedAgent().solve(robot, inst)
    verdict = task.check(inst, final_state(backend, inst, backend.sim_time() - t0))
    assert verdict.ok, (task_id, seed, verdict)
    return inst, backend, robot, trace.n_primitives


@pytest.mark.parametrize("task_id", TASK_IDS)
@pytest.mark.parametrize("seed", list(DEV_SEEDS))
def test_baseline_solves_dev_seeds_on_fake_backend(
    task_id: str, seed: int, config: SceneConfig, params: PrimitiveParams, camera: Camera
) -> None:
    _inst, backend, _robot, n = solve_on_fake(task_id, seed, config, params, camera)
    expected_picks = 3 if task_id == "sort3@1" else 1
    assert n >= 1 + 8 * expected_picks
    assert all(c.z >= CUBE / 2 - 1e-9 for c in backend.cubes)


def test_baseline_clears_the_wall(
    config: SceneConfig, params: PrimitiveParams, camera: Camera
) -> None:
    """Over the wall's footprint the held cube's bottom stays well above the wall top."""
    for seed in DEV_SEEDS:
        inst, backend, robot, _n = solve_on_fake("place_obstacle@1", seed, config, params, camera)
        wall = inst.obstacles[0]
        half_long, half_short = wall.size[0] / 2 + 0.04, wall.size[1] / 2 + 0.04
        c, s = math.cos(wall.yaw), math.sin(wall.yaw)
        lowest = math.inf
        qs = [np.asarray(params.ready_q)] + [np.asarray(q) for q, _ in backend.follow_log]
        for q0, q1 in pairwise(qs):
            for a in np.linspace(0.0, 1.0, 25):
                tcp = robot.tcp_pose(q0 + a * (q1 - q0))
                dx, dy = tcp.x - wall.x, tcp.y - wall.y
                along, across = dx * c + dy * s, -dx * s + dy * c
                if abs(along) < half_long and abs(across) < half_short:
                    lowest = min(lowest, tcp.z - CUBE / 2)
        assert lowest > wall.size[2] + 0.03, (seed, lowest)
        assert lowest < math.inf  # the path really crosses the wall


def test_detect_all_moves_to_observe_pose_when_cubes_hidden(
    config: SceneConfig, params: PrimitiveParams, camera: Camera
) -> None:
    inst = get_task("sort3@1").instance(0, config)
    backend = KinematicBackend(params, camera, cubes=list(inst.scene.cubes))
    robot = Robot(backend, params=params)
    robot.reset()
    found, moves = detect_all(robot, inst.required_colors())
    assert moves == 0  # the fake renderer has no arm: everything is visible from ready_pose
    assert set(found) == set(inst.required_colors())
    backend.camera_available = False
    with pytest.raises(CameraTimeout):
        detect_all(robot, inst.required_colors())
    backend.camera_available = True
    with pytest.raises(LookupError, match="not found"):
        detect_all(robot, ["purple"])
    # every observe pose is reachable without a goal being rejected
    for pose in OBSERVE_POSES:
        robot.move_to(pose)


def test_get_agent() -> None:
    assert get_agent("A").id == "A"
    with pytest.raises(KeyError, match="unknown agent"):
        get_agent("Z")


@settings(max_examples=20, deadline=None)
@given(seed=st.integers(0, 10_000))
def test_any_seed_yields_a_consistent_instance(seed: int) -> None:
    config = load_scene_config()
    for task_id in TASK_IDS:
        inst = get_task(task_id).instance(seed, config)
        assert inst.task == TaskId.parse(task_id)
        for color in inst.required_colors():
            inst.cube(color)
        if inst.obstacles:
            assert isinstance(inst.obstacles[0], Box)


# -- partial detections --------------------------------------------------------------------------


def _det(color: str, x: float, *, complete: bool, y: float = 0.0, yaw: float = 0.0) -> Detection:
    return Detection(
        color=color,
        position=(x, y, 0.0225),
        top_center=(x, y, 0.045),
        yaw_rad=yaw,
        pixel=(320.0, 240.0),
        bbox=(300, 220, 40, 40),
        area_px=700 if complete else 150,
        top_face_fraction=1.0 if complete else 0.2,
        complete=complete,
        depth_m=0.955,
    )


def test_grasp_yaw_turns_the_open_fingers_away_from_a_close_neighbour() -> None:
    """Fingers lie perpendicular to the grasp yaw: at yaw 0 they descend along y, so a cube
    9 cm away along y is inside their sweep and the equivalent yaw -pi/2 is chosen; the same
    neighbour along x, a far one, or none at all keep the detected yaw."""
    target = _det("red", 0.0, complete=True)
    along_fingers = _det("blue", 0.0, complete=True, y=0.09)
    assert grasp_yaw(target, [target, along_fingers]) == pytest.approx(-math.pi / 2)
    assert grasp_yaw(target, [target, _det("blue", 0.09, complete=True)]) == 0.0
    assert grasp_yaw(target, [target, _det("blue", 0.0, complete=True, y=0.3)]) == 0.0
    assert grasp_yaw(target, [target]) == 0.0
    tilted = _det("red", 0.0, complete=True, yaw=0.5)
    assert grasp_yaw(
        tilted, [tilted, _det("blue", -0.09 * math.sin(0.5), complete=True, y=0.09 * math.cos(0.5))]
    ) == pytest.approx(0.5 - math.pi / 2)
    assert 0.055 < FINGER_SWEEP_M < 0.09


def test_merge_prefers_complete_detections_and_keeps_the_first_complete_one() -> None:
    found: dict[str, Detection] = {}
    merge(found, [_det("red", -0.30, complete=False)])
    assert [d.complete for d in found.values()] == [False]
    merge(found, [_det("red", -0.31, complete=True), _det("blue", -0.5, complete=True)])
    assert [(d.complete, d.position[0]) for d in found.values()] == [(True, -0.31), (True, -0.5)]
    merge(found, [_det("red", -0.32, complete=True), _det("blue", -0.6, complete=False)])
    assert [(d.complete, d.position[0]) for d in found.values()] == [(True, -0.31), (True, -0.5)]


@dataclass
class _ArmShadowBackend(KinematicBackend):
    """The first frame has a dark forearm over most of ``shadowed`` cube's top face."""

    shadowed: str = ""
    shadow_frames: int = 1

    def frame(self, timeout_s: float) -> tuple[np.ndarray, np.ndarray] | None:
        out = super().frame(timeout_s)
        if out is None or self.shadow_frames <= 0:
            return out
        self.shadow_frames -= 1
        rgb, depth = out
        cube = next(c for c in self.cubes if c.name == self.shadowed)
        px, _ = self.camera.project_base(np.array([[cube.x, cube.y, cube.size]]))
        ui, vi = round(float(px[0, 0])), round(float(px[0, 1]))
        rgb[vi - 40 : vi + 40, ui - 8 : ui + 40] = ARM_RGB
        depth[vi - 40 : vi + 40, ui - 8 : ui + 40] = 0.6
        return rgb, depth


def test_detect_all_moves_on_when_the_first_view_is_partial(
    config: SceneConfig, params: PrimitiveParams, camera: Camera
) -> None:
    inst = get_task("pick_place@1").instance(418, config)
    assert isinstance(inst.goal, PlaceGoal)
    target = inst.cube(inst.goal.target_color)
    backend = _ArmShadowBackend(params, camera, cubes=list(inst.scene.cubes), shadowed=target.name)
    robot = Robot(backend, params=params)
    robot.reset()
    dets, moves = detect_all(robot, inst.required_colors())
    assert moves >= 1
    d = dets[target.color]
    assert d.complete
    assert math.hypot(d.position[0] - target.x, d.position[1] - target.y) < 0.003

    plain = KinematicBackend(params, camera, cubes=list(inst.scene.cubes))
    robot = Robot(plain, params=params)
    robot.reset()
    _, moves0 = detect_all(robot, inst.required_colors())
    assert moves0 == 0
