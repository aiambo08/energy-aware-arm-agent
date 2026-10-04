"""Primitives (F4) on the ROS-free kinematic backend: contracts, typed errors, planning checks,
speed scaling, gripper semantics and the scripted pick-and-place on seeded scenes."""

from __future__ import annotations

import itertools
import math
import re
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from armbench.kinematics import JOINT_NAMES
from armbench.paths import CONFIG_DIR
from armbench.perception import Camera, PerceptionParams, load_perception_params
from armbench.primitives import (
    ERRORS,
    CameraTimeout,
    Collision,
    MotionOutcome,
    MoveResult,
    NoObjectGrasped,
    Observation,
    OutOfReach,
    Pose,
    PrimitiveError,
    PrimitiveParams,
    Robot,
    Singularity,
    SkillNotAvailable,
    Timeout,
    load_primitive_params,
)
from armbench.primitives.fake import KinematicBackend
from armbench.primitives.scripted import free_spot, grasp_pose, moves, pick_and_place
from armbench.scene import Cube, generate_scene, load_scene_config

CUBE = 0.045
ROOT = Path(__file__).resolve().parents[2]
GRIPPER_XACRO = ROOT / "ros_ws" / "src" / "armbench_description" / "urdf" / "parallel_gripper.xacro"


@pytest.fixture(scope="module")
def params() -> PrimitiveParams:
    return load_primitive_params()


@pytest.fixture(scope="module")
def perception() -> PerceptionParams:
    return load_perception_params()


@pytest.fixture(scope="module")
def camera(perception: PerceptionParams) -> Camera:
    return Camera.from_spec(perception.camera)


@pytest.fixture
def backend(params: PrimitiveParams, camera: Camera) -> KinematicBackend:
    return KinematicBackend(params, camera)


@pytest.fixture
def robot(
    backend: KinematicBackend, params: PrimitiveParams, perception: PerceptionParams, camera: Camera
) -> Robot:
    return Robot(backend, params=params, perception=perception, camera=camera)


def cube(name: str, color: str, x: float, y: float, *, yaw: float = 0.0) -> Cube:
    return Cube(name=name, color=color, x=x, y=y, z=CUBE / 2, size=CUBE, yaw=yaw)


def reachable_poses(params: PrimitiveParams, seed: int, n: int) -> Iterator[Pose]:
    w = params.workspace
    rng = np.random.default_rng(seed)
    for _ in range(n):
        yield Pose(
            x=float(rng.uniform(w.x_min, w.x_max)),
            y=float(rng.uniform(w.y_min, w.y_max)),
            z=float(rng.uniform(w.z_min, w.z_max)),
            yaw=float(rng.uniform(-math.pi / 4, math.pi / 4)),
        )


# -- configuration -----------------------------------------------------------------------------


def test_config_matches_gripper_xacro(params: PrimitiveParams) -> None:
    text = GRIPPER_XACRO.read_text()

    def default(name: str) -> float:
        m = re.search(rf"\b{name}:=([0-9.]+)", text)
        assert m is not None, name
        return float(m.group(1))

    tcp = re.search(r'tcp_joint" type="fixed">\s*<origin xyz="0 0 \$\{(.+?)\}"', text)
    assert tcp is not None
    assert tcp.group(1) == "base_height + finger_length - 0.010"
    inset = 0.010
    assert params.gripper.stroke_m == pytest.approx(default("stroke"))
    assert params.tcp_offset_m == pytest.approx(
        default("base_height") + default("finger_length") - inset
    )
    assert params.finger_tip_below_tcp_m == pytest.approx(inset)
    # the fully open gap is two strokes (fingers at joint position 0 touch the base edges)
    assert params.gripper.max_opening_m == pytest.approx(2 * default("stroke"))


def test_config_matches_sim_check(params: PrimitiveParams) -> None:
    raw = yaml.safe_load((ROOT / "ros_ws/src/armbench_bringup/config/sim_check.yaml").read_text())
    assert params.gripper.open_effort_n == raw["gripper"]["open_effort_n"]
    assert params.gripper.close_effort_n == raw["gripper"]["close_effort_n"]
    assert params.gripper.stroke_m == raw["gripper"]["stroke_m"]


def test_workspace_covers_scene_workspace(params: PrimitiveParams) -> None:
    scene = load_scene_config().workspace
    w = params.workspace
    assert w.x_min <= scene.x_min
    assert scene.x_max <= w.x_max
    assert w.y_min <= scene.y_min
    assert scene.y_max <= w.y_max
    assert w.z_min <= CUBE / 2 <= w.z_max


def test_ready_q_realises_ready_pose(robot: Robot, params: PrimitiveParams) -> None:
    tcp = robot.tcp_pose(params.ready_q)
    x, y, z, yaw = params.ready_pose
    assert tcp.x == pytest.approx(x, abs=1e-3)
    assert tcp.y == pytest.approx(y, abs=1e-3)
    assert tcp.z == pytest.approx(z, abs=1e-3)
    assert tcp.yaw == pytest.approx(yaw, abs=1e-3)


def test_bad_ready_q_rejected(
    backend: KinematicBackend, params: PrimitiveParams, perception: PerceptionParams, camera: Camera
) -> None:
    bad = params.model_copy(update={"ready_q": (0.0, -1.5, -1.5, -1.5, 1.5708, 0.0)})
    with pytest.raises(ValueError, match="ready_q"):
        Robot(backend, params=bad, perception=perception, camera=camera)


def test_unknown_branch_joint_rejected(
    backend: KinematicBackend, params: PrimitiveParams, perception: PerceptionParams, camera: Camera
) -> None:
    ik = params.ik.model_copy(update={"branch": {"nope": (0.0, 1.0)}})
    with pytest.raises(ValueError, match="unknown joint"):
        Robot(
            backend,
            params=params.model_copy(update={"ik": ik}),
            perception=perception,
            camera=camera,
        )


def test_params_schema_rejects_unknown_keys(tmp_path: Path) -> None:
    raw = yaml.safe_load((CONFIG_DIR / "primitives.yaml").read_text())
    raw["surprise"] = 1
    path = tmp_path / "p.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="surprise"):
        load_primitive_params(path)


# -- contracts ---------------------------------------------------------------------------------


def test_pose_helpers() -> None:
    p = Pose(x=-0.5, y=0.1, z=0.02, yaw=0.3)
    up = p.above(0.1)
    assert (up.x, up.y, up.yaw) == (p.x, p.y, p.yaw)
    assert up.z == pytest.approx(0.12)
    assert p.with_yaw(0.0).yaw == 0.0
    assert p.distance_xy(Pose(x=-0.5, y=0.4, z=0.5)) == pytest.approx(0.3)
    with pytest.raises(ValueError, match=r"nan|finite"):
        Pose(x=float("nan"), y=0.0, z=0.0)


def test_error_codes_unique_and_serialisable() -> None:
    codes = [e.code for e in ERRORS]
    assert len(set(codes)) == len(codes)
    err = OutOfReach("far", target={"x": 1.0}, reason="ik")
    assert isinstance(err, PrimitiveError)
    assert err.to_dict() == {
        "code": "out_of_reach",
        "message": "far",
        "details": {"target": {"x": 1.0}, "reason": "ik"},
    }


def test_execute_skill_not_available(robot: Robot) -> None:
    with pytest.raises(SkillNotAvailable) as exc:
        robot.execute_skill("stack", color="red")
    assert exc.value.details == {"name": "stack", "kwargs": {"color": "red"}}


# -- move_to -----------------------------------------------------------------------------------


def test_move_to_reaches_random_poses(robot: Robot, params: PrimitiveParams) -> None:
    for pose in reachable_poses(params, seed=1, n=200):
        res = robot.move_to(pose)
        assert isinstance(res, MoveResult)
        assert res.position_error_m < 5e-4
        assert res.yaw_error_rad < math.radians(0.1)
        assert res.plan.path_min_clearance_m >= params.collision.tip_clearance_m
        assert res.plan.min_singular_value >= params.ik.min_singular_value
        assert robot.kin.within_limits(np.asarray(res.q_final))
        for name, (lo, hi) in params.ik.branch.items():
            assert lo <= res.q_final[JOINT_NAMES.index(name)] <= hi


@pytest.mark.parametrize(
    ("pose", "reason"),
    [
        (Pose(x=-1.2, y=0.0, z=0.1), "outside_workspace"),
        (Pose(x=-0.5, y=0.0, z=0.8), "outside_workspace"),
        (Pose(x=-0.5, y=0.6, z=0.1), "outside_workspace"),
    ],
)
def test_out_of_reach_does_not_move(
    robot: Robot, backend: KinematicBackend, pose: Pose, reason: str
) -> None:
    q0 = backend.q.copy()
    with pytest.raises(OutOfReach) as exc:
        robot.move_to(pose)
    assert exc.value.details["reason"] == reason
    assert np.array_equal(backend.q, q0)
    assert backend.follow_log == []


def test_ik_failure_inside_box_is_out_of_reach(
    backend: KinematicBackend, params: PrimitiveParams, perception: PerceptionParams, camera: Camera
) -> None:
    # widen the box so that a geometrically unreachable pose passes the box check
    wide = params.workspace.model_copy(update={"x_min": -2.0})
    robot = Robot(
        backend,
        params=params.model_copy(update={"workspace": wide}),
        perception=perception,
        camera=camera,
    )
    with pytest.raises(OutOfReach) as exc:
        robot.move_to(Pose(x=-1.5, y=0.0, z=0.1))
    assert exc.value.details["reason"] == "ik"
    assert backend.follow_log == []


def test_target_below_table_is_collision(robot: Robot, backend: KinematicBackend) -> None:
    with pytest.raises(Collision) as exc:
        robot.move_to(Pose(x=-0.5, y=0.0, z=0.0))
    assert exc.value.details["reason"] == "target_below_table"
    assert backend.follow_log == []


def test_path_collision_detected(
    backend: KinematicBackend, params: PrimitiveParams, perception: PerceptionParams, camera: Camera
) -> None:
    # a link clearance larger than the lowest wrist height makes every low move a path collision
    col = params.collision.model_copy(update={"link_clearance_m": 0.20})
    robot = Robot(
        backend,
        params=params.model_copy(update={"collision": col}),
        perception=perception,
        camera=camera,
    )
    with pytest.raises(Collision) as exc:
        robot.move_to(Pose(x=-0.5, y=0.0, z=0.02))
    assert exc.value.details["reason"] == "path"
    assert backend.follow_log == []


def test_singularity_detected(
    backend: KinematicBackend, params: PrimitiveParams, perception: PerceptionParams, camera: Camera
) -> None:
    ik = params.ik.model_copy(update={"min_singular_value": 10.0})
    robot = Robot(
        backend, params=params.model_copy(update={"ik": ik}), perception=perception, camera=camera
    )
    with pytest.raises(Singularity):
        robot.move_to(Pose(x=-0.5, y=0.0, z=0.1))
    assert backend.follow_log == []


def test_invalid_speed_scale(robot: Robot) -> None:
    with pytest.raises(ValueError, match="speed_scale"):
        robot.move_to(Pose(x=-0.5, y=0.0, z=0.1), speed_scale=0.0)
    with pytest.raises(ValueError, match="speed_scale"):
        robot.move_to(Pose(x=-0.5, y=0.0, z=0.1), speed_scale=1.5)


def test_duration_monotone_in_speed_scale(robot: Robot, params: PrimitiveParams) -> None:
    robot.move_to(Pose(x=-0.6, y=-0.2, z=0.1))
    target = Pose(x=-0.4, y=0.2, z=0.2, yaw=0.5)
    scales = [0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
    durations = [robot.plan(target, s).duration_s for s in scales]
    assert all(a > b for a, b in itertools.pairwise(durations))
    assert durations[-1] >= params.motion.min_duration_s
    assert durations[0] == pytest.approx(durations[-1] * 10, rel=0.05) or durations[
        -1
    ] == pytest.approx(params.motion.min_duration_s)


def test_move_timeout_from_controller(robot: Robot, backend: KinematicBackend) -> None:
    backend.motion_outcome = MotionOutcome.ABORTED
    with pytest.raises(Timeout) as exc:
        robot.move_to(Pose(x=-0.5, y=0.0, z=0.1))
    assert exc.value.details["outcome"] == "aborted"


def test_move_goal_tolerance_violation(robot: Robot, backend: KinematicBackend) -> None:
    backend.tracking_noise_rad = 0.1
    with pytest.raises(Timeout) as exc:
        robot.move_to(Pose(x=-0.5, y=0.0, z=0.1))
    assert exc.value.details["outcome"] == "goal_tolerance"


def test_small_tracking_error_reported(robot: Robot, backend: KinematicBackend) -> None:
    backend.tracking_noise_rad = 0.005
    res = robot.move_to(Pose(x=-0.5, y=0.0, z=0.1))
    assert 0 < res.joint_error_rad <= 0.005
    assert 0 < res.position_error_m < 0.01


# -- gripper -----------------------------------------------------------------------------------


def test_grasp_nothing_raises(robot: Robot) -> None:
    robot.move_to(Pose(x=-0.5, y=0.0, z=0.05))
    with pytest.raises(NoObjectGrasped) as exc:
        robot.grasp()
    assert exc.value.details["opening_m"] == pytest.approx(0.0)
    assert not robot.observe().holding


def test_grasp_release_cube(robot: Robot, backend: KinematicBackend) -> None:
    backend.cubes = [cube("cube_0", "red", -0.5, 0.0)]
    robot.move_to(Pose(x=-0.5, y=0.0, z=CUBE / 2))
    g = robot.grasp()
    assert g.holding
    assert g.opening_m == pytest.approx(CUBE)
    assert robot.observe().holding
    robot.move_to(Pose(x=-0.5, y=0.0, z=0.2))
    assert backend.cube("cube_0").z == pytest.approx(0.2, abs=1e-3)
    r = robot.release()
    assert not r.holding
    assert r.opening_m == pytest.approx(2 * 0.0425)
    assert backend.cube("cube_0").z == pytest.approx(CUBE / 2)


def test_reset_opens_and_homes(
    robot: Robot, backend: KinematicBackend, params: PrimitiveParams
) -> None:
    backend.cubes = [cube("cube_0", "blue", -0.45, 0.1)]
    robot.move_to(Pose(x=-0.45, y=0.1, z=CUBE / 2))
    robot.grasp()
    res = robot.reset()
    assert res.primitive == "reset"
    assert res.sim_s > 0
    assert backend.held is None
    assert np.allclose(backend.q, params.ready_q)
    assert backend.fingers == (0.0, 0.0)


def test_reset_from_random_poses_is_fast(robot: Robot, params: PrimitiveParams) -> None:
    for pose in reachable_poses(params, seed=7, n=50):
        robot.move_to(pose)
        res = robot.reset()
        assert res.sim_s < 5.0


# -- camera / observe --------------------------------------------------------------------------


def test_camera_timeout(robot: Robot, backend: KinematicBackend) -> None:
    backend.camera_available = False
    with pytest.raises(CameraTimeout):
        robot.observe()
    with pytest.raises(CameraTimeout):
        robot.detect("red")


def test_observe_contract(robot: Robot, backend: KinematicBackend) -> None:
    backend.cubes = [cube("cube_0", "red", -0.5, 0.1), cube("cube_1", "green", -0.6, -0.1)]
    obs = robot.observe()
    assert isinstance(obs, Observation)
    assert len(obs.q) == 6
    assert obs.tcp.z == pytest.approx(0.25, abs=1e-3)
    assert obs.gripper_opening_m == pytest.approx(0.085)
    assert not obs.holding
    assert [d.color for d in obs.detections] == ["green", "red"]  # sorted by x
    assert len(robot.detect("red")) == 1
    assert robot.detect("yellow") == []
    dumped = obs.model_dump_json()
    assert Observation.model_validate_json(dumped) == obs


# -- scripted pick-and-place ---------------------------------------------------------------------


def test_free_spot(params: PrimitiveParams) -> None:
    w = params.workspace
    spot = free_spot([], w, clearance_m=0.08)
    assert spot is not None
    assert spot == pytest.approx((w.x_min + 0.05, w.y_min + 0.05))
    assert free_spot([spot], w, clearance_m=0.08) != spot
    crowded = [(x, y) for x in np.arange(-0.75, -0.25, 0.02) for y in np.arange(-0.35, 0.35, 0.02)]
    assert free_spot(crowded, w, clearance_m=0.08) is None


@pytest.mark.parametrize("seed", range(10))
def test_scripted_pick_and_place_on_dev_seeds(
    backend: KinematicBackend, robot: Robot, params: PrimitiveParams, seed: int
) -> None:
    scene = generate_scene(seed, load_scene_config())
    backend.cubes = list(scene.cubes)
    robot.reset()
    dets = robot.detect()
    assert len(dets) == len(scene.cubes)
    target = dets[0]
    pick = grasp_pose(target)
    others = [(d.position[0], d.position[1]) for d in dets[1:]]
    spot = free_spot(others, params.workspace, clearance_m=0.08)
    assert spot is not None
    place = Pose(x=spot[0], y=spot[1], z=pick.z)
    log = pick_and_place(robot, pick, place)
    assert len(moves(log)) == 6
    held = [c for c in backend.cubes if math.hypot(c.x - place.x, c.y - place.y) < 0.01]
    assert len(held) == 1
    assert held[0].color == target.color
    assert held[0].z == pytest.approx(CUBE / 2)
    after = robot.detect()
    assert len(after) == len(scene.cubes)
    for d in dets[1:]:  # the other cubes did not move
        assert any(math.dist(d.position, a.position) < 2e-3 for a in after)


@settings(max_examples=30, deadline=None)
@given(
    x=st.floats(-0.70, -0.30),
    y=st.floats(-0.30, 0.30),
    z=st.floats(0.015, 0.30),
    yaw=st.floats(-math.pi, math.pi),
    scale=st.floats(0.1, 1.0),
)
def test_any_pose_in_box_is_reached_or_typed_error(
    x: float, y: float, z: float, yaw: float, scale: float
) -> None:
    params = load_primitive_params()
    perception = load_perception_params()
    camera = Camera.from_spec(perception.camera)
    backend = KinematicBackend(params, camera)
    robot = Robot(backend, params=params, perception=perception, camera=camera)
    try:
        res = robot.move_to(Pose(x=x, y=y, z=z, yaw=yaw), scale)
    except PrimitiveError:
        assert backend.follow_log == []
        return
    assert res.position_error_m < 5e-4
    assert res.yaw_error_rad < 1e-3
