"""Perception: pinhole/extrinsics identities, detect() on synthetic RGB-D renders, contracts."""

from __future__ import annotations

import math
import re
import time

import numpy as np
import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from armbench.paths import CONFIG_DIR
from armbench.perception import (
    ANY_TARGET,
    OPTICAL_IN_SENSOR,
    Camera,
    Extrinsics,
    Intrinsics,
    PerceptionParams,
    UnknownTargetError,
    detect,
    load_perception_params,
    yaw_difference,
)
from armbench.perception.synthetic import (
    ARM_RGB,
    render,
)
from armbench.scene import Cube, generate_scene, load_scene_config


@pytest.fixture(scope="module")
def params() -> PerceptionParams:
    return load_perception_params()


@pytest.fixture(scope="module")
def camera(params: PerceptionParams) -> Camera:
    return Camera.from_spec(params.camera)


def make_cube(
    name: str, color: str, x: float, y: float, *, yaw: float = 0.0, size: float = 0.045
) -> Cube:
    return Cube(name=name, color=color, x=x, y=y, z=size / 2.0, size=size, yaw=yaw)


# -- config ------------------------------------------------------------------------------------


def test_config_matches_world_and_scene(params: PerceptionParams) -> None:
    scene_cfg = load_scene_config()
    assert params.cube_size_m == scene_cfg.cube.size_m
    assert set(params.segmentation.colours) == set(scene_cfg.palette)
    world = (
        CONFIG_DIR.parent / "ros_ws/src/armbench_description/worlds/tabletop.sdf.in"
    ).read_text()
    cam_model = world[world.index('<model name="rgbd_camera">') :]
    pose_text = re.search(r"<pose>([^<]+)</pose>", cam_model)
    assert pose_text is not None
    pose = tuple(float(v) for v in pose_text.group(1).split())
    assert pose == pytest.approx((*params.camera.xyz, *params.camera.rpy), abs=1e-7)
    assert f"<horizontal_fov>{params.camera.horizontal_fov_rad:g}</horizontal_fov>" in world
    assert f"<width>{params.camera.width}</width>" in world
    assert f"<height>{params.camera.height}</height>" in world


def test_config_rejects_bad_hue_range(params: PerceptionParams) -> None:
    raw = yaml.safe_load((CONFIG_DIR / "perception.yaml").read_text())
    raw["segmentation"]["hue_ranges"]["red"] = [[170, 190]]
    with pytest.raises(ValueError, match="hue_ranges"):
        PerceptionParams.model_validate(raw)
    raw["segmentation"]["hue_ranges"]["red"] = [[0, 8]]
    raw["depth"]["max_m"] = raw["depth"]["min_m"]
    with pytest.raises(ValueError, match=r"depth\.max_m"):
        PerceptionParams.model_validate(raw)


# -- camera model --------------------------------------------------------------------------------


def test_intrinsics_match_gazebo_camera_info(params: PerceptionParams) -> None:
    intr = Intrinsics.from_fov(640, 480, 1.0472)
    assert intr.fx == pytest.approx(554.2546911911869, rel=1e-9)
    assert (intr.cx, intr.cy) == (320.0, 240.0)
    k = (554.2546911911869, 0.0, 320.0, 0.0, 554.2546911911869, 240.0, 0.0, 0.0, 1.0)
    from_k = Intrinsics.from_k(k, 640, 480)
    assert (from_k.fx, from_k.fy, from_k.cx, from_k.cy) == pytest.approx(
        (intr.fx, intr.fy, intr.cx, intr.cy), rel=1e-12
    )
    with pytest.raises(ValueError, match="9 entries"):
        Intrinsics.from_k(k[:8], 640, 480)


def test_extrinsics_match_measured_gazebo_pixels(camera: Camera) -> None:
    """Pixels measured in the simulator for cubes at known poses (top face at z = 0.045)."""
    measured = {
        (-0.5, 0.0): (320.0, 240.0),
        (-0.4, -0.15): (407.0, 182.0),
        (-0.6, 0.1): (262.0, 298.0),
    }
    for (x, y), (u, v) in measured.items():
        px, d = camera.project_base(np.array([[x, y, 0.045]]))
        assert d[0] == pytest.approx(0.955, abs=1e-6)
        assert px[0] == pytest.approx((u, v), abs=1.0)


def test_optical_axes_are_right_handed() -> None:
    assert np.allclose(OPTICAL_IN_SENSOR @ OPTICAL_IN_SENSOR.T, np.eye(3))
    assert np.linalg.det(OPTICAL_IN_SENSOR) == pytest.approx(1.0)
    ext = Extrinsics((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    # a point 1 m ahead of an un-rotated sensor is 1 m along optical z
    assert np.allclose(ext.to_optical(np.array([[1.0, 0.0, 0.0]])), [[0.0, 0.0, 1.0]])


@settings(max_examples=300, deadline=None)
@given(
    x=st.floats(-0.9, -0.1),
    y=st.floats(-0.5, 0.5),
    z=st.floats(0.0, 0.5),
)
def test_project_backproject_roundtrip(x: float, y: float, z: float) -> None:
    cam = Camera.from_spec(load_perception_params().camera)
    p = np.array([[x, y, z]])
    px, d = cam.project_base(p)
    back = cam.pixel_to_base(px, d)
    assert np.allclose(back, p, atol=1e-9)


def test_project_rejects_points_behind_camera(camera: Camera) -> None:
    with pytest.raises(ValueError, match="behind"):
        camera.project_base(np.array([[-0.5, 0.0, 1.5]]))


def test_yaw_difference_wraps_quarter_turn() -> None:
    q = math.pi / 2
    assert yaw_difference(0.3, 0.3) == 0.0
    assert yaw_difference(-math.pi / 4, math.pi / 4 - 0.01) == pytest.approx(0.01)
    assert yaw_difference(0.2 + q, 0.2) == pytest.approx(0.0, abs=1e-12)
    assert -math.pi / 4 <= yaw_difference(1.0, -1.0) < math.pi / 4


# -- detect() ------------------------------------------------------------------------------------


def test_detect_recovers_pose_and_yaw(params: PerceptionParams, camera: Camera) -> None:
    cubes = [
        make_cube("cube_0", "red", -0.5, 0.0),
        make_cube("cube_1", "green", -0.6, 0.1, yaw=0.0),
        make_cube("cube_2", "blue", -0.4, -0.15, yaw=0.3),
        make_cube("cube_3", "yellow", -0.65, -0.2, yaw=0.5),
        make_cube("cube_4", "red", -0.33, 0.22, yaw=-0.6),
    ]
    rgb, depth = render(cubes, camera)
    res = detect(rgb, depth, params=params, camera=camera)
    assert len(res.detections) == len(cubes)
    for cube in cubes:
        same = [d for d in res.detections if d.color == cube.color]
        d = min(same, key=lambda d: math.hypot(d.position[0] - cube.x, d.position[1] - cube.y))
        assert math.hypot(d.position[0] - cube.x, d.position[1] - cube.y) < 0.003
        assert d.position[2] == pytest.approx(cube.z, abs=0.002)
        assert d.top_center[2] == pytest.approx(cube.z + cube.size / 2, abs=0.002)
        assert abs(yaw_difference(d.yaw_rad, cube.yaw)) < math.radians(2.0)
        assert d.area_px > params.segmentation.min_area_px


def test_detect_target_filter_and_sorting(params: PerceptionParams, camera: Camera) -> None:
    cubes = [
        make_cube("cube_0", "red", -0.45, 0.1),
        make_cube("cube_1", "red", -0.6, -0.1),
        make_cube("cube_2", "blue", -0.35, -0.2),
    ]
    rgb, depth = render(cubes, camera)
    reds = detect(rgb, depth, "red", params=params, camera=camera).detections
    assert [d.color for d in reds] == ["red", "red"]
    assert reds[0].position[0] < reds[1].position[0]
    assert len(detect(rgb, depth, "blue", params=params, camera=camera).detections) == 1
    assert detect(rgb, depth, "green", params=params, camera=camera).detections == ()
    assert len(detect(rgb, depth, ANY_TARGET, params=params, camera=camera).detections) == 3


def test_detect_empty_scene_returns_empty_list(params: PerceptionParams, camera: Camera) -> None:
    rgb, depth = render([], camera)
    res = detect(rgb, depth, params=params, camera=camera)
    assert res.detections == ()
    assert res.latency_ms >= 0.0


def test_detect_ignores_grey_arm_and_dark_blobs(params: PerceptionParams, camera: Camera) -> None:
    rgb, depth = render([make_cube("cube_0", "red", -0.5, 0.0)], camera)
    rgb[0:40, 280:360] = ARM_RGB  # the robot shoulder at the image top
    depth[0:40, 280:360] = 0.99
    rgb[300:360, 50:110] = (20, 20, 20)  # a black blob on the table
    res = detect(rgb, depth, params=params, camera=camera)
    assert [d.color for d in res.detections] == ["red"]


def test_detect_unknown_target_raises(params: PerceptionParams, camera: Camera) -> None:
    rgb, depth = render([], camera)
    with pytest.raises(UnknownTargetError, match="purple"):
        detect(rgb, depth, "purple", params=params, camera=camera)


def test_detect_rejects_malformed_frames(params: PerceptionParams, camera: Camera) -> None:
    rgb, depth = render([], camera)
    with pytest.raises(ValueError, match="rgb must be"):
        detect(rgb[:, :, :1], depth, params=params, camera=camera)
    with pytest.raises(ValueError, match="rgb must be"):
        detect(rgb.astype(np.float32), depth, params=params, camera=camera)
    with pytest.raises(ValueError, match="depth must be"):
        detect(rgb, depth[1:], params=params, camera=camera)
    with pytest.raises(ValueError, match="depth must be"):
        detect(rgb, (depth * 1000).astype(np.uint16), params=params, camera=camera)


def test_detect_skips_invalid_depth(params: PerceptionParams, camera: Camera) -> None:
    cubes = [make_cube("cube_0", "red", -0.5, 0.0), make_cube("cube_1", "blue", -0.6, 0.15)]
    rgb, depth = render(cubes, camera)
    px, _ = camera.project_base(np.array([[-0.6, 0.15, 0.045]]))
    u, v = (round(c) for c in px[0])
    depth[v - 30 : v + 30, u - 30 : u + 30] = np.inf  # blue cube has no valid depth
    res = detect(rgb, depth, params=params, camera=camera)
    assert [d.color for d in res.detections] == ["red"]
    depth[v - 30 : v + 30, u - 30 : u + 30] = np.nan
    res = detect(rgb, depth, params=params, camera=camera)
    assert [d.color for d in res.detections] == ["red"]


def test_detect_side_faces_do_not_bias_centre(params: PerceptionParams, camera: Camera) -> None:
    """Far from the optical axis the side faces are visible; only the top face must be used."""
    cube = make_cube("cube_0", "yellow", -0.85, 0.4, yaw=0.2)
    rgb, depth = render([cube], camera, side_faces=True)
    rgb_top, depth_top = render([cube], camera, side_faces=False)
    d_side = detect(rgb, depth, params=params, camera=camera).detections[0]
    d_top = detect(rgb_top, depth_top, params=params, camera=camera).detections[0]
    assert math.hypot(d_side.position[0] - cube.x, d_side.position[1] - cube.y) < 0.003
    assert (
        math.hypot(d_side.position[0] - d_top.position[0], d_side.position[1] - d_top.position[1])
        < 0.002
    )


def test_detect_partially_out_of_frame_cube_is_not_hallucinated(
    params: PerceptionParams, camera: Camera
) -> None:
    rgb, depth = render([make_cube("cube_0", "green", -0.5, 0.57)], camera)  # at the image edge
    res = detect(rgb, depth, params=params, camera=camera)
    for d in res.detections:  # either nothing or a detection whose pixels really are green
        assert d.color == "green"
        assert d.bbox[0] == 0


def test_detect_latency_under_budget(params: PerceptionParams, camera: Camera) -> None:
    scene = generate_scene(7, load_scene_config(), n_cubes=6)
    rgb, depth = render(scene.cubes, camera, noise_sigma=3.0)
    detect(rgb, depth, params=params, camera=camera)  # warm-up
    t0 = time.perf_counter()
    n = 10
    for _ in range(n):
        res = detect(rgb, depth, params=params, camera=camera)
    assert (time.perf_counter() - t0) / n * 1e3 < 50.0
    assert len(res.detections) == 6


@settings(max_examples=60, deadline=None)
@given(seed=st.integers(0, 100_000), noise=st.floats(0.0, 6.0))
def test_detect_every_seeded_scene(seed: int, noise: float) -> None:
    """Every generated scene (3-6 cubes, random yaw) is fully recovered, colour by colour."""
    params = load_perception_params()
    camera = Camera.from_spec(params.camera)
    scene = generate_scene(seed, load_scene_config())
    rgb, depth = render(scene.cubes, camera, noise_sigma=noise, seed=seed)
    dets = detect(rgb, depth, params=params, camera=camera).detections
    assert len(dets) == len(scene.cubes)
    for cube in scene.cubes:
        same = [d for d in dets if d.color == cube.color]
        d = min(same, key=lambda d: math.hypot(d.position[0] - cube.x, d.position[1] - cube.y))
        assert math.hypot(d.position[0] - cube.x, d.position[1] - cube.y) < 0.004
        assert abs(d.position[2] - cube.z) < 0.002
        assert abs(yaw_difference(d.yaw_rad, cube.yaw)) < math.radians(5.0)  # ~28 px masks


def test_detect_flags_top_face_occluded_by_the_arm_as_partial(
    params: PerceptionParams, camera: Camera
) -> None:
    cube = make_cube("cube_0", "green", -0.34, -0.12, yaw=-0.32)
    rgb, depth = render([cube], camera)
    (full,) = detect(rgb, depth, params=params, camera=camera).detections
    assert full.complete
    assert 0.85 <= full.top_face_fraction <= 1.15

    px, _ = camera.project_base(np.array([[cube.x, cube.y, cube.size]]))
    ui, vi = round(float(px[0, 0])), round(float(px[0, 1]))
    rgb[vi - 40 : vi + 40, ui - 8 : ui + 40] = ARM_RGB  # forearm over ~70 % of the face
    depth[vi - 40 : vi + 40, ui - 8 : ui + 40] = 0.6
    (part,) = detect(rgb, depth, params=params, camera=camera).detections
    assert not part.complete
    assert part.top_face_fraction < params.segmentation.min_top_face_fraction
    # the visible sliver's centroid is pulled towards it: this is the bias a caller must not trust
    assert math.hypot(part.position[0] - cube.x, part.position[1] - cube.y) > 0.01
    assert math.hypot(full.position[0] - cube.x, full.position[1] - cube.y) < 0.003
