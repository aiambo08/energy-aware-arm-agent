import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from armbench.scene import (
    DEFAULT_SCENE_FILE,
    MAX_PLACEMENT_FAILURES,
    SCENE_MARKER,
    Cube,
    Scene,
    SceneGenerationError,
    cube_inertia,
    generate_scene,
    load_scene_config,
    scene_from_json,
    scene_to_json,
    scene_to_sdf_models,
    scene_to_sdf_world,
)

CONFIG = load_scene_config(DEFAULT_SCENE_FILE)
seeds = st.integers(min_value=0, max_value=10_000)


def parse_xml(text: str) -> ET.Element:
    return ET.fromstring(text)  # noqa: S314 - our own generated XML


# --- config -------------------------------------------------------------------------------


def test_default_config_values() -> None:
    assert CONFIG.schema_version == 1
    assert (CONFIG.workspace.x_min, CONFIG.workspace.x_max) == (-0.70, -0.30)
    assert (CONFIG.workspace.y_min, CONFIG.workspace.y_max) == (-0.25, 0.25)
    assert CONFIG.cube.size_m == 0.045
    assert CONFIG.cube.min_separation_m == 0.09
    assert (CONFIG.n_cubes.min, CONFIG.n_cubes.max) == (3, 6)
    assert set(CONFIG.palette) == {"red", "green", "blue", "yellow"}
    assert CONFIG.palette["red"] == (0.85, 0.10, 0.10, 1.0)


def test_config_hash_is_stable_and_sensitive() -> None:
    assert CONFIG.config_hash() == load_scene_config(DEFAULT_SCENE_FILE).config_hash()
    assert len(CONFIG.config_hash()) == 64
    other = CONFIG.model_copy(update={"n_cubes": CONFIG.n_cubes.model_copy(update={"max": 5})})
    assert other.config_hash() != CONFIG.config_hash()


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda r: r.__setitem__("extra", 1), "extra"),
        (lambda r: r["workspace"].__setitem__("x_max", -0.80), "x_min < x_max"),
        (lambda r: r["n_cubes"].__setitem__("max", 1), "n_cubes.max"),
        (lambda r: r["cube"].__setitem__("edge_margin_m", 0.25), "no room"),
        (lambda r: r["palette"].__setitem__("red", [2.0, 0.0, 0.0, 1.0]), "outside"),
        (lambda r: r.__setitem__("palette", {}), "at least one"),
    ],
)
def test_invalid_config_rejected(tmp_path: Path, mutate: object, match: str) -> None:
    raw = yaml.safe_load(DEFAULT_SCENE_FILE.read_text())
    mutate(raw)  # type: ignore[operator]
    p = tmp_path / "scene.yaml"
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match=match):
        load_scene_config(p)


# --- generator ----------------------------------------------------------------------------


def test_same_seed_same_scene() -> None:
    assert generate_scene(7, CONFIG) == generate_scene(7, CONFIG)
    assert scene_to_json(generate_scene(7, CONFIG)) == scene_to_json(generate_scene(7, CONFIG))


def test_global_rng_is_not_used() -> None:
    np.random.seed(0)
    first = generate_scene(3, CONFIG)
    np.random.seed(123)
    assert generate_scene(3, CONFIG) == first
    state_before = np.random.get_state()[1]
    generate_scene(4, CONFIG)
    assert np.array_equal(np.random.get_state()[1], state_before)


@given(a=seeds, b=seeds)
def test_different_seeds_different_scenes(a: int, b: int) -> None:
    sa, sb = generate_scene(a, CONFIG), generate_scene(b, CONFIG)
    if a == b:
        assert sa == sb
    else:
        assert [(c.x, c.y) for c in sa.cubes] != [(c.x, c.y) for c in sb.cubes]


@settings(max_examples=300, deadline=None)
@given(seed=seeds)
def test_generated_scene_respects_constraints(seed: int) -> None:
    scene = generate_scene(seed, CONFIG)
    ws, spec = CONFIG.workspace, CONFIG.cube
    assert scene.seed == seed
    assert scene.config_hash == CONFIG.config_hash()
    assert CONFIG.n_cubes.min <= len(scene.cubes) <= CONFIG.n_cubes.max
    assert [c.name for c in scene.cubes] == [f"cube_{i}" for i in range(len(scene.cubes))]
    half = spec.size_m / 2
    for cube in scene.cubes:
        assert cube.color in CONFIG.palette
        assert cube.size == spec.size_m
        assert cube.z == pytest.approx(half)
        assert ws.x_min + spec.edge_margin_m <= cube.x <= ws.x_max - spec.edge_margin_m
        assert ws.y_min + spec.edge_margin_m <= cube.y <= ws.y_max - spec.edge_margin_m
        assert ws.x_min <= cube.x - half
        assert cube.x + half <= ws.x_max
        assert ws.y_min <= cube.y - half
        assert cube.y + half <= ws.y_max
        assert -np.pi / 4 <= cube.yaw < np.pi / 4
    for i, a in enumerate(scene.cubes):
        for b in scene.cubes[i + 1 :]:
            assert np.hypot(a.x - b.x, a.y - b.y) >= spec.min_separation_m


def test_n_cubes_override() -> None:
    assert len(generate_scene(0, CONFIG, n_cubes=1).cubes) == 1
    assert len(generate_scene(0, CONFIG, n_cubes=8).cubes) == 8
    with pytest.raises(ValueError, match="n_cubes"):
        generate_scene(0, CONFIG, n_cubes=0)
    with pytest.raises(ValueError, match="seed"):
        generate_scene(-1, CONFIG)


def test_cube_count_distribution_covers_range() -> None:
    counts = {len(generate_scene(seed, CONFIG).cubes) for seed in range(200)}
    assert counts == set(range(CONFIG.n_cubes.min, CONFIG.n_cubes.max + 1))


def test_all_colours_appear_across_seeds() -> None:
    colours = {c.color for seed in range(50) for c in generate_scene(seed, CONFIG).cubes}
    assert colours == set(CONFIG.palette)


def test_too_many_cubes_raise_generation_error() -> None:
    with pytest.raises(SceneGenerationError, match=str(MAX_PLACEMENT_FAILURES)):
        generate_scene(0, CONFIG, n_cubes=200)


def test_scene_names_must_be_sequential() -> None:
    cube = Cube(name="cube_1", color="red", x=-0.5, y=0.0, z=0.0225, yaw=0.0, size=0.045)
    with pytest.raises(ValueError, match="cube names"):
        Scene(seed=0, cubes=[cube], config_hash="x")


def test_json_round_trip() -> None:
    scene = generate_scene(42, CONFIG)
    text = scene_to_json(scene)
    assert json.loads(text)["seed"] == 42
    assert scene_from_json(text) == scene


# --- SDF ----------------------------------------------------------------------------------


def test_cube_inertia_matches_fixture() -> None:
    assert cube_inertia(0.05, 0.045) == pytest.approx(1.6875e-05)


def fixture_scene() -> Scene:
    cube = Cube(name="cube_0", color="red", x=-0.50, y=0.00, z=0.0225, yaw=0.0, size=0.045)
    return Scene(seed=0, cubes=[cube], config_hash=CONFIG.config_hash())


def test_sdf_model_matches_check_scene_fixture_structure() -> None:
    fragment = scene_to_sdf_models(fixture_scene(), CONFIG)
    assert not fragment.lstrip().startswith("<sdf")
    model = parse_xml(fragment)
    assert model.tag == "model"
    assert model.get("name") == "cube_0"
    children = [child.tag for child in model]
    assert children == ["pose", "plugin", "link"]
    assert model.findtext("pose", "").split() == ["-0.5", "0", "0.0225", "0", "0", "0"]
    plugin = model.find("plugin")
    assert plugin is not None
    assert plugin.get("filename") == "gz-sim-pose-publisher-system"
    assert plugin.get("name") == "gz::sim::systems::PosePublisher"
    assert plugin.findtext("publish_link_pose") == "false"
    assert plugin.findtext("publish_model_pose") == "true"
    assert plugin.findtext("use_pose_vector_msg") == "false"
    assert plugin.findtext("update_frequency") == "50"
    link = model.find("link")
    assert link is not None
    assert link.get("name") == "link"
    assert float(link.findtext("inertial/mass", "")) == 0.05
    for axis in ("ixx", "iyy", "izz"):
        assert float(link.findtext(f"inertial/inertia/{axis}", "")) == pytest.approx(1.6875e-05)
    for axis in ("ixy", "ixz", "iyz"):
        assert float(link.findtext(f"inertial/inertia/{axis}", "")) == 0.0
    assert link.findtext("collision/geometry/box/size", "").split() == ["0.045"] * 3
    assert link.findtext("visual/geometry/box/size", "").split() == ["0.045"] * 3
    assert float(link.findtext("collision/surface/friction/ode/mu", "")) == 1.0
    assert float(link.findtext("collision/surface/friction/ode/mu2", "")) == 1.0
    contact = link.find("collision/surface/contact/ode")
    assert contact is not None
    assert float(contact.findtext("kp", "")) == 1_000_000.0
    assert float(contact.findtext("kd", "")) == 100.0
    assert float(contact.findtext("min_depth", "")) == 0.001
    rgba = [float(v) for v in link.findtext("visual/material/ambient", "").split()]
    assert rgba == pytest.approx([0.85, 0.10, 0.10, 1.0])
    assert link.findtext("visual/material/diffuse") == link.findtext("visual/material/ambient")


@settings(max_examples=50, deadline=None)
@given(seed=seeds)
def test_sdf_models_are_valid_xml_for_every_cube(seed: int) -> None:
    scene = generate_scene(seed, CONFIG)
    fragment = scene_to_sdf_models(scene, CONFIG)
    root = parse_xml(f"<world>{fragment}</world>")
    models = root.findall("model")
    assert [m.get("name") for m in models] == [c.name for c in scene.cubes]
    for model, cube in zip(models, scene.cubes, strict=True):
        pose = [float(v) for v in model.findtext("pose", "").split()]
        assert pose[:3] == pytest.approx([cube.x, cube.y, cube.z], abs=1e-5)
        assert pose[3:5] == [0.0, 0.0]
        assert pose[5] == pytest.approx(cube.yaw, abs=1e-5)
        rgba = [float(v) for v in model.findtext("link/visual/material/diffuse", "").split()]
        assert rgba == pytest.approx(list(CONFIG.palette[cube.color]))


def test_sdf_world_replaces_marker_line(tmp_path: Path) -> None:
    template = tmp_path / "tabletop.sdf.in"
    template.write_text(
        '<?xml version="1.0"?>\n<sdf version="1.9">\n  <world name="t">\n'
        f"    {SCENE_MARKER}\n  </world>\n</sdf>\n",
        encoding="utf-8",
    )
    scene = generate_scene(5, CONFIG)
    world = scene_to_sdf_world(scene, CONFIG, template)
    assert SCENE_MARKER not in world
    root = parse_xml(world)
    names = [m.get("name") for m in root.iter("model")]
    assert names == [c.name for c in scene.cubes]
    assert world.startswith('<?xml version="1.0"?>')


def test_sdf_world_without_marker_raises(tmp_path: Path) -> None:
    template = tmp_path / "bad.sdf.in"
    template.write_text('<sdf version="1.9"><world name="t"></world></sdf>\n', encoding="utf-8")
    with pytest.raises(ValueError, match="ARMBENCH_SCENE"):
        scene_to_sdf_world(generate_scene(0, CONFIG), CONFIG, template)


def test_sdf_world_with_real_template_if_present() -> None:
    template = (
        Path(__file__).resolve().parents[2]
        / "ros_ws/src/armbench_description/worlds/tabletop.sdf.in"
    )
    if not template.exists():
        pytest.skip("world template not in this checkout")
    world = scene_to_sdf_world(generate_scene(1, CONFIG), CONFIG, template)
    root = parse_xml(world)
    assert any(m.get("name") == "cube_0" for m in root.iter("model"))
