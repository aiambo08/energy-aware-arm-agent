"""Render a :class:`~armbench.scene.generator.Scene` as Gazebo (SDF 1.9+) cube models."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, field_validator

from armbench.scene.generator import Cube, Scene, SceneConfig

SCENE_MARKER: Final = "<!-- ARMBENCH_SCENE -->"
_MARKER_LINE: Final = re.compile(r"^[ \t]*<!-- ARMBENCH_SCENE -->[ \t]*$", re.MULTILINE)
_INDENT: Final = "    "

# Per-model PosePublisher: the ROS bridge reads ground-truth cube poses from
# /model/cube_<i>/pose (the Pose_V -> TFMessage bridge loses entity names).
_POSE_PUBLISHER: Final = (
    '<plugin filename="gz-sim-pose-publisher-system" name="gz::sim::systems::PosePublisher">',
    "  <publish_link_pose>false</publish_link_pose>",
    "  <publish_model_pose>true</publish_model_pose>",
    "  <use_pose_vector_msg>false</use_pose_vector_msg>",
    "  <update_frequency>50</update_frequency>",
    "</plugin>",
)
_CONTACT: Final = (
    "<contact><ode><kp>1000000.0</kp><kd>100.0</kd><min_depth>0.001</min_depth></ode></contact>"
)


def _num(value: float) -> str:
    return f"{value:.6g}"


def cube_inertia(mass_kg: float, size_m: float) -> float:
    """Moment of inertia of a solid cube about any axis through its centre: ``m s^2 / 6``."""
    return mass_kg * size_m**2 / 6.0


class Box(BaseModel):
    """Any axis-aligned-at-rest box model (cubes, task obstacles): pose, size, mass, colour."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    x: float
    y: float
    z: float
    yaw: float
    size: tuple[float, float, float]
    mass_kg: float = Field(gt=0)
    rgba: tuple[float, float, float, float]
    friction_mu: float = Field(ge=0)

    @field_validator("size")
    @classmethod
    def _positive(cls, value: tuple[float, float, float]) -> tuple[float, float, float]:
        if any(s <= 0 for s in value):
            msg = f"box size must be positive, got {value}"
            raise ValueError(msg)
        return value


def box_inertia(mass_kg: float, size: tuple[float, float, float]) -> tuple[float, float, float]:
    """Principal moments of a solid box: ``m (b^2 + c^2) / 12`` and permutations."""
    a, b, c = size
    return (
        mass_kg * (b**2 + c**2) / 12.0,
        mass_kg * (a**2 + c**2) / 12.0,
        mass_kg * (a**2 + b**2) / 12.0,
    )


def box_to_sdf_model(box: Box) -> str:
    """``<model>`` element for one box (lines indented for insertion under ``<world>``)."""
    colour = " ".join(_num(c) for c in box.rgba)
    size = " ".join(_num(s) for s in box.size)
    ixx, iyy, izz = (_num(v) for v in box_inertia(box.mass_kg, box.size))
    mu = _num(box.friction_mu)
    pose = f"{_num(box.x)} {_num(box.y)} {_num(box.z)} 0 0 {_num(box.yaw)}"
    lines = [
        f'<model name="{box.name}">',
        f"  <pose>{pose}</pose>",
        *(f"  {line}" for line in _POSE_PUBLISHER),
        '  <link name="link">',
        "    <inertial>",
        f"      <mass>{_num(box.mass_kg)}</mass>",
        f"      <inertia><ixx>{ixx}</ixx><iyy>{iyy}</iyy><izz>{izz}</izz>"
        "<ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>",
        "    </inertial>",
        '    <collision name="collision">',
        f"      <geometry><box><size>{size}</size></box></geometry>",
        "      <surface>",
        f"        <friction><ode><mu>{mu}</mu><mu2>{mu}</mu2></ode></friction>",
        f"        {_CONTACT}",
        "      </surface>",
        "    </collision>",
        '    <visual name="visual">',
        f"      <geometry><box><size>{size}</size></box></geometry>",
        f"      <material><ambient>{colour}</ambient><diffuse>{colour}</diffuse></material>",
        "    </visual>",
        "  </link>",
        "</model>",
    ]
    return "\n".join(_INDENT + line for line in lines)


def cube_to_box(cube: Cube, config: SceneConfig) -> Box:
    return Box(
        name=cube.name,
        x=cube.x,
        y=cube.y,
        z=cube.z,
        yaw=cube.yaw,
        size=(cube.size, cube.size, cube.size),
        mass_kg=config.cube.mass_kg,
        rgba=config.palette[cube.color],
        friction_mu=config.cube.friction_mu,
    )


def cube_to_sdf_model(cube: Cube, config: SceneConfig) -> str:
    """``<model>`` element for one cube (lines indented for insertion under ``<world>``)."""
    return box_to_sdf_model(cube_to_box(cube, config))


def scene_to_sdf_models(scene: Scene, config: SceneConfig) -> str:
    """SDF fragment with one ``<model name="cube_i">`` per cube (no ``<sdf>``/``<world>``)."""
    return "\n".join(cube_to_sdf_model(cube, config) for cube in scene.cubes)


def scene_to_sdf_world(scene: Scene, config: SceneConfig, template_path: Path) -> str:
    """Read a world template and replace the ``<!-- ARMBENCH_SCENE -->`` line with the models.

    Raises :class:`ValueError` if the marker line is missing from the template.
    """
    template = template_path.read_text(encoding="utf-8")
    if _MARKER_LINE.search(template) is None:
        msg = f"marker line {SCENE_MARKER!r} not found in {template_path}"
        raise ValueError(msg)
    fragment = scene_to_sdf_models(scene, config)
    return _MARKER_LINE.sub(lambda _: fragment, template, count=1)
