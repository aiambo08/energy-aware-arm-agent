"""Render a :class:`~armbench.scene.generator.Scene` as Gazebo (SDF 1.9+) cube models."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

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


def cube_to_sdf_model(cube: Cube, config: SceneConfig) -> str:
    """``<model>`` element for one cube (lines indented for insertion under ``<world>``)."""
    rgba = config.palette[cube.color]
    colour = " ".join(_num(c) for c in rgba)
    size = " ".join([_num(cube.size)] * 3)
    inertia = _num(cube_inertia(config.cube.mass_kg, cube.size))
    mu = _num(config.cube.friction_mu)
    pose = f"{_num(cube.x)} {_num(cube.y)} {_num(cube.z)} 0 0 {_num(cube.yaw)}"
    lines = [
        f'<model name="{cube.name}">',
        f"  <pose>{pose}</pose>",
        *(f"  {line}" for line in _POSE_PUBLISHER),
        '  <link name="link">',
        "    <inertial>",
        f"      <mass>{_num(config.cube.mass_kg)}</mass>",
        f"      <inertia><ixx>{inertia}</ixx><iyy>{inertia}</iyy><izz>{inertia}</izz>"
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
