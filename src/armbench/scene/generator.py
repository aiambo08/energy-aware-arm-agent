"""Seeded tabletop scene generator (pure Python, deterministic, no ROS).

Coordinates are expressed in the robot ``base_link`` frame. The table surface is at
``z = 0`` and the cubes rest on it, so ``z = size / 2``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Final

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

REPO_ROOT: Final = Path(__file__).resolve().parents[3]
DEFAULT_SCENE_FILE: Final = REPO_ROOT / "configs" / "scene.yaml"
MAX_PLACEMENT_FAILURES: Final = 1_000
YAW_HALF_RANGE: Final = np.pi / 4

Rgba = tuple[float, float, float, float]


class SceneGenerationError(RuntimeError):
    """Raised when no collision-free placement is found after the attempt budget."""


class Workspace(BaseModel):
    """Axis-aligned rectangle on the table (``base_link`` frame) where cubes may be placed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    x_min: float
    x_max: float
    y_min: float
    y_max: float

    @model_validator(mode="after")
    def _ordered(self) -> Workspace:
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            msg = "workspace bounds must satisfy x_min < x_max and y_min < y_max"
            raise ValueError(msg)
        return self


class CubeSpec(BaseModel):
    """Physical parameters shared by every cube."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    size_m: float = Field(gt=0)
    mass_kg: float = Field(gt=0)
    min_separation_m: float = Field(gt=0)
    friction_mu: float = Field(ge=0)
    edge_margin_m: float = Field(ge=0)


class CubeCount(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    min: int = Field(ge=1)
    max: int = Field(ge=1)

    @model_validator(mode="after")
    def _ordered(self) -> CubeCount:
        if self.max < self.min:
            msg = f"n_cubes.max ({self.max}) < n_cubes.min ({self.min})"
            raise ValueError(msg)
        return self


class SceneConfig(BaseModel):
    """Contents of ``configs/scene.yaml``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int
    workspace: Workspace
    cube: CubeSpec
    n_cubes: CubeCount
    palette: dict[str, Rgba]

    @field_validator("palette")
    @classmethod
    def _valid_palette(cls, value: dict[str, Rgba]) -> dict[str, Rgba]:
        if not value:
            msg = "palette must contain at least one colour"
            raise ValueError(msg)
        for name, rgba in value.items():
            if any(not 0.0 <= c <= 1.0 for c in rgba):
                msg = f"palette colour {name!r} has components outside [0, 1]: {rgba}"
                raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def _placeable(self) -> SceneConfig:
        ws, cube = self.workspace, self.cube
        if (
            ws.x_max - ws.x_min <= 2 * cube.edge_margin_m
            or ws.y_max - ws.y_min <= 2 * cube.edge_margin_m
        ):
            msg = "edge_margin_m leaves no room inside the workspace"
            raise ValueError(msg)
        return self

    def config_hash(self) -> str:
        """SHA-256 of the canonical JSON dump; stored in every :class:`Scene`."""
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class Cube(BaseModel):
    """One cube resting on the table; ``yaw`` is the rotation about the vertical."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    color: str
    x: float
    y: float
    z: float
    yaw: float
    size: float = Field(gt=0)


class Scene(BaseModel):
    """A generated scene: ``cubes`` are named ``cube_0 … cube_{n-1}`` in generation order."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: int = Field(ge=0)
    cubes: list[Cube]
    config_hash: str

    @field_validator("cubes")
    @classmethod
    def _sequential_names(cls, value: list[Cube]) -> list[Cube]:
        expected = [f"cube_{i}" for i in range(len(value))]
        if [c.name for c in value] != expected:
            msg = f"cube names must be {expected}, got {[c.name for c in value]}"
            raise ValueError(msg)
        return value


def load_scene_config(path: Path = DEFAULT_SCENE_FILE) -> SceneConfig:
    """Load and validate the scene YAML."""
    with path.open("rb") as fh:
        raw = yaml.safe_load(fh)
    return SceneConfig.model_validate(raw)


def generate_scene(seed: int, config: SceneConfig, *, n_cubes: int | None = None) -> Scene:
    """Generate a deterministic scene from ``seed`` using only ``numpy.random.default_rng``.

    The number of cubes is uniform in ``[n_cubes.min, n_cubes.max]`` unless ``n_cubes`` is
    given. Colours are sampled from the palette with replacement; positions are
    rejection-sampled inside the workspace shrunk by ``edge_margin_m`` with centre-to-centre
    distance >= ``min_separation_m``; yaw is uniform in ``[-pi/4, pi/4)``.

    Raises :class:`SceneGenerationError` after ``MAX_PLACEMENT_FAILURES`` rejected samples.
    """
    if seed < 0:
        msg = f"seed must be >= 0, got {seed}"
        raise ValueError(msg)
    rng = np.random.default_rng(seed)
    if n_cubes is None:
        n_cubes = int(rng.integers(config.n_cubes.min, config.n_cubes.max + 1))
    elif n_cubes < 1:
        msg = f"n_cubes must be >= 1, got {n_cubes}"
        raise ValueError(msg)

    ws, spec = config.workspace, config.cube
    x_lo, x_hi = ws.x_min + spec.edge_margin_m, ws.x_max - spec.edge_margin_m
    y_lo, y_hi = ws.y_min + spec.edge_margin_m, ws.y_max - spec.edge_margin_m
    colours = list(config.palette)
    placed: list[tuple[float, float]] = []
    cubes: list[Cube] = []
    failures = 0
    for i in range(n_cubes):
        colour = colours[int(rng.integers(len(colours)))]
        while True:
            x = float(rng.uniform(x_lo, x_hi))
            y = float(rng.uniform(y_lo, y_hi))
            if all(np.hypot(x - px, y - py) >= spec.min_separation_m for px, py in placed):
                break
            failures += 1
            if failures >= MAX_PLACEMENT_FAILURES:
                msg = (
                    f"seed {seed}: could not place cube {i} of {n_cubes} after "
                    f"{MAX_PLACEMENT_FAILURES} rejected samples"
                )
                raise SceneGenerationError(msg)
        yaw = float(rng.uniform(-YAW_HALF_RANGE, YAW_HALF_RANGE))
        placed.append((x, y))
        cubes.append(
            Cube(
                name=f"cube_{i}",
                color=colour,
                x=x,
                y=y,
                z=spec.size_m / 2,
                yaw=yaw,
                size=spec.size_m,
            )
        )
    return Scene(seed=seed, cubes=cubes, config_hash=config.config_hash())


def scene_to_json(scene: Scene) -> str:
    """Serialise a scene (stored per episode by the runner)."""
    return scene.model_dump_json(indent=2)


def scene_from_json(text: str) -> Scene:
    """Inverse of :func:`scene_to_json`."""
    return Scene.model_validate_json(text)
