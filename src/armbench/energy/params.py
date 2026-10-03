"""Energy model parameters (``configs/energy.yaml``)."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

REPO_ROOT: Final = Path(__file__).resolve().parents[3]
DEFAULT_ENERGY_FILE: Final = REPO_ROOT / "configs" / "energy.yaml"


class JointElectricalParams(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    torque_constant_nm_per_a: float = Field(gt=0, description="joint-level torque constant")
    winding_resistance_ohm: float = Field(ge=0)


class EnergyParams(BaseModel):
    """Everything the model needs; ``eta`` excludes copper losses (ADR-004)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int
    eta: float = Field(gt=0, le=1)
    eta_sensitivity: tuple[float, ...] = ()
    p0_w: float = Field(ge=0)
    joint_order: tuple[str, ...]
    joints: dict[str, JointElectricalParams]

    @model_validator(mode="after")
    def _consistent(self) -> EnergyParams:
        if len(self.joint_order) != len(set(self.joint_order)):
            msg = "joint_order has duplicates"
            raise ValueError(msg)
        missing = set(self.joint_order) - set(self.joints)
        extra = set(self.joints) - set(self.joint_order)
        if missing or extra:
            msg = f"joints/joint_order mismatch: missing={sorted(missing)} extra={sorted(extra)}"
            raise ValueError(msg)
        if any(not 0 < e <= 1 for e in self.eta_sensitivity):
            msg = "eta_sensitivity values must be in (0, 1]"
            raise ValueError(msg)
        return self

    @property
    def n_joints(self) -> int:
        return len(self.joint_order)

    def torque_constants(self) -> np.ndarray:
        return np.array([self.joints[j].torque_constant_nm_per_a for j in self.joint_order])

    def winding_resistances(self) -> np.ndarray:
        return np.array([self.joints[j].winding_resistance_ohm for j in self.joint_order])

    def etas(self) -> tuple[float, ...]:
        """Nominal eta first, then the sensitivity values (deduplicated, order kept)."""
        seen: list[float] = []
        for e in (self.eta, *self.eta_sensitivity):
            if e not in seen:
                seen.append(e)
        return tuple(seen)


def load_energy_params(path: Path = DEFAULT_ENERGY_FILE) -> EnergyParams:
    with path.open("rb") as fh:
        raw = yaml.safe_load(fh)
    return EnergyParams.model_validate(raw)
