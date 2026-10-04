"""Energy model (phase F2): variants A/B, Joule losses, base power, sensitivity."""

from armbench.energy.log import episode_from_jsonl, read_samples_jsonl, write_samples_jsonl
from armbench.energy.model import (
    J_PER_WH,
    EnergyBreakdown,
    EnergyMeter,
    Variant,
    copper_power,
    electrical_power,
    episode_energy,
    integrate,
    mechanical_power,
    sensitivity,
)
from armbench.energy.params import (
    DEFAULT_ENERGY_FILE,
    EnergyParams,
    JointElectricalParams,
    load_energy_params,
)
from armbench.energy.reference import (
    REFERENCE_FILE,
    EnergyReference,
    EnergyReferenceError,
    ReferenceSample,
    ReferenceSource,
    TaskReference,
)

__all__ = [
    "DEFAULT_ENERGY_FILE",
    "J_PER_WH",
    "REFERENCE_FILE",
    "EnergyBreakdown",
    "EnergyMeter",
    "EnergyParams",
    "EnergyReference",
    "EnergyReferenceError",
    "JointElectricalParams",
    "ReferenceSample",
    "ReferenceSource",
    "TaskReference",
    "Variant",
    "copper_power",
    "electrical_power",
    "episode_energy",
    "episode_from_jsonl",
    "integrate",
    "load_energy_params",
    "mechanical_power",
    "read_samples_jsonl",
    "sensitivity",
    "write_samples_jsonl",
]
