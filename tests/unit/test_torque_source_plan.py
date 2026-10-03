"""The F2 torque-source plan must be seeded, deterministic and collision-safe by construction."""

import importlib.util
import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from armbench.kinematics import UR5eModel

_SPEC = importlib.util.spec_from_file_location(
    "torque_source", Path(__file__).resolve().parents[2] / "scripts" / "torque_source.py"
)
assert _SPEC is not None
assert _SPEC.loader is not None
torque_source = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(torque_source)


def test_plan_is_deterministic_and_sized() -> None:
    a = torque_source.make_plan(0, 20, 10)
    b = torque_source.make_plan(0, 20, 10)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert len(a["trajectories"]) == 20
    assert a["repeat"] == {"trajectory": "traj_00", "n": 10}
    assert json.dumps(torque_source.make_plan(1, 20, 10)) != json.dumps(a)


def test_every_waypoint_is_safe_and_times_increase() -> None:
    model = UR5eModel()
    plan = torque_source.make_plan(3, 10, 2)
    for hold in plan["static_holds"]:
        assert torque_source.is_safe(model, np.asarray(hold["q"]))
    for traj in plan["trajectories"]:
        times = [p["t"] for p in traj["points"]]
        assert times[0] == 0.0
        assert all(t1 > t0 for t0, t1 in pairwise(times))
        for p in traj["points"]:
            q = np.asarray(p["q"])
            assert len(q) == 6
            assert torque_source.is_safe(model, q)
            assert model.fk(q)[2, 3] >= torque_source.Z_MIN_TOOL


@pytest.mark.parametrize(
    "q",
    [
        [0.0, 0.5, 0.0, 0.0, 0.0, 0.0],  # upper arm pointing below the table plane
        [0.0, -1.57, 0.0, -1.57, 0.0, 9.0],  # wrist_3 outside its joint limits
    ],
)
def test_unsafe_configurations_are_rejected(q: list[float]) -> None:
    assert not torque_source.is_safe(UR5eModel(), np.asarray(q))
