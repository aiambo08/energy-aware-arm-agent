"""Energy model: analytic cases (rel. error < 1e-6), properties (hypothesis) and the log format."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from pydantic import ValidationError

from armbench.energy import (
    DEFAULT_ENERGY_FILE,
    J_PER_WH,
    EnergyMeter,
    EnergyParams,
    Variant,
    copper_power,
    electrical_power,
    episode_energy,
    episode_from_jsonl,
    load_energy_params,
    mechanical_power,
    read_samples_jsonl,
    sensitivity,
    write_samples_jsonl,
)
from armbench.energy.log import Sample

REL = 1e-6
N = 6


@pytest.fixture(scope="module")
def params() -> EnergyParams:
    return load_energy_params()


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-12)


# ----------------------------------------------------------------------------- parameters
def test_default_params_load_and_are_sane(params: EnergyParams) -> None:
    assert params.n_joints == N
    assert 0 < params.eta <= 1
    assert params.p0_w >= 0
    assert params.etas()[0] == params.eta
    assert set(params.eta_sensitivity) <= set(params.etas())
    assert DEFAULT_ENERGY_FILE.exists()


def test_params_reject_mismatch(params: EnergyParams) -> None:
    raw = params.model_dump()
    raw["joint_order"] = raw["joint_order"][:-1]
    with pytest.raises(ValidationError, match="mismatch"):
        EnergyParams.model_validate(raw)
    raw = params.model_dump()
    raw["eta"] = 1.2
    with pytest.raises(ValidationError):
        EnergyParams.model_validate(raw)
    raw = params.model_dump()
    raw["joints"]["elbow_joint"]["torque_constant_nm_per_a"] = 0.0
    with pytest.raises(ValidationError):
        EnergyParams.model_validate(raw)


# ----------------------------------------------------------------------------- analytic
def test_constant_velocity_and_torque(params: EnergyParams) -> None:
    """Steady motion: E_mech = sum(tau*omega)*T, copper = sum(R (tau/kt)^2) T, base = P0 T."""
    duration, n = 4.0, 401
    t = np.linspace(0.0, duration, n)
    tau = np.tile([30.0, -20.0, 10.0, 5.0, -3.0, 2.0], (n, 1))
    omega = np.tile([0.5, 0.4, -0.3, 0.2, 0.1, -0.6], (n, 1))
    p = tau[0] * omega[0]
    expected_a = np.maximum(p, 0).sum() * duration
    expected_b = np.abs(p).sum() * duration
    expected_cu = (params.winding_resistances() * (tau[0] / params.torque_constants()) ** 2).sum()
    expected_cu *= duration
    a = episode_energy(t, omega, tau, params, variant=Variant.A)
    b = episode_energy(t, omega, tau, params, variant=Variant.B)
    assert _rel(a.mechanical_j, expected_a) < REL
    assert _rel(b.mechanical_j, expected_b) < REL
    assert _rel(a.copper_j, expected_cu) < REL
    assert _rel(a.base_j, params.p0_w * duration) < REL
    assert _rel(a.total_j, expected_a / params.eta + expected_cu + params.p0_w * duration) < REL
    assert _rel(a.total_wh, a.total_j / J_PER_WH) < REL
    assert a.duration_s == duration
    assert a.n_samples == n


def test_static_hold_is_copper_plus_base_only(params: EnergyParams) -> None:
    t = np.linspace(0.0, 2.0, 51)
    tau = np.tile([0.0, 40.0, 15.0, 1.0, 0.5, 0.0], (51, 1))
    omega = np.zeros_like(tau)
    for v in Variant:
        e = episode_energy(t, omega, tau, params, variant=v)
        assert e.mechanical_j == 0.0
        cu = (params.winding_resistances() * (tau[0] / params.torque_constants()) ** 2).sum() * 2.0
        assert _rel(e.copper_j, cu) < REL
        assert _rel(e.total_j, cu + params.p0_w * 2.0) < REL


def test_negative_power_is_zero_in_a_and_positive_in_b(params: EnergyParams) -> None:
    t = np.linspace(0.0, 1.0, 11)
    tau = np.tile([10.0] * N, (11, 1))
    omega = np.tile([-1.0] * N, (11, 1))
    a = episode_energy(t, omega, tau, params, variant=Variant.A)
    b = episode_energy(t, omega, tau, params, variant=Variant.B)
    assert a.mechanical_j == 0.0
    assert _rel(b.mechanical_j, 10.0 * N * 1.0) < REL
    assert np.all(mechanical_power(tau, omega, Variant.A) == 0.0)


def test_trapezoid_is_exact_for_linear_power(params: EnergyParams) -> None:
    """tau constant, omega linear in t -> power linear -> trapezoid exact: integral = tau*w1*T/2."""
    t = np.linspace(0.0, 3.0, 7)
    tau = np.tile([2.0, 0, 0, 0, 0, 0], (7, 1))
    omega = np.zeros((7, N))
    omega[:, 0] = t / 3.0  # 0 -> 1 rad/s
    e = episode_energy(t, omega, tau, params, variant=Variant.A)
    assert _rel(e.mechanical_j, 2.0 * 1.0 * 3.0 / 2.0) < REL


def test_eta_sensitivity_ordering(params: EnergyParams) -> None:
    t = np.linspace(0.0, 1.0, 101)
    rng = np.random.default_rng(0)
    tau = rng.normal(size=(101, N)) * 20
    omega = rng.normal(size=(101, N))
    table = sensitivity(t, omega, tau, params)
    assert len(table) == len(Variant) * len(params.etas())
    assert table[0].variant is Variant.A
    assert table[0].eta == params.eta
    by = {(r.variant, r.eta): r for r in table}
    for v in Variant:
        etas = sorted(params.etas())
        totals = [by[(v, e)].total_j for e in etas]
        assert totals == sorted(totals, reverse=True)  # lower eta -> more energy
    for e in params.etas():
        assert by[(Variant.B, e)].total_j >= by[(Variant.A, e)].total_j


def test_electrical_power_matches_components(params: EnergyParams) -> None:
    tau = np.array([[10.0, -5.0, 3.0, 1.0, 0.0, 2.0]])
    omega = np.array([[1.0, 1.0, -1.0, 0.0, 2.0, 0.5]])
    p = electrical_power(tau, omega, params, Variant.A, eta=0.5)
    mech = mechanical_power(tau, omega, Variant.A)
    assert np.allclose(p, mech / 0.5 + copper_power(tau, params) + params.p0_w)


def test_degenerate_inputs(params: EnergyParams) -> None:
    e = episode_energy(np.array([]), np.zeros((0, N)), np.zeros((0, N)), params)
    assert e.total_j == 0.0
    assert e.n_samples == 0
    e1 = episode_energy(np.array([1.0]), np.ones((1, N)), np.ones((1, N)), params)
    assert e1.total_j == 0.0
    with pytest.raises(ValueError, match="shape"):
        episode_energy(np.array([0.0, 1.0]), np.zeros((2, 5)), np.zeros((2, 5)), params)
    with pytest.raises(ValueError, match="non-decreasing"):
        episode_energy(np.array([1.0, 0.0]), np.zeros((2, N)), np.zeros((2, N)), params)
    with pytest.raises(ValueError, match="non-finite"):
        episode_energy(np.array([0.0, 1.0]), np.full((2, N), np.nan), np.zeros((2, N)), params)
    with pytest.raises(ValueError, match="disagree"):
        episode_energy(np.array([0.0, 1.0]), np.zeros((2, N)), np.zeros((3, N)), params)


# ----------------------------------------------------------------------------- properties
samples = st.integers(min_value=2, max_value=60)
finite = st.floats(min_value=-200.0, max_value=200.0, allow_nan=False, allow_infinity=False)


@st.composite
def trajectories(draw: st.DrawFn) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = draw(samples)
    dts = draw(
        hnp.arrays(
            float,
            (n,),
            elements=st.floats(min_value=0.0, max_value=0.1, allow_nan=False),
        )
    )
    t = np.cumsum(dts)
    tau = draw(hnp.arrays(float, (n, N), elements=finite))
    omega = draw(hnp.arrays(float, (n, N), elements=finite))
    return t, omega, tau


@settings(max_examples=1000, deadline=None)
@given(trajectories())
def test_energy_nonnegative_b_ge_a_and_monotone_in_time(
    traj: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    params = load_energy_params()
    t, omega, tau = traj
    a = episode_energy(t, omega, tau, params, variant=Variant.A)
    b = episode_energy(t, omega, tau, params, variant=Variant.B)
    assert a.total_j >= 0
    assert a.mechanical_j >= 0
    assert a.copper_j >= 0
    assert a.base_j >= 0
    assert b.mechanical_j >= a.mechanical_j
    assert b.total_j >= a.total_j
    # monotone: energy of a prefix never exceeds the energy of the whole episode
    k = len(t) // 2
    for v in Variant:
        prefix = episode_energy(t[:k], omega[:k], tau[:k], params, variant=v)
        full = episode_energy(t, omega, tau, params, variant=v)
        assert prefix.total_j <= full.total_j * (1 + 1e-12) + 1e-9


@settings(max_examples=300, deadline=None)
@given(trajectories())
def test_meter_matches_batch(traj: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    params = load_energy_params()
    t, omega, tau = traj
    meter = EnergyMeter(params)
    for i in range(len(t)):
        meter.update(float(t[i]), omega[i], tau[i])
    for row in meter.table():
        ref = episode_energy(t, omega, tau, params, variant=row.variant, eta=row.eta)
        assert row.n_samples == ref.n_samples
        assert abs(row.total_j - ref.total_j) <= 1e-9 * max(1.0, abs(ref.total_j))
        assert abs(row.mechanical_j - ref.mechanical_j) <= 1e-9 * max(1.0, ref.mechanical_j)


def test_meter_guards(params: EnergyParams) -> None:
    m = EnergyMeter(params)
    assert m.duration_s == 0.0
    assert m.breakdown().total_j == 0.0
    m.update(1.0, np.zeros(N), np.zeros(N))
    with pytest.raises(ValueError, match="backwards"):
        m.update(0.5, np.zeros(N), np.zeros(N))
    with pytest.raises(ValueError, match="joints"):
        m.update(2.0, np.zeros(5), np.zeros(N))
    with pytest.raises(ValueError, match="non-finite"):
        m.update(2.0, np.full(N, np.inf), np.zeros(N))
    m.reset()
    assert m.n_samples == 0


# ----------------------------------------------------------------------------- log format
def test_jsonl_roundtrip_and_episode(tmp_path: Path, params: EnergyParams) -> None:
    path = tmp_path / "ep.jsonl"
    rows = [Sample(t=0.1 * i, q=(0.0,) * N, qd=(0.5,) * N, tau=(10.0,) * N) for i in range(21)]
    assert write_samples_jsonl(path, rows) == 21
    s = read_samples_jsonl(path, N)
    assert s.n == 21
    assert s.tau.shape == (21, N)
    table = episode_from_jsonl(path, params, full_sensitivity=True)
    assert len(table) == len(Variant) * len(params.etas())
    single = episode_from_jsonl(path, params, variant=Variant.B, eta=0.6)
    assert single[0].variant is Variant.B
    assert single[0].eta == 0.6
    assert _rel(single[0].mechanical_j, 10.0 * 0.5 * N * 2.0) < 1e-9


def test_jsonl_rejects_wrong_width(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(Sample(t=0.0, q=(0.0,) * 5, qd=(0.0,) * 5, tau=(0.0,) * 5).model_dump_json())
    with pytest.raises(ValueError, match="expected 6 joints"):
        read_samples_jsonl(path, N)
