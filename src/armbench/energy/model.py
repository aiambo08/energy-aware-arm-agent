"""Electrical energy of a joint-torque trajectory.

Per sample, with joint torques ``tau`` [N m] and joint velocities ``omega`` [rad/s]::

    P_mech_A = sum_i max(tau_i * omega_i, 0)      # no regeneration (variant A)
    P_mech_B = sum_i |tau_i * omega_i|            # upper bound (variant B), B >= A
    P_cu     = sum_i R_i * (tau_i / kt_i) ** 2    # Joule losses in the windings
    P_el     = P_mech / eta + P_cu + P0

Energies are trapezoidal integrals over sample time (exact for piecewise-linear power).
``eta`` is the gearbox + electronics efficiency and must not include copper losses,
otherwise ``P_cu`` would be counted twice (ADR-004).
"""

from __future__ import annotations

from enum import StrEnum

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from armbench.energy.params import EnergyParams

J_PER_WH = 3600.0


class Variant(StrEnum):
    A = "A"  # no regeneration: negative mechanical power is dissipated, counts as 0
    B = "B"  # |tau*omega|: braking costs as much as driving (upper bound)


class EnergyBreakdown(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    variant: Variant
    eta: float = Field(gt=0, le=1)
    duration_s: float = Field(ge=0)
    n_samples: int = Field(ge=0)
    mechanical_j: float = Field(ge=0, description="shaft-side mechanical work (before /eta)")
    copper_j: float = Field(ge=0)
    base_j: float = Field(ge=0)
    total_j: float = Field(ge=0)

    @property
    def total_wh(self) -> float:
        return self.total_j / J_PER_WH


def _as_2d(x: np.ndarray, n_joints: int, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != n_joints:
        msg = f"{name} must have shape (T, {n_joints}), got {arr.shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(arr)):
        msg = f"{name} contains non-finite values"
        raise ValueError(msg)
    return arr


def _check_time(t: np.ndarray, n: int) -> np.ndarray:
    tt = np.asarray(t, dtype=float)
    if tt.ndim != 1 or tt.shape[0] != n:
        msg = f"t must have shape ({n},), got {tt.shape}"
        raise ValueError(msg)
    if n and (not np.all(np.isfinite(tt)) or np.any(np.diff(tt) < 0)):
        msg = "t must be finite and non-decreasing"
        raise ValueError(msg)
    return tt


def mechanical_power(tau: np.ndarray, omega: np.ndarray, variant: Variant) -> np.ndarray:
    """Per-sample mechanical power summed over joints, shape (T,)."""
    p = np.asarray(tau, dtype=float) * np.asarray(omega, dtype=float)
    if variant is Variant.A:
        return np.asarray(np.maximum(p, 0.0).sum(axis=-1))
    return np.asarray(np.abs(p).sum(axis=-1))


def copper_power(tau: np.ndarray, params: EnergyParams) -> np.ndarray:
    """Per-sample Joule losses summed over joints, shape (T,)."""
    current = np.asarray(tau, dtype=float) / params.torque_constants()
    return np.asarray((params.winding_resistances() * current**2).sum(axis=-1))


def electrical_power(
    tau: np.ndarray,
    omega: np.ndarray,
    params: EnergyParams,
    variant: Variant,
    eta: float | None = None,
) -> np.ndarray:
    """Per-sample electrical power P_el, shape (T,)."""
    eta_v = params.eta if eta is None else eta
    p_el = mechanical_power(tau, omega, variant) / eta_v + copper_power(tau, params) + params.p0_w
    return np.asarray(p_el)


def integrate(t: np.ndarray, p: np.ndarray) -> float:
    """Trapezoidal integral of ``p`` over ``t`` (0 for fewer than 2 samples)."""
    if len(t) < 2:
        return 0.0
    tt = np.asarray(t, dtype=float)
    pp = np.asarray(p, dtype=float)
    return float(np.sum(0.5 * (pp[1:] + pp[:-1]) * np.diff(tt)))


def episode_energy(  # noqa: PLR0913
    t: np.ndarray,
    omega: np.ndarray,
    tau: np.ndarray,
    params: EnergyParams,
    *,
    variant: Variant = Variant.A,
    eta: float | None = None,
) -> EnergyBreakdown:
    """Energy of a whole episode from sampled ``t`` (s), ``omega`` (rad/s) and ``tau`` (N m)."""
    n = params.n_joints
    omega2 = _as_2d(omega, n, "omega")
    tau2 = _as_2d(tau, n, "tau")
    if omega2.shape[0] != tau2.shape[0]:
        msg = f"omega and tau disagree on T: {omega2.shape[0]} vs {tau2.shape[0]}"
        raise ValueError(msg)
    tt = _check_time(t, tau2.shape[0])
    eta_v = params.eta if eta is None else eta
    mech = integrate(tt, mechanical_power(tau2, omega2, variant))
    cu = integrate(tt, copper_power(tau2, params))
    duration = float(tt[-1] - tt[0]) if len(tt) else 0.0
    base = params.p0_w * duration
    return EnergyBreakdown(
        variant=variant,
        eta=eta_v,
        duration_s=duration,
        n_samples=int(tau2.shape[0]),
        mechanical_j=mech,
        copper_j=cu,
        base_j=base,
        total_j=mech / eta_v + cu + base,
    )


def sensitivity(
    t: np.ndarray, omega: np.ndarray, tau: np.ndarray, params: EnergyParams
) -> list[EnergyBreakdown]:
    """All (variant, eta) combinations the protocol requires; nominal eta first."""
    return [
        episode_energy(t, omega, tau, params, variant=v, eta=e)
        for v in Variant
        for e in params.etas()
    ]


class EnergyMeter:
    """Streaming trapezoidal integrator; equals :func:`episode_energy` on the same samples.

    Keeps every (variant, eta) combination so a single pass over the joint states yields the
    complete sensitivity table. ``update`` must be called with non-decreasing time.
    """

    def __init__(self, params: EnergyParams) -> None:
        self.params = params
        self._kt = params.torque_constants()
        self._r = params.winding_resistances()
        self._combos = [(v, e) for v in Variant for e in params.etas()]
        self.reset()

    def reset(self) -> None:
        self.n_samples = 0
        self.t_first = float("nan")
        self.t_last = float("nan")
        self._prev_mech: dict[Variant, float] = dict.fromkeys(Variant, 0.0)
        self._prev_cu = 0.0
        self.mechanical_j: dict[Variant, float] = dict.fromkeys(Variant, 0.0)
        self.copper_j = 0.0

    def update(self, t: float, omega: np.ndarray, tau: np.ndarray) -> None:
        om = np.asarray(omega, dtype=float)
        ta = np.asarray(tau, dtype=float)
        if om.shape != (self.params.n_joints,) or ta.shape != (self.params.n_joints,):
            msg = f"expected {self.params.n_joints} joints, got {om.shape} and {ta.shape}"
            raise ValueError(msg)
        if not (np.isfinite(t) and np.all(np.isfinite(om)) and np.all(np.isfinite(ta))):
            msg = "non-finite sample"
            raise ValueError(msg)
        mech = {v: float(mechanical_power(ta, om, v)) for v in Variant}
        cu = float((self._r * (ta / self._kt) ** 2).sum())
        if self.n_samples:
            if t < self.t_last:
                msg = f"time went backwards: {t} < {self.t_last}"
                raise ValueError(msg)
            dt = t - self.t_last
            for v in Variant:
                self.mechanical_j[v] += 0.5 * (self._prev_mech[v] + mech[v]) * dt
            self.copper_j += 0.5 * (self._prev_cu + cu) * dt
        else:
            self.t_first = t
        self._prev_mech = mech
        self._prev_cu = cu
        self.t_last = t
        self.n_samples += 1

    @property
    def duration_s(self) -> float:
        return self.t_last - self.t_first if self.n_samples else 0.0

    def breakdown(self, variant: Variant = Variant.A, eta: float | None = None) -> EnergyBreakdown:
        eta_v = self.params.eta if eta is None else eta
        base = self.params.p0_w * self.duration_s
        mech = self.mechanical_j[variant]
        return EnergyBreakdown(
            variant=variant,
            eta=eta_v,
            duration_s=self.duration_s,
            n_samples=self.n_samples,
            mechanical_j=mech,
            copper_j=self.copper_j,
            base_j=base,
            total_j=mech / eta_v + self.copper_j + base,
        )

    def table(self) -> list[EnergyBreakdown]:
        return [self.breakdown(v, e) for v, e in self._combos]
