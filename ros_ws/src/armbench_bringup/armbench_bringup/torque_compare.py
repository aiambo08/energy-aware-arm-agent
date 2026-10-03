"""Gazebo joint effort vs inverse dynamics (Pinocchio) on the segments recorded by torque_probe.

For every ``<segment>.jsonl``:

* static segments: mean Gazebo effort vs gravity torque ``rnea(q, 0, 0)`` per joint;
* motion segments: ``rnea(q, qd_f, qdd_f)`` with ``qd_f``/``qdd_f`` from a Savitzky-Golay
  filter of the published velocity, compared sample-by-sample (RMSE, normalised RMSE, R^2);
* writes ``<segment>.id.jsonl`` with ``tau`` replaced by the inverse-dynamics torque so the
  host can integrate energy from either source.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pinocchio as pin
from ament_index_python.packages import get_package_share_directory
from scipy.signal import savgol_filter

ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


@dataclass(frozen=True)
class IdModel:
    """Pinocchio model plus the q/v indices of the six arm joints."""

    model: pin.Model
    q_idx: list[int]
    v_idx: list[int]


def build_model() -> IdModel:
    desc = get_package_share_directory("armbench_description")
    bringup = get_package_share_directory("armbench_bringup")
    xacro = shutil.which("xacro") or "/opt/ros/jazzy/bin/xacro"
    urdf = subprocess.run(  # noqa: S603 - fixed arguments, no user input
        [
            xacro,
            f"{desc}/urdf/armbench_ur5e.urdf.xacro",
            "name:=armbench_ur5e",
            f"simulation_controllers:={bringup}/config/controllers.yaml",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    model = pin.buildModelFromXML(urdf)
    q_idx = [model.joints[model.getJointId(j)].idx_q for j in ARM_JOINTS]
    v_idx = [model.joints[model.getJointId(j)].idx_v for j in ARM_JOINTS]
    return IdModel(model, q_idx, v_idx)


def load(path: Path) -> dict[str, np.ndarray]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {k: np.array([r[k] for r in rows], dtype=float) for k in ("t", "q", "qd", "tau")}


def inverse_dynamics(idm: IdModel, q: np.ndarray, qd: np.ndarray, qdd: np.ndarray) -> np.ndarray:
    model, q_idx, v_idx = idm.model, idm.q_idx, idm.v_idx
    data = model.createData()
    out = np.zeros_like(q)
    qq = np.zeros(model.nq)
    vv = np.zeros(model.nv)
    aa = np.zeros(model.nv)
    for k in range(q.shape[0]):
        qq[q_idx] = q[k]
        vv[v_idx] = qd[k]
        aa[v_idx] = qdd[k]
        tau = pin.rnea(model, data, qq, vv, aa)
        out[k] = tau[v_idx]
    return out


def filtered_derivatives(
    t: np.ndarray, qd: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray]:
    n = qd.shape[0]
    win = min(window, n if n % 2 else n - 1)
    if win < 5:
        return qd, np.gradient(qd, t, axis=0)
    dt = float(np.median(np.diff(t)))
    qd_f = savgol_filter(qd, win, 3, axis=0)
    qdd = savgol_filter(qd, win, 3, deriv=1, delta=dt, axis=0)
    return qd_f, qdd


def compare_motion(tau_gz: np.ndarray, tau_id: np.ndarray) -> dict:
    err = tau_gz - tau_id
    rmse = np.sqrt((err**2).mean(axis=0))
    span = np.maximum(tau_gz.max(axis=0) - tau_gz.min(axis=0), 1e-9)
    ss_res = (err**2).sum(axis=0)
    ss_tot = np.maximum(((tau_gz - tau_gz.mean(axis=0)) ** 2).sum(axis=0), 1e-12)
    return {
        "rmse_nm": rmse.round(4).tolist(),
        "nrmse": (rmse / span).round(4).tolist(),
        "r2": (1 - ss_res / ss_tot).round(4).tolist(),
        "tau_gz_span_nm": span.round(3).tolist(),
        "mean_abs_gz_nm": np.abs(tau_gz).mean(axis=0).round(3).tolist(),
        "mean_abs_id_nm": np.abs(tau_id).mean(axis=0).round(3).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", default="/tmp/f2_torque")  # noqa: S108
    parser.add_argument("--window", type=int, default=31, help="Savitzky-Golay window (samples)")
    args = parser.parse_args()
    in_dir = Path(args.in_dir)
    idm = build_model()
    out: dict = {"static": {}, "motion": {}, "window": args.window}
    for path in sorted(in_dir.glob("*.jsonl")):
        if path.name.endswith(".id.jsonl"):
            continue
        d = load(path)
        if d["t"].shape[0] < 5:
            continue
        name = path.stem
        if name.startswith("static_"):
            q_mean = d["q"].mean(axis=0)
            tau_g = inverse_dynamics(idm, q_mean[None], np.zeros((1, 6)), np.zeros((1, 6)))[0]
            tau_gz = d["tau"].mean(axis=0)
            diff = tau_gz - tau_g
            out["static"][name] = {
                "q": q_mean.round(4).tolist(),
                "tau_gz_mean_nm": tau_gz.round(4).tolist(),
                "tau_gz_std_nm": d["tau"].std(axis=0).round(4).tolist(),
                "tau_gravity_id_nm": tau_g.round(4).tolist(),
                "abs_diff_nm": np.abs(diff).round(4).tolist(),
                "rel_diff": (np.abs(diff) / np.maximum(np.abs(tau_g), 1e-9)).round(4).tolist(),
            }
            tau_id_series = np.tile(tau_g, (d["t"].shape[0], 1))
        else:
            qd_f, qdd = filtered_derivatives(d["t"], d["qd"], args.window)
            tau_id_series = inverse_dynamics(idm, d["q"], qd_f, qdd)
            out["motion"][name] = compare_motion(d["tau"], tau_id_series)
            out["motion"][name]["max_abs_qd"] = np.abs(d["qd"]).max(axis=0).round(3).tolist()
            out["motion"][name]["max_abs_qdd"] = np.abs(qdd).max(axis=0).round(3).tolist()
        with path.with_suffix(".id.jsonl").open("w") as fh:
            for k in range(d["t"].shape[0]):
                fh.write(
                    json.dumps(
                        {
                            "t": d["t"][k],
                            "q": d["q"][k].tolist(),
                            "qd": d["qd"][k].tolist(),
                            "tau": tau_id_series[k].tolist(),
                        }
                    )
                    + "\n"
                )
    Path(in_dir, "compare.json").write_text(json.dumps(out, indent=2, sort_keys=True))
    print(json.dumps(out, indent=2, sort_keys=True))  # noqa: T201


if __name__ == "__main__":
    main()
