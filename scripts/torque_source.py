"""F2 gate: where does the joint torque come from? (critical gate 2)

Generates a seeded plan (static holds, N scripted joint-space trajectories checked with the
UR5e FK to stay above the table, and R repeats of the first trajectory), runs it inside the
simulation image with the ``energy_meter`` node alive, confronts Gazebo's joint effort with
Pinocchio inverse dynamics (``torque_compare``), integrates the energy of every segment from
both torque sources on the host and writes ``reports/f2_torque_source.json``.

    uv run python scripts/torque_source.py --image armbench-sim:dev \\
        --out reports/f2_torque_source.json

Thresholds (docs/plan.es.md, F2): static gravity torque Gazebo vs inverse dynamics < 10 % per
joint (joints carrying >= 1 N m; |diff| < 0.5 N m otherwise), Wh repeatability CV < 2 % over
the repeats, energy_meter CPU < 5 %. The chosen torque source is recorded in ADR-004.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from armbench.energy import (
    EnergyBreakdown,
    EnergyParams,
    Variant,
    episode_energy,
    episode_from_jsonl,
    load_energy_params,
    read_samples_jsonl,
)
from armbench.kinematics import HOME_Q, UR5eModel

CONTAINER = "armbench-f2-torque"
ENTRYPOINT = "/usr/local/bin/entrypoint.sh"  # docker exec does not source ROS by itself
IN_DIR = "/tmp/f2_torque"  # noqa: S108 - inside the throwaway container
WORLD = (
    "/work/ros_ws/install/armbench_description/share/armbench_description/worlds/check_scene.sdf"
)
LAUNCH_CMD = f"ros2 launch armbench_bringup sim.launch.py headless:=true world:={WORLD}"
THRESHOLDS: dict[str, float] = {
    "static_rel_diff_max": 0.10,
    "static_abs_diff_floor_nm": 0.5,
    "static_min_torque_nm": 1.0,
    "wh_cv_max": 0.02,
    "meter_cpu_share_percent_max": 5.0,  # meter CPU time / whole-container CPU time
    "motion_nrmse_median_max": 0.15,
}
STATIC_HOLDS = {
    "home": list(HOME_Q),
    "stretched": [0.0, -1.57, 0.0, 0.0, 0.0, 0.0],
    "half": [0.3, -1.2, -1.0, -1.0, 1.57, 0.5],
    "pregrasp": [0.2699, -1.6612, -1.9823, -1.0690, 1.5708, 1.8407],
    "elbow_up": [-0.6, -2.0, 1.2, -1.5, -1.2, 0.8],
}
Z_MIN_LINKS = 0.08
Z_MIN_TOOL = 0.12
Z_MAX = 1.35
XY_MAX = 0.85
CAMERA_XY = (-0.5, 0.0)  # RGB-D camera + post at z >= 0.85 (worlds/tabletop.sdf.in)
CAMERA_R = 0.15
CAMERA_Z = 0.85


def run(cmd: list[str], timeout: float | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)


def stop_container() -> None:
    run(["docker", "rm", "-f", CONTAINER], timeout=60)


# ----------------------------------------------------------------------------- plan
def is_safe(model: UR5eModel, q: np.ndarray) -> bool:
    if not model.within_limits(q):
        return False
    frames = model.fk_all(q)
    for i, f in enumerate(frames):
        x, y, z = f[:3, 3]
        if i >= 1 and z < Z_MIN_LINKS:  # frame 0 (shoulder) sits on the base
            return False
        if z > Z_MAX or abs(x) > XY_MAX or abs(y) > XY_MAX:
            return False
        if z > CAMERA_Z and np.hypot(x - CAMERA_XY[0], y - CAMERA_XY[1]) < CAMERA_R:
            return False
    return bool(frames[-1][2, 3] >= Z_MIN_TOOL)


def sample_waypoint(rng: np.random.Generator, model: UR5eModel, around: np.ndarray) -> np.ndarray:
    span = np.array([0.9, 0.6, 0.7, 0.9, 0.9, 1.2])
    for _ in range(1000):
        q = np.asarray(around + rng.uniform(-span, span))
        if is_safe(model, q):
            return q
    msg = "could not sample a safe waypoint"
    raise RuntimeError(msg)


def make_plan(seed: int, n_traj: int, n_rep: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    model = UR5eModel()
    for name, q in STATIC_HOLDS.items():
        if not is_safe(model, np.asarray(q)):
            msg = f"static hold {name} is not safe"
            raise RuntimeError(msg)
    home = np.asarray(HOME_Q)
    trajectories = []
    for i in range(n_traj):
        v_max = float(rng.uniform(0.6, 1.8))
        q_prev = sample_waypoint(rng, model, home)
        points = [{"q": q_prev.round(4).tolist(), "t": 0.0}]
        t = 0.0
        for _ in range(3):
            q_next = sample_waypoint(rng, model, q_prev)
            t += max(0.8, float(np.abs(q_next - q_prev).max()) / v_max)
            points.append({"q": q_next.round(4).tolist(), "t": round(t, 3)})
            q_prev = q_next
        trajectories.append(
            {"name": f"traj_{i:02d}", "v_max_rad_s": round(v_max, 3), "points": points}
        )
    return {
        "seed": seed,
        "hold_s": 2.0,
        "settle_s": 1.0,
        "move_s": 3.0,
        "static_holds": [{"name": n, "q": q} for n, q in STATIC_HOLDS.items()],
        "trajectories": trajectories,
        "repeat": {"trajectory": "traj_00", "n": n_rep},
    }


# ----------------------------------------------------------------------------- container
def run_in_container(image: str, plan_dir: Path, work_dir: Path) -> dict[str, Any]:
    stop_container()
    started = run(
        [
            "docker", "run", "-d", "--network", "none", "--name", CONTAINER,
            "-v", f"{plan_dir.resolve()}:/probe:ro", image, "bash", "-c", LAUNCH_CMD,
        ]
    )  # fmt: skip
    if started.returncode != 0:
        return {"ok": False, "error": f"docker run failed: {started.stderr.strip()}"}
    t_launch = time.time()
    meter_log = f"{IN_DIR}/meter.jsonl"
    run(["docker", "exec", CONTAINER, "mkdir", "-p", IN_DIR])
    run(
        ["docker", "exec", "-d", CONTAINER, ENTRYPOINT, "bash", "-c",
         f"ros2 run armbench_bringup energy_meter --log {meter_log} > {IN_DIR}/meter.log 2>&1"]
    )  # fmt: skip
    pids: list[str] = []
    for _ in range(30):
        time.sleep(1.0)
        pids = run(
            ["docker", "exec", CONTAINER, "pgrep", "-f", "armbench_bringup/energy_meter"]
        ).stdout.split()
        if pids:
            break
    cpu_pid = pids[0] if pids else ""
    probe = run(
        ["docker", "exec", CONTAINER, ENTRYPOINT, "bash", "-c",
         f"ros2 run armbench_bringup torque_probe --plan /probe/plan.json --out-dir {IN_DIR} "
         f"--timeout 90 {'--cpu-pid ' + cpu_pid if cpu_pid else ''} > {IN_DIR}/probe.log 2>&1"],
        timeout=3600,
    )  # fmt: skip
    compare = run(
        ["docker", "exec", CONTAINER, ENTRYPOINT, "bash", "-c",
         f"ros2 run armbench_bringup torque_compare --in-dir {IN_DIR} > {IN_DIR}/compare.log 2>&1"],
        timeout=1800,
    )  # fmt: skip
    if work_dir.exists():
        shutil.rmtree(work_dir)
    cp = run(["docker", "cp", f"{CONTAINER}:{IN_DIR}", str(work_dir)], timeout=120)
    logs = run(["docker", "logs", "--tail", "40", CONTAINER], timeout=30)
    stop_container()
    return {
        "ok": probe.returncode == 0 and compare.returncode == 0 and cp.returncode == 0,
        "probe_rc": probe.returncode,
        "compare_rc": compare.returncode,
        "cp_rc": cp.returncode,
        "meter_pid_found": bool(cpu_pid),
        "wall_s": round(time.time() - t_launch, 1),
        "sim_log_tail": logs.stdout.splitlines()[-15:],
    }


# ----------------------------------------------------------------------------- analysis
DECIMATION = 5  # 500 Hz /joint_states -> 100 Hz energy_state_broadcaster


def decimated_energy(path: Path, params: EnergyParams, every: int) -> EnergyBreakdown:
    """Energy of ``path`` keeping one sample out of ``every`` (what the 100 Hz meter sees)."""
    s = read_samples_jsonl(path, params.n_joints)
    return episode_energy(s.t[::every], s.qd[::every], s.tau[::every], params, variant=Variant.A)


def cv(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    arr = np.asarray(values)
    return float(arr.std(ddof=1) / arr.mean()) if arr.mean() > 0 else None


def analyse(work_dir: Path, plan: dict[str, Any]) -> dict[str, Any]:
    params = load_energy_params()
    probe = json.loads((work_dir / "probe.json").read_text())
    compare = json.loads((work_dir / "compare.json").read_text())
    segments = {s["file"]: s for s in probe["segments"]}

    # static gravity torque
    static_rows = []
    static_ok = True
    for name, row in sorted(compare["static"].items()):
        per_joint = []
        for j, (tg, ad, rd) in enumerate(
            zip(row["tau_gravity_id_nm"], row["abs_diff_nm"], row["rel_diff"], strict=True)
        ):
            if abs(tg) >= THRESHOLDS["static_min_torque_nm"]:
                ok = rd <= THRESHOLDS["static_rel_diff_max"]
            else:
                ok = ad <= THRESHOLDS["static_abs_diff_floor_nm"]
            per_joint.append(
                {"joint": j, "tau_id_nm": tg, "abs_diff_nm": ad, "rel_diff": rd, "ok": ok}
            )
            static_ok = static_ok and ok
        static_rows.append(
            {"name": name, "joints": per_joint, "ok": all(p["ok"] for p in per_joint)}
        )

    # motion: inverse dynamics vs effort, and energy from both sources
    motion_rows = []
    nrmse_all: list[float] = []
    for name, row in sorted(compare["motion"].items()):
        seg = segments.get(f"{name}.jsonl", {})
        gz = episode_from_jsonl(work_dir / f"{name}.jsonl", params, variant=Variant.A)[0]
        idd = episode_from_jsonl(work_dir / f"{name}.id.jsonl", params, variant=Variant.A)[0]
        dec = decimated_energy(work_dir / f"{name}.jsonl", params, DECIMATION)
        dec_err = abs(dec.total_wh - gz.total_wh) / gz.total_wh if gz.total_wh > 0 else None
        nrmse_all += row["nrmse"]
        motion_rows.append(
            {
                "name": name,
                "ok": seg.get("ok"),
                "n_samples": gz.n_samples,
                "duration_s": round(gz.duration_s, 3),
                "rtf": seg.get("rtf"),
                "nrmse": row["nrmse"],
                "r2": row["r2"],
                "rmse_nm": row["rmse_nm"],
                "wh_gazebo_A": round(gz.total_wh, 5),
                "wh_gazebo_A_100hz": round(dec.total_wh, 5),
                "decimation_rel_err": dec_err,
                "wh_inverse_dynamics_A": round(idd.total_wh, 5),
                "mech_j_gazebo_A": round(gz.mechanical_j, 3),
                "mech_j_inverse_dynamics_A": round(idd.mechanical_j, 3),
            }
        )
    reps = [r for r in motion_rows if r["name"].startswith("rep_")]
    trajs = [r for r in motion_rows if r["name"].startswith("traj_")]
    cv_gz = cv([r["wh_gazebo_A"] for r in reps])
    cv_id = cv([r["wh_inverse_dynamics_A"] for r in reps])
    cpu = [c for c in probe.get("cpu_samples", []) if c is not None]
    share_max = max((c["meter_share_percent"] for c in cpu), default=None)
    core_max = max((c["meter_core_percent"] for c in cpu), default=None)
    container_med = float(np.median([c["container_core_percent"] for c in cpu])) if cpu else None
    motion_median = float(np.median(nrmse_all)) if nrmse_all else None

    gazebo_usable = (
        static_ok
        and motion_median is not None
        and (motion_median <= THRESHOLDS["motion_nrmse_median_max"])
    )
    summary = {
        "n_static": len(static_rows),
        "n_trajectories": len(trajs),
        "n_repeats": len(reps),
        "segments_all_ok": bool(probe.get("all_ok")),
        "static_gravity_ok": static_ok,
        "static_rel_diff_max": max(
            (
                p["rel_diff"]
                for r in static_rows
                for p in r["joints"]
                if abs(p["tau_id_nm"]) >= THRESHOLDS["static_min_torque_nm"]
            ),
            default=None,
        ),
        "motion_nrmse_median": motion_median,
        "motion_r2_median": float(np.median([v for r in trajs for v in r["r2"]]))
        if trajs
        else None,
        "decimation_100hz_rel_err_max": max(
            (r["decimation_rel_err"] for r in motion_rows if r["decimation_rel_err"] is not None),
            default=None,
        ),
        "wh_cv_gazebo": cv_gz,
        "wh_cv_inverse_dynamics": cv_id,
        "wh_cv_ok": cv_gz is not None
        and cv_gz <= THRESHOLDS["wh_cv_max"]
        and cv_id is not None
        and cv_id <= THRESHOLDS["wh_cv_max"],
        "meter_cpu_share_percent_max": share_max,
        "meter_cpu_core_percent_max": core_max,
        "container_cpu_core_percent_median": container_med,
        "meter_cpu_ok": share_max is not None
        and share_max <= THRESHOLDS["meter_cpu_share_percent_max"],
        "trajectory_wh_gazebo_A": [r["wh_gazebo_A"] for r in trajs],
        "trajectory_wh_inverse_dynamics_A": [r["wh_inverse_dynamics_A"] for r in trajs],
        "torque_source": "gazebo_effort" if gazebo_usable else "inverse_dynamics",
    }
    summary["passed"] = bool(
        summary["segments_all_ok"] and summary["wh_cv_ok"] and summary["meter_cpu_ok"]
    )
    return {
        "summary": summary,
        "static": static_rows,
        "motion": motion_rows,
        "savgol_window": compare.get("window"),
        "plan_seed": plan["seed"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--image", default="armbench-sim:dev")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-traj", type=int, default=20)
    parser.add_argument("--n-rep", type=int, default=10)
    parser.add_argument("--work-dir", type=Path, default=Path("reports/_f2_torque_work"))
    parser.add_argument("--out", type=Path, default=Path("reports/f2_torque_source.json"))
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)

    plan = make_plan(args.seed, args.n_traj, args.n_rep)
    plan_dir = args.work_dir.parent / "_f2_plan"
    plan_dir.mkdir(parents=True, exist_ok=True)
    (plan_dir / "plan.json").write_text(json.dumps(plan, indent=2))
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return 0
    container = run_in_container(args.image, plan_dir, args.work_dir)
    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "image": args.image,
        "host": {"cpus": __import__("os").cpu_count(), "platform": platform.platform()},
        "thresholds": THRESHOLDS,
        "container": container,
    }
    if container.get("ok") or (args.work_dir / "compare.json").exists():
        report.update(analyse(args.work_dir, plan))
    else:
        report["summary"] = {"passed": False}
    report["passed"] = bool(report["summary"].get("passed")) and bool(container.get("ok"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    print(f"passed={report['passed']} -> {args.out}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
