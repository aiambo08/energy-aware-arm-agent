"""F4 gate script: summaries and verdict against the thresholds on synthetic node output."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

Raw = dict[str, object]

SPEC = importlib.util.spec_from_file_location(
    "primitives_eval", Path(__file__).resolve().parents[2] / "scripts" / "primitives_eval.py"
)
assert SPEC is not None
assert SPEC.loader is not None
pe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pe
SPEC.loader.exec_module(pe)


def move(pos_mm: float = 1.0, yaw_deg: float = 0.1, tip_z: float = 0.01, speed: float = 1.0) -> Raw:
    return {
        "pos_err_mm": pos_mm,
        "yaw_err_deg": yaw_deg,
        "min_tip_z_m": tip_z,
        "min_frame_z_m": 0.1,
        "sim_s": 1.0 / speed,
        "planned_s": 1.0 / speed,
        "speed_scale": speed,
    }


def contracts(
    *,
    n_moves: int = 200,
    n_unreachable: int = 100,
    n_reset: int = 100,
    raised: str = "out_of_reach",
    dq: float = 0.0,
    reset_s: float = 2.0,
    scales: tuple[float, ...] = (0.1, 0.5, 1.0),
    moves: list[Raw] | None = None,
) -> Raw:
    if moves is None:
        moves = [move() for _ in range(n_moves)]
    return {
        "accuracy": {"moves": moves, "rejected": []},
        "unreachable": {
            "cases": [
                {"raised": raised, "max_dq_rad": dq, "goals_sent": 0} for _ in range(n_unreachable)
            ]
        },
        "speed": {"moves": [move(speed=s) for s in scales]},
        "reset": {
            "resets": [
                {"sim_s": reset_s, "wall_s": 3.0, "q_err_rad": 0.001, "opening_m": 0.085}
                for _ in range(n_reset)
            ]
        },
    }


def episode(seed: int, ok: bool = True, code: str | None = None) -> Raw:
    e: Raw = {"seed": seed, "ok": ok, "n_cubes": 4, "n_detections": 4, "place_err_mm": 1.0}
    if code is not None:
        e["error"] = {"code": code}
    return e


def test_parse_seeds() -> None:
    assert pe.parse_seeds("400-403") == [400, 401, 402, 403]
    assert pe.parse_seeds("5,7") == [5, 7]


def test_thresholds_match_plan() -> None:
    t = pe.THRESHOLDS
    assert (t["moves_min"], t["pos_p95_mm_max"], t["yaw_p95_deg_max"]) == (200, 5.0, 2.0)
    assert (t["collisions_max"], t["unreachable_min"], t["unreachable_detected_min"]) == (
        0,
        100,
        1.0,
    )
    assert (t["resets_min"], t["reset_sim_s_max"]) == (100, 5.0)
    assert (t["pick_seeds_min"], t["pick_success_min"]) == (100, 0.95)


def test_pass() -> None:
    c = pe.summarize_contracts(contracts())
    p = pe.summarize_pick([{"episodes": [episode(s) for s in range(100)]}])
    checks = pe.judge(c, p)
    assert checks["passed"] is True
    assert c["speed"]["strictly_decreasing"] is True
    assert p["success"] == 1.0


def test_each_threshold_fails() -> None:
    good_pick = pe.summarize_pick([{"episodes": [episode(s) for s in range(100)]}])

    def fail(c: Raw) -> dict[str, bool]:
        checks: dict[str, bool] = pe.judge(pe.summarize_contracts(c), good_pick)
        return checks

    assert fail(contracts(n_moves=199))["moves_n"] is False
    assert fail(contracts(raised="collision"))["unreachable_detected"] is False
    assert fail(contracts(dq=0.01))["unreachable_no_motion"] is False
    assert fail(contracts(reset_s=6.0))["reset_time"] is False
    assert fail(contracts(scales=(1.0, 0.5)))["speed_monotone"] is False
    hit = [move(tip_z=-0.01), *(move() for _ in range(199))]
    assert fail(contracts(moves=hit))["collisions"] is False
    coarse = [*(move(pos_mm=6.0) for _ in range(20)), *(move() for _ in range(180))]
    assert fail(contracts(moves=coarse))["pos_p95"] is False
    assert fail({"error": "boom"})["contracts_complete"] is False
    pick = pe.summarize_pick(
        [
            {
                "episodes": [
                    *(episode(s) for s in range(94)),
                    *(episode(s, False, "timeout") for s in range(94, 100)),
                ]
            }
        ]
    )
    checks = pe.judge(pe.summarize_contracts(contracts()), pick)
    assert checks["pick_success"] is False
    assert pick["failures_by_code"] == {"timeout": 6}
    assert pick["failed_seeds"] == list(range(94, 100))


def test_empty_pick() -> None:
    p = pe.summarize_pick([])
    assert p["n"] == 0
    assert math.isnan(p["success"])
    assert pe.judge(pe.summarize_contracts(contracts()), p)["pick_n"] is False
