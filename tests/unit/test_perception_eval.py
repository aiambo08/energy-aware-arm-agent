"""F3 gate script: seed parsing, matching summary and verdict against the thresholds."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "perception_eval", Path(__file__).resolve().parents[2] / "scripts" / "perception_eval.py"
)
assert SPEC is not None
assert SPEC.loader is not None
pe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pe  # dataclasses resolve postponed annotations via sys.modules
SPEC.loader.exec_module(pe)


def scene(seed: int, n: int, matched: int, fp: int = 0, latency: float = 8.0) -> object:
    return pe.SceneResult(
        seed=seed,
        n_cubes=n,
        n_visible=n,
        n_detected=matched + fp,
        n_matched=matched,
        n_false_positive=fp,
        latency_ms=latency,
        missed=[f"cube_{i}" for i in range(matched, n)],
    )


def match(seed: int, xy: float, z: float = 0.1, color: str = "red") -> object:
    return pe.Match(
        seed=seed,
        name="cube_0",
        color=color,
        n_cubes=3,
        xy_mm=xy,
        z_mm=z,
        yaw_deg=0.5,
        pixel_r=100.0,
    )


def test_parse_seeds() -> None:
    assert pe.parse_seeds("200-203") == [200, 201, 202, 203]
    assert pe.parse_seeds("5,7") == [5, 7]


def test_thresholds_match_plan() -> None:
    t = pe.THRESHOLDS
    assert t["min_scenes"] == 200
    assert t["recall_min"] == 0.99
    assert t["false_positive_rate_max"] == 0.01
    assert (t["xy_median_mm_max"], t["xy_p95_mm_max"], t["z_p95_mm_max"]) == (5.0, 10.0, 10.0)
    assert t["latency_p95_ms_max"] == 50.0


def test_summary_and_verdict_pass() -> None:
    scenes = [scene(s, 4, 4) for s in range(200)]
    matches = [match(s, xy=1.0 + 0.01 * s) for s in range(200)]
    metrics = pe.summarise(scenes, matches, 0.0005)
    assert metrics["recall"] == 1.0
    assert metrics["false_positive_rate"] == 0.0
    assert metrics["n_cubes_gt"] == 800
    assert metrics["gazebo_vs_requested_pose_max_mm"] == pytest.approx(0.5)
    checks = pe.verdict(metrics)
    assert checks["passed"] is True


def test_verdict_fails_on_each_threshold() -> None:
    good = [scene(s, 3, 3) for s in range(200)]
    good_m = [match(s, xy=1.0) for s in range(200)]
    assert pe.verdict(pe.summarise(good[:199], good_m, 0.0))["scenes_ok"] is False
    missed = [*good[:190], *(scene(s, 3, 2) for s in range(190, 200))]  # 10 misses / 600
    assert pe.verdict(pe.summarise(missed, good_m, 0.0))["recall_ok"] is False
    fps = [*good[:190], *(scene(s, 3, 3, fp=1) for s in range(190, 200))]  # 10 FP / 600 > 1 %
    assert pe.verdict(pe.summarise(fps, good_m, 0.0))["false_positive_ok"] is False
    coarse = [match(s, xy=6.0) for s in range(200)]
    assert pe.verdict(pe.summarise(good, coarse, 0.0))["xy_median_ok"] is False
    tail = [*good_m[:180], *(match(s, xy=11.0) for s in range(20))]
    v = pe.verdict(pe.summarise(good, tail, 0.0))
    assert v["xy_median_ok"] is True
    assert v["xy_p95_ok"] is False
    tall = [match(s, xy=1.0, z=-12.0) for s in range(200)]
    assert pe.verdict(pe.summarise(good, tall, 0.0))["z_p95_ok"] is False
    slow = [scene(s, 3, 3, latency=60.0) for s in range(200)]
    assert pe.verdict(pe.summarise(slow, good_m, 0.0))["latency_p95_ok"] is False


def test_summary_handles_no_matches() -> None:
    metrics = pe.summarise([], [], 0.0)
    assert metrics["recall"] == 0.0
    assert metrics["xy_mm"] == {"n": 0}
    assert pe.verdict(metrics)["passed"] is False
    assert math.isinf(pe.THRESHOLDS.get("missing", math.inf))
