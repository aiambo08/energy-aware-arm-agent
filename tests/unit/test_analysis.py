"""F9 pre-registered analysis: episode selection, paired bootstrap, decision rules."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from armbench.agents import AgentTrace
from armbench.analysis import AnalysisError, bootstrap, markdown, select
from armbench.analysis import analyse as _analyse
from armbench.energy import Variant
from armbench.protocol import load_manifest
from armbench.runner import EnergyRecord, EnergyRow, EpisodeRecord, Software

TASKS = ("pick_place@1", "stack2@1", "sort3@1", "place_obstacle@1")
ROWS = [(str(v), e) for v in Variant for e in (0.70, 0.60, 0.80)]
MANIFEST = load_manifest()
SEEDS = range(100, 120)


def _energy(wh: float) -> EnergyRecord:
    rows = tuple(
        EnergyRow(variant=v, eta=e, duration_s=5.0, n_samples=500, mechanical_j=1.0,
                  copper_j=1.0, base_j=500.0, total_j=wh * 3600, total_wh=wh * (2 - e))
        for v in Variant for e in (0.70, 0.60, 0.80)
    )  # fmt: skip
    return EnergyRecord(source="gazebo_effort", topic="/e", n_samples=500, rate_hz=100.0, rows=rows)


def _rec(agent: str, task: str, seed: int, *, ok: bool = True, wh: float = 0.2,
         infra: bool = False, at: str = "1") -> EpisodeRecord:  # fmt: skip
    return EpisodeRecord(
        run_id=agent, episode_id=f"{agent}{task}{seed}{at}", task=task, task_version=1,
        agent=agent, seed=seed, repeat=0, backend="sim", instance_sha256="x", started_at=at,
        ok=ok, reason="" if ok else "fail", infra_failure=infra, sim_s=6.0, wall_s=20.0,
        energy=_energy(wh), trace=AgentTrace(prompt_tokens=10, completion_tokens=5),
        software=Software(armbench="test"),
    )  # fmt: skip


def _write(root: Path, agent: str, recs: list[EpisodeRecord], sha: str | None = "p") -> Path:
    d = root / agent.replace("+", "S")
    d.mkdir()
    meta = {"run_id": agent, "agent": agent, "protocol_sha256": sha, "seeds": list(SEEDS)}
    (d / "run.json").write_text(json.dumps(meta))
    (d / "episodes.jsonl").write_text("".join(r.line() + "\n" for r in recs))
    return d


def _runs(root: Path, wh: dict[str, float], fail: dict[str, int] | None = None) -> list[Path]:
    rng = np.random.default_rng(0)
    fail = fail or {}
    dirs = []
    for agent in MANIFEST.agents:
        recs = [
            _rec(agent, t, s, ok=(s - 100) >= fail.get(agent, 0),
                 wh=wh.get(agent, 0.2) * (1 + 0.002 * rng.standard_normal()))
            for t in TASKS for s in SEEDS
        ]  # fmt: skip
        dirs.append(_write(root, agent, recs))
    return dirs


def analyse(*args: Any) -> Any:
    """The report as plain JSON (also checks it serialises)."""
    return json.loads(json.dumps(_analyse(*args)))


def test_selection_skips_infra_failures_and_keeps_latest() -> None:
    recs = [
        _rec("B", TASKS[0], 100, infra=True, at="1"),
        _rec("B", TASKS[0], 100, ok=False, at="2"),
        _rec("B", TASKS[0], 101, ok=False, at="1"),
        _rec("B", TASKS[0], 101, ok=True, at="2"),
    ]
    scored, infra = select(recs)
    assert not scored["B"][(TASKS[0], 100)].ok
    assert scored["B"][(TASKS[0], 101)].ok
    assert len(infra["B"]) == 1


def test_bootstrap_is_seeded_and_brackets_the_point() -> None:
    data = [np.arange(20, dtype=float), np.arange(20, dtype=float) + 5]
    a = bootstrap(data, lambda x: np.median(x, axis=-1), MANIFEST.analysis,
                  np.random.default_rng(1))  # fmt: skip
    b = bootstrap(data, lambda x: np.median(x, axis=-1), MANIFEST.analysis,
                  np.random.default_rng(1))  # fmt: skip
    assert a == b
    assert a[1] <= a[0] <= a[2]


def test_clear_saving_supports_h2_h3(tmp_path: Path) -> None:
    dirs = _runs(tmp_path, {"C": 0.18, "C+S": 0.18}, fail={"B": 5})
    rep = analyse(dirs, MANIFEST, ROWS, "p")
    v = rep["verdicts"]
    assert v == {"H1": "supported", "H2": "supported", "H3": "supported",
                 "scenario": "favourable"}  # fmt: skip
    h2 = rep["comparisons"]["H2"]
    assert len(h2["energy"]) == 6
    assert h2["energy"][0]["median_rel_diff"] == pytest.approx(0.18 / 0.2 - 1, abs=0.01)
    assert rep["runs_match_protocol"]
    assert rep["n_scored"] == 400
    assert "H2" in markdown(_analyse(dirs, MANIFEST, ROWS, "p"))


def test_saving_bought_with_failures_is_not_supported(tmp_path: Path) -> None:
    dirs = _runs(tmp_path, {"C": 0.18}, fail={"C": 6})
    v = analyse(dirs, MANIFEST, ROWS, "p")["verdicts"]
    assert v["H2"] == "mixed"
    assert v["H3"] == "not supported"
    assert v["H1"] == "not supported"
    assert v["scenario"] == "mixed"


def test_no_difference_is_null(tmp_path: Path) -> None:
    v = analyse(_runs(tmp_path, {}), MANIFEST, ROWS, "p")["verdicts"]
    assert v["scenario"] == "null"


def test_report_is_deterministic(tmp_path: Path) -> None:
    dirs = _runs(tmp_path, {"C": 0.19})
    assert (analyse(dirs, MANIFEST, ROWS, "p")["report_sha256"]
            == analyse(dirs, MANIFEST, ROWS, "p")["report_sha256"])  # fmt: skip


def test_missing_cells_or_infra_make_a_technical_failure(tmp_path: Path) -> None:
    dirs = _runs(tmp_path, {})
    log = dirs[1] / "episodes.jsonl"
    lines = log.read_text().splitlines()
    bad = EpisodeRecord.model_validate_json(lines[0]).model_copy(update={"infra_failure": True})
    log.write_text("\n".join([bad.line(), *lines[1:]]) + "\n")
    rep = analyse(dirs, MANIFEST, ROWS, "other")
    assert rep["verdicts"] == {"scenario": "technical failure"}
    assert rep["missing"] == ["B/pick_place@1/100"]
    assert not rep["runs_match_protocol"]


def test_not_a_run_directory(tmp_path: Path) -> None:
    with pytest.raises(AnalysisError):
        _analyse([tmp_path], MANIFEST, ROWS, None)
