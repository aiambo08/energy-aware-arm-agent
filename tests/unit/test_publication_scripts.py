"""F10 publication scripts: deterministic dataset bundle, replay demo helpers, results page."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tarfile
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "data" / "f9-v1.tar.gz"
REPORT = ROOT / "reports" / "f9_eval.json"


def load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


export = load("export_dataset")
demo = load("demo")
site = load("build_site")

REPORT_STUB: dict[str, object] = {"protocol_sha256": "p" * 64, "report_sha256": "r" * 64}


def fake_run(root: Path, name: str) -> Path:
    run = root / name
    (run / "programs").mkdir(parents=True)
    (run / "llm" / "cache" / "ab").mkdir(parents=True)
    (run / "samples").mkdir()
    (run / "run.json").write_text('{"agent": "B"}\n')
    (run / "episodes.jsonl").write_text('{"task": "pick_place@1", "seed": 100}\n')
    (run / "programs" / "e1.py").write_text("robot.observe()\n")
    (run / "llm" / "cache" / "ab" / "abcd.json").write_text("{}\n")
    (run / "samples" / "e1.jsonl.gz").write_bytes(b"\x1f\x8b")
    return run


@pytest.fixture(scope="module")
def extracted(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("f9")
    with tarfile.open(BUNDLE) as tar:
        tar.extractall(out, filter="data")
    return out / "f9-v1"


def test_bundle_is_deterministic_and_manifest_matches(tmp_path: Path) -> None:
    runs = [fake_run(tmp_path, "r1"), fake_run(tmp_path, "r2")]
    a = export.build(runs, tmp_path / "a.tar.gz", full=False, report=REPORT_STUB)
    (runs[0] / "run.json").touch()  # mtime changes must not change the archive
    b = export.build(runs, tmp_path / "b.tar.gz", full=False, report=REPORT_STUB)
    assert a == b == hashlib.sha256((tmp_path / "a.tar.gz").read_bytes()).hexdigest()
    with tarfile.open(tmp_path / "a.tar.gz") as tar:
        members = tar.getmembers()
        assert all(m.mtime == 0 and m.uid == 0 and m.gid == 0 for m in members)
        names = [m.name for m in members]
        manifest = json.loads(tar.extractfile("f9-v1/MANIFEST.json").read())  # type: ignore[union-attr]
        for name, digest in manifest["files"].items():
            data = tar.extractfile(name).read()  # type: ignore[union-attr]
            assert hashlib.sha256(data).hexdigest() == digest
    assert names == sorted(names)
    assert not any("/samples/" in n for n in names)
    assert "f9-v1/r1/llm/cache/ab/abcd.json" in names
    assert manifest["protocol_sha256"] == "p" * 64


def test_full_bundle_adds_samples(tmp_path: Path) -> None:
    runs = [fake_run(tmp_path, "r1")]
    export.build(runs, tmp_path / "full.tar.gz", full=True, report=REPORT_STUB)
    with tarfile.open(tmp_path / "full.tar.gz") as tar:
        assert "f9-v1/r1/samples/e1.jsonl.gz" in tar.getnames()


def test_export_refuses_missing_runs(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text(json.dumps({**REPORT_STUB, "runs": [{"dir": "nope", "agent": "B"}]}))
    rc = export.main(["--report", str(report), "--runs-root", str(tmp_path)])
    assert rc == 2


def test_committed_bundle_matches_the_f9_report(extracted: Path) -> None:
    report = json.loads(REPORT.read_text())
    manifest = json.loads((extracted / "MANIFEST.json").read_text())
    assert manifest["protocol_sha256"] == report["protocol_sha256"]
    assert manifest["report_sha256"] == report["report_sha256"]
    for r in report["runs"]:
        assert (extracted / str(r["dir"]) / "episodes.jsonl").is_file()
    assert set(demo.RUN_DIRS.values()) == {str(r["dir"]) for r in report["runs"]}


def test_demo_find_and_wh(extracted: Path) -> None:
    rec = demo.find(extracted / "f9_C" / "episodes.jsonl", "pick_place@1", 100)
    assert rec["seed"] == 100
    value = demo.wh(rec)
    assert value is not None
    assert 0.1 < value < 1.0
    assert demo.wh({"energy": None}) is None
    with pytest.raises(SystemExit):
        demo.find(extracted / "f9_C" / "episodes.jsonl", "pick_place@1", 5)


def test_demo_rejects_unknown_task() -> None:
    with pytest.raises(SystemExit):
        demo.main(["--task", "juggle"])


def test_seed_cache_copies_once(extracted: Path, tmp_path: Path) -> None:
    run = extracted / "f9_B"
    n = demo.seed_cache(run, tmp_path / "cache")
    assert n == len(list((run / "llm" / "cache").rglob("*.json"))) > 0
    assert demo.seed_cache(run, tmp_path / "cache") == 0


def test_site_success_matches_the_report(extracted: Path) -> None:
    report = json.loads(REPORT.read_text())
    runs = site.load_runs(extracted, report)
    points = {(p["task"], p["agent"]): p for p in site.pareto(runs)}
    expected = {"A": 80, "B": 62, "B+S": 78, "C": 76, "C+S": 80}
    for agent, ok in expected.items():
        assert points[("all", agent)]["n"] == 80
        assert points[("all", agent)]["success"] == pytest.approx(ok / 80)
        assert points[("all", agent)]["wh_median"] is not None


def test_site_scene_is_seeded() -> None:
    a = site.scene("pick_place@1", 100)
    assert a == site.scene("pick_place@1", 100)
    assert a["cubes"]
    assert a["obstacles"] == []
    assert site.scene("place_obstacle@1", 100)["obstacles"]


def test_committed_page_embeds_the_report_hashes() -> None:
    html = (ROOT / "site" / "index.html").read_text()
    report = json.loads(REPORT.read_text())
    assert str(report["report_sha256"]) in html
    assert str(report["protocol_sha256"]) in html
    assert "/*__DATA__*/" not in html
    assert "<script src=" not in html
