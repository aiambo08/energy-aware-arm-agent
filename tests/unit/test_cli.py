import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from armbench import __version__
from armbench.cli import app
from armbench.energy.log import Sample, write_samples_jsonl

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.output.strip() == __version__


def test_seeds_table() -> None:
    result = runner.invoke(app, ["seeds"])
    assert result.exit_code == 0
    assert "final_eval" in result.output
    assert "[LOCKED]" in result.output


def test_seeds_json() -> None:
    result = runner.invoke(app, ["seeds", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["splits"]["final_eval"]["locked"] is True


def test_scene_sdf_fragment() -> None:
    result = runner.invoke(app, ["scene", "--seed", "3"])
    assert result.exit_code == 0, result.output
    assert result.output.lstrip().startswith('<model name="cube_0">')
    assert "<sdf" not in result.output
    assert "gz::sim::systems::PosePublisher" in result.output


def test_scene_json_and_n_cubes() -> None:
    result = runner.invoke(app, ["scene", "--seed", "3", "--n-cubes", "2", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["seed"] == 3
    assert [c["name"] for c in data["cubes"]] == ["cube_0", "cube_1"]


def test_scene_is_deterministic() -> None:
    a = runner.invoke(app, ["scene", "--seed", "9"]).output
    b = runner.invoke(app, ["scene", "--seed", "9"]).output
    assert a == b


def test_scene_world_from_template(tmp_path: Path) -> None:
    template = tmp_path / "w.sdf.in"
    template.write_text(
        '<sdf version="1.9"><world name="t">\n    <!-- ARMBENCH_SCENE -->\n</world></sdf>\n'
    )
    out = tmp_path / "world.sdf"
    result = runner.invoke(
        app, ["scene", "--seed", "1", "--template", str(template), "--out", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert "wrote" in result.output
    text = out.read_text()
    assert "ARMBENCH_SCENE" not in text
    assert text.startswith("<sdf")
    assert '<model name="cube_0">' in text


def test_scene_missing_marker_fails(tmp_path: Path) -> None:
    template = tmp_path / "w.sdf.in"
    template.write_text("<sdf/>\n")
    result = runner.invoke(app, ["scene", "--seed", "1", "--template", str(template)])
    assert result.exit_code != 0


def test_scene_requires_seed() -> None:
    result = runner.invoke(app, ["scene"])
    assert result.exit_code != 0


def test_fk_zero() -> None:
    result = runner.invoke(app, ["fk", "0", "0", "0", "0", "0", "0"])
    assert result.exit_code == 0, result.output
    xyz_line, rotvec_line = result.output.strip().splitlines()
    xyz = [float(v) for v in xyz_line.split()[1:]]
    assert xyz == pytest.approx([0.8172, 0.2329, 0.0628], abs=1e-4)
    assert rotvec_line.startswith("rotvec_rad:")


def test_fk_accepts_negative_angles() -> None:
    result = runner.invoke(app, ["fk", "0", "-1.57", "0", "-1.57", "0", "0"])
    assert result.exit_code == 0, result.output
    xyz = [float(v) for v in result.output.splitlines()[0].split()[1:]]
    assert xyz == pytest.approx([0.0008, 0.2329, 1.0794], abs=2e-3)


def test_fk_wrong_arity() -> None:
    result = runner.invoke(app, ["fk", "0", "0", "0"])
    assert result.exit_code != 0


def test_energy_command_table_and_json(tmp_path: Path) -> None:
    log = tmp_path / "ep.jsonl"
    write_samples_jsonl(
        log, [Sample(t=0.1 * i, q=(0.0,) * 6, qd=(0.5,) * 6, tau=(10.0,) * 6) for i in range(11)]
    )
    result = runner.invoke(app, ["energy", str(log)])
    assert result.exit_code == 0, result.output
    assert "A " in result.output
    assert "Wh" in result.output
    result = runner.invoke(app, ["energy", str(log), "--full", "--json"])
    assert result.exit_code == 0, result.output
    rows = json.loads(result.output)
    assert len(rows) == 6
    assert {r["variant"] for r in rows} == {"A", "B"}
    result = runner.invoke(app, ["energy", str(log), "--variant", "B", "--eta", "0.5", "--json"])
    assert json.loads(result.output)[0]["eta"] == 0.5
