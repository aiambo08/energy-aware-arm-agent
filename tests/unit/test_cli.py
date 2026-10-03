import json

from typer.testing import CliRunner

from armbench import __version__
from armbench.cli import app

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
