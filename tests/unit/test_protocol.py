"""F9 protocol freeze: manifest, artefact hashes and the --final-eval unlock."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from armbench.cli import app
from armbench.paths import REPO_ROOT
from armbench.protocol import (
    PROTOCOL_FILE,
    ProtocolError,
    authorise_final_eval,
    code_tree_sha256,
    current_artefacts,
    load_manifest,
    protocol_sha256,
    verify,
)

GIT = shutil.which("git") or "git"


def _git(root: Path, *args: str) -> None:
    subprocess.run([GIT, "-C", str(root), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A committed mini repository with the real skills, energy reference and LLM profile."""
    for d in ("skills", "energy_ref"):
        shutil.copytree(REPO_ROOT / d, tmp_path / d)
    (tmp_path / "configs").mkdir()
    shutil.copy(REPO_ROOT / "configs/llm.nebius.yaml", tmp_path / "configs")
    shutil.copy(REPO_ROOT / "configs/llm.yaml", tmp_path / "configs")
    (tmp_path / "src/armbench").mkdir(parents=True)
    (tmp_path / "src/armbench/x.py").write_text("X = 1\n")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init")
    text = PROTOCOL_FILE.read_text()
    pinned = load_manifest().artefacts.code_tree_sha256
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/protocol.md").write_text(text.replace(pinned, code_tree_sha256(tmp_path)))
    return tmp_path


def test_repository_protocol_pins_the_frozen_library_and_reference() -> None:
    m = load_manifest()
    cur = current_artefacts()
    assert m.split == "final_eval"
    assert m.agents == ("A", "B", "B+S", "C", "C+S")
    assert m.artefacts.skills_sha256 == cur.skills_sha256
    assert m.artefacts.energy_reference_sha256 == cur.energy_reference_sha256
    assert set(m.hypotheses) == {"H1", "H2", "H3"}
    assert m.analysis.bootstrap_resamples == 10_000


def test_clean_repo_unlocks_and_any_change_refuses(repo: Path) -> None:
    proto = repo / "docs/protocol.md"
    h = protocol_sha256(proto)
    cfg = repo / "configs/llm.nebius.yaml"
    assert verify(proto, repo) == []
    tasks = ["pick_place@1"]
    authorise_final_eval(h, cfg, "C+S", tasks, path=proto, root=repo)
    with pytest.raises(ProtocolError, match="protocol-hash"):
        authorise_final_eval("deadbeef", cfg, "C+S", tasks, path=proto, root=repo)
    with pytest.raises(ProtocolError, match="llm-config"):
        authorise_final_eval(h, repo / "configs/llm.yaml", "B", tasks, path=proto, root=repo)
    with pytest.raises(ProtocolError, match="not in the protocol"):
        authorise_final_eval(h, cfg, "D", ["other@1"], path=proto, root=repo)
    (repo / "src/armbench/x.py").write_text("X = 2\n")
    problems = verify(proto, repo)
    assert any("uncommitted" in p for p in problems)
    assert any("code_tree_sha256" in p for p in problems)
    with pytest.raises(ProtocolError):
        authorise_final_eval(h, cfg, "A", tasks, path=proto, root=repo)


def test_untracked_file_under_a_root_refuses(repo: Path) -> None:
    (repo / "src/armbench/new.py").write_text("")
    assert verify(repo / "docs/protocol.md", repo)


def test_protocol_must_have_one_manifest(tmp_path: Path) -> None:
    p = tmp_path / "p.md"
    p.write_text("# nothing\n")
    with pytest.raises(ProtocolError, match="exactly one"):
        load_manifest(p)


def test_cli_refuses_final_eval_without_the_registered_hash(tmp_path: Path) -> None:
    res = CliRunner().invoke(app, [
        "run", "--task", "pick_place@1", "--agent", "A", "--seeds", "final_eval",
        "--backend", "fake", "--final-eval", "--protocol-hash", "not-the-hash",
        "--out", str(tmp_path / "r"),
    ])  # fmt: skip
    assert res.exit_code == 2
    assert "refused" in res.output
    assert not (tmp_path / "r").exists()


def test_cli_protocol_hash_matches_the_file() -> None:
    res = CliRunner().invoke(app, ["protocol", "hash"])
    assert res.exit_code == 0
    assert res.output.strip() == protocol_sha256()
