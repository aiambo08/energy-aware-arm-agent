"""Pre-registered evaluation protocol: manifest, artefact hashes and the final-eval unlock.

``docs/protocol.md`` carries a fenced ``armbench-protocol`` YAML block that pins the provider,
model, agents, tasks, analysis parameters and the hashes of everything an episode depends on
(``code_tree_sha256`` over the tracked files under :data:`TREE_ROOTS`, the frozen skill library
and the energy reference). The protocol hash is the sha256 of the protocol file's bytes; the
runner unlocks the ``final_eval`` seeds only when ``--protocol-hash`` equals it and every pinned
artefact still matches the working tree.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from pathlib import Path
from typing import Final

import yaml
from pydantic import BaseModel, ConfigDict, Field

from armbench.energy import EnergyReference
from armbench.llm import load_llm_params
from armbench.paths import REPO_ROOT
from armbench.skills import SkillLibrary

PROTOCOL_FILE: Final = REPO_ROOT / "docs" / "protocol.md"
TREE_ROOTS: Final[tuple[str, ...]] = (
    "configs",
    "docker",
    "energy_ref",
    "ros_ws/src",
    "skills",
    "src/armbench",
)
"""Everything that decides what an episode does or how it is scored (tests and docs excluded)."""
_BLOCK = re.compile(r"```armbench-protocol\n(.*?)```", re.DOTALL)


class ProtocolError(Exception):
    """The protocol file is missing, malformed, or no longer matches the repository."""


class Analysis(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    bootstrap_resamples: int = Field(ge=100)
    bootstrap_seed: int
    ci_level: float = Field(gt=0, lt=1)
    noninferiority_margin: float = Field(ge=0, lt=1)
    min_valid_fraction: float = Field(gt=0, le=1)


class Artefacts(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    code_tree_sha256: str
    skills_sha256: str
    energy_reference_sha256: str


class Manifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int
    protocol_id: str
    split: str
    llm_config: str
    model: str
    agents: tuple[str, ...]
    tasks: tuple[str, ...]
    hypotheses: dict[str, tuple[str, str]]
    """Hypothesis id -> (treatment agent, control agent)."""
    analysis: Analysis
    artefacts: Artefacts


def protocol_sha256(path: Path = PROTOCOL_FILE) -> str:
    if not path.is_file():
        msg = f"no protocol at {path}"
        raise ProtocolError(msg)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest(path: Path = PROTOCOL_FILE) -> Manifest:
    if not path.is_file():
        msg = f"no protocol at {path}"
        raise ProtocolError(msg)
    blocks = _BLOCK.findall(path.read_text())
    if len(blocks) != 1:
        msg = f"{path} must contain exactly one ```armbench-protocol block, found {len(blocks)}"
        raise ProtocolError(msg)
    try:
        return Manifest.model_validate(yaml.safe_load(blocks[0]))
    except ValueError as exc:
        msg = f"malformed protocol manifest in {path}: {exc}"
        raise ProtocolError(msg) from exc


def _git(root: Path, *args: str) -> str:
    try:
        proc = subprocess.run(  # noqa: S603 - fixed git invocation
            [shutil.which("git") or "git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        msg = f"git {' '.join(args)} failed in {root}: {exc}"
        raise ProtocolError(msg) from exc
    return proc.stdout


def tracked_files(root: Path = REPO_ROOT, roots: tuple[str, ...] = TREE_ROOTS) -> list[str]:
    out = _git(root, "ls-files", "-z", "--", *roots)
    return sorted(p for p in out.split("\0") if p)


def code_tree_sha256(root: Path = REPO_ROOT, roots: tuple[str, ...] = TREE_ROOTS) -> str:
    """sha256 over ``path\\0sha256(contents)\\n`` for every tracked file under ``roots``,
    read from the working tree (so uncommitted edits change the hash)."""
    h = hashlib.sha256()
    for rel in tracked_files(root, roots):
        p = root / rel
        digest = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else "missing"
        h.update(f"{rel}\0{digest}\n".encode())
    return h.hexdigest()


def dirty_files(root: Path = REPO_ROOT, roots: tuple[str, ...] = TREE_ROOTS) -> list[str]:
    """Tracked or untracked changes under ``roots`` that are not committed."""
    out = _git(root, "status", "--porcelain", "--untracked-files=all", "--", *roots)
    return [line[3:] for line in out.splitlines() if line.strip()]


def current_artefacts(root: Path = REPO_ROOT) -> Artefacts:
    return Artefacts(
        code_tree_sha256=code_tree_sha256(root),
        skills_sha256=SkillLibrary.load(root / "skills", require_frozen=True).sha256(),
        energy_reference_sha256=EnergyReference.load(root / "energy_ref").sha256(),
    )


def verify(path: Path = PROTOCOL_FILE, root: Path = REPO_ROOT) -> list[str]:
    """Problems that forbid a final-eval run (empty list = the repository matches)."""
    manifest = load_manifest(path)
    problems = [f"uncommitted change: {f}" for f in dirty_files(root)]
    current = current_artefacts(root)
    for field, pinned in manifest.artefacts.model_dump().items():
        now = getattr(current, field)
        if now != pinned:
            problems.append(f"{field}: protocol pins {pinned[:12]}…, repository has {now[:12]}…")
    return problems


def authorise_final_eval(  # noqa: PLR0913
    protocol_hash: str | None,
    llm_config: Path,
    agent: str,
    tasks: list[str],
    *,
    path: Path = PROTOCOL_FILE,
    root: Path = REPO_ROOT,
) -> Manifest:
    """Raise :class:`ProtocolError` unless a final-eval run matches the frozen protocol."""
    expected = protocol_sha256(path)
    if protocol_hash != expected:
        msg = f"--protocol-hash must be the sha256 of {path.name} ({expected[:12]}…)"
        raise ProtocolError(msg)
    manifest = load_manifest(path)
    problems = verify(path, root)
    if agent not in manifest.agents:
        problems.append(f"agent {agent!r} is not in the protocol")
    if agent != "A" and Path(llm_config).resolve() != (root / manifest.llm_config).resolve():
        problems.append(f"--llm-config must be {manifest.llm_config}")
    elif agent != "A" and load_llm_params(Path(llm_config)).model != manifest.model:
        problems.append(f"the LLM profile does not pin {manifest.model}")
    extra = sorted(set(tasks) - set(manifest.tasks))
    if extra:
        problems.append(f"tasks not in the protocol: {extra}")
    if problems:
        raise ProtocolError("; ".join(problems))
    return manifest
