"""Skill library: a JSON file of validated skills, deduplicated by signature, retrievable by
description, and frozen (hash recorded) before any evaluation run (D7 in docs/plan.es.md)."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from armbench import __version__
from armbench.paths import SKILLS_DIR
from armbench.skills.spec import Skill

LIBRARY_FILE: Final = "library.json"
FROZEN_FILE: Final = "FROZEN.json"

_WORD_RE = re.compile(r"[a-z]{3,}")
STOPWORDS: Final[frozenset[str]] = frozenset(
    {
        "the", "and", "for", "with", "that", "its", "this", "not", "are", "does", "from",
        "into", "onto", "where", "each", "other", "them", "then", "than", "which", "metres",
        "base", "link", "centre", "centred", "resting", "given", "point", "identified", "one",
    }
)  # fmt: skip


class LibraryError(RuntimeError):
    pass


class FrozenLibraryError(LibraryError):
    """Mutation attempted on a frozen library."""


class Manifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    sha256: str
    n_skills: int = Field(ge=0)
    frozen_at: str
    armbench: str
    skills: dict[str, str]
    """name -> skill sha256, so a reader can tell which skill changed."""


def tokens(text: str) -> frozenset[str]:
    return frozenset(w for w in _WORD_RE.findall(text.lower()) if w not in STOPWORDS)


class Retrieved(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    skill: Skill
    score: float = Field(ge=0, le=1)
    overlap: tuple[str, ...]


class SkillLibrary:
    """Ordered by insertion; names are unique and so are signatures."""

    def __init__(self, skills: Iterable[Skill] = (), *, frozen: bool = False) -> None:
        self._skills: dict[str, Skill] = {}
        self._frozen = False
        for s in skills:
            self.add(s)
        self._frozen = frozen

    # -- contents --------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._skills)

    def __contains__(self, name: object) -> bool:
        return name in self._skills

    def __iter__(self) -> Iterator[Skill]:
        return iter(self._skills.values())

    @property
    def skills(self) -> tuple[Skill, ...]:
        return tuple(self._skills.values())

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._skills)

    @property
    def frozen(self) -> bool:
        return self._frozen

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def by_signature(self, signature: str) -> Skill | None:
        return next((s for s in self._skills.values() if s.signature() == signature), None)

    def add(self, skill: Skill) -> Skill:
        """Insert unless a skill with the same signature exists (then the existing one is
        returned untouched: first validated wins). A new *body* under an existing *name* with a
        different signature is a different skill and must carry a different name."""
        if self._frozen:
            msg = "library is frozen; it is read-only during evaluation"
            raise FrozenLibraryError(msg)
        if (dup := self.by_signature(skill.signature())) is not None:
            return dup
        if skill.name in self._skills:
            msg = f"skill name {skill.name!r} already used with signature "
            msg += f"{self._skills[skill.name].signature()!r}"
            raise LibraryError(msg)
        self._skills[skill.name] = skill
        return skill

    def remove(self, name: str) -> None:
        if self._frozen:
            msg = "library is frozen; it is read-only during evaluation"
            raise FrozenLibraryError(msg)
        del self._skills[name]

    # -- retrieval -------------------------------------------------------------------------
    def retrieve(self, query: str, k: int = 3) -> list[Retrieved]:
        """Shortlist by word overlap between the query and the skill's description and
        signature (Jaccard-like, deterministic, no embeddings; plan F7). Skills with no
        overlapping word are not returned."""
        q = tokens(query)
        out: list[Retrieved] = []
        for s in self._skills.values():
            d = tokens(s.description + " " + s.name.replace("_", " "))
            overlap = q & d
            if not overlap:
                continue
            score = len(overlap) / len(q | d)
            out.append(Retrieved(skill=s, score=score, overlap=tuple(sorted(overlap))))
        out.sort(key=lambda r: (-r.score, r.skill.name))
        return out[:k]

    # -- hashing and freezing --------------------------------------------------------------
    def sha256(self) -> str:
        """Order-independent hash of the skills' behaviour hashes."""
        items = sorted((s.name, s.sha256()) for s in self._skills.values())
        blob = json.dumps(items, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    def manifest(self) -> Manifest:
        return Manifest(
            sha256=self.sha256(),
            n_skills=len(self._skills),
            frozen_at=datetime.now(UTC).isoformat(timespec="seconds"),
            armbench=__version__,
            skills={s.name: s.sha256() for s in self._skills.values()},
        )

    def freeze(self) -> Manifest:
        self._frozen = True
        return self.manifest()

    # -- persistence -----------------------------------------------------------------------
    def to_json(self) -> str:
        payload = {
            "schema_version": 1,
            "skills": [s.model_dump(mode="json") for s in self._skills.values()],
        }
        return json.dumps(payload, indent=1, sort_keys=True) + "\n"

    def save(self, directory: Path = SKILLS_DIR, *, manifest: Manifest | None = None) -> Path:
        """Write ``library.json``; with ``manifest`` also ``FROZEN.json`` (the frozen state)."""
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / LIBRARY_FILE
        path.write_text(self.to_json())
        frozen = directory / FROZEN_FILE
        if manifest is not None:
            frozen.write_text(manifest.model_dump_json(indent=1) + "\n")
        elif frozen.exists():
            frozen.unlink()
        return path

    @classmethod
    def from_json(cls, text: str, *, frozen: bool = False) -> SkillLibrary:
        raw = json.loads(text)
        if raw.get("schema_version") != 1:
            msg = f"unsupported library schema {raw.get('schema_version')!r}"
            raise LibraryError(msg)
        return cls((Skill.model_validate(s) for s in raw["skills"]), frozen=frozen)

    @classmethod
    def load(cls, directory: Path = SKILLS_DIR, *, require_frozen: bool = False) -> SkillLibrary:
        """Read the library; if ``FROZEN.json`` is present its hash must match the contents
        (a tampered frozen library is refused). ``require_frozen`` refuses an unfrozen one."""
        path = directory / LIBRARY_FILE
        if not path.exists():
            if require_frozen:
                msg = f"no skill library at {path}"
                raise LibraryError(msg)
            return cls()
        manifest_path = directory / FROZEN_FILE
        lib = cls.from_json(path.read_text())
        if manifest_path.exists():
            manifest = Manifest.model_validate_json(manifest_path.read_text())
            if manifest.sha256 != lib.sha256():
                msg = f"{path} does not match {manifest_path}: {lib.sha256()} != {manifest.sha256}"
                raise LibraryError(msg)
            lib._frozen = True
        elif require_frozen:
            msg = f"skill library at {directory} is not frozen (no {FROZEN_FILE})"
            raise LibraryError(msg)
        return lib

    @staticmethod
    def frozen_sha256(directory: Path = SKILLS_DIR) -> str | None:
        manifest_path = directory / FROZEN_FILE
        if not manifest_path.exists():
            return None
        return Manifest.model_validate_json(manifest_path.read_text()).sha256
