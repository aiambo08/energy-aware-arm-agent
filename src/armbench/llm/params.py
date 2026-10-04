"""Validated LLM-agent configuration (``configs/llm.yaml``): provider, sampling, caps, sandbox."""

from __future__ import annotations

from pathlib import Path
from typing import Final, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from armbench.llm.budget import Prices
from armbench.llm.openai_compat import OpenAICompatSpec
from armbench.paths import CONFIG_DIR
from armbench.sandbox import ProgramLimits, SandboxLimits

DEFAULT_LLM_FILE: Final = CONFIG_DIR / "llm.yaml"

ProviderKind = Literal["template", "openai", "replay"]
SPEND_FILE: Final = "spend.json"


class SandboxSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    cpu_s: float = Field(gt=0, default=10.0)
    memory_mb: int = Field(gt=0, default=512)
    wall_s: float = Field(gt=0, default=300.0)
    sim_s: float = Field(gt=0, default=60.0)
    max_calls: int = Field(gt=0, default=200)
    stdout_kb: int = Field(gt=0, default=16)
    max_source_chars: int = Field(gt=0, default=20_000)
    max_nodes: int = Field(gt=0, default=5_000)

    def limits(self) -> SandboxLimits:
        return SandboxLimits(
            cpu_s=self.cpu_s,
            memory_mb=self.memory_mb,
            wall_s=self.wall_s,
            sim_s=self.sim_s,
            max_calls=self.max_calls,
            stdout_kb=self.stdout_kb,
            program=ProgramLimits(max_source_chars=self.max_source_chars, max_nodes=self.max_nodes),
        )


class LLMParams(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1]
    provider: ProviderKind = "template"
    model: str = "template-v1"
    """Model identifier sent to the provider and part of the cache key."""
    temperature: float = Field(ge=0.0, le=2.0, default=0.0)
    max_tokens: int = Field(gt=0, default=1500)
    seed: int | None = 0
    max_attempts: int = Field(ge=1, default=1)
    """Program generations per episode (D5: one in the main evaluation)."""
    max_tokens_per_episode: int = Field(gt=0, default=8000)
    max_usd_per_run: float = Field(ge=0, default=5.0)
    max_usd_total: float | None = Field(ge=0, default=None)
    """Cap over every run sharing ``cache_dir`` (ledger ``<cache_dir>/spend.json``); off if None."""
    reasoning_effort: str | None = None
    """Sent to thinking models (part of the cache key); ``None`` keeps the provider default."""
    prices_usd_per_1m: Prices = Prices()
    cache_dir: Path = Path("cache/llm")
    openai: OpenAICompatSpec = OpenAICompatSpec()
    sandbox: SandboxSpec = SandboxSpec()

    def endpoint(self) -> str | None:
        """Identity of the answering service in the cache key: the configured base URL for
        ``openai`` (and ``replay``, which must find what ``openai`` stored); none for template."""
        if self.provider == "template":
            return None
        return self.openai.base_url.rstrip("/")

    def spend_ledger_path(self) -> Path:
        return self.cache_dir / SPEND_FILE


def load_llm_params(path: Path = DEFAULT_LLM_FILE) -> LLMParams:
    with path.open(encoding="utf-8") as fh:
        return LLMParams.model_validate(yaml.safe_load(fh))
