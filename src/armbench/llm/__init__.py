"""LLM providers behind one interface, with a disk cache, replay and spending control (F6)."""

from __future__ import annotations

from pathlib import Path

from armbench.llm.budget import BudgetedProvider, Ledger, Prices
from armbench.llm.cache import CachedProvider, CacheEntry, ReplayProvider, ResponseCache
from armbench.llm.fake import StaticProvider, TemplateProvider, template_program
from armbench.llm.openai_compat import OpenAICompatProvider, OpenAICompatSpec
from armbench.llm.params import (
    DEFAULT_LLM_FILE,
    LLMParams,
    ProviderKind,
    SandboxSpec,
    load_llm_params,
)
from armbench.llm.provider import BudgetExceeded, CacheMiss, MissingAPIKey, Provider, ProviderError
from armbench.llm.types import LLMRequest, LLMResponse, Message, Usage, sha256_text


def make_provider(
    params: LLMParams,
    *,
    kind: ProviderKind | None = None,
    cache_dir: Path | None = None,
    ledger_path: Path | None = None,
) -> Provider:
    """Provider stack for a run: ``replay`` is cache-only; the others are
    ``CachedProvider(BudgetedProvider(base))`` so hits cost nothing and misses are capped."""
    which = kind or params.provider
    cache = ResponseCache(cache_dir or params.cache_dir)
    if which == "replay":
        return ReplayProvider(cache)
    base: Provider = (
        TemplateProvider() if which == "template" else OpenAICompatProvider(params.openai)
    )
    budgeted = BudgetedProvider(
        base, params.prices_usd_per_1m, max_usd=params.max_usd_per_run, ledger_path=ledger_path
    )
    return CachedProvider(budgeted, cache)


__all__ = [
    "DEFAULT_LLM_FILE",
    "BudgetExceeded",
    "BudgetedProvider",
    "CacheEntry",
    "CacheMiss",
    "CachedProvider",
    "LLMParams",
    "LLMRequest",
    "LLMResponse",
    "Ledger",
    "Message",
    "MissingAPIKey",
    "OpenAICompatProvider",
    "OpenAICompatSpec",
    "Prices",
    "Provider",
    "ProviderError",
    "ProviderKind",
    "ReplayProvider",
    "ResponseCache",
    "SandboxSpec",
    "StaticProvider",
    "TemplateProvider",
    "Usage",
    "load_llm_params",
    "make_provider",
    "sha256_text",
    "template_program",
]
