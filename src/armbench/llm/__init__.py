"""LLM providers behind one interface, with a disk cache, replay and spending control (F6)."""

from __future__ import annotations

from pathlib import Path

from armbench.llm.budget import BudgetedProvider, Ledger, Prices
from armbench.llm.cache import CachedProvider, CacheEntry, ReplayProvider, ResponseCache
from armbench.llm.estimate import Estimate, estimate
from armbench.llm.fake import StaticProvider, TemplateProvider, template_program
from armbench.llm.openai_compat import OpenAICompatProvider, OpenAICompatSpec
from armbench.llm.params import (
    DEFAULT_LLM_FILE,
    SPEND_FILE,
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
    ``BudgetedProvider(CachedProvider(base))`` so the ledger sees every answer (hits are
    counted, cost nothing and are never re-priced) and misses are priced and capped."""
    which = kind or params.provider
    root = cache_dir or params.cache_dir
    cache = ResponseCache(root)
    if which == "replay":
        return ReplayProvider(cache)
    base: Provider = (
        TemplateProvider() if which == "template" else OpenAICompatProvider(params.openai)
    )
    return BudgetedProvider(
        CachedProvider(base, cache),
        params.prices_usd_per_1m,
        max_usd=params.max_usd_per_run,
        ledger_path=ledger_path,
        max_usd_total=params.max_usd_total,
        total_path=root / SPEND_FILE,
    )


__all__ = [
    "DEFAULT_LLM_FILE",
    "SPEND_FILE",
    "BudgetExceeded",
    "BudgetedProvider",
    "CacheEntry",
    "CacheMiss",
    "CachedProvider",
    "Estimate",
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
    "estimate",
    "load_llm_params",
    "make_provider",
    "sha256_text",
    "template_program",
]
