"""Cost of a run before it is made: which requests the cache already answers and the
worst-case price of the rest under the configured caps (``armbench llm estimate``)."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from armbench.llm.budget import CHARS_PER_TOKEN_ESTIMATE, Ledger
from armbench.llm.cache import ResponseCache
from armbench.llm.params import LLMParams
from armbench.llm.types import LLMRequest


class Estimate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str
    endpoint: str | None
    n_requests: int = Field(ge=0)
    n_cached: int = Field(ge=0)
    n_live: int = Field(ge=0)
    """First-turn requests the cache cannot answer (each is one paid call)."""
    max_attempts: int = Field(ge=1)
    prompt_tokens_est: int = Field(ge=0)
    """Prompt tokens of the live requests, estimated from characters (high side)."""
    completion_tokens_max: int = Field(ge=0)
    """``max_tokens`` x live requests x attempts: the most the run can be billed for output."""
    usd_worst: float = Field(ge=0)
    max_usd_per_run: float = Field(ge=0)
    spent_usd_total: float = Field(ge=0)
    max_usd_total: float | None
    fits: bool
    """Worst case under both caps (the run would never be cut by the budget)."""


def estimate(requests: Sequence[LLMRequest], params: LLMParams, cache: ResponseCache) -> Estimate:
    live = [r for r in requests if cache.get(r) is None]
    attempts = params.max_attempts
    prompt = sum(r.prompt_chars() for r in live) / CHARS_PER_TOKEN_ESTIMATE * attempts
    completion = sum(r.max_tokens for r in live) * attempts
    prices = params.prices_usd_per_1m
    usd = (prompt * prices.input + completion * prices.output) / 1e6
    spent = Ledger.load(params.spend_ledger_path()).usd
    fits = usd <= params.max_usd_per_run and (
        params.max_usd_total is None or spent + usd <= params.max_usd_total
    )
    return Estimate(
        model=params.model,
        endpoint=params.endpoint(),
        n_requests=len(requests),
        n_cached=len(requests) - len(live),
        n_live=len(live),
        max_attempts=attempts,
        prompt_tokens_est=round(prompt),
        completion_tokens_max=completion,
        usd_worst=round(usd, 6),
        max_usd_per_run=params.max_usd_per_run,
        spent_usd_total=round(spent, 6),
        max_usd_total=params.max_usd_total,
        fits=fits,
    )
