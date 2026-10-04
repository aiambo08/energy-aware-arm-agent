"""Spending control: per-token prices, a persisted ledger per run and a hard cap per run."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from armbench.llm.provider import BudgetExceeded, Provider
from armbench.llm.types import LLMRequest, LLMResponse, Usage

CHARS_PER_TOKEN_ESTIMATE = 3.0
"""Conservative (low) chars-per-token guess used only to bound the cost of a call *before*
it is made; the ledger records the provider's real token counts afterwards."""


class Prices(BaseModel):
    """USD per million tokens."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    input: float = Field(ge=0, default=0.0)
    output: float = Field(ge=0, default=0.0)

    def cost_usd(self, usage: Usage) -> float:
        return (usage.prompt_tokens * self.input + usage.completion_tokens * self.output) / 1e6

    def worst_case_usd(self, request: LLMRequest) -> float:
        prompt = request.prompt_chars() / CHARS_PER_TOKEN_ESTIMATE
        return (prompt * self.input + request.max_tokens * self.output) / 1e6


class Ledger(BaseModel):
    """Running totals of one run; written after every live call."""

    model_config = ConfigDict(extra="forbid")

    n_calls: int = 0
    """Live calls made (cache hits are not calls)."""
    n_cached: int = 0
    n_failed: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    usd: float = 0.0
    latency_s_total: float = 0.0
    latencies_s: list[float] = Field(default_factory=list)
    models: dict[str, int] = Field(default_factory=dict)

    def record(self, response: LLMResponse) -> None:
        if response.cached:
            self.n_cached += 1
            return
        self.n_calls += 1
        self.prompt_tokens += response.usage.prompt_tokens
        self.completion_tokens += response.usage.completion_tokens
        self.usd += response.cost_usd
        self.latency_s_total += response.latency_s
        self.latencies_s.append(round(response.latency_s, 4))
        self.models[response.model] = self.models.get(response.model, 0) + 1

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.model_dump(mode="json"), indent=1, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: Path) -> Ledger:
        if not path.is_file():
            return cls()
        return cls.model_validate_json(path.read_text())


class BudgetedProvider:
    """Refuses a live call that could push the run past ``max_usd``; prices every answer."""

    def __init__(
        self,
        inner: Provider,
        prices: Prices,
        *,
        max_usd: float,
        ledger: Ledger | None = None,
        ledger_path: Path | None = None,
    ) -> None:
        self.inner = inner
        self.prices = prices
        self.max_usd = max_usd
        self.ledger = ledger or (Ledger.load(ledger_path) if ledger_path else Ledger())
        self.ledger_path = ledger_path
        self.id = inner.id
        if ledger_path is not None:
            self.ledger.save(ledger_path)

    def complete(self, request: LLMRequest) -> LLMResponse:
        worst = self.prices.worst_case_usd(request)
        if self.ledger.usd + worst > self.max_usd:
            msg = (
                f"run spent {self.ledger.usd:.4f} USD; this call could cost {worst:.4f} and the "
                f"cap is {self.max_usd:.2f} USD"
            )
            raise BudgetExceeded(msg)
        try:
            response = self.inner.complete(request)
        except Exception:
            self.ledger.n_failed += 1
            self._flush()
            raise
        priced = response.model_copy(update={"cost_usd": self.prices.cost_usd(response.usage)})
        self.ledger.record(priced)
        self._flush()
        return priced

    def _flush(self) -> None:
        if self.ledger_path is not None:
            self.ledger.save(self.ledger_path)
