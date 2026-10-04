"""The single interface every LLM backend implements, and its typed failures."""

from __future__ import annotations

from typing import Protocol

from armbench.llm.types import LLMRequest, LLMResponse


class ProviderError(Exception):
    """A completion could not be produced; ``code`` is stable for the episode log."""

    code = "provider_error"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code


class CacheMiss(ProviderError):
    """Replay mode found no cached answer for this request."""

    code = "cache_miss"


class BudgetExceeded(ProviderError):
    """The run's spending cap would be crossed by this call."""

    code = "budget_exceeded"


class MissingAPIKey(ProviderError):
    code = "missing_api_key"


class Provider(Protocol):
    id: str

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Answer one chat request or raise a :class:`ProviderError`."""
        ...
