"""Provider-neutral request/response records; the hashes here are what the episode log keeps."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["system", "user", "assistant"]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Message(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Role
    content: str


class LLMRequest(BaseModel):
    """One chat completion request. ``key()`` is the cache identity: everything that changes
    what a deterministic provider would answer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str
    messages: tuple[Message, ...] = Field(min_length=1)
    temperature: float = Field(ge=0.0, le=2.0, default=0.0)
    max_tokens: int = Field(gt=0, default=1500)
    seed: int | None = None
    reasoning_effort: str | None = None
    """Forwarded as ``reasoning_effort`` (thinking models, e.g. Gemini); None = provider default."""
    endpoint: str | None = None
    """Base URL of the provider that answers; two providers serving the same model name never
    share a cache entry. ``None`` for the template provider."""

    def canonical(self) -> str:
        payload = self.model_dump(mode="json")
        for optional in ("reasoning_effort", "endpoint"):
            if payload[optional] is None:
                del payload[optional]
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def key(self) -> str:
        return sha256_text(self.canonical())

    def prompt_sha256(self) -> str:
        """Hash of the messages alone (model- and sampling-independent)."""
        body = json.dumps([m.model_dump() for m in self.messages], separators=(",", ":"))
        return sha256_text(body)

    def prompt_chars(self) -> int:
        return sum(len(m.content) for m in self.messages)

    def last_user(self) -> str:
        for m in reversed(self.messages):
            if m.role == "user":
                return m.content
        return ""


class Usage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    prompt_tokens: int = Field(ge=0, default=0)
    completion_tokens: int = Field(ge=0, default=0)

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )


class LLMResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    model: str
    """Exact model identifier the provider reported (may be more specific than requested)."""
    provider: str
    usage: Usage = Usage()
    latency_s: float = Field(ge=0, default=0.0)
    cost_usd: float = Field(ge=0, default=0.0)
    cached: bool = False
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    finish_reason: str | None = None
    request_key: str

    def response_sha256(self) -> str:
        return sha256_text(self.text)

    def as_cached(self) -> LLMResponse:
        """The same answer served from the cache: no latency, no cost, flagged."""
        return self.model_copy(update={"cached": True, "latency_s": 0.0, "cost_usd": 0.0})
