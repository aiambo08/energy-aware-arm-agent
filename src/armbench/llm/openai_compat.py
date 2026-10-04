"""Chat-completions client for any OpenAI-compatible endpoint (OpenAI, Nebius Token Factory,
vLLM, Ollama ...) over the standard library: no SDK, one POST per request, bounded retries.

The key is read from the environment (``api_key_env``) at call time, so an agent can be
constructed, cached and replayed on a machine that has no key at all.
"""

from __future__ import annotations

import json
import math
import os
import time
import urllib.error
import urllib.request

from pydantic import BaseModel, ConfigDict, Field

from armbench.llm.provider import MissingAPIKey, ProviderError
from armbench.llm.types import LLMRequest, LLMResponse, Usage

RETRY_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


class OpenAICompatSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "ARMBENCH_LLM_API_KEY"
    timeout_s: float = Field(gt=0, default=60.0)
    max_retries: int = Field(ge=0, default=2)
    backoff_s: float = Field(ge=0, default=2.0)
    min_interval_s: float = Field(ge=0, default=0.0)
    """Minimum spacing between requests (free tiers with a requests-per-minute quota)."""
    models_query: str = ""
    """Query for ``GET /models`` (Nebius: ``verbose=true`` adds prices; Gemini rejects it)."""
    max_retry_after_s: float = Field(ge=0, default=120.0)
    """Longest ``Retry-After`` honoured on a 429/503; longer waits fail the call instead."""


class OpenAICompatProvider:
    id = "openai"

    def __init__(self, spec: OpenAICompatSpec) -> None:
        self.spec = spec
        self._last_request = -math.inf

    def _pace(self) -> None:
        wait = self._last_request + self.spec.min_interval_s - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._key()}",
            "Content-Type": "application/json",
            "User-Agent": "armbench",
        }

    def list_models(self) -> bytes:
        """Raw ``GET /models`` with the profile's ``models_query``."""
        query = f"?{self.spec.models_query}" if self.spec.models_query else ""
        url = self.spec.base_url.rstrip("/") + "/models" + query
        req = urllib.request.Request(url, method="GET", headers=self._headers())  # noqa: S310
        try:
            with urllib.request.urlopen(req, timeout=self.spec.timeout_s) as resp:  # noqa: S310
                return bytes(resp.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read()[:500].decode("utf-8", "replace")
            raise ProviderError(f"HTTP {exc.code}: {detail}", code=f"http_{exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ProviderError(f"network: {exc}", code="network") from exc

    def _key(self) -> str:
        key = os.environ.get(self.spec.api_key_env, "")
        if not key:
            msg = f"environment variable {self.spec.api_key_env} is not set"
            raise MissingAPIKey(msg)
        return key

    def _body(self, request: LLMRequest) -> bytes:
        payload: dict[str, object] = {
            "model": request.model,
            "messages": [m.model_dump() for m in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.seed is not None:
            payload["seed"] = request.seed
        if request.reasoning_effort is not None:
            payload["reasoning_effort"] = request.reasoning_effort
        return json.dumps(payload).encode("utf-8")

    def complete(self, request: LLMRequest) -> LLMResponse:
        headers = self._headers()
        url = self.spec.base_url.rstrip("/") + "/chat/completions"
        body = self._body(request)
        last: ProviderError | None = None
        for attempt in range(self.spec.max_retries + 1):
            req = urllib.request.Request(  # noqa: S310 - https endpoint from the config
                url, data=body, method="POST", headers=headers
            )
            self._pace()
            delay = self.spec.backoff_s * (2**attempt)
            t0 = time.monotonic()
            try:
                with urllib.request.urlopen(req, timeout=self.spec.timeout_s) as resp:  # noqa: S310
                    raw = resp.read()
                latency = time.monotonic() - t0
                return parse_completion(raw, request, latency)
            except urllib.error.HTTPError as exc:
                detail = exc.read()[:500].decode("utf-8", "replace")
                last = ProviderError(f"HTTP {exc.code}: {detail}", code=f"http_{exc.code}")
                if exc.code not in RETRY_STATUSES:
                    raise last from exc
                hinted = retry_after_s(exc.headers.get("Retry-After"))
                if hinted is not None:
                    if hinted > self.spec.max_retry_after_s:
                        raise last from exc
                    delay = max(delay, hinted)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last = ProviderError(f"network: {exc}", code="network")
            if attempt < self.spec.max_retries:
                time.sleep(delay)
        assert last is not None  # noqa: S101 - loop ran at least once
        raise last


def retry_after_s(value: str | None) -> float | None:
    """Seconds from a ``Retry-After`` header (delta-seconds form only)."""
    if value is None:
        return None
    try:
        seconds = float(value.strip())
    except ValueError:
        return None
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


def parse_completion(raw: bytes, request: LLMRequest, latency_s: float) -> LLMResponse:
    try:
        data = json.loads(raw)
        choice = data["choices"][0]
        text = choice["message"]["content"]
        usage = data.get("usage") or {}
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        msg = f"malformed completion payload: {exc}"
        raise ProviderError(msg, code="bad_payload") from exc
    if not isinstance(text, str):
        msg = "completion content is not text"
        raise ProviderError(msg, code="bad_payload")
    return LLMResponse(
        text=text,
        model=str(data.get("model") or request.model),
        provider=OpenAICompatProvider.id,
        usage=Usage(
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
        ),
        latency_s=latency_s,
        finish_reason=choice.get("finish_reason"),
        request_key=request.key(),
    )
