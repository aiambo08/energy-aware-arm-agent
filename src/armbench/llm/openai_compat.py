"""Chat-completions client for any OpenAI-compatible endpoint (OpenAI, Nebius Token Factory,
vLLM, Ollama ...) over the standard library: no SDK, one POST per request, bounded retries.

The key is read from the environment (``api_key_env``) at call time, so an agent can be
constructed, cached and replayed on a machine that has no key at all.
"""

from __future__ import annotations

import json
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


class OpenAICompatProvider:
    id = "openai"

    def __init__(self, spec: OpenAICompatSpec) -> None:
        self.spec = spec

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
        return json.dumps(payload).encode("utf-8")

    def complete(self, request: LLMRequest) -> LLMResponse:
        key = self._key()
        url = self.spec.base_url.rstrip("/") + "/chat/completions"
        body = self._body(request)
        last: ProviderError | None = None
        for attempt in range(self.spec.max_retries + 1):
            req = urllib.request.Request(  # noqa: S310 - https endpoint from the config
                url,
                data=body,
                method="POST",
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                    "User-Agent": "armbench",
                },
            )
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
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last = ProviderError(f"network: {exc}", code="network")
            if attempt < self.spec.max_retries:
                time.sleep(self.spec.backoff_s * (2**attempt))
        assert last is not None  # noqa: S101 - loop ran at least once
        raise last


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
