"""Disk cache of completions keyed by ``hash(model, temperature, seed, max_tokens, messages)``.

One JSON file per request under ``<dir>/<key[:2]>/<key>.json`` holding both the request and the
response, so a cache directory is a self-describing, diff-able record of every prompt and
answer in a run (``armbench run`` copies it next to the episode log). Writes are atomic.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from armbench.llm.provider import CacheMiss, Provider
from armbench.llm.types import LLMRequest, LLMResponse


class CacheEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = 1
    request: LLMRequest
    response: LLMResponse


class ResponseCache:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, request: LLMRequest) -> LLMResponse | None:
        key = request.key()
        p = self.path(key)
        if not p.is_file():
            return None
        try:
            entry = CacheEntry.model_validate_json(p.read_text(encoding="utf-8"))
        except (ValidationError, ValueError, OSError):
            return None
        if entry.request.key() != key or entry.response.request_key != key:
            return None
        return entry.response

    def put(self, request: LLMRequest, response: LLMResponse) -> Path:
        key = request.key()
        if response.request_key != key:
            msg = "response does not belong to this request"
            raise ValueError(msg)
        p = self.path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        entry = CacheEntry(request=request, response=response)
        fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=".tmp-", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(entry.model_dump_json(indent=1) + "\n")
        os.replace(tmp, p)
        return p

    def entries(self) -> Iterator[CacheEntry]:
        if not self.root.is_dir():
            return
        for p in sorted(self.root.glob("*/*.json")):
            try:
                yield CacheEntry.model_validate_json(p.read_text(encoding="utf-8"))
            except (ValidationError, ValueError, OSError):
                continue

    def __len__(self) -> int:
        return sum(1 for _ in self.entries())


class CachedProvider:
    """Serve from the cache when possible, otherwise ask ``inner`` and remember the answer."""

    def __init__(self, inner: Provider, cache: ResponseCache) -> None:
        self.inner = inner
        self.cache = cache
        self.id = inner.id

    def complete(self, request: LLMRequest) -> LLMResponse:
        hit = self.cache.get(request)
        if hit is not None:
            return hit.as_cached()
        response = self.inner.complete(request)
        self.cache.put(request, response)
        return response


class ReplayProvider:
    """Cache only: a request that was never answered is an error, never a network call."""

    id = "replay"

    def __init__(self, cache: ResponseCache) -> None:
        self.cache = cache

    def complete(self, request: LLMRequest) -> LLMResponse:
        hit = self.cache.get(request)
        if hit is None:
            msg = f"no cached response for request {request.key()[:12]} ({request.model})"
            raise CacheMiss(msg)
        return hit.as_cached()


def dump_json(value: object) -> str:
    return json.dumps(value, indent=1, sort_keys=True)
