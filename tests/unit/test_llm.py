"""Provider interface, cache/replay, budget ledger, OpenAI payload parsing, template provider."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from armbench.agents.llm import extract_program
from armbench.llm import (
    DEFAULT_LLM_FILE,
    BudgetedProvider,
    BudgetExceeded,
    CachedProvider,
    CacheMiss,
    Ledger,
    LLMParams,
    LLMRequest,
    LLMResponse,
    MissingAPIKey,
    OpenAICompatProvider,
    OpenAICompatSpec,
    Prices,
    ProviderError,
    ReplayProvider,
    ResponseCache,
    StaticProvider,
    TemplateProvider,
    Usage,
    load_llm_params,
    make_provider,
    template_program,
)
from armbench.llm.openai_compat import parse_completion
from armbench.sandbox import check_program
from armbench.scene import load_scene_config
from armbench.tasks import TASK_IDS, get_task


def req(text: str = "hello", **kw: object) -> LLMRequest:
    return LLMRequest.model_validate(
        {"model": "m", "messages": [{"role": "user", "content": text}], **kw}
    )


def test_request_key_depends_on_sampling_but_prompt_hash_does_not() -> None:
    a, b = req(temperature=0.0), req(temperature=0.5)
    assert a.key() != b.key()
    assert a.prompt_sha256() == b.prompt_sha256()
    assert req().key() == req().key()
    assert req("other").key() != req().key()
    assert req(seed=1).key() != req(seed=2).key()


def test_cache_roundtrip_and_corruption(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path)
    r = req()
    assert cache.get(r) is None
    resp = LLMResponse(text="x", model="m", provider="p", request_key=r.key(), latency_s=1.5)
    path = cache.put(r, resp)
    assert path.parent.name == r.key()[:2]
    hit = cache.get(r)
    assert hit is not None and hit.text == "x" and hit.latency_s == 1.5
    assert hit.as_cached().cached and hit.as_cached().latency_s == 0.0
    assert len(cache) == 1
    with pytest.raises(ValueError, match="belong"):
        cache.put(req("z"), resp)
    path.write_text("{not json")
    assert cache.get(r) is None
    # an entry whose stored request does not match its file name is ignored
    other = req("tampered")
    entry = json.loads(
        cache.put(other, resp.model_copy(update={"request_key": other.key()})).read_text()
    )
    entry["request"]["messages"][0]["content"] = "swapped"
    cache.path(other.key()).write_text(json.dumps(entry))
    assert cache.get(other) is None


def test_cached_provider_calls_inner_once_and_replay_needs_a_hit(tmp_path: Path) -> None:
    inner = StaticProvider("print(1)\n")
    cache = ResponseCache(tmp_path)
    cp = CachedProvider(inner, cache)
    r = req()
    first, second = cp.complete(r), cp.complete(r)
    assert inner.calls == 1
    assert not first.cached and second.cached
    assert first.text == second.text and first.request_key == second.request_key
    replay = ReplayProvider(cache)
    assert replay.complete(r).cached
    with pytest.raises(CacheMiss):
        replay.complete(req("never asked"))


def test_budget_prices_ledger_and_cap(tmp_path: Path) -> None:
    prices = Prices(input=1.0, output=10.0)
    assert prices.cost_usd(Usage(prompt_tokens=1_000_000, completion_tokens=100_000)) == 2.0
    ledger_path = tmp_path / "ledger.json"
    bp = BudgetedProvider(StaticProvider("x"), prices, max_usd=0.001, ledger_path=ledger_path)
    r = req("a" * 300, max_tokens=50)
    assert prices.worst_case_usd(r) < 0.001
    resp = bp.complete(r)
    assert resp.cost_usd == prices.cost_usd(resp.usage) > 0
    saved = Ledger.load(ledger_path)
    assert saved.n_calls == 1 and saved.usd == pytest.approx(resp.cost_usd)
    with pytest.raises(BudgetExceeded):
        bp.complete(req("b" * 3000, max_tokens=1500))
    assert Ledger.load(ledger_path).n_calls == 1
    # cached answers are not calls and cost nothing
    bp2 = BudgetedProvider(StaticProvider("x"), prices, max_usd=1.0)
    cp = CachedProvider(bp2, ResponseCache(tmp_path / "c"))
    cp.complete(r)
    assert cp.complete(r).cost_usd == 0.0
    assert bp2.ledger.n_calls == 1


def test_openai_parse_payloads() -> None:
    r = req()
    raw = json.dumps(
        {
            "model": "gpt-x-2025",
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3},
        }
    ).encode()
    resp = parse_completion(raw, r, 0.2)
    assert resp.model == "gpt-x-2025" and resp.usage.total == 10 and resp.finish_reason == "stop"
    assert resp.request_key == r.key() and resp.provider == "openai"
    with pytest.raises(ProviderError) as exc:
        parse_completion(b'{"choices": []}', r, 0.0)
    assert exc.value.code == "bad_payload"
    with pytest.raises(ProviderError):
        parse_completion(b"garbage", r, 0.0)
    with pytest.raises(ProviderError):
        parse_completion(b'{"choices": [{"message": {"content": null}}]}', r, 0.0)


def test_openai_without_key_fails_before_any_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ARMBENCH_TEST_KEY", raising=False)
    p = OpenAICompatProvider(OpenAICompatSpec(api_key_env="ARMBENCH_TEST_KEY"))
    with pytest.raises(MissingAPIKey):
        p.complete(req())


def test_default_config_loads_and_is_strict(tmp_path: Path) -> None:
    params = load_llm_params(DEFAULT_LLM_FILE)
    assert params.provider == "template" and params.max_attempts == 1
    assert params.sandbox.limits().max_calls == params.sandbox.max_calls
    with pytest.raises(ValueError, match="extra"):
        LLMParams.model_validate({"schema_version": 1, "bogus": 1})
    assert make_provider(params, kind="replay", cache_dir=tmp_path).id == "replay"
    assert make_provider(params, cache_dir=tmp_path).id == "template"
    assert make_provider(params, kind="openai", cache_dir=tmp_path).id == "openai"
    assert DEFAULT_LLM_FILE.name == "llm.yaml"


def test_run_provider_ledger_counts_cache_hits_without_repricing(tmp_path: Path) -> None:
    params = load_llm_params(DEFAULT_LLM_FILE)
    provider = make_provider(params, cache_dir=tmp_path, ledger_path=tmp_path / "ledger.json")
    assert isinstance(provider, BudgetedProvider)
    first = provider.complete(req("same prompt"))
    second = provider.complete(req("same prompt"))
    assert not first.cached and second.cached
    assert second.cost_usd == 0.0 and second.latency_s == 0.0
    assert provider.ledger.n_calls == 1 and provider.ledger.n_cached == 1
    assert Ledger.load(tmp_path / "ledger.json").n_cached == 1


def test_template_programs_pass_the_static_check_for_every_task() -> None:
    cfg = load_scene_config()
    for tid in TASK_IDS:
        for seed in range(5):
            inst = get_task(tid).instance(seed, cfg)
            src = template_program(inst.prompt)
            assert src is not None, (tid, seed)
            check = check_program(src)
            assert check.ok, (tid, seed, check.summary())
    assert template_program("Juggle the cubes.") is None


def test_template_provider_is_deterministic_and_fenced() -> None:
    cfg = load_scene_config()
    inst = get_task("stack2@1").instance(3, cfg)
    r = req(inst.prompt)
    tp = TemplateProvider()
    a, b = tp.complete(r), tp.complete(r)
    assert a.text == b.text and a.response_sha256() == b.response_sha256()
    assert a.text.startswith("```python\n") and a.usage.completion_tokens > 0
    assert extract_program(a.text) == template_program(inst.prompt)
    assert "```" not in extract_program(a.text)
    unknown = tp.complete(req("Juggle the cubes."))
    assert "```" not in unknown.text


def test_extract_program_variants() -> None:
    assert extract_program("x = 1") == "x = 1\n"
    assert extract_program("text\n```py\nx = 1\n```\nmore") == "x = 1\n"
    assert extract_program("```\na\n```\n```python\nb\n```") == "b\n"
