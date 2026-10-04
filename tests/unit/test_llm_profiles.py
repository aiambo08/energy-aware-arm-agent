"""Provider profiles (Gemini / Nebius), cache identity per endpoint, pacing and Retry-After,
the cross-run spending cap, cost estimates and resuming an interrupted prefetch."""

from __future__ import annotations

import email.message
import io
import json
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from typer.testing import CliRunner

from armbench.agents import LLMAgent, get_agent, prefetch
from armbench.cli import app
from armbench.llm import (
    SPEND_FILE,
    BudgetedProvider,
    BudgetExceeded,
    CachedProvider,
    Ledger,
    LLMParams,
    LLMRequest,
    LLMResponse,
    OpenAICompatProvider,
    OpenAICompatSpec,
    Prices,
    ProviderError,
    ResponseCache,
    StaticProvider,
    Usage,
    estimate,
    load_llm_params,
    make_provider,
)
from armbench.llm.openai_compat import retry_after_s
from armbench.paths import CONFIG_DIR
from armbench.primitives import load_primitive_params
from armbench.scene import load_scene_config
from armbench.tasks import TASK_IDS, TaskInstance, get_task

runner = CliRunner()
GEMINI = CONFIG_DIR / "llm.gemini.yaml"
NEBIUS = CONFIG_DIR / "llm.nebius.yaml"


def req(text: str = "hello", **kw: object) -> LLMRequest:
    return LLMRequest.model_validate(
        {"model": "m", "messages": [{"role": "user", "content": text}], **kw}
    )


def test_optional_fields_keep_old_cache_keys_and_separate_endpoints() -> None:
    legacy = (
        '{"max_tokens":1500,"messages":[{"content":"hello","role":"user"}],'
        '"model":"m","seed":null,"temperature":0.0}'
    )
    assert req().canonical() == legacy
    a = req(endpoint="https://generativelanguage.googleapis.com/v1beta/openai")
    b = req(endpoint="https://api.tokenfactory.nebius.com/v1")
    assert len({req().key(), a.key(), b.key()}) == 3
    assert a.prompt_sha256() == b.prompt_sha256() == req().prompt_sha256()
    assert req(reasoning_effort="low").key() != req().key()


@pytest.mark.parametrize("path", [GEMINI, NEBIUS])
def test_profiles_load_and_point_at_their_own_key(path: Path) -> None:
    p = load_llm_params(path)
    assert p.provider == "openai" and p.max_attempts == 1 and p.temperature == 0.0
    assert p.openai.base_url.startswith("https://")
    assert p.openai.api_key_env.startswith("ARMBENCH_") and p.openai.api_key_env.endswith("_KEY")
    assert p.endpoint() == p.openai.base_url.rstrip("/")
    assert p.spend_ledger_path() == p.cache_dir / SPEND_FILE


def test_profiles_differ_where_they_should() -> None:
    g, n = load_llm_params(GEMINI), load_llm_params(NEBIUS)
    assert g.endpoint() != n.endpoint()
    assert g.openai.api_key_env != n.openai.api_key_env
    assert g.openai.min_interval_s > 0 and g.reasoning_effort is not None
    assert n.max_usd_total is not None and n.max_usd_total <= 5.0
    assert n.prices_usd_per_1m.input > 0 and n.prices_usd_per_1m.output > 0


def test_template_requests_have_no_endpoint_but_openai_and_replay_agree() -> None:
    base = LLMParams(schema_version=1)
    assert base.endpoint() is None
    live = base.model_copy(update={"provider": "openai"})
    replay = base.model_copy(update={"provider": "replay"})
    assert live.endpoint() == replay.endpoint() == "https://api.openai.com/v1"


def agent_for(params: LLMParams, cache: Path, agent: str = "B") -> LLMAgent:
    provider = make_provider(params, cache_dir=cache)
    a = get_agent(agent, llm=params, provider=provider)
    assert isinstance(a, LLMAgent)
    return a


def test_agent_requests_carry_profile_identity(tmp_path: Path) -> None:
    g = load_llm_params(GEMINI)
    inst = get_task(TASK_IDS[0]).instance(0, load_scene_config())
    r = agent_for(g, tmp_path).request(load_primitive_params(), inst)
    assert r.endpoint == g.endpoint() and r.reasoning_effort == "low"
    assert r.model == g.model and r.seed is None and r.max_tokens == g.max_tokens


def test_body_forwards_reasoning_effort_only_when_set() -> None:
    p = OpenAICompatProvider(OpenAICompatSpec())
    assert "reasoning_effort" not in json.loads(p._body(req()))
    body = json.loads(p._body(req(reasoning_effort="low", seed=3)))
    assert body["reasoning_effort"] == "low" and body["seed"] == 3
    assert "endpoint" not in body


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, None), ("7", 7.0), (" 1.5 ", 1.5), ("-1", None), ("nan", None), ("soon", None),
     ("Wed, 21 Oct 2026 07:28:00 GMT", None)],
)  # fmt: skip
def test_retry_after_parsing(value: str | None, expected: float | None) -> None:
    assert retry_after_s(value) == expected


class FakeHTTP:
    """``urlopen`` stand-in: a scripted list of status codes (200 = a valid completion)."""

    def __init__(self, statuses: list[int], retry_after: str | None = None) -> None:
        self.statuses = list(statuses)
        self.retry_after = retry_after
        self.calls = 0

    def __call__(self, request: urllib.request.Request, timeout: float) -> io.BytesIO:
        self.calls += 1
        status = self.statuses.pop(0)
        if status != 200:
            headers = email.message.Message()
            if self.retry_after is not None:
                headers["Retry-After"] = self.retry_after
            raise urllib.error.HTTPError(
                request.full_url, status, "err", headers, io.BytesIO(b"quota")
            )
        payload = {"model": "m", "choices": [{"message": {"content": "ok"}}],
                   "usage": {"prompt_tokens": 3, "completion_tokens": 1}}  # fmt: skip
        return io.BytesIO(json.dumps(payload).encode())


def patch_net(monkeypatch: pytest.MonkeyPatch, http: FakeHTTP) -> list[float]:
    sleeps: list[float] = []
    monkeypatch.setenv("ARMBENCH_TEST_KEY", "k")
    monkeypatch.setattr(urllib.request, "urlopen", http)
    monkeypatch.setattr("armbench.llm.openai_compat.time.sleep", sleeps.append)
    return sleeps


def test_429_honours_retry_after(monkeypatch: pytest.MonkeyPatch) -> None:
    http = FakeHTTP([429, 200], retry_after="30")
    sleeps = patch_net(monkeypatch, http)
    spec = OpenAICompatSpec(api_key_env="ARMBENCH_TEST_KEY", backoff_s=1.0, max_retries=2)
    resp = OpenAICompatProvider(spec).complete(req())
    assert resp.text == "ok" and http.calls == 2
    assert sleeps == [30.0]


def test_retry_after_beyond_limit_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    http = FakeHTTP([429, 200], retry_after="3600")
    sleeps = patch_net(monkeypatch, http)
    spec = OpenAICompatSpec(api_key_env="ARMBENCH_TEST_KEY", max_retry_after_s=60.0)
    with pytest.raises(ProviderError) as exc:
        OpenAICompatProvider(spec).complete(req())
    assert exc.value.code == "http_429" and http.calls == 1 and sleeps == []


def test_non_retryable_status_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    http = FakeHTTP([400, 200])
    patch_net(monkeypatch, http)
    with pytest.raises(ProviderError) as exc:
        OpenAICompatProvider(OpenAICompatSpec(api_key_env="ARMBENCH_TEST_KEY")).complete(req())
    assert exc.value.code == "http_400" and http.calls == 1


def test_min_interval_spaces_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    http = FakeHTTP([200, 200])
    sleeps = patch_net(monkeypatch, http)
    clock = iter([100.0, 100.0, 100.0, 101.0, 101.0, 107.0, 107.0, 107.0])
    monkeypatch.setattr("armbench.llm.openai_compat.time.monotonic", lambda: next(clock))
    p = OpenAICompatProvider(OpenAICompatSpec(api_key_env="ARMBENCH_TEST_KEY", min_interval_s=6.0))
    p.complete(req("a"))
    p.complete(req("b"))
    assert sleeps == [pytest.approx(5.0)]


def test_total_cap_spans_runs(tmp_path: Path) -> None:
    prices = Prices(input=1000.0, output=1000.0)
    total = tmp_path / SPEND_FILE
    r = req("a" * 30, max_tokens=10)
    worst = prices.worst_case_usd(r)
    first = BudgetedProvider(StaticProvider("x"), prices, max_usd=1.0, max_usd_total=worst * 1.5,
                             total_path=total)  # fmt: skip
    first.complete(r)
    spent = Ledger.load(total).usd
    assert spent > 0 and Ledger.load(total).n_calls == 1
    # a new run (fresh per-run ledger) still sees the shared total and is refused
    second = BudgetedProvider(StaticProvider("x"), prices, max_usd=1.0, max_usd_total=worst * 1.5,
                              total_path=total)  # fmt: skip
    with pytest.raises(BudgetExceeded, match="all runs"):
        second.complete(req("b" * 30, max_tokens=10))
    assert second.ledger.n_calls == 0 and Ledger.load(total).usd == spent


def test_cache_hits_never_touch_the_total(tmp_path: Path) -> None:
    prices = Prices(input=1.0, output=1.0)
    total = tmp_path / SPEND_FILE
    inner = CachedProvider(StaticProvider("x"), ResponseCache(tmp_path / "c"))
    bp = BudgetedProvider(inner, prices, max_usd=1.0, max_usd_total=1.0, total_path=total)
    bp.complete(req())
    once = Ledger.load(total)
    bp.complete(req())
    assert Ledger.load(total) == once and once.n_calls == 1


def nebius_like(tmp_path: Path) -> LLMParams:
    return load_llm_params(NEBIUS).model_copy(update={"cache_dir": tmp_path / "cache"})


def instances(n: int) -> list[TaskInstance]:
    cfg = load_scene_config()
    return [get_task(TASK_IDS[0]).instance(s, cfg) for s in range(n)]


def test_estimate_counts_cache_hits_as_free(tmp_path: Path) -> None:
    params = nebius_like(tmp_path)
    cache = ResponseCache(params.cache_dir)
    agent = agent_for(params, params.cache_dir)
    reqs = [agent.request(load_primitive_params(), i) for i in instances(4)]
    cold = estimate(reqs, params, cache)
    assert cold.n_live == 4 and cold.n_cached == 0 and cold.usd_worst > 0
    assert cold.completion_tokens_max == 4 * params.max_tokens
    warm_cache = CachedProvider(StaticProvider("print(1)\n"), cache)
    for r in reqs[:3]:
        warm_cache.complete(r)
    warm = estimate(reqs, params, cache)
    assert warm.n_cached == 3 and warm.n_live == 1
    assert warm.usd_worst == pytest.approx(cold.usd_worst / 4, rel=0.05)
    assert warm.fits and warm.endpoint == params.endpoint()


def test_estimate_flags_a_total_cap_already_used(tmp_path: Path) -> None:
    params = nebius_like(tmp_path)
    assert params.max_usd_total is not None
    Ledger(usd=params.max_usd_total).save(params.spend_ledger_path())
    agent = agent_for(params, params.cache_dir)
    reqs = [agent.request(load_primitive_params(), i) for i in instances(1)]
    est = estimate(reqs, params, ResponseCache(params.cache_dir))
    assert not est.fits and est.spent_usd_total == params.max_usd_total


def test_interrupted_prefetch_resumes_without_repaying(tmp_path: Path) -> None:
    """A run cut by the cap (or a quota) keeps every answer it got: re-running it pays only
    for the requests that were never answered."""
    params = LLMParams(schema_version=1, cache_dir=tmp_path / "cache")
    cache = ResponseCache(params.cache_dir)
    insts = instances(4)
    calls: list[str] = []

    class Counting(StaticProvider):
        def complete(self, request: LLMRequest) -> LLMResponse:
            calls.append(request.key())
            usage = Usage(prompt_tokens=0, completion_tokens=request.max_tokens)
            return super().complete(request).model_copy(update={"usage": usage})

    prices = Prices(input=0.0, output=1000.0)
    one_call = params.max_tokens * 1000.0 / 1e6
    cut = BudgetedProvider(CachedProvider(Counting("print(1)\n"), cache), prices,
                           max_usd=one_call * 2.5)  # fmt: skip
    agent = get_agent("B", llm=params, provider=cut)
    assert isinstance(agent, LLMAgent)
    with pytest.raises(BudgetExceeded):
        prefetch(agent, load_primitive_params(), insts)
    assert len(calls) == 2 and len(cache) == 2
    resumed = BudgetedProvider(CachedProvider(Counting("print(1)\n"), cache), prices,
                               max_usd=one_call * 2.5)  # fmt: skip
    agent2 = get_agent("B", llm=params, provider=resumed)
    assert isinstance(agent2, LLMAgent)
    pairs = prefetch(agent2, load_primitive_params(), insts)
    assert len(pairs) == 4 and sum(r.cached for _, r in pairs) == 2
    assert len(calls) == 4 and len(set(calls)) == 4


def test_cli_dry_run_and_estimate(tmp_path: Path) -> None:
    cfg = tmp_path / "llm.yaml"
    text = NEBIUS.read_text().replace("cache_dir: cache/llm", f"cache_dir: {tmp_path / 'c'}")
    cfg.write_text(text)
    res = runner.invoke(app, ["run", "--task", "all", "--agent", "C", "--seeds", "dev",
                              "--llm-config", str(cfg), "--dry-run",
                              "--out", str(tmp_path / "run")])  # fmt: skip
    assert res.exit_code == 0, res.output
    assert "40 live" in res.output and "fits the caps" in res.output
    assert not (tmp_path / "run").exists()
    res = runner.invoke(app, ["llm", "estimate", "--task", "all", "--agent", "B+S",
                              "--llm-config", str(cfg), "--json"])  # fmt: skip
    assert res.exit_code == 0, res.output
    data = json.loads(res.output)
    assert data["n_requests"] == 40 and data["model"] == load_llm_params(NEBIUS).model
    res = runner.invoke(app, ["llm", "estimate", "--task", "all", "--seeds", "final_eval",
                              "--llm-config", str(cfg)])  # fmt: skip
    assert res.exit_code == 2  # locked seeds stay locked for estimates too
    res = runner.invoke(app, ["run", "--task", "all", "--agent", "A", "--dry-run"])
    assert res.exit_code == 0 and "nothing to price" in res.output


def test_cli_estimate_exits_1_when_over_the_cap(tmp_path: Path) -> None:
    cfg = tmp_path / "llm.yaml"
    text = NEBIUS.read_text().replace("cache_dir: cache/llm", f"cache_dir: {tmp_path / 'c'}")
    cfg.write_text(text.replace("max_usd_per_run: 2.0", "max_usd_per_run: 0.0001"))
    res = runner.invoke(app, ["llm", "estimate", "--task", "all", "--llm-config", str(cfg)])
    assert res.exit_code == 1 and "EXCEEDS" in res.output


def test_cli_check_and_models_without_key_fail_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ARMBENCH_NEBIUS_API_KEY", raising=False)
    res = runner.invoke(app, ["llm", "check", "--llm-config", str(NEBIUS)])
    assert res.exit_code == 1 and "ARMBENCH_NEBIUS_API_KEY" in res.output
    res = runner.invoke(app, ["llm", "models", "--llm-config", str(NEBIUS)])
    assert res.exit_code == 1 and "ARMBENCH_NEBIUS_API_KEY" in res.output
