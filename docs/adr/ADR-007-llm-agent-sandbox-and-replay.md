# ADR-007: Agent B — provider interface, response cache/replay and the program sandbox (F6)

- Status: accepted
- Date: 2026-10-03
- Phase: F6

## Context

B is the first LLM configuration of the protocol: per episode the model reads a task prompt and
writes a Python program over the F4 primitives; the program is executed and judged by the F5
success checker exactly like baseline A. Three properties of the plan constrain the design:
the whole evaluation must be reproducible without a GPU or an API key (`docs/plan.es.md`,
"público nivel 1"), the generated code is untrusted, and the Gazebo containers run with
`--network none` (F1: a second DDS participant corrupted the gate). No API key is available
during this phase; the user chose to build the infrastructure first and plug the key later.

## Decisions

### One `Provider` protocol, five implementations, no SDK

`armbench.llm.Provider` is `complete(LLMRequest) -> LLMResponse`. `LLMRequest` is a frozen
model (`model`, `messages`, `temperature`, `max_tokens`, `seed`) and its `key()` is the SHA-256
of its canonical JSON, so two hosts build the same key for the same prompt. Implementations:
`TemplateProvider` (deterministic programs derived from the task text; the only provider used in
this phase), `StaticProvider` (tests), `OpenAICompatProvider` (`urllib.request` against any
`/v1/chat/completions`, key from `ARMBENCH_LLM_API_KEY`, never called in CI), `CachedProvider`
(disk cache `<dir>/<key[:2]>/<key>.json`, atomic writes) and `ReplayProvider` (cache only;
`CacheMiss` is a typed error). Prices, the per-run `max_usd` guard and the `Ledger`
(tokens, USD, calls, failures) live in `BudgetedProvider`, which also prices cached answers so
the extrapolated cost of the final evaluation is visible before a single live call.

**Rejected:** the `openai` SDK (adds a dependency to the simulation image for one HTTP call) and
a vector store for prompts (not needed: the key is exact, replay is exact).

### `TemplateProvider` is infrastructure, not evidence

Its programs are hand-derived from the task family and hit the same structure as baseline A.
Numbers obtained with it (success, Wh, tokens) validate the pipeline — prompt → program →
sandbox → log → report — and nothing about an LLM's competence. `reports/f6_agent_b.json`
and the README say so explicitly; `AgentTrace.llm_provider` records the provider per episode.

### Programs run inside containers from a per-run bundle

The containers cannot reach any API, so `armbench run --agent B --backend sim` fetches every
first-turn program on the host (`prefetch`), writes `<out>/llm/{llm.yaml, cache/, ledger.json}`
and copies that bundle into each container (`docker cp`), where `episode_runner` builds
`get_agent("B", provider=ReplayProvider)`. Keys match because the prompt depends only on
`PrimitiveParams`, the `TaskInstance` and the sandbox limits, all of which are part of the
protocol. Retry turns (`max_attempts > 1`) are *not* prefetched: they depend on the run-time
error, so they are only available with a live provider on the fake backend, or in a future
phase with a host-side RPC. F6 ships `max_attempts: 1`, as the plan prescribes.

### Four-layer sandbox, honest about what each layer gives

1. **AST whitelist** (`armbench.guest.ast_check`): statements and expressions limited to an
   explicit set; no `import`, `global`/`nonlocal`, `async`, `with`, `try`, decorators, classes,
   `yield`, names or attributes starting with `_`, and a deny list of dunder-reaching attributes
   (`__class__`, `__dict__`, `gi_frame`, `format`, ...). Builtins are a whitelist (`len`,
   `range`, `min`, `sorted`, `abs`, ...); `eval`, `exec`, `compile`, `open`, `getattr`,
   `setattr`, `vars`, `type`, `__import__` are absent from the namespace *and* rejected by the
   AST pass. Caps: 20 000 chars, 5 000 AST nodes.
2. **Child process** `python -I -m armbench.guest.worker`: `RLIMIT_AS`, `RLIMIT_CPU`,
   `RLIMIT_FSIZE=0`, `RLIMIT_NPROC=0`, `RLIMIT_CORE=0`, empty environment, read-only temporary
   cwd, `unshare -rn` (new user+network namespace) when the kernel allows it. The guest package
   has no heavy imports (no NumPy/OpenCV/Pydantic), so a 512 MB address-space cap is viable
   (~35 MB resident).
3. **Caps** enforced by the parent: `max_calls`, `wall_s`, `sim_s` (robot clock), `stdout_kb`.
4. **Typed RPC**: the child only has `RobotProxy`; each primitive is a JSON line
   `{"id", "call", "args"}` answered by the parent with the serialised `Result` or a typed
   `PrimitiveError` code. The parent validates every message with Pydantic and refuses unknown
   calls, so a program can never reach the real `Robot`, `Backend` or ROS.

`ProgramRun.isolation` records which layers were actually active (`ast`, `builtins`,
`rlimits:<applied>`, `unshare_net`, `container`). On this host `unshare -rn` is refused inside Docker (user
namespaces disabled), so the containers rely on `--network none` + rlimits; **the sandbox is
process-level, not a VM boundary**, and `docs/plan.es.md` keeps "ejecución en VM dedicada"
as the requirement for a public live demo (F11).

**Rejected:** `RestrictedPython` (blacklists bytecode, still shares the interpreter) and
running programs in a second container per episode (≈ 2 s per episode of overhead and a Docker
socket inside the runner).

### Outcomes are data, never exceptions

The sandbox returns `ProgramRun.outcome ∈ {completed, rejected, primitive_error, exception,
call_limit, sim_limit, timeout, killed, protocol_error}`; agent B maps all of them into
`AgentTrace` (`program_outcome`, `program_calls`, `program_error`, hashes, tokens, latency,
cost, `sandbox_isolation`) and only raises `AgentError` for provider/budget failures, which
`run_episode` logs as `Failure(stage="agent")`. Attack programs therefore appear in the
episode log as failed episodes of B, not as infrastructure failures.

## Consequences

- F7 (skills) can store `program_sha256` + `response_sha256` and re-run a skill with the same
  sandbox; the replay bundle format (`llm.yaml` + cache) is the artefact the dataset of F10 ships.
- Adding a real model is configuration only: `provider: openai_compat`, `openai.base_url`,
  `openai.model`, the env var, prices in `prices_usd_per_1m`; the test battery, the bundle and
  the report need no change.
- The report gains an `llm` block per task (tokens, cost, cost/400 episodes, cache rate,
  latency p95, attempts, program outcomes, isolation layers) so the F9 budget decision is made
  on measured numbers.
