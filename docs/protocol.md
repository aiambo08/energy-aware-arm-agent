# F9 pre-registered evaluation protocol

Status: **frozen once merged into `main`**. The protocol hash is the sha256 of this file's bytes
(`uv run armbench protocol hash`). `armbench run --final-eval` refuses the locked seeds unless
`--protocol-hash` equals it and every artefact pinned in §10 still matches the committed
repository (`uv run armbench protocol verify`). Any change after the freeze is an amendment
(§11): a new protocol id, a new hash, and a written reason, made *before* any final-eval episode
is observed — never after.

## 1. Question and hypotheses

Does giving an LLM agent energy information, and letting it reuse validated programs as skills,
reduce the electrical energy a simulated UR5e spends per task while keeping it successful? All
agents write one Python program over the same closed, typed primitives, run in the same sandbox.

| Id | Comparison (treatment vs control) | Primary endpoint | Supported if |
|---|---|---|---|
| H1 — skills | B+S vs B | episode success | paired success difference: 95 % CI lower bound > 0 |
| H2 — energy information | C vs B | Wh per episode | paired Wh relative difference: 95 % CI upper bound < 0 in **all 6** sensitivity rows (§6.3) **and** success non-inferior (CI lower bound > −0.10) |
| H3 — energy information with skills | C+S vs B+S | Wh per episode | same rule as H2 |

H1's Wh, tokens and time, and every C+S vs B and X vs A comparison, are secondary and reported
without a decision rule. The hypotheses are about agents, not acceptance criteria of the
software: a null result is a valid result.

## 2. Configurations

| Agent | Writes the program | Skills (`skills/`, frozen) | `# Energy` block (baseline A's Wh, D8) |
|---|---|---|---|
| A | scripted baseline (no LLM) | — | — |
| B | LLM | no | no |
| B+S | LLM | yes | no |
| C | LLM | no | yes |
| C+S | LLM | yes | yes |

The only differences between B, B+S, C and C+S are the prompt sections the agent ids add; the
model, decoding parameters, API reference, sandbox, primitives, tasks and seeds are identical.

## 3. Tasks and seeds

- Tasks: `pick_place@1`, `stack2@1`, `sort3@1`, `place_obstacle@1` (success checkers in
  `src/armbench/tasks/`, unchanged since F5).
- Seeds: the locked `final_eval` split, 100–119 (`configs/seeds.yaml`), 20 per task, one episode
  per (agent, task, seed): 5 × 4 × 20 = **400 episodes**, 320 of them with an LLM call.
- No final-eval seed has been run, prompted or inspected before the freeze. Dev (0–9),
  skill-validation (20–39) and dev-extended (400–499) data were used for development only.

## 4. LLM

- Provider: Nebius Token Factory, OpenAI-compatible endpoint, profile `configs/llm.nebius.yaml`.
- Model: `Qwen/Qwen3-235B-A22B-Instruct-2507` (D6), for every LLM agent. No fallback model or
  provider: if the model becomes unavailable mid-evaluation the evaluation stops and is
  reported as a technical failure (§8), nothing is substituted.
- Decoding: temperature 0.0, seed 0, `max_tokens` 1500, `max_attempts` 1 (single turn, no
  adaptive retries), `max_tokens_per_episode` 8000 (above it the episode fails as
  `agent:token_budget`).
- HTTP transport retries (`max_retries` 3, timeouts, 429 back-off) resend the same request and
  are not agent attempts.
- Every request/response pair is stored in the content-addressed cache `cache/llm` and copied
  into each run directory; the cache is published with the dataset and never deleted. Replay
  (`--provider replay`) re-runs the evaluation without any call.

## 5. Execution

Prerequisites (on the merge commit of this protocol, clean working tree):

```bash
docker build -f docker/Dockerfile -t armbench-sim:dev .   # the image must contain this commit
uv run armbench protocol verify                           # exit 0 required
H=$(uv run armbench protocol hash)
```

Cost check before any live call (worst case must fit both caps, otherwise stop):

```bash
for a in B B+S C C+S; do
  uv run armbench llm estimate --task all --agent "$a" --seeds final_eval \
    --final-eval --protocol-hash "$H" --llm-config configs/llm.nebius.yaml
done
```

Runs, in this order (A has no LLM, so it carries no provider risk):

```bash
uv run armbench run --task all --agent A --seeds final_eval --backend sim \
  --final-eval --protocol-hash "$H" --out runs/f9_A
for a in B B+S C C+S; do
  d=runs/f9_$(echo "$a" | tr -d '+')$( [[ $a == *+S ]] && echo S )
  uv run armbench run --task all --agent "$a" --seeds final_eval --backend sim \
    --final-eval --protocol-hash "$H" --llm-config configs/llm.nebius.yaml --out "$d"
done
```

- Each task runs in fresh containers (`--chunk 25`, so one container per task and agent), with
  `--network none`, one container at a time.
- `run.json` records the protocol hash, git SHA, image, `armbench` version, LLM profile and
  model, the frozen-library hash and the energy-reference hash; every episode record carries
  its software versions, instance hash, prompt/response/program hashes, tokens, cost and
  sandbox outcome.
- Spend caps: `max_usd_per_run` 2.0 USD and `max_usd_total` 3.0 USD (ledger
  `cache/llm/spend.json`, all Nebius calls of the project included). A run stops *before* a
  live call that could cross a cap. Estimated worst case of the 320 calls: ≈ 0.42 USD.

## 6. Analysis

`uv run armbench analyze runs/f9_A runs/f9_B runs/f9_BS runs/f9_C runs/f9_CS
--out reports/f9_eval.json --md reports/f9_eval.md` (code: `src/armbench/analysis.py`, pinned
by §10). It is deterministic: the same logs give the same `report_sha256`.

### 6.1 Episode selection

Per (agent, task, seed) the latest episode without an infrastructure failure is scored.
Episodes with `infra_failure` are listed and counted in the valid fraction, never scored as a
success or a failure.

### 6.2 Per-cell summaries (agent × task)

Success k/n with Wilson 95 % interval; Wh (variant A, η = 0.70) over successful episodes:
median and IQM; simulated time of successful episodes and wall time: median; tokens per
episode: median; cost; sandbox outcomes; episodes that called a skill; episodes with
`speed_scale < 1`.

### 6.3 Paired comparisons (one per hypothesis)

Over the (task, seed) pairs both agents ran:

- **Success:** mean of `ok_treatment − ok_control` (pooled over tasks).
- **Energy:** over the pairs where both episodes succeeded, the per-pair relative difference
  `Wh_treatment / Wh_control − 1`; statistic = median (pooled over tasks). Computed for each of
  the **6 sensitivity rows**: model variant A (no regeneration) and B (|τω|) × η ∈ {0.60, 0.70,
  0.80} (`configs/energy.yaml`; P0 = 100 W; torque source = Gazebo `effort`, ADR-004; η excludes
  copper losses). The plan's "4 variants" (A/B × 0.60/0.80) are a subset; requiring the nominal
  η too only makes the rule stricter.
- **Intervals:** percentile bootstrap, 95 %, 10 000 resamples, stratified by task (seeds are
  resampled within each task), NumPy `default_rng(20261004)`.

Restricting energy to jointly successful pairs avoids crediting an agent with the low energy of
a program that gave up early; the success guard in H2/H3 prevents an energy "saving" bought
with failures.

### 6.4 Decision rules and scenario

- H1: supported if the success-difference CI lower bound > 0, otherwise not supported.
- H2, H3: supported if every energy row saves (CI upper bound < 0) and success is non-inferior
  (CI lower bound > −0.10); **mixed** if at least one row saves but not all (or success is
  inferior); otherwise **not supported**.
- Scenario: **favourable** if H2 and H3 are supported; **mixed** if either is supported or
  mixed; **null** otherwise; **technical failure** if §8 applies (then no hypothesis is judged).
- A saving is only claimed under "supported". Effects are reported as medians with intervals;
  no causal claim beyond the paired design on these four tasks, this simulator and this model.

## 7. Reported regardless of outcome

All 400 episodes, the per-cell table, the three comparisons with all six energy rows, the
secondary comparisons, every exclusion with its reason and episode id, total tokens and cost,
and every deviation from this protocol.

## 8. Exclusions, re-runs and technical failure

- Infrastructure failure (container, Gazebo, controller, bridge, DDS; `infra_failure = true`):
  the episode is re-run once in a fresh container with the same cached LLM response
  (`--task <task> --seeds <seed>` for that agent, same `--protocol-hash`, into a new directory
  such as `runs/f9_<agent>_rerun1` that is added to the `analyze` arguments), and both
  attempts are reported. Agent failures (rejected program, exception, timeout, budget, wrong final state)
  are **never** re-run or removed.
- Technical failure of the evaluation: valid fraction < 0.99 after re-runs, an (agent, task,
  seed) cell without a scored episode, a spend cap hit, or the model unavailable. It is
  reported as such and `analyze` sets the scenario to `technical failure`.
- No interim analysis: no comparison is computed until all five runs are complete.

## 9. Reproduction

- From logs: `armbench analyze` as in §6 regenerates `reports/f9_eval.json` with the same
  `report_sha256`.
- From replay on another machine: the same runs with `--provider replay` and the published
  cache must give identical LLM requests, responses and primitive calls; Gazebo physics is
  not bit-reproducible (F6), so the per-episode verdict agreement and Wh within ±3 % are
  reported, not assumed.

## 10. Manifest

Machine-readable; `armbench run --final-eval` and `armbench analyze` read it.
`code_tree_sha256` covers every tracked file under `configs/`, `docker/`, `energy_ref/`,
`ros_ws/src/`, `skills/` and `src/armbench/` (prompt builder, sandbox and its builtin list —
`next` allowed, `iter` forbidden —, primitives, tasks, energy model, analysis);
`uv run armbench protocol artefacts` prints the current values.

```armbench-protocol
schema_version: 1
protocol_id: f9-v1
split: final_eval
llm_config: configs/llm.nebius.yaml
model: Qwen/Qwen3-235B-A22B-Instruct-2507
agents: [A, B, B+S, C, C+S]
tasks: [pick_place@1, stack2@1, sort3@1, place_obstacle@1]
hypotheses:
  H1: [B+S, B]
  H2: [C, B]
  H3: [C+S, B+S]
analysis:
  bootstrap_resamples: 10000
  bootstrap_seed: 20261004
  ci_level: 0.95
  noninferiority_margin: 0.10
  min_valid_fraction: 0.99
artefacts:
  code_tree_sha256: 0d0f71bc01d9601f0872613ed6228c5c38c8bbd8aec4810de45d5c5c7e4236a0
  skills_sha256: 18ba25b859cee9b33cec8a43c44f4ddf1c2006c6dea17b8b391f430b4ab7c6a3
  energy_reference_sha256: 4213da2feb590782d50345c0990db8695a1ee0b9664160167be8eb2eb3c8f0c5
```

## 11. Amendments

None.
