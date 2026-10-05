# armbench: the benchmark

How to run it, what is measured, what the logs contain, and how to add an agent, a task or a
model. Results of the pre-registered evaluation are in [`report.md`](report.md); the protocol
they follow is [`protocol.md`](protocol.md).

## 1. Setup

| Need | For |
|---|---|
| [uv](https://docs.astral.sh/uv/) ≥ 0.8 | everything on the host (installs Python 3.12) |
| Docker Engine ≥ 24 | Gazebo episodes (`--backend sim`), or the `docker compose` demo |
| an OpenAI-compatible LLM endpoint + key | new LLM episodes (not needed for replay) |

No GPU is needed: Gazebo renders the camera with Mesa llvmpipe.

```bash
git clone https://github.com/aiambo08/energy-aware-arm-agent.git
cd energy-aware-arm-agent
uv sync
docker build -f docker/Dockerfile -t armbench-sim:dev .     # ~5.7 GB, 15-25 min the first time
```

Windows: use WSL2 (Ubuntu 24.04) with Docker Desktop's WSL integration and run the commands
above inside the WSL shell. This path is documented but has not been tested by the authors.

## 2. Replay the F9 evaluation (no key, no GPU, no ROS)

```bash
uv run python scripts/demo.py --task pick_place --agent C --seed 100
# or, without uv on the host:
docker compose run --rm demo --task sort3 --agent B+S --seed 107
```

`scripts/demo.py` extracts `data/f9-v1.tar.gz` into `.armbench/`, copies the logged LLM answers
into `cache/llm`, runs `armbench run --provider replay --backend fake --final-eval` for that
episode and compares it with the Gazebo log. It exits 1 if the prompt or the program differ
(that would mean the code changed what the agent sees). The primitive calls and the verdict
may differ: the fake backend is kinematic, so a program that branches on what it detects can
take another branch. Watt-hours are only measured in Gazebo and are printed from the log.

Re-run the statistics from the dataset:

```bash
mkdir -p /tmp/f9 && tar -xzf data/f9-v1.tar.gz -C /tmp/f9
uv run armbench analyze /tmp/f9/f9-v1/f9_A /tmp/f9/f9-v1/f9_B /tmp/f9/f9-v1/f9_BSS \
  /tmp/f9/f9-v1/f9_C /tmp/f9/f9-v1/f9_CSS --out /tmp/f9_eval.json
python3 -c "import json; print(json.load(open('/tmp/f9_eval.json'))['report_sha256'])"
# 87390f0bfe0be03eb58b137831987aebb81190498f8ba26a0d35d16512559028
```

## 3. Run episodes

```bash
uv run armbench seeds                                  # dev 0-9, skill_validation 20-39, final_eval 100-119 (locked)
uv run armbench run --task pick_place@1 --agent A --seeds 0 --backend sim --out runs/a_smoke
uv run armbench run --task all --agent C --seeds dev --backend sim \
  --provider openai --llm-config configs/llm.nebius.yaml --dry-run      # tokens and USD, no call
uv run armbench run --task all --agent C --seeds dev --backend sim \
  --provider openai --llm-config configs/llm.nebius.yaml --out runs/c_dev
uv run armbench report runs/c_dev --out reports/c_dev.json
```

- `--backend fake` runs the same agents on a kinematic backend in-process (no Docker, no Wh).
- `--provider template` is a deterministic stand-in (no key); `replay` serves only cached answers.
- Every response is cached by (provider, endpoint, model, parameters, prompt) in `cache/llm`;
  a run cut by a rate limit or the spend cap resumes without paying again.
- `final_eval` seeds refuse to run without `--final-eval --protocol-hash "$(uv run armbench protocol hash)"`
  and a clean tree matching the protocol manifest.

LLM profiles: `configs/llm.nebius.yaml` (the F9 model), `configs/llm.gemini.yaml` (Google AI
Studio free tier, ~20 requests/day) and `examples/llm.ollama.yaml` (a local model through
Ollama's OpenAI-compatible endpoint). Results with another model are a different experiment
and must not be pooled with `f9-v1`.

## 4. Tasks, agents, primitives

| Task | Goal | Success (ground truth) |
|---|---|---|
| `pick_place@1` | move the named cube to a goal on the table | cube within tolerance of the goal |
| `stack2@1` | put one cube on top of another | top cube resting on the bottom one |
| `sort3@1` | three cubes to the goals of their colours | all three within tolerance |
| `place_obstacle@1` | place a cube behind a wall | at goal and the wall untouched |

Instances are pure functions of `(task, seed)` (`src/armbench/tasks`).

| Agent | Writes the program | Skills | Energy block |
|---|---|---|---|
| A | scripted baseline | – | – |
| B | LLM | – | – |
| B+S | LLM | frozen library (`skills/`) | – |
| C | LLM | – | baseline A's Wh per task (`energy_ref/`) |
| C+S | LLM | frozen library | baseline A's Wh |

The program sees only `robot.observe()`, `robot.detect(color)`, `robot.move_to(pose, speed_scale)`,
`robot.grasp()`, `robot.release()`, `robot.execute_skill(name, **kwargs)` (B+S/C+S), `Pose`,
the typed errors and a short builtin list. It runs in the sandbox described in
[`../SECURITY.md`](../SECURITY.md).

## 5. What is measured

Per episode (`EpisodeRecord`, schema v1, one JSON line in `episodes.jsonl`):

- `ok`, `reason`, `failure` (`stage`: agent / robot / judge, typed error), `infra_failure`;
- `energy.rows`: Wh for mechanical-power variants A (Σ max(τ·q̇, 0): no regeneration) and B (Σ|τ·q̇|: braking
  costs as much as driving) ×
  drive efficiency η 0.6 / 0.7 / 0.8, plus copper losses and idle power P₀ (`configs/energy.yaml`,
  ADR-004). Torque is Gazebo's joint effort sampled at 100 Hz. The primary row is A, η 0.7;
- `sim_s`, `wall_s`, `metrics` (task-specific errors), `min_tip_z_m`;
- `trace`: prompt/completion tokens, cost, latency, cache hit, `prompt_sha256`,
  `program_sha256`, sandbox outcome and isolation layers, primitive call sequence,
  `skills_used`, moves and slow moves, `speed_scale_min`, energy reference;
- `software`, `instance_sha256`, `run_id`, `started_at`.

`run.json` holds the agent, git sha, LLM profile, frozen-library and energy-reference hashes
and, for final runs, the protocol hash. With `--backend sim`, `samples/<episode>.jsonl.gz`
holds the 100 Hz `{t, q, qd, tau}` samples the energy was integrated from.

## 6. Extending

New results that are meant to be compared with `f9-v1` need a new protocol (copy
`docs/protocol.md`, new id and seeds) because the code tree hash is pinned.

- **Agent**: implement the `Agent` protocol (`src/armbench/agents/base.py`, `solve(robot, instance) -> AgentTrace`),
  add its id to `AGENT_IDS` and to `get_agent` in `src/armbench/agents/__init__.py`, unit-test it
  on `--backend fake`.
- **Task**: implement the `TaskSpec` protocol (`src/armbench/tasks/spec.py`: `instance(seed, config)`,
  `check(instance, final)`), register it in `src/armbench/tasks/__init__.py` with a versioned id (`name@1`)
  and add a baseline-A program; bump the version
  whenever the checker or the goals change.
- **Model**: copy a profile from `configs/` or `examples/`, set `model`, `base_url`,
  `api_key_env` and real prices, then `armbench llm check` and `armbench run --dry-run`.

## 7. Releasing

`release.yml` runs on a pushed tag `vX.Y.Z` equal to `pyproject.toml`'s version: tests, `uv build`,
PyPI upload via Trusted Publishing and the simulation image to
`ghcr.io/aiambo08/energy-aware-arm-agent` (`X.Y.Z`, `X.Y`, `latest`). One-time setup:

1. pypi.org → *Your projects* → *Publishing* → add a pending publisher: owner `aiambo08`,
   repository `energy-aware-arm-agent`, workflow `release.yml`, environment `pypi`.
2. GitHub → *Settings* → *Environments* → create `pypi`.
3. GitHub → *Settings* → *Pages* → Source "GitHub Actions"; `pages.yml` deploys `site/` on
   each published release or by hand.
4. Zenodo → *GitHub* → enable the repository; a published GitHub release then gets a DOI.
   Upload `armbench-f9-v1-full.tar.gz` (`uv run python scripts/export_dataset.py --full`, needs
   the local `runs/`) to the same record as the full dataset.

Rebuild the committed artefacts after changing runs or the page template:

```bash
uv run python scripts/export_dataset.py         # data/f9-v1.tar.gz, deterministic sha256
uv run python scripts/build_site.py             # site/index.html (needs runs/ with samples/)
```
