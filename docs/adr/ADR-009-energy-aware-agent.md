# ADR-009: Energy-aware agents C and C+S — the reference budget is baseline A's Wh (F8)

- Status: accepted
- Date: 2026-10-04
- Phase: F8
- Decides: D8 (meaning of the "previous Wh" agent C receives)

## Context

Configuration C of the protocol is agent B plus energy information in the prompt; C+S is C
plus the frozen skill library of F7. The informe said C receives "the Wh of the previous
attempt", which is undefined under D5 (`max_attempts: 1`: there is no previous attempt) and
ambiguous otherwise: the previous *episode* of the same task would make C's prompt depend on
the order of the evaluation seeds and break the pairing with B, and the previous *attempt* of
the same episode would mix the effect of retries with the effect of the energy information.
`docs/plan.es.md` (F8) proposed a frozen "energy memory" from development seeds. The user chose,
among (1) retry Wh, (2) baseline A's Wh as a reference budget and (3) both pre-registered,
option 2.

## Decisions

### The energy block quotes baseline A on the same task, from a frozen artefact

`energy_ref/reference.json` (`armbench.energy.EnergyReference`) holds, per task, the
distribution of baseline A's electrical energy over the development seeds — median, IQM, min,
max, median simulated time and median primitive calls — for one protocol variant and one
efficiency (A, nominal eta), plus the provenance (run id, agent, backend, seeds, split, git
sha) and a build timestamp. `armbench energy-ref build --run runs/f5_dev` writes it from a
finished baseline-A **simulation** run (a fake-backend run or any other agent is refused);
`armbench energy-ref show` prints it. Its identity is the SHA-256 of the content without the
timestamp, so rebuilding it from the same run gives the same hash (a test checks the committed
file against `runs/f5_dev`). `build` refuses to overwrite: a new reference is a new directory
and a new hash. The reference is a protocol artefact like the frozen library: F9 registers its
hash in the pre-registered protocol and never rebuilds it from evaluation seeds.

**Rejected:** quoting C's own previous attempt (needs retries, confounds two effects); quoting
C's previous episode (order dependence, unpaired with B); computing the reference at run time
from whatever A run is on disk (not auditable).

### The prompt: model constants, budget, levers — nothing else changes

C's messages are B's messages with an `# Energy` section prepended to the task message
(`armbench.agents.prompt.energy_section`); the system prompt is byte-identical across A/B/B+S/
C/C+S and B's prompt is byte-identical to F7 (tests pin both). The section states the energy
model as scored (`P_el = P_mech / eta + copper losses + P0`, with the configured eta and P0),
the budget (baseline A's median Wh, IQM, range, n, simulated time and calls for *this* task),
the rule "aim at or below the budget without failing the task: a failed episode saves
nothing", and the two levers the API exposes, ranked honestly for this arm: time first (P0 is
paid every second: fewer moves, shorter paths, lower lifts, observe only when needed) and
`speed_scale` second (lower peak power, longer move, pays off only when the motion term
dominates). C+S prepends the `# Skills` section before `# Energy`, so it is exactly B+S plus
the energy block. No Wh of the episode itself is ever fed back: `EpisodeRecord.energy` is
computed after the episode and the agent never sees it.

### Traceability: what the agent saw, and whether it pulled the lever

`AgentTrace` gains `energy_reference_wh` and `energy_reference_sha256` (the budget quoted and
the reference it came from; `None` for A/B/B+S) and the lever accounting `n_moves`,
`n_slow_moves` (moves with `speed_scale < 1`) and `speed_scale_min`, derived from the sandbox
`CallRecord.speed_scale` (read from the `MoveResult.plan`) and, for skills, from
`SkillResult.speed_scales` (every `move_to` of the body). `run.json` records the reference
directory, hash, variant, eta, source run and the per-task budgets; the Docker bundle ships
`energy_ref/reference.json` into the `--network none` containers next to `llm.yaml` and the
frozen library, and the ROS `episode_runner` loads it for C/C+S only. The report adds, per
task and agent, the share of prompts with the energy block, the quoted reference, measured
Wh over reference, moves per episode, the share of episodes with `speed_scale < 1` and the
minimum speed used. A task missing from the reference is refused before any LLM call (exit 2
in the CLI), never silently run without a budget.

### The template provider gets a fixed energy recipe, declared as such

`TemplateProvider` answers an `# Energy` prompt with the B program after three literal edits
(`ENERGY_EDITS`): approach/carry height 10 → 6 cm (still above a 4.5 cm neighbour), the slow
transfer lift at full speed, and `SLOW` 0.5 → 0.7 for the descents it keeps. The task sentence
is parsed from the text after `# Task`, so energy words in the block cannot change the task
template. This is a deterministic stand-in that exercises the lever and the accounting
end-to-end; it is **not** a model's reasoning, and F8 claims nothing about energy savings
from it — H2 is evaluated in F9 with a real provider.

## Consequences

- C and C+S run today with `provider: template` and no API key; a real model changes nothing
  in the pipeline (the energy block is part of the cached request key).
- `scripts/f8_gate.py` checks: F8 tests green; reference built from baseline A on the
  simulation backend and rebuilt bit for bit; its hash registered in the C/C+S `run.json` and
  in every C/C+S trace; the energy block in every C/C+S prompt and in no B/B+S prompt; lever
  usage reported; the B vs C and B+S vs C+S Wh comparison reported with **no** improvement
  threshold; 0 cost aborts; 0 infra failures; complete energy tables.
- A change in the reference (another A run, another eta, another variant) is a new hash and a
  new cache key; a response cached under an identical prompt by an earlier code revision is
  replayed, so live gate runs start from an empty `cache/llm` (as in F6/F7).
- F9's pre-registered protocol must name the reference hash (`4213da2f…`, built from
  `runs/f5_dev`, dev seeds 0–9, variant A, eta 0.70) next to the library hash; evaluation
  seeds 100–119 never enter it.
