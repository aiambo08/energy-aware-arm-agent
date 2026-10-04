# Project state

Single source of truth for where the project is. Updated at the end of every
phase PR.

| Field | Value |
|---|---|
| Current phase | F8 done (energy-aware agents C and C+S on a frozen baseline-A energy reference, D8 option 2; Gazebo pilot B/C/B+S/C+S 40/40 each with the deterministic template provider); F9 next |
| Last green commit | `main` after PR F6 — see `git log -1 main` |
| Next phase | F9 — pre-registered evaluation (protocol frozen in a commit before the 400 episodes on the locked `final_eval` seeds 100–119, H1–H3, Wilson/IQM/bootstrap, η sensitivity). Provider and model fixed (D6): Nebius `Qwen/Qwen3-235B-A22B-Instruct-2507`; free Gemini pilot on dev seeds first |
| Blockers | F9 needs the keys (`ARMBENCH_GEMINI_API_KEY` for pilots, `ARMBENCH_NEBIUS_API_KEY` for the evaluation), both verified with `armbench llm check` on 2026-10-04; everything up to F8 runs without one. Spending tools are ready: `armbench llm check/models/estimate`, `run --dry-run`, `max_usd_total`, resumable prefetch |

## Phase status

| Phase | Status | Evidence |
|---|---|---|
| F0 skeleton, Docker, quality | merged (PR #1) | `reports/f0_quality.json` (ruff, format, mypy strict, 13 tests, 1.2 s), `reports/f0_sim.json` (boot 2.5 s, image 4.46 GB) |
| F1 arm simulation | done (PR #3) | `reports/f1_sim.json` (3 runs) and `reports/f1_sim_50runs.json` (50 runs: boots 50/50, ready ≤ 18.7 s, RTF ≥ 0.77, camera ≥ 10.6 FPS, grasp 50/50, slip ≤ 0.25 mm); kinematics + scene generator unit tests |
| F2 torque source and energy model | done (PR #5) | `reports/f2_torque_source.json` (static effort vs gravity ≤ 1.6 %, motion NRMSE median 0.06, Wh CV 1.1 %, meter 2.3 % of container CPU); energy model unit + hypothesis tests; ADR-004 |
| F3 perception `detect()` | done (PR #6) | `reports/f3_perception.json` — 200 scenes / 891 cubes: recall 100 %, 0 FP, XY median 1.39 mm, p95 2.34 mm, Z p95 < 0.001 mm, latency p95 14.4 ms; perception unit + hypothesis tests on synthetic renders |
| F4 primitives with contracts | done (PR #7) | `reports/f4_primitives.json` — 200 moves pos p95 0.23 mm / yaw p95 0.03°, 0 collisions; 100/100 `OutOfReach` with 0 goals; speed_scale monotone (8.01 → 0.88 s); 100 resets ≤ 1.96 s; pick-and-place 98/100 (seeds 418 and 489 placed > 15 mm off); 46 primitive unit/hypothesis tests on the kinematic backend; ADR-005 |
| F5 tasks, baseline A, runner | done (PR F5) | `reports/f5_baseline.json` — 250 Gazebo episodes (4 tasks × dev 0–9, 10 repeats of `pick_place@1` seed 0, 4 tasks × dev_extended 400–449): success 250/250, energy table complete, 0 infra failures, wall p95 ≤ 26.6 s, repeat Wh CV 0.36 %; 82 task/runner/report unit tests (tests/unit/test_tasks.py, test_runner.py); ADR-006 |
| F6 LLM agent, sandbox, cost control | done (PR F6) | `reports/f6_agent_b.json` — 40 Gazebo episodes of agent B (4 tasks × dev 0–9) with the deterministic `template` provider: success 40/40, 0 infra failures, every program `completed` in the sandbox, energy table complete, wall p95 ≤ 24.5 s; the replay run reproduces the LLM side of every episode exactly (same response, program and primitive-call sequence, 40/40 served from the cache) and judged 40/40 the same — in an earlier run of the same programs one `place_obstacle` episode judged differently with identical calls (place error 3 mm vs 63 mm), so the gate requires call-sequence equality and only reports outcome agreement (physics is Gazebo's, not bit-reproducible); host ledger 38 live calls + 2 cache hits (two `stack2` seeds share the goal, hence the prompt), 68 812 tokens, 0 USD (template prices are 0; extrapolation to 400 episodes reported); attack battery 86 cases, 82 stopped (≥ 40 required), 364 host tests; isolation layers recorded per episode (`ast,builtins,container,rlimits` — `unshare -rn` is not permitted inside the container, so no network namespace layer); ADR-007. These numbers validate the infrastructure, not an LLM: the template provider is not a language model |
| F7 skill library (B+S) | done (PR F7) | `reports/f7_skill_validation.json` — 4 candidates proposed from the 40 completed B programs of `runs/f6_live` (one per task, goal constants parametrised), each validated on the 20 `skill_validation` seeds 20–39 in the sandbox with pre/post-conditions and the task checker: 20/20 accepted (`pick_place_cube`, `stack_cube`, `sort_cubes`, `place_cube_over_wall`); `skills/FROZEN.json` sha256 `18ba25b8…b6a3`, recorded in every B+S `run.json`. `reports/f7_skills.json` — Gazebo pilot 4 tasks × dev 0–9, cache emptied first: B 40/40 and B+S 40/40, 0 infra failures, energy tables complete; B+S called a skill in 40/40 episodes (one `execute_skill` per program) and the trace counts the inner primitives (median 9/9/27/11 per task for both agents — the skill bodies *are* B's programs); tokens/episode median 1 653–1 731 for B+S vs 1 753–1 920 for B (the `# Skills` section costs less than the inline steps the template writes); Wh A medians within −0.2 % … +2.3 % of B (same motions; differences are Gazebo run-to-run noise, no improvement claimed — H1 is tested in F9); fault-injection tests (5 cm offset, held cube, tampered frozen file, unknown skill, primitive error inside a skill) 7/7; 393 host tests; ADR-008. The template provider is not a language model: these numbers validate the library, the contract checks and the accounting, not skill reuse by an LLM |
| F8 energy-aware agent (C, C+S) | done (PR F8) | `energy_ref/reference.json` — baseline A's Wh per task from `runs/f5_dev` (agent A, sim, dev 0–9, variant A, η 0.7): medians 0.2105 / 0.2257 / 0.5842 / 0.2279 Wh for `pick_place` / `stack2` / `sort3` / `place_obstacle`; sha256 `4213da2f…3c5` (timestamp excluded from the hash; the gate rebuilds it from the run bit for bit) and recorded in every C/C+S `run.json` and trace. `reports/f8_energy_aware.json` — Gazebo pilot 4 tasks × dev 0–9, cache emptied first: B 40/40, C 40/40, B+S 40/40, C+S 40/40, 0 infra failures, 0 cost aborts, energy tables complete; the `# Energy` block is in 40/40 C and C+S prompts and in 0/80 B and B+S prompts; `speed_scale < 1` used in 40/40 episodes of every agent (min 0.5 for B/B+S/C+S — the frozen skill bodies carry their own `SLOW = 0.5` — and 0.7 for C). Wh A medians C/B = 0.95–0.98 (template's energy edits: lower approach height, faster slow segment, one fewer slow move) and C+S/B+S = 0.99–1.00 (with a skill the program is a single `execute_skill` call, so the template has nothing to edit inside the frozen body); same move counts per pair; tokens/episode +~350 for the energy block. No improvement is claimed: there is no threshold, these are medians over 10 seeds with a stand-in provider, and H2/H3 are tested in F9. 21 F8 unit tests (`tests/unit/test_agent_c.py`), 414 host tests; ADR-009 |
| F9 pre-registered evaluation | not started | — |
| F10 publication | not started | — |

## Decisions log

| ADR | Title | Status |
|---|---|---|
| ADR-001 | Simulation stack, repository name and package name | accepted |
| ADR-002 | Simulation image size threshold raised to 5 GB | accepted |
| ADR-003 | Simple parallel-jaw gripper instead of Robotiq 2F-85 (D2) | accepted |
| ADR-004 | Torque source = Gazebo effort, η excludes copper losses (D3, D4) | accepted |
| ADR-005 | Pre-flight checks and typed errors of the robot primitives | accepted |
| ADR-006 | Tasks, scripted baseline A and the episode log (gripper readiness, partial detections) | accepted |
| ADR-007 | LLM agent B: provider interface, host prefetch + container replay, four-layer process sandbox (D5) | accepted |
| ADR-008 | Skill library: validated parametrised programs, closed-vocabulary conditions, frozen by hash before evaluation, nested sandboxed `execute_skill`, agent B+S (D7) | accepted |
| ADR-009 | Energy-aware agents C and C+S: baseline A's Wh as the frozen, hashed reference budget in the prompt; `speed_scale` accounting (D8) | accepted |

## Open decisions (from `docs/plan.es.md` §7)

| # | Decision | Due |
|---|---|---|
| D5 | Single-turn vs retries — single turn (`max_attempts: 1`), retries kept configurable (ADR-007) | decided F6 |
| D6 | LLM model and provider — interface and replay decided (ADR-007); hybrid plan: Gemini (AI Studio free tier, `configs/llm.gemini.yaml`) for development and pilots, Nebius Token Factory (`configs/llm.nebius.yaml`, one fixed model, total cap 3 USD) for the pre-registered evaluation; Nebius model `Qwen/Qwen3-235B-A22B-Instruct-2507` at 0.20/0.60 USD per 1M tokens (instruct, no thinking: short, predictable completions; strong at code; cheap). `meta-llama/Llama-3.3-70B-Instruct`, the earlier placeholder, is not offered. Worst case for the full F9 LLM grid (4 agents × 4 tasks × 20 seeds) ≈ 0.42 USD, inside the 3 USD cap | decided 2026-10-04 |
| D7 | Frozen skill library — built from dev programs, validated on 20–39, frozen by hash and read-only for evaluation (ADR-008) | decided F7 |
| D8 | Meaning of "previous Wh" in C — option 2: baseline A's Wh for the same task from a frozen, hashed reference (ADR-009); no retry/previous-episode memory | decided F8 |
| D10 | Whether to build F11 (live demo) | F10 |
