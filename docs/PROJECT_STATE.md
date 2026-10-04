# Project state

Single source of truth for where the project is. Updated at the end of every
phase PR.

| Field | Value |
|---|---|
| Current phase | F3 merged (perception gate passed); F4 next |
| Last green commit | `main` after PR #6 (F3) — see `git log -1 main` |
| Next phase | F4 — primitives with Pydantic contracts (`observe`, `detect`, `move_to`, `grasp`, `release`), IK, typed errors, simulation tests |
| Blockers | none |

## Phase status

| Phase | Status | Evidence |
|---|---|---|
| F0 skeleton, Docker, quality | merged (PR #1) | `reports/f0_quality.json` (ruff, format, mypy strict, 13 tests, 1.2 s), `reports/f0_sim.json` (boot 2.5 s, image 4.46 GB) |
| F1 arm simulation | done (PR #3) | `reports/f1_sim.json` (3 runs) and `reports/f1_sim_50runs.json` (50 runs: boots 50/50, ready ≤ 18.7 s, RTF ≥ 0.77, camera ≥ 10.6 FPS, grasp 50/50, slip ≤ 0.25 mm); kinematics + scene generator unit tests |
| F2 torque source and energy model | done (PR #5) | `reports/f2_torque_source.json` (static effort vs gravity ≤ 1.6 %, motion NRMSE median 0.06, Wh CV 1.1 %, meter 2.3 % of container CPU); energy model unit + hypothesis tests; ADR-004 |
| F3 perception `detect()` | done (PR #6) | `reports/f3_perception.json` — 200 scenes / 891 cubes: recall 100 %, 0 FP, XY median 1.39 mm, p95 2.34 mm, Z p95 < 0.001 mm, latency p95 14.4 ms; perception unit + hypothesis tests on synthetic renders |
| F4 primitives with contracts | not started | — |
| F5 tasks, baseline A, runner | not started | — |
| F6 LLM agent, sandbox, cost control | not started | — |
| F7 skill library (B+S) | not started | — |
| F8 energy-aware agent (C, C+S) | not started | — |
| F9 pre-registered evaluation | not started | — |
| F10 publication | not started | — |

## Decisions log

| ADR | Title | Status |
|---|---|---|
| ADR-001 | Simulation stack, repository name and package name | accepted |
| ADR-002 | Simulation image size threshold raised to 5 GB | accepted |
| ADR-003 | Simple parallel-jaw gripper instead of Robotiq 2F-85 (D2) | accepted |
| ADR-004 | Torque source = Gazebo effort, η excludes copper losses (D3, D4) | accepted |

## Open decisions (from `docs/plan.es.md` §7)

| # | Decision | Due |
|---|---|---|
| D5 | Single-turn vs retries | F6 |
| D6 | LLM model and provider | F6 |
| D7 | Frozen skill library | F7 |
| D8 | Meaning of "previous Wh" in C | F8 |
| D10 | Whether to build F11 (live demo) | F10 |
