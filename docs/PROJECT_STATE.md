# Project state

Single source of truth for where the project is. Updated at the end of every
phase PR.

| Field | Value |
|---|---|
| Current phase | F0 — skeleton, Docker and quality gates |
| Last green commit | (set after the first CI run on `main`) |
| Next phase | F1 — arm simulation (critical gate 1) |
| Blockers | none |

## Phase status

| Phase | Status | Evidence |
|---|---|---|
| F0 skeleton, Docker, quality | PR open, gates green locally | `reports/f0_quality.json` (ruff, format, mypy strict, 13 tests, 1.2 s), `reports/f0_sim.json` (boot 2.5 s, image 4.46 GB) |
| F1 arm simulation | not started | — |
| F2 torque source and energy model | not started | — |
| F3 perception `detect()` | not started | — |
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

## Open decisions (from `docs/plan.es.md` §7)

| # | Decision | Due |
|---|---|---|
| D2 | Arm and gripper (UR5e + Robotiq 2F-85 vs Panda) | F1 |
| D3 | Torque source (Gazebo effort vs inverse dynamics) | F2 |
| D4 | η excludes copper losses | F2 |
| D5 | Single-turn vs retries | F6 |
| D6 | LLM model and provider | F6 |
| D7 | Frozen skill library | F7 |
| D8 | Meaning of "previous Wh" in C | F8 |
| D10 | Whether to build F11 (live demo) | F10 |
