# Energy information does not make an LLM robot agent cheaper; validated skills make it more reliable

Technical report, pre-registered evaluation `f9-v1` of armbench (UR5e in Gazebo Harmonic).
Short version first; the long version follows. Numbers come from `reports/f9_eval.json`
(report sha256 `87390f0bfe0be03eb58b137831987aebb81190498f8ba26a0d35d16512559028`), produced
by `armbench analyze` under protocol sha256
`845c11742d269d36cfcdde5610be15c6933829079fa7b6f3c10838075cb502b5` ([`protocol.md`](protocol.md)).

## Short version

**Question.** An LLM writes a Python program over typed primitives to solve a tabletop task
with a simulated UR5e. Does (H1) reusing a frozen library of validated skills raise success, and
do (H2, H3) telling it the reference energy of a scripted baseline reduce the watt-hours spent,
without and with skills?

**Setup.** 4 tasks × 20 locked seeds (100–119) × 5 agents = 400 Gazebo episodes. One model
(`Qwen/Qwen3-235B-A22B-Instruct-2507`, Nebius Token Factory, temperature 0, one attempt).
Energy from Gazebo joint effort at 100 Hz, two mechanical-power variants × η ∈ {0.6, 0.7, 0.8}.
Hypotheses, decision rules and exclusions were fixed and hashed before the seeds were unlocked.

**Results.** 400/400 episodes scored, 0 infrastructure failures, 320 LLM calls ≈ 0.16 USD.

| Agent | Success | Wilson 95 % (pooled) |
|---|---:|---|
| A (scripted) | 80/80 | [0.95, 1.00] |
| B (LLM) | 62/80 | [0.67, 0.85] |
| B+S (LLM + skills) | 78/80 | [0.91, 0.99] |
| C (LLM + energy reference) | 76/80 | [0.88, 0.98] |
| C+S (LLM + skills + energy reference) | 80/80 | [0.95, 1.00] |

- **H1 supported.** Skills raise success by +0.200 [+0.100, +0.300] (B+S vs B). They cost
  ~0.8 % more Wh (CI excludes 0 in all six energy rows).
- **H2 not supported.** C vs B: median relative Wh difference +0.1 %, every CI contains 0.
  C *is* more successful (+0.175 [+0.075, +0.275]), mostly because it wrote fewer programs the
  sandbox rejected; that was not a pre-registered hypothesis.
- **H3 not supported.** C+S vs B+S: Wh difference ≈ 0.0 % [−0.2 %, +0.1 %].
- **Scenario: null.** Putting energy information in the prompt did not reduce consumption.

**Takeaway.** In this setting the energy of a pick-and-place episode is dominated by the idle
draw P₀ over the episode duration and by the motions the task requires; an LLM that is told a
budget does not find cheaper motions on its own. Validated skills are the lever that mattered,
and they act on reliability, not energy.

## Long version

### 1. System

- **Simulation**: ROS 2 Jazzy, Gazebo Harmonic, UR5e (`ur_description` 3.5.1), a parallel-jaw
  gripper (ADR-003), a fixed RGB-D camera, cubes spawned per seed. Docker, no GPU.
- **Perception**: HSV + depth `detect()` returning colour, position and yaw in `base_link`
  (F3 gate: recall 100 %, XY p95 2.3 mm over 200 scenes).
- **Primitives**: `observe`, `detect`, `move_to(pose, speed_scale)`, `grasp`, `release`,
  `execute_skill`, with Pydantic contracts and pre-flight checks (IK, singularity, table
  clearance) that raise typed errors before any motion (ADR-005).
- **Sandbox**: AST whitelist, restricted builtins, rlimited child process, typed RPC, network-less
  container (ADR-007, `SECURITY.md`).
- **Skills (B+S, C+S)**: 4 parametrised programs proposed from B's dev-seed programs, accepted
  at 20/20 on seeds 20–39, frozen by hash before evaluation (ADR-008).
- **Energy reference (C, C+S)**: baseline A's median Wh per task on dev seeds, frozen by hash
  (decision D8, ADR-009), shown with the energy model and its levers (speed, path length,
  number of moves).

### 2. Energy model

P_el(t) = P_mech(t)/η + Σᵢ Rᵢ (τᵢ/k_tᵢ)² + P₀, integrated over the episode. P_mech is
Σ max(τ·q̇, 0) (variant A, no regeneration) or Σ|τ·q̇| (variant B). τ is Gazebo's joint effort,
validated against gravity torques (1.6 %) and Pinocchio inverse dynamics (NRMSE 0.06) in F2
(ADR-004). The primary row is variant A, η = 0.7; H2/H3 require a saving in all six rows.

### 3. Protocol

Paired design: every agent sees the same 80 `(task, seed)` instances. Success differences use
paired bootstrap intervals; Wh differences are the median of paired relative differences over
episodes both agents solved, with stratified bootstrap 95 % intervals. A hypothesis is
supported only if the pre-registered threshold holds; the scenario (favourable / mixed / null /
technical failure) is a function of H2 and H3. The runner refuses the locked seeds unless the
protocol hash and the pinned code tree, skill library and energy reference all match.

### 4. Per-task results

| Agent | pick_place | stack2 | sort3 | place_obstacle |
|---|---:|---:|---:|---:|
| A | 20/20 · 0.2090 | 20/20 · 0.2172 | 20/20 · 0.5806 | 20/20 · 0.2242 |
| B | 19/20 · 0.2102 | 15/20 · 0.2179 | 17/20 · 0.5883 | 11/20 · 0.2323 |
| B+S | 20/20 · 0.2136 | 20/20 · 0.2213 | 20/20 · 0.5854 | 18/20 · 0.2278 |
| C | 20/20 · 0.2103 | 20/20 · 0.2154 | 19/20 · 0.5938 | 17/20 · 0.2150 |
| C+S | 20/20 · 0.2134 | 20/20 · 0.2214 | 20/20 · 0.5862 | 20/20 · 0.2271 |

Cells: successes · median Wh of successful episodes (variant A, η 0.7).

Failures of B (18): 9 programs rejected by the sandbox before running, 2 runtime exceptions in
the program, 5 programs that completed but left the scene wrong, 2 typed robot errors. C failed
4 times (2 rejected, 1 wrong final state, 1 robot error), B+S twice (both rejected), C+S never.

### 5. Why no energy effect

- **P₀ dominates.** With P₀ = 100 W, a 7 s episode costs ~0.19 Wh just for idling (≈ 89 % of
  the total for `pick_place` seed 100 of agent A); mechanical power divided by η is ≈ 4 %. Slower moves (`speed_scale < 1`) lower mechanical power
  but lengthen the episode, so they rarely pay off.
- **The task fixes most of the motion.** Approach, grasp and place poses are dictated by the
  goal; the agent can only trim approach heights and the number of observation moves.
- **Skills freeze the motion.** With skills the program is mostly one `execute_skill` call, so
  C+S has nothing left to change (H3 ≈ 0.0 %).

These are explanations consistent with the logs, not tested hypotheses.

### 6. Limitations

- One model, one provider, temperature 0, one attempt: other models may use the information.
- Simulation only; P₀, η and copper resistances are assumptions (`configs/energy.yaml`), not
  measurements of a real UR5e.
- Gazebo is not bit-reproducible: rerunning the same program can change the trajectory, the Wh
  in the third decimal and, rarely, the verdict. Replay guarantees the same prompts and programs,
  not the same physics.
- 20 seeds per task: small effects (< ~1 % Wh) are below the resolution of the design.
- Skills were built from the same model's dev programs; their benefit may be smaller for a model
  that already writes correct programs.

### 7. Reproducing

`docs/benchmark.md` §2 replays any episode and re-runs the analysis from `data/f9-v1.tar.gz`
(report sha256 above). Re-running the Gazebo episodes needs the simulation image and a Nebius
key; the cached responses make the LLM side free.
