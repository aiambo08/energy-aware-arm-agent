# ADR-006: Task contract, scripted baseline A and the episode log (F5)

- Status: accepted
- Date: 2026-10-03
- Phase: F5

## Context

F5 is the cut-off milestone of `docs/plan.es.md`: four versioned tasks with a success checker,
a hand-written baseline (A) that uses only the F4 primitives, and a runner whose JSONL log
records everything the later agents (F6–F8) and the pre-registered evaluation (F9) need. The
F4 gate left two facts that shape this phase: from `ready_pose` the forearm hides part of the
table from the camera (all cubes visible in only 42/100 scenes), and Gazebo ground-truth poses
are available for every spawned model through `/model/<name>/pose`.

## Decisions

### Tasks are pure functions of `(task_id, seed)`

`armbench.tasks.Task.instance(seed, config)` derives everything from the seeded scene of F1
plus a task-specific RNG (`numpy.random.default_rng([seed, hash(task_id)])`): recoloured cubes
so goals are unambiguous by colour, goal points, bins, and for `place_obstacle@1` the wall.
Instances are frozen Pydantic models, so `sha256(instance.json)` in the log identifies the
exact problem an episode solved; a change in geometry or prompt is a new `version`.

| Task | Goal | Success |
|---|---|---|
| `pick_place@1` | one cube (by colour) to a free point | XY error ≤ 2 cm, cube on the table |
| `stack2@1` | top cube on base cube (both by colour) | XY ≤ 2 cm, Z = base + one cube ± 1 cm |
| `sort3@1` | three cubes, one bin per colour | every cube within 2 cm of its bin |
| `place_obstacle@1` | one cube to a point behind a 10 cm wall | as `pick_place@1`, wall moved < 5 mm |

Every checker additionally requires: ≤ 60 s of simulated time, no non-target cube moved
> 2 cm, finger tips never below the table top (`min_tip_z_m` from the clearance monitor).
Checkers are pure (`Task.check(instance, FinalState) -> Verdict`) and run on Gazebo poses in
the container and on the kinematic fake in unit tests, so the same rule judges both.

The obstacle is a grey box (`OBSTACLE_RGBA`) by design: the HSV detector of F3 only knows the
cube palette, so the wall is invisible to `detect()` and the agent has to rely on the prompt
("a wall … do not touch it") and the task instance. Its placement (perpendicular to the
pick→place segment, with clearance from every cube and the goal) is searched over the goal
candidates, the target cube and finally scenes with fewer cubes, so every seed has a valid
instance. The box is 16 × 4 × 10 cm with its *long* side across the segment (the SDF box's x
axis points along `yaw`); with every cube ≥ 12 cm from the wall centre the gripper's vertical
approach to the target stays clear of it. The first Gazebo gate caught the two sizes swapped
(a 4 cm-wide wall lying *along* the segment ended 1.75 cm from the target cube and the open
fingers hit it on descent), which is why the geometry test checks the SDF extents explicitly.

### Baseline A observes from three poses

The scripted agent (`armbench.agents.ScriptedAgent`, id `A`) calls `detect()` from
`ready_pose` and, only if a colour it needs is missing, moves to up to three observation poses
at `x = −0.30 m` (above the robot column, where the forearm occludes nothing) and detects
again. The number of extra observation moves is logged (`AgentTrace.n_observe_moves`) because
they cost energy and time that the LLM agents will also have to pay. For `place_obstacle@1`
the carry height is raised to 0.18 m so the cube clears the 0.10 m wall; everything else is
the F4 `pick_and_place` sequence.

### One episode = one JSONL line, schema version 1

`armbench.runner.EpisodeRecord` (`schema_version: 1`) records task/version, agent, seed,
repeat index, backend, `instance_sha256`, verdict (`ok`, `reason`, checker metrics), a typed
`failure` with its stage (`infra` | `agent` | `robot` | `judge`), simulated and wall time,
the agent trace (attempts, primitives, tokens, latency, skills, prompt/response/program
hashes — zeros or `null` for A), the energy table and the software versions. Logs are
append-only; `armbench report` recomputes every statistic from the lines, nothing is stored
pre-aggregated.

Failure stages matter for the DoD: `infra` (scene spawn/settle, camera bridge timeouts) is
counted against the < 1 % infrastructure threshold; `robot` (a typed primitive error) and
`judge` (the arm finished but the goal is not met) count against the agent.

### Energy comes from the 100 Hz effort tap, with raw samples kept

Inside the container the runner subscribes to `/energy_state_broadcaster/joint_states`
(ADR-004) for the duration of the episode and integrates with `armbench.energy.sensitivity`:
variants A and B × η ∈ {0.70, 0.60, 0.80}, six rows per episode. The raw `{t, q, qd, tau}`
samples are written to `samples/<episode>.jsonl.gz` next to the log so Wh can be recomputed
offline with other parameters (`armbench energy episode`); the recomputation matches the
logged value to the last digit. The kinematic fake has no torques, so `backend: fake` records
carry `energy: null` and the energy thresholds only bind for `backend: sim` runs.

MCAP recording (`--mcap`, `ros2 bag record -s mcap` of clock, joint states, the energy
topic and model poses; no camera images) is optional: it works and is used for debugging and
for the replays of F9, but it is off by default in gates because the bag process competes for
CPU with the simulation.

### Fresh, isolated containers per chunk of seeds

`armbench run --backend sim` starts one `--network none` container per chunk of 25 seeds
(F3: a long-lived world degrades after ~35 spawn/remove cycles; isolation prevents DDS
cross-talk between parallel runs), launches the simulation, runs `episode_runner` inside and
copies `episodes.jsonl`, `samples/` and `mcap/` out. Chunks are appended to one log per run
directory.

### Locked seeds need `--final-eval` and a protocol hash

`armbench run` goes through `check_seeds_allowed` (F0): seeds 100–119 are refused with exit
code 2 unless `--final-eval --protocol-hash <sha>` is given; the hash is recorded in the log
by F9. Development uses `dev` (0–9) and `dev_extended` (400–499).

## Consequences

- Any agent that implements `Agent.solve(robot, instance) -> AgentTrace` plugs into the same
  runner, log and report; F6–F8 add agents, not infrastructure.
- Reports (`armbench report`) carry Wilson 95 % intervals for success, medians/IQM/p95 for
  Wh and durations, Wh CV across repeats of the same seed, infra rate and the per-threshold
  verdict used as the phase gate.
- A task change is a version bump; old logs remain interpretable because the record carries
  the version and the instance hash.
