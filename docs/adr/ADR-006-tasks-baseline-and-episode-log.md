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

### The gripper command channel is part of "ready"

The first dev gate lost one episode per container now and then (`pick_place@1` seed 0 in one
run, `stack2@1` seed 0 in the next): the arm stopped on the descent with the finger tips at
the cube's top face, every joint pushing at its effort limit, zero velocity, until the
controller deadline fired — and a retried descent stalled at exactly the same configuration.
Instrumenting a fresh container showed the fingers after the boot `reset()` at
`(0.000, 0.0425)`: one pad fully closed on the centre line, so it landed squarely on the cube.
The cause is a start-up race, not physics: `release()` publishes its effort command right
after `ready()`, which checked the clock, joint states, camera and trajectory server but not
the gripper controller; a `Float64MultiArray` published before the controller's subscription
has matched is simply lost, the fingers stay effort-free, and at the spawn configuration
(arm horizontal, finger axis vertical) gravity drops the lower finger to its closed limit
while the arm moves to `ready_q`. Later episodes always passed because every subsequent
`release()` found the controller listening. `RosBackend.ready()` now also waits for a
subscriber on `/gripper_controller/commands`, and `gripper()` waits (bounded) for one before
publishing, so the first command of a container is never dropped. The baseline stays the plain
F4 sequence: no retry logic hides this class of fault.

### A partially visible cube is a partial detection, not a pose

The extended split failed deterministically on seed 418 (`pick_place@1` 21 mm off,
`stack2@1` 55 mm off) and the F4 gate had already flagged seeds 418 and 489. The green cube
sits at x = −0.33 m, inside the ~30 % of the table the forearm shadows from `ready_pose`
(ADR-005). From there `detect()` saw a 116 px sliver of its top face (a full face is ~680 px
at that depth) and reported its centroid 19 mm from the cube centre — almost half a cube —
so the grasp closed on an edge, the cube turned in the pads and landed off the goal (or jammed
against the table and the descent timed out). `detect_all()` accepted that detection because
the colour was "found" from the first view. `Detection` now carries
`top_face_fraction = area_px / expected_px(depth)` and `complete = fraction >= 0.7`
(`segmentation.min_top_face_fraction`): the top face of a cube is parallel to the image plane
of the top-down camera, so a whole face always covers ~1.0 of the expected area regardless of
where it lies, while an occluded or frame-cut face falls well below (0.17–0.45 in the seed-418
frame). `merge()` lets a complete detection replace a partial one and `detect_all()` keeps
visiting observe poses until every required colour is complete (a colour partial from every
pose keeps its best detection rather than failing). The flag is part of the primitive
contract the LLM agents will see in F6, so they can make the same decision.

### The grasp yaw keeps the open fingers clear of neighbours

A square cube has two equivalent top-down grasps, `yaw` and `yaw -/+ pi/2`. The first full
dev_extended run (seeds 400–449) lost two episodes, both of scene 443: a second cube sat 93 mm
from the target almost exactly along the finger axis of the detected yaw, and the open gripper
(outer finger faces 54.5 mm from the TCP, 25 mm wide; ADR-003) pushed it 45 mm aside on the
way down. `place_obstacle@1` then failed the "no other cube moved" check and `sort3@1` closed on
empty air at the stale detection of that cube. `grasp_yaw()` scores both grasps by the distance
from the nearest other detected cube to the open-finger segment (`FINGER_SWEEP_M`, saturated at
`CLEAR_ENOUGH_M` so far cubes never decide) and switches only when the alternative gains more
than 1 cm; distractors count because `detect()` returns every colour. Re-detecting after each
pick was rejected for baseline A: it adds observation moves (and Wh) to every episode to cover
a case the yaw choice removes at no cost.

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
