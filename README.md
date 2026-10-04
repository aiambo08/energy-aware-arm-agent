# Energy-Aware Arm Agent (`armbench`)

> Does an LLM agent that writes robot programs over a small set of primitives,
> reuses validated skills, and receives the arm's energy (Wh) as feedback,
> solve manipulation tasks with **fewer watt-hours** than one that does not?

A reproducible benchmark around a simulated **UR5e** (ROS 2 Jazzy + Gazebo
Harmonic, Docker, no GPU required) to answer that question with paired seeds,
confidence intervals and a pre-registered protocol.

**Status: phase F2 (energy model and torque source) merged.** There are no benchmark results yet.
Anything in this repository that looks like a number is either a configuration
value or a measurement stored under `reports/` with the script that produced it.

## What exists today

| Area | State |
|---|---|
| Python package `armbench` (`src/armbench`) | seed-split loader with the final-evaluation lock, UR5e kinematics (FK/Jacobian/IK, numpy only), seeded scene generator → SDF, `armbench` CLI |
| Quality gates | `ruff`, `ruff format`, `mypy --strict`, `pytest` (+ `hypothesis`), `pre-commit` |
| Simulation image | `docker/Dockerfile`: ROS 2 Jazzy + Gazebo Harmonic + UR5e/Robotiq/Pinocchio packages, `ros_ws` built in the image |
| Energy model (F2) | `src/armbench/energy`: P_mech variants A/B, copper losses, P₀, η sensitivity, trapezoidal integration (batch and streaming `EnergyMeter`), JSONL episode logs, `armbench energy` CLI; `configs/energy.yaml` holds the electrical assumptions |
| Perception (F3) | `src/armbench/perception`: pinhole + camera extrinsics (`Camera`), HSV + depth `detect()` → `Detection` (colour, position and yaw in `base_link`); `configs/perception.yaml`; `armbench_bringup` node `scene_capture` (RGB-D + Gazebo ground truth per seed); `scripts/perception_eval.py` runs the F3 gate → `reports/f3_perception.json` |
| Tasks and baseline A (F5) | `src/armbench/tasks`: four versioned tasks (`pick_place@1`, `stack2@1`, `sort3@1`, `place_obstacle@1`) as pure functions of `(task, seed)` with ground-truth success checkers; `src/armbench/agents`: `Agent` protocol, `AgentTrace`, `ScriptedAgent` (baseline A, primitives only, observes from extra poses when the forearm occludes a cube); ADR-006 |
| Episode runner and reports (F5) | `src/armbench/runner`: `EpisodeRecord` JSONL schema v1 (verdict, typed failure stage, sim/wall time, Wh A/B × η, agent trace, instance hash), `run_episode` over a `World` protocol (`FakeWorld` without ROS, Gazebo in `armbench_bringup` `episode_runner`), Docker chunk orchestration (`--network none`, fresh container per 25 seeds), `armbench run` / `armbench report` (Wilson CI, Wh median/IQM/p95, repeat CV, infra rate, thresholds) → `reports/f5_baseline.json` |
| LLM agent B, sandbox, cache/replay (F6) | `src/armbench/llm`: `Provider` protocol with `TemplateProvider` (deterministic, no key), `OpenAICompatProvider` (`urllib`, key from `ARMBENCH_LLM_API_KEY`, unused until a key exists), `CachedProvider`/`ReplayProvider` (content-addressed disk cache), `BudgetedProvider` (token/USD ledger); `src/armbench/agents`: reproducible prompt, `LLMAgent` (B) with tokens/cost/latency/hash trace; `src/armbench/sandbox` + `src/armbench/guest`: AST whitelist, rlimited child interpreter, typed RPC to `robot`, call/sim/wall caps; host prefetch → replay inside the network-less containers; `scripts/f6_gate.py` → `reports/f6_agent_b.json`; ADR-007 |
| Skill library and agent B+S (F7) | `src/armbench/skills`: `Skill` (typed params, closed-vocabulary pre/post-conditions, origin, validation record, order-independent hash), `SkillLibrary` (dedup by signature, lexical retrieval, `library.json` + `FROZEN.json` manifest, read-only once frozen), proposer (B's completed dev programs → parametrised candidates), `Validator` (sandbox + conditions + task checker on seeds 20–39, ≥ 90 %), `SkillRunner` behind `Robot.execute_skill()` (nested sandbox, `SkillPreconditionFailed`/`SkillPostconditionFailed`/`SkillFailed`, inner calls accounted in the trace); agent `B+S` (frozen library required, `# Skills` prompt section, `trace.skills_used`); `armbench skills propose|validate|freeze|show`; frozen library in `skills/`; `scripts/f7_gate.py` → `reports/f7_skills.json`; ADR-008 |
| Energy-aware agents C and C+S (F8) | `src/armbench/energy/reference.py`: `EnergyReference` (baseline A's Wh per task — median/IQM/range, sim time, calls — with provenance and a content hash; D8), `armbench energy-ref build|show`, frozen reference in `energy_ref/`; `# Energy` prompt section (model constants, reference budget, levers) for `C`/`C+S` only; sandbox `CallRecord.speed_scale` and `SkillResult.speed_scales` → `trace.n_moves`/`n_slow_moves`/`speed_scale_min`, `trace.energy_reference_wh`/`_sha256`; report `llm` block with `energy_prompt_rate`, `wh_over_reference`, `slow_move_episode_rate`; bundle `energy_ref/` into the containers; `scripts/f8_gate.py` → `reports/f8_energy_aware.json`; ADR-009 |
| Primitives (F4) | `src/armbench/primitives`: `Robot` with `observe()`, `detect()`, `move_to()`, `grasp()`, `release()`, `reset()`, `execute_skill()` (wired to the F7 library); Pydantic `Pose`/`Observation`/`Detection`/`Result`; typed errors (`OutOfReach`, `Singularity`, `Collision`, `Timeout`, `NoObjectGrasped`, `CameraTimeout`); pre-flight IK/branch/singularity/table-clearance checks (ADR-005); `KinematicBackend` (no ROS) and `RosBackend` (`armbench_bringup`); `configs/primitives.yaml`; `scripts/primitives_eval.py` runs the F4 gate → `reports/f4_primitives.json` |
| Torque source (F2) | `armbench_bringup` nodes `energy_meter` (online Wh on `/armbench/energy`), `torque_probe` and `torque_compare` (Pinocchio inverse dynamics); `scripts/torque_source.py` runs critical gate 2 → `reports/f2_torque_source.json`, decision in ADR-004 |
| Simulation (F1) | `ros_ws/src/armbench_description` (UR5e + parallel gripper xacro, tabletop world with fixed RGB-D camera), `ros_ws/src/armbench_bringup` (`sim.launch.py`, controllers, `sim_check` node) |
| CI | `verify` (lint, types, tests, no ROS) and `sim-image` (build image, F0 boot gate, F1 self-check gate) |
| Docs | `docs/plan.es.md` (full phased plan, Spanish), ADRs in `docs/adr/`, `docs/PROJECT_STATE.md` |

Roadmap (one PR per phase, see `docs/plan.es.md` and `docs/PROJECT_STATE.md`):
~~F1 arm simulation~~ · ~~F2 torque source and energy model~~ · ~~F3 perception~~ ·
~~F4 primitives~~ · ~~F5 tasks, baseline and runner~~ · ~~F6 LLM agent and sandbox~~ ·
~~F7 skill library~~ · ~~F8 energy-aware agent~~ · F9 pre-registered evaluation ·
F10 publication.

## Quick start (host, no ROS)

Requirements: [uv](https://docs.astral.sh/uv/) ≥ 0.8 (it installs Python 3.12 itself).

```bash
git clone https://github.com/aiambo08/energy-aware-arm-agent.git
cd energy-aware-arm-agent
uv sync                 # creates .venv with runtime + dev dependencies
uv run armbench seeds   # show the seed split
uv run pytest -q        # unit tests (no ROS needed)
uv run pre-commit install
```

Expected output of `uv run armbench seeds`:

```
dev                  0-9       Development (prompt and code tuning). Free use.
skill_validation    20-39      Only for accepting or rejecting skills (phase F7).
final_eval         100-119  [LOCKED]  Locked. The runner refuses them unless --final-eval and a registered protocol hash are given.
```

## Simulation image

Requirements: Docker Engine ≥ 24 with BuildKit. No GPU is needed; rendering
uses Mesa software (llvmpipe) when cameras are added in F1.

```bash
docker build -f docker/Dockerfile -t armbench-sim:dev .
docker run --rm armbench-sim:dev gz sim --version
# Boot an empty headless world and measure time to the first /clock message
# (runs on the host: the script drives `docker run`/`docker exec` itself)
uv run python scripts/check_sim_boot.py --image armbench-sim:dev --repeats 3 \
  --out reports/f0_sim.json
```

`check_sim_boot.py` starts `gz sim -s -r empty.sdf` inside the container,
bridges `/clock` with `ros_gz_bridge`, waits for the first message and writes
a JSON report with the boot time, image size and pass/fail against the F0
thresholds (boot ≤ 30 s, image ≤ 5 GB uncompressed; see ADR-002).
Measured on 2026-10-03: boot 2.5 s (3/3), image 4.46 GB.

Both check scripts start their container with `--network none`: ROS 2 (DDS) discovery
otherwise crosses the Docker bridge, so two simulation containers on the same host see
each other's `/controller_manager` and `/clock` and the controller spawners of the second
one fail to configure. Everything the checks need runs inside the container, so no network
is required.

## Arm simulation (F1)

`sim.launch.py` starts Gazebo Harmonic with the tabletop world, spawns the UR5e with the
parallel gripper (ADR-003), loads `joint_state_broadcaster` (position, velocity **and
effort** for the 8 joints), `joint_trajectory_controller` (6 arm joints, position command)
and `gripper_controller` (`effort_controllers/JointGroupEffortController`, N per finger),
and bridges `/clock`, `/camera/image` (640×480 `rgb8`, 15 Hz), `/camera/depth_image`
(`32FC1`), `/camera/camera_info` and the ground-truth pose of every cube
(`/model/cube_<i>/pose`).

```bash
docker build -f docker/Dockerfile -t armbench-sim:dev .
# headless simulation with the F1 check scene (one red cube in front of the robot)
docker run --rm --network none --name armbench-sim armbench-sim:dev \
  ros2 launch armbench_bringup sim.launch.py headless:=true \
  world:=/work/ros_ws/install/armbench_description/share/armbench_description/worlds/check_scene.sdf
# in another terminal: list topics / controllers
docker exec armbench-sim /usr/local/bin/entrypoint.sh ros2 topic list
docker exec armbench-sim /usr/local/bin/entrypoint.sh ros2 control list_controllers
# GUI (needs an X server on the host): headless:=false
```

The F1 gate runs from the host and drives Docker itself:

```bash
uv run python scripts/check_sim.py --image armbench-sim:dev --repeats 3 --out reports/f1_sim.json
```

For every repetition it boots a fresh container, waits for the three controllers to be
active, then runs `ros2 run armbench_bringup sim_check` inside it, which (1) verifies
`/clock` and `/joint_states` with effort for all joints, (2) measures RGB/depth FPS in
simulated time and the real-time factor idle and while moving, (3) moves each of the six
joints ±0.4 rad and back, (4) opens/closes the gripper, and (5) performs a fixed-waypoint
pick-and-place of `cube_0` (waypoints in `armbench_bringup/config/sim_check.yaml`), reading
the cube's ground-truth pose from Gazebo to compute lift height and slip relative to the
TCP during transport. Thresholds: ready ≤ 60 s, RTF ≥ 0.5, camera ≥ 10 FPS, grasp success
≥ 96 % with slip ≤ 5 mm.
`--camera-fps-min` / `--rtf-min` / `--ready-timeout-s` override them and the effective values are
recorded in the report; CI passes `--camera-fps-min 8` because GitHub's 4-vCPU runners render the
camera at ~9.6 FPS with software ogre2.

Measured on 2026-10-03 (8 vCPU, no GPU, `reports/f1_sim.json`, 3 runs): boots 3/3,
controllers active in ≤ 18.2 s, RTF while moving ≥ 0.96, camera ≥ 10.9 FPS, grasp 3/3
with slip ≤ 0.05 mm, lift 14.5 cm.
50-run tirada on the same host (`reports/f1_sim_50runs.json`, containers isolated with
`--network none`): boots 50/50, controllers active in ≤ 18.7 s (mean 8.9 s), RTF while moving
≥ 0.77, camera ≥ 10.6 FPS, grasp 50/50 with slip ≤ 0.25 mm, lift 14.5 cm in every run.

## Energy model and torque source (F2)

The electrical power of the arm is modelled from joint torque τᵢ and velocity ωᵢ
(`docs/adr/ADR-004-torque-source.md`, `configs/energy.yaml`):

```text
P_mech,A = Σ max(τᵢ ωᵢ, 0)        # no regeneration (default)
P_mech,B = Σ |τᵢ ωᵢ|              # upper bound
P_cu     = Σ Rᵢ (τᵢ / k_t,i)²     # copper losses
P_el     = P_mech / η + P_cu + P₀  # η = 0.70 (sensitivity 0.60 / 0.80), P₀ = 100 W
E        = ∫ P_el dt  (trapezoidal), reported in Wh
```

η excludes the copper losses (otherwise they would be counted twice). k_t (10 N·m/A base
joints, 4 N·m/A wrists), R (0.5 Ω / 1.0 Ω) and P₀ are **assumptions**, not UR5e data:
absolute Wh depend on them, comparisons between methods on the same episodes do not.

Recompute the energy of any episode log (JSONL rows `{"t", "q", "qd", "tau"}`) on the host:

```bash
# e.g. a segment left by the gate run below (reports/_f2_torque_work/ is not committed)
uv run armbench energy reports/_f2_torque_work/traj_00.jsonl                 # table, variant A
uv run armbench energy reports/_f2_torque_work/traj_00.jsonl --full --json   # A/B × η sensitivity
```

Inside the simulation `ros2 run armbench_bringup energy_meter --log episode.jsonl` integrates
the 100 Hz `energy_state_broadcaster` stream online and publishes `/armbench/energy`
(`[t_sim, Wh_A, Wh_B, Wh_mech_A, Wh_cu, Wh_p0]`) and a human-readable `/armbench/energy_table`;
`/armbench/energy/reset` (std_srvs/Trigger) starts a new episode.

Critical gate 2 — is Gazebo's `effort` a usable torque? — runs from the host:

```bash
uv run python scripts/torque_source.py --image armbench-sim:dev --out reports/f2_torque_source.json
```

It boots one isolated container (`--network none`), records 5 static holds, 20 scripted
trajectories and 10 repeats of the first one (`torque_probe`, windows delimited in simulation
time), computes Pinocchio inverse dynamics for every segment (`torque_compare`) and compares:
static effort vs gravity torque (rel. diff < 10 %), effort vs inverse dynamics during motion
(NRMSE median < 0.15), Wh repeatability over the 10 repeats (CV < 2 %) and the CPU of the
`energy_meter` process (< 5 % of the container's CPU time). Thresholds live in
`THRESHOLDS` in the script and are copied into the report.

Measured on 2026-10-03 (8 vCPU, `reports/f2_torque_source.json`): static rel. diff ≤ 1.6 %,
motion NRMSE median 0.06, Wh CV 1.1 % (Gazebo) / 0.07 % (inverse dynamics), meter 5.2 % of
one core = 2.3 % of the container's CPU time, 100 Hz decimation changes Wh by ≤ 2.6 %. Decision: **torque source = Gazebo effort**, inverse dynamics kept as
cross-check (ADR-004).

## Perception `detect()` (F3)

`armbench.perception` turns one RGB-D frame of the simulated camera into cube poses in
`base_link` (`configs/perception.yaml`): HSV thresholds per colour → depth-valid mask →
morphological opening → connected components → **top face only** (pixels within
`depth.top_face_band_m` of the nearest depth of the blob, so visible side faces do not bias
the centre) → centroid + median depth → pinhole back-projection (`CameraInfo.K`, or
`fx = fy = (W/2)/tan(hfov/2)` from the SDF) → camera extrinsics from the world file
(`xyz`, `rpy`, Gazebo sensor frame x-forward → optical frame z-forward) → cube centre
(`top − size/2` in z). Yaw comes from the minimum-area rectangle of the top face, folded to
`[−π/4, π/4)` (a cube is symmetric every 90°).

```python
import numpy as np
from armbench.perception import Camera, detect, load_perception_params

params = load_perception_params()  # configs/perception.yaml
camera = Camera.from_spec(params.camera)  # or Camera.from_spec(params.camera, k=CameraInfo.k)
rgb: np.ndarray = ...  # (480, 640, 3) uint8, rgb8 from /camera/image
depth: np.ndarray = ...  # (480, 640) float32 metres, 32FC1 from /camera/depth_image
result = detect(rgb, depth, "red", params=params, camera=camera)  # or target="any"
for d in result.detections:  # sorted by x, then y (deterministic)
    print(d.color, d.position, d.yaw_rad, d.pixel, d.area_px)
print(result.latency_ms)
```

Contract: an empty scene returns an empty tuple (no exception); an unknown colour raises
`UnknownTargetError`; a frame with the wrong shape or dtype raises `ValueError`; depth
pixels that are non-finite or outside `[depth.min_m, depth.max_m]` are ignored.

F3 gate — recall, false positives, XY/Z error and latency on 200 seeded scenes with Gazebo
ground truth — runs from the host:

```bash
uv run python scripts/perception_eval.py --seeds 200-399 --out reports/f3_perception.json
# re-evaluate an already captured dataset (reports/_f3_dataset/ is not committed)
uv run python scripts/perception_eval.py --skip-capture
```

It boots one isolated container, runs `ros2 run armbench_bringup scene_capture` (per seed:
spawn the cubes of `armbench.scene.generate_scene`, wait until every `/model/cube_i/pose`
is at rest at the requested pose + 1 s of simulation time, save the first RGB + depth pair
rendered after that instant with `CameraInfo.K` and the PosePublisher poses as ground truth,
remove the cubes), then runs `detect()` on the host and matches detections to cubes of the same
colour within 3 cm. Thresholds (`THRESHOLDS` in the script, copied into the report): ≥ 200
scenes, recall ≥ 99 %, false positives ≤ 1 % of cubes, XY median < 5 mm and p95 < 10 mm,
|Z| p95 < 10 mm, latency p95 < 50 ms on the host CPU.

**Measured (seeds 200–399, 200 scenes, 891 cubes, `reports/f3_perception.json`):** recall 100 % (891/891 visible cubes matched), 0 false positives, XY error median 1.39 mm / p95 2.34 mm / max 2.83 mm, Z error p95 < 0.001 mm (Gazebo depth is noise-free), yaw error (folded to the square's 90° symmetry) median 0.15° / p95 1.0°, `detect()` latency median 9.7 ms / p95 14.4 ms on CPU. Ground truth comes from Gazebo's `PosePublisher` (max 0.7 µm from the requested poses). The capture runs 25 seeds per fresh simulation container: a long-lived world degrades after a few hundred spawn/remove cycles (gz-transport service calls start timing out), which chunking avoids; one scene needed the built-in single retry.

## Robot primitives (F4)

`armbench.primitives.Robot` is the closed instruction set that agent programs, skills and the
scripted baseline use. Every call returns a Pydantic model or raises a typed
`PrimitiveError` (`code` + `details`, see `to_dict()`); the arm never receives a goal that
the pre-flight checks reject (ADR-005). `configs/primitives.yaml` holds the workspace box,
the IK branch, the singularity threshold, the table clearances and the gripper efforts.

```python
from armbench.perception import Camera, load_perception_params
from armbench.primitives import KinematicBackend, Pose, Robot, load_primitive_params
from armbench.primitives.scripted import grasp_pose
from armbench.scene import generate_scene, load_scene_config

params = load_primitive_params()  # configs/primitives.yaml
camera = Camera.from_spec(load_perception_params().camera)
scene = generate_scene(0, load_scene_config())
backend = KinematicBackend(params, camera, cubes=list(scene.cubes))  # no ROS
robot = Robot(backend, params=params)
obs = robot.observe()  # t_sim, q, qd, tcp, gripper opening, holding, detections
cube = robot.detect("red")[0]  # armbench.perception.Detection, sorted by x then y
target = grasp_pose(cube)  # pads centred on the cube, yaw aligned
robot.move_to(target.above(0.10))  # may raise OutOfReach / Singularity / Collision / Timeout
robot.move_to(target, speed_scale=0.5)
robot.grasp()  # NoObjectGrasped if the fingers close on nothing
robot.move_to(Pose(x=-0.5, y=0.0, z=0.25))
robot.release()
robot.reset()  # home, open gripper; the recovery entry point after any error
```

- `move_to(pose, speed_scale)`: top-down grasp orientation (`yaw` about z); `speed_scale` ∈
  [0.1, 1] scales the joint speed so duration is strictly monotone in it — it is the energy
  lever of F8. Checks, in order: workspace box → table → IK (`UR5eModel.ik_top_down`, joints
  unwrapped towards the current configuration) → IK branch → joint limits → Jacobian
  singular value → sampled joint-space path clearance (links and finger tips above the table).
  After the controller succeeds the robot waits until the joints are at rest before
  measuring the final pose.
- `grasp()` / `release()`: effort command to the parallel gripper, settle time, opening check.
- `observe()` / `detect(target)`: one fresh RGB-D pair (stamped after the call) → F3
  `detect()`; `CameraTimeout` if none arrives in `camera_timeout_s`.
- `reset()`: open the gripper and return to `ready_pose`; also the first thing the gate does.
- `KinematicBackend` executes motions exactly, renders the cubes with
  `armbench.perception.synthetic` and moves a held cube with the gripper, so the whole stack
  (perception → planning → scripted pick-and-place) runs in the unit tests in seconds;
  `RosBackend` (`ros_ws/src/armbench_bringup/armbench_bringup/ros_backend.py`) drives the
  Gazebo simulation through `/joint_trajectory_controller/follow_joint_trajectory`,
  `/gripper_controller/commands`, `/joint_states` and the camera bridges, and monitors the
  minimum link and finger-tip height during every motion.

F4 gate (`primitives_check` node in the container, orchestrated from the host; one fresh
simulation per section and per 25 pick-and-place seeds):

```bash
uv run python scripts/primitives_eval.py --seeds 400-499 --out reports/f4_primitives.json
# re-judge without re-running the simulation
uv run python scripts/primitives_eval.py --skip-contracts --skip-pick
```

Thresholds (`THRESHOLDS` in the script, copied into the report): 200 random reachable poses
with position error p95 < 5 mm and yaw p95 < 2° and 0 table contacts (finger tips never
below the table top, from `/joint_states` at 100 Hz); 100 unreachable poses, 100 % raised
as `OutOfReach` with no goal sent and max |Δq| < 1 mrad; duration strictly decreasing over
`speed_scale` ∈ {0.1, 0.2, 0.35, 0.5, 0.75, 1}; 100 `reset()` from random poses in < 5 s of
simulation; scripted pick-and-place (`armbench.primitives.scripted`: detect → approach →
descend → grasp → lift → move → descend → release → retreat, Gazebo poses as ground truth,
placement error < 15 mm, other cubes undisturbed) ≥ 95 % on the 100 `dev_extended` seeds.

**Measured (`reports/f4_primitives.json`, seeds 400–499, isolated containers):** 200/200
reachable poses reached with position error median 0.13 mm / p95 0.23 mm / max 0.30 mm and
yaw p95 0.03°, 0 table contacts (lowest finger tip 7.7 mm above the table); 100/100
unreachable poses raised `OutOfReach` with 0 goals sent and max |Δq| 0.3 mrad; `speed_scale`
0.1 → 1.0 gives 8.01 → 0.88 s of simulation, strictly decreasing; 100/100 resets in
≤ 1.96 s of simulation with joint error ≤ 0.3 mrad and the gripper open; pick-and-place
98/100 (placement error median 2.2 mm, p95 7.4 mm; the two failures, seeds 418 and 489,
placed the cube 22.6 and 15.5 mm off, reproducibly across three runs; 0 scene/infrastructure
errors). `detect()` from the ready pose saw every cube in only 42/100 episodes because the
forearm and wrist sit under the camera (ADR-005); the scripted pick is unaffected, but F5
tasks that need every cube must look from elsewhere.

## Tasks, baseline A and the episode runner (F5)

Four tasks, each a pure function of `(task_id, seed)` on top of the seeded scenes, with a
success checker that judges Gazebo ground truth (ADR-006). The same checker runs on the
kinematic fake in the unit tests.

| Task | Goal | Success (plus: ≤ 60 s sim, no other cube moved > 2 cm, tips above the table) |
|---|---|---|
| `pick_place@1` | cube of a colour → free point | XY ≤ 2 cm |
| `stack2@1` | top cube on base cube | XY ≤ 2 cm, Z = base + cube ± 1 cm |
| `sort3@1` | three cubes → bin per colour | every cube within 2 cm of its bin |
| `place_obstacle@1` | cube → point behind a 10 cm grey wall | XY ≤ 2 cm, wall moved < 5 mm |

```bash
# task instance for a seed (goals, colours, obstacle) and the prompt the LLM agents will see
uv run python -c "from armbench.tasks import get_task; t = get_task('place_obstacle@1'); i = t.instance(7); print(i.model_dump_json(indent=1)); print(t.prompt(i))"

# baseline A on the kinematic backend, no ROS: 4 tasks × dev seeds, JSONL log + report
uv run armbench run --task all --agent A --backend fake --seeds dev --repeat 2 --out runs/fake
uv run armbench report runs/fake

# the same in Gazebo (fresh --network none container per chunk of 25 seeds, energy from the
# 100 Hz effort tap; add --mcap to also record a bag per episode)
uv run armbench run --task all --agent A --backend sim --seeds dev --out runs/f5_dev
uv run armbench run --task pick_place@1 --seeds 0 --repeat 10 --out runs/f5_repeat   # Wh CV
uv run armbench report runs/f5_dev runs/f5_repeat --out reports/f5_baseline.json --strict

# locked seeds are refused (exit 2) without --final-eval --protocol-hash <sha>
uv run armbench run --task pick_place@1 --backend fake --seeds final_eval
```

Each episode is one line of `episodes.jsonl` (`schema_version: 1`): task and version, agent,
seed, repeat, backend, `instance_sha256`, `ok`/`reason`/checker metrics, a typed `failure`
with its stage (`infra` | `agent` | `robot` | `judge`), `sim_s`, `wall_s`, the energy table
(variants A/B × η 0.70/0.60/0.80, J and Wh, sample count and rate) with the raw
`{t, q, qd, tau}` samples in `samples/<episode>.jsonl.gz` for offline recomputation, the
agent trace (attempts, primitives, observation moves, tokens, latency, skills,
prompt/response/program hashes — empty for A) and software versions. `armbench report`
recomputes everything from the lines: success with Wilson 95 % CI, Wh median/IQM/p05/p95 per
variant and η, Wh CV across repeats of the same seed, infra rate, sim/wall distributions
(one row per task × agent × backend × seed split, so dev and extended seeds never mix), and
the F5 thresholds (success ≥ 95 %, infra < 1 %, wall < 120 s, energy table complete, repeat
CV < 3 %).

Measured (`reports/f5_baseline.json`, 250 Gazebo episodes, image `armbench-sim:dev`, 8 vCPU host,
one fresh container per 25 episodes, `--network none`):

| task / split | n | success (Wilson 95 %) | Wh A η=0.70 median | Wh B η=0.70 | Wh A η=0.60 / 0.80 | sim s median | wall s p95 |
|---|---:|---|---:|---:|---|---:|---:|
| `pick_place@1` dev (+10 repeats of seed 0) | 20 | 100 % [83.9, 100] | 0.214 | 0.228 | 0.216 / 0.213 | 6.76 | 13.4 |
| `pick_place@1` dev_extended 400–449 | 50 | 100 % [92.9, 100] | 0.210 | 0.225 | 0.212 / 0.209 | 6.59 | 15.2 |
| `stack2@1` dev | 10 | 100 % [72.2, 100] | 0.226 | 0.241 | 0.228 / 0.224 | 7.18 | 14.3 |
| `stack2@1` dev_extended | 50 | 100 % [92.9, 100] | 0.216 | 0.234 | 0.218 / 0.214 | 6.73 | 16.0 |
| `sort3@1` dev | 10 | 100 % [72.2, 100] | 0.584 | 0.625 | 0.590 / 0.580 | 18.3 | 24.3 |
| `sort3@1` dev_extended | 50 | 100 % [92.9, 100] | 0.581 | 0.624 | 0.588 / 0.577 | 18.2 | 26.6 |
| `place_obstacle@1` dev | 10 | 100 % [72.2, 100] | 0.228 | 0.247 | 0.231 / 0.226 | 7.17 | 12.9 |
| `place_obstacle@1` dev_extended | 50 | 100 % [92.9, 100] | 0.224 | 0.241 | 0.226 / 0.222 | 7.02 | 15.3 |

Energy table complete in 250/250 episodes, 0 infra failures, 0 duplicates, Wh CV across the 10
repeats of `pick_place@1` seed 0: 0.36 % (A) — the simulation is effectively deterministic once
the gripper controller is up (ADR-006). Baseline A needed one extra observation move in 102/250
episodes (the forearm shadows ~30 % of the table from the ready pose). The ranking of the
variants is the same everywhere: B > A by 6–9 % and η=0.60 / 0.80 shift Wh by about ±1 % only,
because the base power `P0·T` dominates these short episodes. An earlier run of the same 250
episodes lost `sort3@1` and `place_obstacle@1` on seed 443, where the open gripper swept a
neighbouring cube on the way down; baseline A now picks the equivalent grasp yaw that keeps the
fingers clear (ADR-006) and that run is not the one reported here.

## Agent B: LLM-written programs, sandbox, cache and replay (F6)

Configuration B is the first LLM agent of the protocol: per episode it receives the task prompt
plus a fixed system prompt (the primitive API, typed errors, workspace limits, the energy
levers and one worked example — `armbench.agents.prompt`) and must answer with a Python
program that drives `robot` through the F4 primitives. The program runs in a sandbox and the
F5 checker judges the result exactly as for baseline A (ADR-007).

```bash
# agent B on the kinematic backend with the bundled deterministic provider (no key, no network)
uv run armbench run --task all --agent B --backend fake --seeds dev --out runs/b_fake
uv run armbench report runs/b_fake            # adds an `llm` block per task: tokens, cost, latency, outcomes

# the same answers again, from the cache only (exit 1 on a cache miss)
uv run armbench run --task all --agent B --backend fake --seeds dev --out runs/b_replay --provider replay

# Gazebo: programs are fetched on the host, the bundle runs/b_sim/llm/ is copied into each
# --network none container and replayed there; replay the whole run later without any provider
rm -rf cache/llm   # the gate's live run must really call the provider (cost and latency are measured)
uv run armbench run --task all --agent B --backend sim --seeds dev --out runs/b_sim
uv run armbench run --task all --agent B --backend sim --seeds dev --out runs/b_sim_replay --provider replay
uv run python scripts/f6_gate.py --live runs/b_sim --replay runs/b_sim_replay   # → reports/f6_agent_b.json
# replay reproduces the LLM side exactly (response, program, primitive calls); the success verdict
# is Gazebo's and can differ between two physics runs of the same program (see the report)

# try a program against the sandbox by hand
uv run python -c "
from armbench.perception import Camera, load_perception_params
from armbench.primitives import KinematicBackend, Robot, load_primitive_params
from armbench.sandbox import SandboxLimits, run_program
from armbench.scene import load_scene_config
from armbench.tasks import get_task
p = load_primitive_params(); inst = get_task('pick_place@1').instance(0, load_scene_config())
robot = Robot(KinematicBackend(p, Camera.from_spec(load_perception_params().camera), cubes=list(inst.scene.cubes)), params=p); robot.reset()
run = run_program('import os\n', robot, SandboxLimits(cpu_s=2, memory_mb=256, wall_s=20, sim_s=30, max_calls=12, stdout_kb=1))
print(run.outcome, run.describe())"
```

**Providers** (`configs/llm.yaml`, `armbench.llm`): `template` (deterministic programs derived
from the task text — the only one used so far), `openai_compat` (any `/v1/chat/completions`,
key in `ARMBENCH_LLM_API_KEY`, `openai.base_url`/`openai.model`), `replay` (cache only).
Every provider is wrapped in a disk cache keyed by the SHA-256 of the canonical request
(model, messages, temperature, max_tokens, seed) and in a budget guard (`max_usd_per_run`,
`max_tokens_per_episode`, prices per million tokens) that writes `<out>/llm/ledger.json`.
Switching to a real model is configuration only; no code path changes, and CI never makes a
live call.

**Sandbox** (`armbench.sandbox` on the host side, `armbench.guest` inside the child): AST
whitelist (no imports, classes, `try`, `with`, decorators, dunder/underscore names, `eval`/
`exec`/`open`/`getattr`/`type`…; 20 k chars, 5 k nodes), a `python -I` child with
`RLIMIT_AS/CPU/FSIZE/NPROC/CORE`, empty env, read-only cwd and `unshare -rn` where the kernel
allows it, parent-side caps (`max_calls`, wall, simulated seconds, stdout) and a typed
JSON-lines RPC so the program only ever talks to a `RobotProxy`. Outcomes are data
(`completed | rejected | primitive_error | exception | call_limit | sim_limit | timeout |
killed | protocol_error`) and land in the episode trace with the prompt/response/program
hashes, tokens, latency, cost and the isolation layers that were active. Inside the Docker
image `unshare` is refused, so there the layers are `--network none` + rlimits + AST + RPC:
a process-level sandbox, not a VM boundary (ADR-007). `tests/unit/test_sandbox.py` holds the
attack battery (imports, files, sockets, introspection, frame walks, CPU/memory bombs,
call floods, malformed code, random mutations).

> `TemplateProvider` exercises the pipeline, not a model: its success rate and Wh say nothing
> about what an LLM would achieve. The numbers below are an infrastructure gate.
## Skill library and agent B+S (F7)

A *skill* is a validated program of agent B whose goal constants became typed parameters,
with executable pre/post-conditions, its origin (run, task, seeds, program hash) and its
validation record (`armbench.skills`, ADR-008). B+S is agent B plus a **frozen** library: the
prompt gains a `# Skills` section with the top-3 skills retrieved by the task wording, and the
program may call `robot.execute_skill("name", param=value, ...)`, which runs the skill body in a
nested sandbox, checks the contract and raises `SkillPreconditionFailed` /
`SkillPostconditionFailed` / `SkillFailed` (or the primitive error that stopped it).

```bash
# 1. candidates from agent B's completed dev programs (one per task, goal constants parametrised)
uv run armbench skills propose --run runs/f6_live --out skills/candidates.json

# 2. validate on the skill_validation seeds 20-39: sandbox + pre/post-conditions + task checker,
#    accepted at >= 90 % (18/20); writes skills/library.json (unfrozen) and a per-seed report
uv run armbench skills validate --candidates skills/candidates.json --out skills \
    --report reports/f7_skill_validation.json

# 3. freeze (D7): skills/FROZEN.json with the library hash; nothing can be added afterwards
uv run armbench skills freeze --dir skills
uv run armbench skills show --dir skills           # contracts, origin, validation, hash (--json)

# 4. B vs B+S pilot — kinematic backend, then Gazebo (the bundle runs/<out>/llm/skills/ carries the
#    frozen library into the --network none containers; an unfrozen or edited library is refused)
rm -rf cache/llm      # live pilot: the prompts must really reach the provider
uv run armbench run --task all --agent B   --backend fake --seeds dev --out runs/f7_fake_B
uv run armbench run --task all --agent B+S --backend fake --seeds dev --out runs/f7_fake_BS
uv run armbench run --task all --agent B+S --backend sim  --seeds dev --out runs/f7_live_BS
uv run armbench run --task all --agent B   --backend sim  --seeds dev --out runs/f7_live_B
uv run armbench report runs/f7_live_BS     # the llm block adds skills used per task and primitives/episode
uv run python scripts/f7_gate.py --b runs/f7_live_B --bs runs/f7_live_BS \
    --fake-b runs/f7_fake_B --fake-bs runs/f7_fake_BS      # → reports/f7_skills.json
```

**Library** (`skills/`): `library.json` (skills sorted by name) + `FROZEN.json` (SHA-256 of the
library, per-skill hashes, timestamp). Deduplication is by signature (`name(param: kind, ...)`);
a name reused with another signature is an error. `SkillLibrary.load(..., require_frozen=True)`
recomputes the hash and refuses an unfrozen or edited library; `armbench run --agent B+S`,
`skills validate` and `skills freeze` refuse to modify a frozen one — a new library is a new
directory with a new hash, recorded in every `run.json` (`skills.sha256`).

**Conditions** are a closed vocabulary (`not_holding`, `holding`, `tcp_above`, `cube_near`),
not Python. Robot-state conditions are checked by `execute_skill` on the live robot; world
conditions (`cube_near`) are checked by the validator and the task checker and reported as
*deferred* inside an episode (the agent's robot has no ground truth). Fault-injection tests in
`tests/unit/test_skills.py` show a body that places 5 cm off, or ends holding the cube, is
rejected on every validation seed.

**Accounting**: one `execute_skill` is one call for the agent and several primitives for the
arm; the sandbox run keeps the inner calls, so `trace.n_primitives` counts what the arm did and
`trace.skills_used` the skills that completed. The report's `llm` block adds `skills_used` and
`skill_episode_rate`. Retrieval is deterministic word overlap (no embeddings); `TemplateProvider`
answers a B+S prompt with one skill call when an offered skill matches the task kind.

## Energy-aware agents C and C+S (F8)

C is agent B with an `# Energy` section prepended to the task message (the system prompt is
identical for every configuration): the energy model as scored (`P_el = P_mech / eta + copper
+ P0`, configured eta and P0), the **reference budget** — baseline A's Wh for the *same task*
(median, IQM, range over the dev seeds, simulated time and primitive calls) — and the two
levers the API exposes (fewer/shorter moves and lower lifts first, `speed_scale` second).
C+S is B+S plus the same block. Decision D8 (ADR-009): the budget is baseline A, read from a
frozen artefact, never the agent's own previous attempt or episode.

```bash
# 1. the reference (once): per-task Wh of a finished baseline-A Gazebo run, variant A, nominal eta
uv run armbench energy-ref build --run runs/f5_dev --out energy_ref     # refuses to overwrite
uv run armbench energy-ref show                                         # provenance, hash, budgets (--json)

# 2. C and C+S on the kinematic backend (no API key: provider template, a fixed recipe, not a model)
uv run armbench run --task all --agent C   --backend fake --seeds dev --out runs/f8_fake_C
uv run armbench run --task all --agent C+S --backend fake --seeds dev --out runs/f8_fake_CS

# 3. Gazebo pilot B vs C and B+S vs C+S (the bundle runs/<out>/llm/energy_ref/ ships the reference
#    into the --network none containers; a task missing from the reference is refused, exit 2)
rm -rf cache/llm
for a in B C B+S C+S; do
  uv run armbench run --task all --agent $a --backend sim --seeds dev --out runs/f8_live_${a//+/}
done
uv run armbench report runs/f8_live_C      # llm block: energy block rate, reference Wh, Wh/reference,
                                           # moves/episode, episodes with speed_scale < 1
uv run python scripts/f8_gate.py --b runs/f8_live_B --c runs/f8_live_C \
    --bs runs/f8_live_BS --cs runs/f8_live_CS          # → reports/f8_energy_aware.json
```

**Reference** (`energy_ref/reference.json`, `armbench.energy.EnergyReference`): one entry per
task (`wh_median`, `wh_iqm`, `wh_min`, `wh_max`, `n`, `seeds`, `sim_s_median`,
`n_primitives_median`), the variant and eta quoted, the source run (id, agent A, backend sim,
seeds, split, git sha) and a timestamp. The SHA-256 covers everything but the timestamp, so a
rebuild from the same run is the same hash; `run.json` of every C/C+S run records it
(`energy_ref.sha256`, `reference_wh` per task) and so does every trace
(`trace.energy_reference_sha256`, `trace.energy_reference_wh`). A reference from another agent
or from the fake backend is refused.

**Lever accounting**: the sandbox records the `speed_scale` of every `move_to` (and
`SkillResult.speed_scales` those inside a skill body), so `trace.n_moves`, `trace.n_slow_moves`
(`speed_scale < 1`) and `trace.speed_scale_min` are exact for all four LLM agents; the report's
`llm` block adds `energy_prompt_rate`, `energy_reference_wh`, `wh_over_reference`,
`moves_per_episode`, `slow_move_episode_rate` and `speed_scale_min`. The pilot comparison
C vs B / C+S vs B+S is reported without an improvement threshold: that is hypothesis H2, to be
evaluated in F9 with a real provider.

**Pilot (`reports/f8_energy_aware.json`)**: 4 tasks × dev 0–9 in Gazebo, cache emptied first —
B, C, B+S and C+S each 40/40, 0 infra failures, 0 cost aborts; the `# Energy` block appears in
40/40 C and C+S prompts and in no B/B+S prompt; the reference hash is in every C/C+S run and
trace and the gate rebuilds the reference from `runs/f5_dev` bit for bit. Wh A medians: C/B
0.95–0.98 (the template's energy recipe lowers the approach height, raises the slow segment to
0.7 and drops one slow move), C+S/B+S 0.99–1.00 (with a skill the program is one `execute_skill`
call, so the frozen body — `SLOW = 0.5` inside — is executed unchanged). Same move counts per
pair; ~350 more tokens per episode for the block. These numbers validate the plumbing and the
accounting with a fixed recipe, not energy awareness in a language model.

## Seeded scenes and UR5e kinematics (host, no ROS)

```bash
# deterministic tabletop scene (3-6 coloured cubes) for a seed: as JSON, or as a full Gazebo world
uv run armbench scene --seed 0 --json
uv run armbench scene --seed 0 \
  --template ros_ws/src/armbench_description/worlds/tabletop.sdf.in --out /tmp/scene0.sdf
docker run --rm --network none --name armbench-sim \
  -v /tmp/scene0.sdf:/scene/scene0.sdf:ro armbench-sim:dev \
  ros2 launch armbench_bringup sim.launch.py headless:=true world:=/scene/scene0.sdf
# forward kinematics of tool0 (base_link frame) for a joint vector, radians
uv run armbench fk 0 -1.57 0 -1.57 0 0
```

`armbench.scene` (`configs/scene.yaml`) draws the number of cubes, colours, positions and
yaws from `numpy.random.default_rng(seed)` only, rejects placements closer than 9 cm or
outside the workspace in front of the robot, and renders each cube as an SDF `<model>`
(`cube_0`…`cube_{n-1}`) with inertia, friction/contact parameters and a per-model
`PosePublisher`, inserted at the `<!-- ARMBENCH_SCENE -->` marker of the world template.
The JSON form carries a hash of the config so a scene can be re-created exactly later.

`armbench.kinematics` (`configs/ur5e_kinematics.yaml`, values from `ur_description` 3.5.1)
models the UR5e chain exactly as the xacro does (`base_link → … → tool0`), with forward
kinematics, the geometric Jacobian and a damped-least-squares IK (`ik`, `ik_top_down` with a
TCP offset) that respects joint limits and never raises on unreachable targets. Tests check
FK against reference poses, the Jacobian against finite differences and IK round trips with
Hypothesis.

Verified on 2026-10-03 with the seed-0 world (6 cubes): headless boot with the three
controllers active, every `/model/cube_<i>/pose` equal to the generated pose to 1e-6 m
after settling, camera at 12.7 FPS.

## Seed policy

| Split | Seeds | Rule |
|---|---|---|
| `dev` | 0–9 | free |
| `skill_validation` | 20–39 | only to accept/reject skills |
| `dev_extended` | 400–499 | free; F4 primitives gate, F5 baseline check (400–449) |
| `final_eval` | 100–119 | **locked**: `check_seeds_allowed` raises unless `final_eval=True` and a protocol hash are given |

The split is data (`configs/seeds.yaml`), validated for disjointness, and
tested in `tests/unit/test_seeds.py`. Changing it after pre-registration
requires an ADR.

## Repository layout

```
docker/            Dockerfile, entrypoint
ros_ws/src/        ROS 2 packages: armbench_description (xacro, worlds), armbench_bringup (launch, controllers, sim_check)
src/armbench/      pure-Python package (energy, perception, primitives, tasks, agents, llm, sandbox, skills, runner), testable without ROS
configs/           seeds.yaml, ur5e_kinematics.yaml, scene.yaml, energy.yaml, perception.yaml, primitives.yaml
scripts/           check scripts that write reports/*.json
skills/            the frozen skill library (library.json + FROZEN.json, D7) used by agent B+S
tests/unit/        tests without ROS; tests marked `sim` need the image
reports/           committed JSON metrics per phase (the DoD evidence)
docs/              plan, ADRs, PROJECT_STATE.md
```

## Contributing and licence

See `CONTRIBUTING.md`. Code is MIT (`LICENSE`). Robot meshes and third-party
ROS packages keep their own licences; they will be listed in
`THIRD_PARTY_LICENSES.md` before any image is published (phase F10).
