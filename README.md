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
| Torque source (F2) | `armbench_bringup` nodes `energy_meter` (online Wh on `/armbench/energy`), `torque_probe` and `torque_compare` (Pinocchio inverse dynamics); `scripts/torque_source.py` runs critical gate 2 → `reports/f2_torque_source.json`, decision in ADR-004 |
| Simulation (F1) | `ros_ws/src/armbench_description` (UR5e + parallel gripper xacro, tabletop world with fixed RGB-D camera), `ros_ws/src/armbench_bringup` (`sim.launch.py`, controllers, `sim_check` node) |
| CI | `verify` (lint, types, tests, no ROS) and `sim-image` (build image, F0 boot gate, F1 self-check gate) |
| Docs | `docs/plan.es.md` (full phased plan, Spanish), ADRs in `docs/adr/`, `docs/PROJECT_STATE.md` |

Roadmap (one PR per phase, see `docs/plan.es.md` and `docs/PROJECT_STATE.md`):
~~F1 arm simulation~~ · ~~F2 torque source and energy model~~ · F3 perception ·
F4 primitives · F5 tasks, baseline and runner · F6 LLM agent and sandbox ·
F7 skill library · F8 energy-aware agent · F9 pre-registered evaluation ·
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
| `final_eval` | 100–119 | **locked**: `check_seeds_allowed` raises unless `final_eval=True` and a protocol hash are given |

The split is data (`configs/seeds.yaml`), validated for disjointness, and
tested in `tests/unit/test_seeds.py`. Changing it after pre-registration
requires an ADR.

## Repository layout

```
docker/            Dockerfile, entrypoint
ros_ws/src/        ROS 2 packages: armbench_description (xacro, worlds), armbench_bringup (launch, controllers, sim_check)
src/armbench/      pure-Python package, testable without ROS
configs/           seeds.yaml, ur5e_kinematics.yaml, scene.yaml (energy.yaml, tasks.yaml… later)
scripts/           check scripts that write reports/*.json
tests/unit/        tests without ROS; tests marked `sim` need the image
reports/           committed JSON metrics per phase (the DoD evidence)
docs/              plan, ADRs, PROJECT_STATE.md
```

## Contributing and licence

See `CONTRIBUTING.md`. Code is MIT (`LICENSE`). Robot meshes and third-party
ROS packages keep their own licences; they will be listed in
`THIRD_PARTY_LICENSES.md` before any image is published (phase F10).
