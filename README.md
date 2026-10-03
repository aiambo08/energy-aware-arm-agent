# Energy-Aware Arm Agent (`armbench`)

> Does an LLM agent that writes robot programs over a small set of primitives,
> reuses validated skills, and receives the arm's energy (Wh) as feedback,
> solve manipulation tasks with **fewer watt-hours** than one that does not?

A reproducible benchmark around a simulated **UR5e** (ROS 2 Jazzy + Gazebo
Harmonic, Docker, no GPU required) to answer that question with paired seeds,
confidence intervals and a pre-registered protocol.

**Status: phase F0 (skeleton).** There are no results yet. Anything in this
repository that looks like a number is either a configuration value or a
measurement stored under `reports/` with the script that produced it.

## What exists today

| Area | State |
|---|---|
| Python package `armbench` (`src/armbench`) | seed-split loader with the final-evaluation lock, `armbench` CLI |
| Quality gates | `ruff`, `ruff format`, `mypy --strict`, `pytest` (+ `hypothesis`), `pre-commit` |
| Simulation image | `docker/Dockerfile`: ROS 2 Jazzy + Gazebo Harmonic + UR5e/Robotiq/Pinocchio packages; empty headless world boot check |
| CI | `verify` (lint, types, tests, no ROS) and `sim-image` (build image, boot empty world, publish `reports/f0_sim.json`) |
| Docs | `docs/plan.es.md` (full phased plan, Spanish), ADRs in `docs/adr/`, `docs/PROJECT_STATE.md` |

Roadmap (one PR per phase, see `docs/plan.es.md` and `docs/PROJECT_STATE.md`):
F1 arm simulation · F2 torque source and energy model · F3 perception ·
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
docker/            Dockerfile, entrypoint (compose added in F1)
ros_ws/src/        ROS 2 packages (description, bringup, bridge) — from F1
src/armbench/      pure-Python package, testable without ROS
configs/           seeds.yaml (energy.yaml, tasks.yaml… in later phases)
scripts/           check scripts that write reports/*.json
tests/unit/        tests without ROS; tests marked `sim` need the image
reports/           committed JSON metrics per phase (the DoD evidence)
docs/              plan, ADRs, PROJECT_STATE.md
```

## Contributing and licence

See `CONTRIBUTING.md`. Code is MIT (`LICENSE`). Robot meshes and third-party
ROS packages keep their own licences; they will be listed in
`THIRD_PARTY_LICENSES.md` before any image is published (phase F10).
