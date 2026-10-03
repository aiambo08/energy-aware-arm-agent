# ADR-001: Simulation stack, repository name and package name

- Status: accepted
- Date: 2026-10-03
- Phase: F0
- Closes plan decisions D1 (stack) and D9 (name)

## Context

The technical report and the phased plan (`docs/plan.es.md`) leave two
decisions for F0: the simulation stack and the project name. The repository
was created by the owner as `aiambo08/energy-aware-arm-agent`; the plan used
`armbench` as a provisional name.

## Decision

1. **Stack:** ROS 2 Jazzy Jalisco on Ubuntu 24.04 with Gazebo Harmonic via the
   `ros-jazzy-ros-gz` vendor packages, inside Docker (`ros:jazzy-ros-base`
   pinned by digest). Arm: UR5e from `ros-jazzy-ur-description` /
   `ros-jazzy-ur-simulation-gz`. Gripper candidate: Robotiq 2F-85 from
   `ros-jazzy-robotiq-description` (to be verified in F1, D2). Inverse-dynamics
   fallback: `ros-jazzy-pinocchio` (D3, F2). All of these were confirmed
   installable from `packages.ros.org` for `noble` on 2026-10-03.
2. **Names:** the GitHub repository and the public project are
   *Energy-Aware Arm Agent*. The Python package, CLI and Docker image are
   `armbench` (`src/armbench`, `uv run armbench`, `armbench-sim:<tag>`), which
   is shorter to type and matches the plan. No PyPI release is planned before
   F10; the PyPI name will be checked then.
3. **Python:** 3.12 (the system interpreter in Noble, required by `rclpy`).
   Pure-Python code lives in `src/armbench` and is tested on the host without
   ROS; ROS-dependent code lives in `ros_ws/src` and is tested inside the image
   (`pytest -m sim`).

## Alternatives considered

- ROS 2 Lyrical / newer Gazebo: smaller ecosystem for UR packages (report §3.2).
- Gazebo Classic: end-of-life; the UR Classic repository does not target Jazzy.
- Naming the package after the repository (`energy_aware_arm_agent`): too
  long for a CLI used in every command.

## Consequences

- One image serves F0–F10; its size is a tracked metric (threshold 5 GB, see ADR-002,
  reported in `reports/f0_sim.json`).
- Any change of distro, simulator or arm requires a new ADR.
