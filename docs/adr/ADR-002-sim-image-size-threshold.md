# ADR-002: Simulation image size threshold raised to 5 GB

- Status: accepted
- Date: 2026-10-03
- Phase: F0

## Context

The plan set a provisional F0 gate of "simulation image < 4 GB" and marked it
adjustable. The first build of `docker/Dockerfile` measured 4.46 GB
uncompressed (sum of layer sizes from `docker history`; 1.24 GB compressed).

Measured contributions (uncompressed, same base image
`ros:jazzy-ros-base` = 1.32 GB):

| Variant | Packages | Layer sum |
|---|---|---|
| full | ros_gz, gz_ros2_control, ros2_control(lers), ur_description, **ur_simulation_gz**, robotiq_description, pinocchio, cv_bridge, image_transport, rosbag2 mcap, xacro, colcon, python3-opencv, Mesa | 4.461 GB |
| slim | same without `ur_simulation_gz` and `mesa-utils` | 4.251 GB |

Dropping `ur_simulation_gz` (which pulls rviz2 and MoveIt) saves only 0.21 GB,
far less than the ~1 GB initially estimated. The bulk of the image is
Gazebo Harmonic itself, Boost/LLVM/VTK runtime libraries and the `-dev`
headers that `colcon build` needs in F1.

## Decision

1. Keep `ur_simulation_gz` in the image: the F1 gate is precisely that its
   launch files bring up the UR5e headless inside Docker (risk R1 of the plan).
2. Raise the image threshold to **5 GB uncompressed** (`IMAGE_THRESHOLD_GB`
   in `scripts/check_sim_boot.py`). The metric stays the conservative
   `docker history` layer sum, not the compressed registry size.
3. Revisit in F10 (public release): a multi-stage build that strips `-dev`
   headers from the runtime image is the obvious lever if a smaller public
   image is wanted.

## Consequences

- `reports/f0_sim.json` records `image_size_gb` against a 5 GB threshold.
- Further package additions must be justified against this budget in the
  phase that adds them.
