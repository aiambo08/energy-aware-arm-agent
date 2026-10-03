# ADR-003: Simple parallel-jaw gripper instead of Robotiq 2F-85 (decision D2)

- Status: accepted
- Date: 2026-10-03
- Phase: F1

## Context

Decision D2 of `docs/plan.es.md` asked for the arm and gripper. The arm is settled
(UR5e from `ur_description` 3.5.1, ADR-001). For the gripper the plan listed Robotiq 2F-85
as the default with "verify in Gazebo Harmonic" attached.

`ros-jazzy-robotiq-description 0.0.1` ships the 2F-85 meshes and a `ros2_control`
block for the real hardware, but its Gazebo support relies on mimic joints (one actuated
joint driving five passive ones). In Gazebo Harmonic + `gz_ros2_control` 1.2.x the mimic
joints are implemented as position constraints that fight the contact solver; the
community reports jittery or slipping grasps, and the finger-pad geometry is complex to
tune. F1 needs a gripper whose grasp is reliable enough to be a *fixture* for F2–F9
(energy and task metrics must not be polluted by gripper physics).

## Decision

1. `armbench_description/urdf/parallel_gripper.xacro`: a two-finger parallel gripper of our
   own — a box base on `tool0`, two prismatic fingers (stroke 42.5 mm each, 85 mm total,
   like a 2F-85), simple box collision pads with μ = 1.0, and a `tcp` frame 125 mm below
   `tool0`.
2. Fingers are controlled by `effort_controllers/JointGroupEffortController`
   (`gripper_controller/commands`, N per finger): +20 N closes, −10 N opens. A force
   command closes until contact, which is the behaviour `grasp()` needs in F4, and the
   finger effort becomes part of the F2 energy model with no extra work.
3. The gripper declares its own `<ros2_control>` system next to the UR one; `gz_ros2_control`
   loads both into the same `controller_manager`.
4. Robotiq 2F-85 is kept in the image (`robotiq_description`) for a later visual swap; it is
   not on the critical path.

## Consequences

- Measured in F1 (`reports/f1_sim.json`): the fixed-waypoint pick-and-place of a 45 mm,
  50 g cube lifts it 14.5 cm, transports it 18 cm and places it back with sub-millimetre
  slip; fingers stop at 19.8–20.2 mm (= stroke − cube/2) when closing on the cube.
- The gripper is not a physically calibrated model of any product; energy figures for the
  gripper are therefore indicative, and the paper must say so.
- If a photorealistic gripper is ever needed, the swap is confined to the xacro and the
  `tcp` offset.
