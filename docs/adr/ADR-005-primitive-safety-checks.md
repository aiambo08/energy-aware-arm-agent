# ADR-005: Pre-flight checks and typed errors of the robot primitives (F4)

- Status: accepted
- Date: 2026-10-03
- Phase: F4

## Context

`armbench.primitives.Robot` is the only way agent programs (F6), skills (F7) and the
scripted baseline (F5) touch the arm. The plan asks for a closed set of primitives
(`observe`, `detect`, `move_to`, `grasp`, `release`, `execute_skill`) with typed errors and a
simulation gate (`docs/plan.es.md`, F4): random reachable poses are reached with p95 position
error < 5 mm and orientation < 2°, unreachable poses are refused 100/100 *without moving*,
0 table collisions in 200 valid moves, monotone duration in `speed_scale`, scripted
pick-and-place ≥ 95 % and `reset()` < 5 s.

Gazebo has no table contact sensor and the UR5e `joint_trajectory_controller` executes any
joint goal it is given, so "safe" has to be decided before a goal is sent.

## Decision

`Robot.move_to(pose, speed_scale)` plans and validates the whole motion before the backend
sees it (`Robot.plan()`), in this order; the first failing check raises and **no goal is
sent** (the gate measures `max |Δq|` and the number of goals during refusals):

1. **Workspace box** (`configs/primitives.yaml: workspace`): a target outside it raises
   `OutOfReach(reason="outside_workspace")`. The box is deliberately the table area the scenes
   use plus a small margin, not the kinematic reach of the arm: everything the tasks need is
   inside, and everything outside is where the arm would sweep through the robot column or
   leave the camera's view.
2. **Table**: a target whose finger tips would be below the table top raises
   `Collision(reason="target_below_table")`.
3. **IK** (`UR5eModel.ik_top_down`, damped least squares from the current configuration,
   joints unwrapped by 2π towards the current configuration): no convergence raises
   `OutOfReach(reason="ik")`; a solution outside the joint limits raises
   `OutOfReach(reason="joint_limits")`.
4. **IK branch** (`ik: branch`): the elbow-up/wrist-down family that keeps the forearm above
   the table is enforced per joint; another branch raises `OutOfReach(reason="branch")`.
   Without it the solver happily returns an elbow-down solution whose path goes through the
   table even when the final pose is fine.
5. **Singularity**: the smallest singular value of the geometric Jacobian at the target below
   `ik.min_singular_value` raises `Singularity`.
6. **Path clearance**: the joint-space interpolation from the current configuration to the
   target (`collision.path_samples` samples, which is what the controller executes for a
   single-waypoint goal) must keep every arm frame above `collision.link_clearance_m` and the
   finger tips above `collision.tip_clearance_m`; otherwise `Collision(reason="path")`.
   Checking only the final configuration was not enough: a first experiment on the pure
   kinematic model reached every pose but the path minimum was −0.15 m.
7. **Duration** = `max(min_duration_s, max|Δq| / (max_joint_speed_rad_s · speed_scale))`:
   monotone in `speed_scale` by construction, so the energy lever of F8 is well defined.

After the controller reports success the robot waits until the joints are at rest
(`motion.settle_*`) and only then measures the final pose: the controller's goal tolerance
(0.05 rad) is reached while the joints are still converging, and measuring at that instant
gave 1–7 mm errors in the first live run. A final joint error above
`motion.goal_tolerance_rad`, a rejected/aborted goal or a controller deadline raises
`Timeout` with the outcome in `details`.

`grasp()` closes with `gripper.close_effort_n`, waits `gripper.settle_s` and raises
`NoObjectGrasped` when the pad opening is below `gripper.min_object_width_m` (the fingers met).
`release()` never fails. `observe()`/`detect()` raise `CameraTimeout` when no RGB-D pair
stamped after the call arrives within `camera_timeout_s`. `execute_skill()` raises
`SkillNotAvailable` until the skill library exists (F7).

Errors derive from `PrimitiveError` with a stable `code` and a `details` dict
(`to_dict()` is what the agent and the episode log see). The names are the ones the plan
fixes (`OutOfReach`, `Singularity`, `Collision`, `Timeout`, `NoObjectGrasped`,
`CameraTimeout`), so ruff's `N818` is disabled for that module only.

The same `Robot` runs on two backends behind the `Backend` protocol: `RosBackend`
(rclpy, the live simulation) and `KinematicBackend` (no ROS; exact motions, synthetic
RGB-D rendered by `armbench.perception.synthetic`, cubes that follow the gripper). Unit
tests, the Hypothesis property "any pose in the box is reached or a typed error is raised
without a goal being sent" and the F6 replay mode use the kinematic backend; the gate uses
the ROS backend.

## Consequences

- An agent program can only fail in typed, logged ways; the arm never receives a goal that
  the checks reject. Refusals cost no simulation time or energy.
- The workspace box is smaller than the reachable set; tasks (F5) must place everything
  inside it. The branch constraint rules out some reachable configurations near the base.
- The path check assumes the controller interpolates in joint space between the current
  and the goal configuration (true for a single-waypoint `FollowJointTrajectory` goal);
  multi-waypoint trajectories would need per-segment checks.
- The table is the only obstacle modelled in F4; the obstacle task of F5 needs an
  additional check (box obstacles against the same sampled path), not a new mechanism.
- `ready_pose` (-0.5, 0, 0.25) puts the forearm and wrist right under the camera
  (-0.5, 0, 1.0), so `observe()`/`detect()` from the ready pose miss the cubes in the arm's
  shadow (every cube was visible in only 42 of the 100 gate episodes; a ray cast from the camera against the ready-pose links modelled as
  3 cm cylinders predicts the exact number of missed cubes in 91 of 98 episodes). The
  scripted pick-and-place is unaffected (it grasps the first detection), but F5 tasks that need every cube must look from a pose outside the camera
  cone or re-detect after moving away.
- Spawning/removing cubes goes through the `gz service` CLI with hard timeouts. The CLI
  reply is lost now and then while the world steps, so each call is attempted up to three
  times (checking the bridged `/model/<cube>/pose` topic to see whether the world already
  applied it), and the gate retries an episode once when the failure is still a `scene`
  error (spawn/settle infrastructure, not the robot), reporting the count as
  `scene_retries`.
