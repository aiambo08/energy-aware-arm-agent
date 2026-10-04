"""The prompt agent B sends: API reference, environment facts, rules, one example, the task.

The prompt is a pure function of the task instance and the configuration — it never contains
live perception, so it can be answered on the host (where the key is), cached, and replayed
inside the network-less simulation container with the same hash.
"""

from __future__ import annotations

import textwrap

from armbench.llm.types import Message
from armbench.primitives.params import PrimitiveParams
from armbench.sandbox import ALLOWED_BUILTINS, ProgramRun
from armbench.tasks import TaskInstance

OBSERVE_POSES_HINT = "Pose(-0.30, 0.0, 0.15), Pose(-0.30, 0.25, 0.15), Pose(-0.30, -0.25, 0.15)"

API_REFERENCE = """\
robot.detect(target="any") -> tuple of detections, one per visible cube (optionally only cubes
    of colour `target`). Each detection has: color (str), position (x, y, z of the cube CENTRE in
    metres), top_center (x, y, z), yaw_rad (top-face yaw in [-pi/4, pi/4)), complete (bool: the
    whole top face was visible; a partial face biases the position by up to ~1 cm), depth_m.
    Camera only: it does not move the arm and costs no energy.
robot.observe() -> record with t_sim, q, qd, tcp (Pose), gripper_opening_m, holding (bool) and
    detections (same as detect()).
robot.move_to(pose, speed_scale=1.0) -> moves the TCP to `pose` (top-down, gripper pointing down)
    along a joint-space path. speed_scale in [{smin}, 1.0] scales the joint speed: slower moves
    use less peak power but take longer. Raises OutOfReach, Singularity or Collision BEFORE moving
    if the target is unsafe; Timeout if the controller fails.
robot.grasp() -> closes the fingers (fingers lie along the gripper axis perpendicular to the pose
    yaw); raises NoObjectGrasped if they close on nothing.
robot.release() -> opens the fingers.
robot.reset() -> opens the gripper and returns to the ready pose.
Pose(x, y, z, yaw=0.0) with .above(dz) -> Pose, .with_yaw(yaw) -> Pose, .distance_xy(other).
Errors: PrimitiveError (base), OutOfReach, Singularity, Collision, Timeout, NoObjectGrasped,
    CameraTimeout, SkillNotAvailable; each has .code and .message.
"""

ENVIRONMENT = """\
- Frame base_link: the table surface is z = 0, the robot base is at the origin, the table lies
  towards negative x. Cubes have {cube_cm:.1f} cm edges, so a resting cube's centre is at
  z ~= {cube_half:.4f} and a cube on top of another is {cube_m:.3f} m higher.
- Reachable TCP box (anything else raises OutOfReach): x in [{xmin:.2f}, {xmax:.2f}],
  y in [{ymin:.2f}, {ymax:.2f}], z in [{zmin:.3f}, {zmax:.2f}]. Finger tips are 1 cm below the TCP.
- The gripper opens to 85 mm. To grasp a cube, put the TCP at the detected centre
  (z = position[2]) with yaw = the detection's yaw_rad (or yaw +/- pi/2, equivalent for a cube);
  approach from ~10 cm above, descend, grasp, lift. The open fingers sweep ~6 cm either side of
  the grasp point along the finger axis: pick the yaw whose fingers stay clear of neighbours.
- Only the colours named in the task are unique; other cubes may repeat colours. Cubes are at
  least 9 cm apart. Release ~5 mm above the resting height so the cube settles.
- The arm starts at the ready pose (x={rx:.2f}, y={ry:.2f}, z={rz:.2f}) with the gripper open.
  From there the forearm shadows ~30 % of the table: detections may be missing or partial.
  Observation poses at the near edge that keep the arm out of the camera view: {observe}.
- Task limit: {sim_s:.0f} simulated seconds and {max_calls} primitive calls; a Gazebo move takes
  about 1-2 s. Energy (Wh) is integrated over the whole episode, including observation moves.
"""

RULES = """\
Write ONE Python program that solves the task when executed once, top to bottom, in a sandbox.
- Allowed: the names robot, Pose, math and the error classes above; functions, loops, if,
  try/except, arithmetic, comparisons, lists/tuples/dicts/sets, comprehensions, f-strings, and
  these builtins: {builtins}.
- Not allowed (the program is rejected before running): import, class, with, yield, async,
  decorators, any identifier or attribute starting with an underscore, .format(), and any other
  builtin (open, eval, exec, getattr, type, globals ...).
- Do not call robot.reset() at the end; leave the cube where the task wants it.
- Answer with the program only, inside one ```python fenced block, no prose.
"""

EXAMPLE = """\
```python
def best(dets, color):
    hits = [d for d in dets if d.color == color]
    full = [d for d in hits if d.complete]
    return (full or hits)[0] if hits else None

d = best(robot.detect(), "red")
if d is None or not d.complete:
    robot.move_to(Pose(-0.30, 0.0, 0.15))
    d = best(robot.detect(), "red") or d
pick = Pose(d.position[0], d.position[1], d.position[2], d.yaw_rad)
place = Pose(-0.45, 0.10, pick.z)
robot.move_to(pick.above(0.10))
robot.move_to(pick, 0.5)
robot.grasp()
robot.move_to(pick.above(0.10), 0.5)
robot.move_to(place.above(0.10))
robot.move_to(place.above(0.005), 0.5)
robot.release()
robot.move_to(place.above(0.10))
```"""


def system_prompt(
    params: PrimitiveParams, cube_size_m: float, *, sim_s: float, max_calls: int
) -> str:
    ws = params.workspace
    rx, ry, rz = params.ready_pose[0], params.ready_pose[1], params.ready_pose[2]
    env = ENVIRONMENT.format(
        cube_cm=cube_size_m * 100, cube_half=cube_size_m / 2, cube_m=cube_size_m,
        xmin=ws.x_min, xmax=ws.x_max, ymin=ws.y_min, ymax=ws.y_max, zmin=ws.z_min, zmax=ws.z_max,
        rx=rx, ry=ry, rz=rz, observe=OBSERVE_POSES_HINT, sim_s=sim_s, max_calls=max_calls,
    )  # fmt: skip
    api = API_REFERENCE.format(smin=params.motion.speed_scale_min)
    rules = RULES.format(builtins=", ".join(ALLOWED_BUILTINS))
    return (
        "You control a simulated UR5e arm with a parallel gripper and an RGB-D camera above a "
        "table of coloured cubes by writing a Python program over a small, closed API.\n\n"
        "# API\n"
        + api
        + "\n# Environment\n"
        + env
        + "\n# Rules\n"
        + rules
        + "\n# Example (move the red cube to x=-0.45, y=0.10)\n"
        + EXAMPLE
        + "\n"
    )


def task_message(instance: TaskInstance) -> str:
    n = len(instance.scene.cubes)
    return f"# Task\n{instance.prompt}\n\nThere are {n} cubes on the table. Write the program."


def build_messages(instance: TaskInstance, system: str) -> tuple[Message, ...]:
    return (
        Message(role="system", content=system),
        Message(role="user", content=task_message(instance)),
    )


def feedback_message(run: ProgramRun) -> str:
    """Retry turn (only when ``max_attempts > 1``): what the previous program did wrong."""
    out = run.stdout.strip()
    tail = f"\nIts printed output was:\n{textwrap.indent(out[-1500:], '    ')}" if out else ""
    return (
        f"The previous program did not complete: {run.describe()}.{tail}\n"
        "Write a corrected full program in one ```python block."
    )
