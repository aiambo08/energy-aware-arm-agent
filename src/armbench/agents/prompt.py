"""The prompt agent B sends: API reference, environment facts, rules, one example, the task.
B+S appends a ``# Skills`` section and C/C+S an ``# Energy`` section to the task message.

The prompt is a pure function of the task instance and the configuration — it never contains
live perception, so it can be answered on the host (where the key is), cached, and replayed
inside the network-less simulation container with the same hash.
"""

from __future__ import annotations

import textwrap
from collections.abc import Sequence

from armbench.energy import EnergyParams, EnergyReference, TaskReference
from armbench.llm.types import Message
from armbench.primitives.params import PrimitiveParams
from armbench.sandbox import ALLOWED_BUILTINS, ProgramRun
from armbench.skills.spec import Skill
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


SKILLS_HEADER = """\
# Skills
Validated skills (agent B+S). Each is a whole validated program with the goal as parameters;
calling one is a single call for you, several robot moves for the arm, and it ends with the
gripper open. robot.execute_skill("name", param=value, ...) runs it and returns a record with
.skill, .calls (primitives it ran), .t_start_sim, .t_end_sim; it raises SkillPreconditionFailed
(nothing moved), SkillPostconditionFailed (ran, but the stated outcome does not hold), SkillFailed
(stopped early) or the primitive error that stopped it. Prefer a skill whose contract matches the
task; otherwise write the steps yourself. <color> is a colour name string, <float> metres.
"""


def skills_section(skills: Sequence[Skill]) -> str:
    lines = [SKILLS_HEADER]
    for s in skills:
        pre = ", ".join(c.describe() for c in s.preconditions) or "nothing"
        post = ", ".join(c.describe() for c in s.postconditions) or "nothing stated"
        contract = f"requires: {pre}; ensures: {post}"
        lines.append(f"- {s.call_syntax()}\n  {s.description}\n  {contract}")
    return "\n".join(lines) + "\n"


ENERGY_HEADER = "# Energy"

ENERGY_SECTION = """\
# Energy
Your program is also scored on the electrical energy (Wh) the arm draws over the whole episode,
from the first primitive call to the last, observation moves included. Model:
P_el = P_mech / eta + copper losses + P0, with eta = {eta:.2f} (drive-train efficiency),
P0 = {p0:.0f} W (controller, brakes and electronics: drawn every second, moving or not) and
copper losses = sum_i R_i * (tau_i / kt_i)^2 (grow with the square of the joint torques).
Reference budget: a scripted baseline solved this task in a median of {wh:.3f} Wh
(interquartile mean {iqm:.3f}, range {lo:.3f}-{hi:.3f} Wh over {n} development scenes), using
{sim_s:.1f} s of simulated time and {calls:.0f} primitive calls per episode. Aim at or below
{wh:.3f} Wh without failing the task: a failed episode saves nothing.
Levers, in order of effect for this arm:
- Time. P0 is paid every second, so every move and every second of motion costs energy:
  use fewer moves, shorter paths and lower lift heights (keep clearance above neighbouring cubes
  and obstacles), and move to an observation pose only when a needed cube is missing or partial
  (detect() itself costs nothing).
- speed_scale. A lower value cuts peak mechanical and copper power but lengthens the move, so
  P0 is paid for longer; it pays off only when the motion term dominates (fast, heavy moves).
  Use it where precision matters (final descent, release), not as a default.
"""


def energy_section(ref: EnergyReference, task: TaskReference, params: EnergyParams) -> str:
    """The ``# Energy`` block of agents C and C+S: the model constants, baseline A's Wh for the
    same task as a budget (D8) and the levers the program controls."""
    return ENERGY_SECTION.format(
        eta=params.eta, p0=params.p0_w, wh=task.wh_median, iqm=task.wh_iqm, lo=task.wh_min,
        hi=task.wh_max, n=task.n, sim_s=task.sim_s_median, calls=task.n_primitives_median,
    )  # fmt: skip


def task_message(
    instance: TaskInstance, skills: Sequence[Skill] = (), energy: str | None = None
) -> str:
    n = len(instance.scene.cubes)
    head = skills_section(skills) + "\n" if skills else ""
    if energy:
        head += energy + "\n"
    return (
        f"{head}# Task\n{instance.prompt}\n\nThere are {n} cubes on the table. Write the program."
    )


def build_messages(
    instance: TaskInstance,
    system: str,
    skills: Sequence[Skill] = (),
    energy: str | None = None,
) -> tuple[Message, ...]:
    """Agent B's two messages; B+S prepends the retrieved skills to the task message and C/C+S
    the energy block (the system prompt is identical for all four, so the configurations differ
    only by those sections)."""
    return (
        Message(role="system", content=system),
        Message(role="user", content=task_message(instance, skills, energy)),
    )


def feedback_message(run: ProgramRun) -> str:
    """Retry turn (only when ``max_attempts > 1``): what the previous program did wrong."""
    out = run.stdout.strip()
    tail = f"\nIts printed output was:\n{textwrap.indent(out[-1500:], '    ')}" if out else ""
    return (
        f"The previous program did not complete: {run.describe()}.{tail}\n"
        "Write a corrected full program in one ```python block."
    )
