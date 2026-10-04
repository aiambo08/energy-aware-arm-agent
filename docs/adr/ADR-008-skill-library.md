# ADR-008: Skill library — format, validation, freezing and agent B+S (F7)

- Status: accepted
- Date: 2026-10-04
- Phase: F7
- Decides: D7 (frozen skill library)

## Context

B+S and C+S reuse validated programs as *skills*. The plan fixes three constraints
(`docs/plan.es.md`, F7): a skill enters the library only if it passes >= 90 % of the
`skill_validation` seeds (20–39) **and** the sandbox; the library is built with development and
validation seeds and frozen (hash) before F9, read-only during evaluation, so the order of the
evaluation seeds cannot change what an agent knows; retrieval is a short list by description,
deduplicated by signature, with no vector store. The informe left open where skills come from
and how `execute_skill()` (a stub since F4) runs them.

## Decisions

### A skill is a validated program with the goal constants as parameters

`armbench.skills.Skill` is a frozen Pydantic model: `name`, `description`, typed `params`
(`color` or `float`), executable `preconditions`/`postconditions` (`Condition`), `origin`
(agent, task, run, seeds, program hash), `version`, the `body` (a program over the F4
primitives in which the goal constants became parameter names) and `validation` (split,
seeds, n_ok, failed seeds with the first reason, backend, armbench version). The identity is
the `signature()` — `name(param: kind, ...)` — and `sha256()` covers signature, body,
conditions and version. The body is bound by **AST-safe substitution**: `bind(kwargs)` prefixes
the program with `name = <literal>` assignments after `check_args` has typed every value, so a
skill call can never inject code. The bound program must pass the same `check_program`
whitelist as any agent program (`tests/unit/test_skills.py` checks it).

Conditions are a closed vocabulary, not free Python: `not_holding`, `holding`, `tcp_above`,
`cube_near(color, x, y, tol)`. Robot-state conditions are checked against `Robot.state()`
(joint/gripper snapshot, no camera); world conditions (`cube_near`) are checked where the
world is known — the validator and the F5 checker — and *deferred* inside `execute_skill`
(reported in `SkillResult.postconditions_deferred`) because the agent's robot has no ground
truth. **Rejected:** arbitrary Python pre/post-conditions (another sandbox to audit, and the
agent could state tautologies).

### Candidates come from agent B's completed dev programs

`armbench skills propose --run <dir>` reads an `armbench run` directory of agent B (dev seeds),
keeps the episodes whose program `completed` and whose verdict is `ok`, groups them by task,
and **parametrises** the program: every goal constant of the task instance (colours, target
coordinates as written in the prompt, also `round(x, 3)`) is replaced by a parameter name;
programs where a constant never appears are rejected with the reason. The origin records the
run, task and seeds; the body is the program of the lowest seed. Four candidates came out of
`runs/f6_live` (`pick_place_cube`, `stack_cube`, `sort_cubes`, `place_cube_over_wall`).

### Validation: sandbox + pre/post-conditions + the task checker on seeds 20–39

`armbench skills validate` runs each candidate on every `skill_validation` seed with
`Validator`: bind the task instance's goal to the signature, check preconditions on the fresh
robot, run the body through `run_program` (sandbox limits `SKILL_LIMITS`), check every
postcondition against the kinematic world, and finally ask the task's own success checker.
A seed passes only if all four hold; the skill is accepted at `MIN_PASS_RATE = 0.9`
(18/20) and its `validation` block is stored. Fault-injection tests prove the conditions bite:
a body that places 5 cm off fails `cube_near` on every seed and is rejected; a body that
stops holding fails `not_holding`; a frozen library whose JSON is edited is refused by its hash.
The validator runs on the kinematic backend, so what it certifies is the program logic and the
contract, not Gazebo physics; the Gazebo pilot of F7 exercises the same bodies in simulation.

### One library file, dedup by signature, frozen by hash, read-only once frozen

`skills/library.json` holds the skills (sorted by name; the file hash is order-independent)
and `skills/FROZEN.json` the manifest: library SHA-256, per-skill hashes, timestamp, version.
`add()` returns the existing skill when the signature already exists (first validated wins)
and refuses a name reused with another signature. After `armbench skills freeze` every
mutation raises `FrozenLibraryError`; `load(require_frozen=True)` recomputes the hash and
refuses a tampered or unfrozen library. The B+S agent, the CLI (`--skills-dir`) and the ROS
`episode_runner` all require a frozen library; `run.json` records its hash, size and names,
and the Docker bundle (`<out>/llm/skills/`) ships the two files into the `--network none`
containers. `armbench skills validate`/`freeze` refuse to touch a frozen directory: to change
the library you create a new one, and its hash changes with it.

### Retrieval is lexical and deterministic

`SkillLibrary.retrieve(query, k)` scores skills by word overlap between the task wording and
the skill's description/name (normalised tokens, stop words dropped), ties broken by name.
The agent shows the top `SKILLS_PER_PROMPT = 3` in a `# Skills` section with signature,
description, `requires`/`ensures`. **Rejected:** embeddings (a model dependency and a
non-deterministic ranking for a four-skill library).

### `execute_skill` is a nested sandboxed run, accounted in the outer trace

`Robot.execute_skill(name, **kwargs)` delegates to `SkillRunner` (the `SkillExecutor`
protocol): look-up (`SkillNotAvailable` with the available names), argument typing,
preconditions (`SkillPreconditionFailed`, nothing moved), `run_program` of the bound body
with the skill limits, propagation of the primitive error that stopped the body or
`SkillFailed` for any other outcome, postconditions (`SkillPostconditionFailed`, with the
inner calls in `details`), and a `SkillResult` with `skill`, `skill_sha256`, `calls` and the
sim-time span. The guest API exposes the three errors so programs can catch them. The outer
sandbox `Run` keeps one record per `execute_skill` with `skill` and `inner_calls`, so
`n_primitive_calls` counts the primitives the arm really executed and `skills_used` only the
skills that completed; `AgentTrace.n_primitives`/`skills_used` and the report's `llm.skills_used`
and `skill_episode_rate` are derived from them. From the agent's point of view a skill is one
call; from the energy meter's it is the same arm motion as the program it came from.

### Agent B+S is agent B plus a library

`get_agent("B+S", ..., skills_dir=...)` builds the same `LLMAgent` with a `SkillLibrary`; B
has none and its prompt has no `# Skills` section (a test pins this). `TemplateProvider`
answers B+S prompts with a single `robot.execute_skill(...)` when an offered skill's parameter
set matches the task kind (and prefers the "wall" skill only when the task mentions one),
otherwise it falls back to the plain B program. This is deliberately the simplest possible
B+S: no retries, no composition of several skills, no skill learning during a run.

## Consequences

- The library is built once from dev programs, validated on 20–39 and frozen; F9 will run
  B+S/C+S against `skills/FROZEN.json` with its hash in the pre-registered protocol.
- `scripts/f7_gate.py` checks: fault-injection tests green, library frozen and hash registered
  in the B+S run, every skill validated on the right split, B+S exercised the library in
  Gazebo with inner calls accounted, no infra failures, energy tables complete, and reports
  the B vs B+S pilot (success, tokens, latency, primitive calls, Wh) with **no** improvement
  threshold — that is hypothesis H1, and the template provider is not a language model.
- The cache is keyed by the prompt; the `# Skills` section is part of it, so a change in the
  library (or in retrieval) is a different request and never replays a stale answer. A
  response cached under the same prompt by an earlier code revision *is* replayed — a live
  gate run therefore starts from an empty `cache/llm`, as in F6.
