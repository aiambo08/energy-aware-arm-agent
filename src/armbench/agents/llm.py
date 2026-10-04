"""Agent B: one LLM call (by default) writes a program over the primitives; the sandbox runs it.

The agent is provider-agnostic (:mod:`armbench.llm`) and never sees the robot directly: the
program it obtains is checked, executed in a child process and accounted for by
:func:`armbench.sandbox.run_program`. Everything the episode log needs for audit and cost comes
back in the :class:`AgentTrace` (hashes of prompt, response and program, tokens, latency,
cost, sandbox outcome and the primitive call sequence).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Final

from armbench.agents.base import AgentError, AgentTrace
from armbench.agents.prompt import build_messages, feedback_message, system_prompt
from armbench.llm import (
    LLMParams,
    LLMRequest,
    LLMResponse,
    Message,
    Provider,
    ProviderError,
    Usage,
    sha256_text,
)
from armbench.primitives import PrimitiveParams, Robot
from armbench.sandbox import ProgramRun, SandboxLimits, run_program
from armbench.scene import load_scene_config
from armbench.skills import Skill, SkillLibrary
from armbench.tasks import TaskInstance

SKILLS_PER_PROMPT: Final = 3

FENCE_RE: Final = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL)


def extract_program(text: str) -> str:
    """The last fenced code block, or the whole answer when there is none."""
    blocks = FENCE_RE.findall(text)
    body = blocks[-1] if blocks else text
    return body.strip("\n") + "\n"


class LLMAgent:
    def __init__(  # noqa: PLR0913 - keyword-only knobs, all optional
        self,
        provider: Provider,
        params: LLMParams,
        *,
        agent_id: str = "B",
        limits: SandboxLimits | None = None,
        artifacts_dir: Path | None = None,
        cube_size_m: float | None = None,
        library: SkillLibrary | None = None,
    ) -> None:
        self.id = agent_id
        self.provider = provider
        self.params = params
        self.library = library
        """B+S: the frozen library whose skills are offered in the prompt (B has none)."""
        self.limits = limits or params.sandbox.limits()
        self.artifacts_dir = artifacts_dir
        self.cube_size_m = (
            cube_size_m if cube_size_m is not None else load_scene_config().cube.size_m
        )
        self._system: str | None = None

    def system(self, params: PrimitiveParams) -> str:
        if self._system is None:
            self._system = system_prompt(
                params, self.cube_size_m, sim_s=self.limits.sim_s, max_calls=self.limits.max_calls
            )
        return self._system

    def skills_for(self, instance: TaskInstance) -> tuple[Skill, ...]:
        """Skills retrieved by the task wording (deterministic word overlap, top-k)."""
        if self.library is None:
            return ()
        hits = self.library.retrieve(instance.prompt, k=SKILLS_PER_PROMPT)
        return tuple(h.skill for h in hits)

    def request(self, params: PrimitiveParams, instance: TaskInstance) -> LLMRequest:
        """The first-turn request for an instance (what the host pre-fetches and caches)."""
        return LLMRequest(
            model=self.params.model,
            messages=build_messages(instance, self.system(params), self.skills_for(instance)),
            temperature=self.params.temperature,
            max_tokens=self.params.max_tokens,
            seed=self.params.seed,
        )

    def solve(self, robot: Robot, instance: TaskInstance) -> AgentTrace:
        p = self.params
        req = self.request(robot.params, instance)
        prompt_sha = req.prompt_sha256()
        usage = Usage()
        latency = 0.0
        cost = 0.0
        cached = True
        model: str | None = None
        origin: str | None = None
        runs: list[ProgramRun] = []
        response_sha = program_sha = None
        for attempt in range(1, p.max_attempts + 1):
            try:
                resp = self.provider.complete(req)
            except ProviderError as exc:
                raise AgentError(exc.code, exc.message) from exc
            usage = usage + resp.usage
            latency += resp.latency_s
            cost += resp.cost_usd
            cached = cached and resp.cached
            model = resp.model
            origin = resp.provider
            response_sha = resp.response_sha256()
            if usage.total > p.max_tokens_per_episode:
                msg = f"{usage.total} tokens over {p.max_attempts} attempt(s) exceed the cap"
                raise AgentError("token_budget", msg)
            program = extract_program(resp.text)
            program_sha = sha256_text(program)
            run = run_program(program, robot, self.limits)
            runs.append(run)
            self._save(instance, attempt, program, run)
            if run.outcome == "completed" or attempt == p.max_attempts:
                break
            req = req.model_copy(
                update={
                    "messages": (
                        *req.messages,
                        Message(role="assistant", content=resp.text),
                        Message(role="user", content=feedback_message(run)),
                    )
                }
            )
        last = runs[-1]
        skills_used = tuple(dict.fromkeys(n for r in runs for n in r.skills_used))
        trace = AgentTrace(
            attempts=len(runs),
            n_primitives=sum(r.n_primitive_calls for r in runs),
            skills_used=skills_used,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            llm_latency_s=latency,
            prompt_sha256=prompt_sha,
            response_sha256=response_sha,
            program_sha256=program_sha,
            llm_provider=origin or self.provider.id,
            llm_model=model,
            llm_cached=cached,
            cost_usd=cost,
            program_outcome=last.outcome,
            program_calls=last.primitives,
            program_error=last.error_message if last.outcome != "completed" else None,
            sandbox_isolation=last.isolation,
            notes=f"{last.describe()}; isolation={','.join(last.isolation)}",
        )
        if last.outcome != "completed":
            err = last.primitive_error()
            if err is not None:
                raise err
            raise AgentError(last.error_code or "program_failed", last.describe(), trace=trace)
        return trace

    def _save(self, instance: TaskInstance, attempt: int, program: str, run: ProgramRun) -> None:
        if self.artifacts_dir is None:
            return
        d = self.artifacts_dir / "programs"
        d.mkdir(parents=True, exist_ok=True)
        stem = f"{instance.task.name}_v{instance.task.version}_s{instance.seed}_a{attempt}"
        (d / f"{stem}.py").write_text(program, encoding="utf-8")
        (d / f"{stem}.run.json").write_text(run.model_dump_json(indent=1) + "\n", encoding="utf-8")


def prefetch(
    agent: LLMAgent, params: PrimitiveParams, instances: Iterable[TaskInstance]
) -> list[tuple[LLMRequest, LLMResponse]]:
    """Ask the provider for every first-turn program up front (on the host, where the key and
    the network are) so the simulation containers can run agent B in replay mode."""
    out: list[tuple[LLMRequest, LLMResponse]] = []
    for inst in instances:
        req = agent.request(params, inst)
        out.append((req, agent.provider.complete(req)))
    return out
