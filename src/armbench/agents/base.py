"""What every agent (scripted baseline, LLM programs, skill users) must provide."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from armbench.primitives import Robot
from armbench.tasks import TaskInstance


class AgentTrace(BaseModel):
    """Bookkeeping an agent returns after acting; the runner adds the ground-truth verdict."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    attempts: int = Field(ge=1, default=1)
    n_primitives: int = Field(ge=0, default=0)
    n_observe_moves: int = Field(ge=0, default=0)
    """Extra ``move_to`` calls made only to see the table (cost of perception)."""
    prompt_tokens: int = Field(ge=0, default=0)
    completion_tokens: int = Field(ge=0, default=0)
    llm_latency_s: float | None = None
    skills_used: tuple[str, ...] = ()
    prompt_sha256: str | None = None
    response_sha256: str | None = None
    program_sha256: str | None = None
    """Hash of the code that ran (the baseline module for A, the generated program for B+)."""
    llm_provider: str | None = None
    """Provider that produced the answer (the origin, also when it came back from the cache)."""
    llm_model: str | None = None
    """Exact model identifier the provider reported."""
    llm_cached: bool | None = None
    """Every completion of the episode came from the cache (replay or repeated prompt)."""
    cost_usd: float | None = None
    program_outcome: str | None = None
    """Sandbox outcome of the last program (``completed``, ``rejected``, ``exception`` ...)."""
    program_calls: tuple[str, ...] = ()
    """Primitive names the last program invoked, in order (replay compares these)."""
    program_error: str | None = None
    sandbox_isolation: tuple[str, ...] = ()
    notes: str = ""


class AgentError(Exception):
    """The agent could not act (no program, provider failure, budget, rejected code); the
    runner logs it as a failure of stage ``agent`` with this ``code`` and keeps ``trace``."""

    def __init__(self, code: str, message: str, *, trace: AgentTrace | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.trace = trace


class Agent(Protocol):
    id: str

    def solve(self, robot: Robot, instance: TaskInstance) -> AgentTrace:
        """Act on the robot until the agent believes the task is done.

        Primitive errors propagate: the runner records them as the episode's failure.
        """
        ...
