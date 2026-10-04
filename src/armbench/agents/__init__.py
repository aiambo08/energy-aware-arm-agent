"""Agents that act on the primitives: the scripted baseline A and the LLM program agent B."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from armbench.agents.base import Agent, AgentError, AgentTrace
from armbench.agents.llm import LLMAgent, extract_program, prefetch
from armbench.agents.scripted import ScriptedAgent, detect_all
from armbench.llm import LLMParams, Provider, load_llm_params, make_provider

AGENT_IDS: Final[tuple[str, ...]] = ("A", "B")
AGENTS: dict[str, Agent] = {ScriptedAgent.id: ScriptedAgent()}


def get_agent(
    agent_id: str,
    *,
    llm: LLMParams | None = None,
    provider: Provider | None = None,
    artifacts_dir: Path | None = None,
) -> Agent:
    """``A`` needs nothing; ``B`` takes the LLM configuration (``configs/llm.yaml`` by default)
    and optionally a ready-made provider (the runner passes ``replay`` inside containers)."""
    if agent_id in AGENTS:
        return AGENTS[agent_id]
    if agent_id == "B":
        params = llm or load_llm_params()
        return LLMAgent(provider or make_provider(params), params, artifacts_dir=artifacts_dir)
    msg = f"unknown agent {agent_id!r}; known: {', '.join(AGENT_IDS)}"
    raise KeyError(msg)


__all__ = [
    "AGENTS",
    "AGENT_IDS",
    "Agent",
    "AgentError",
    "AgentTrace",
    "LLMAgent",
    "ScriptedAgent",
    "detect_all",
    "extract_program",
    "get_agent",
    "prefetch",
]
