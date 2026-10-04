"""Agents that act on the primitives: the scripted baseline A and the LLM program agent B."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from armbench.agents.base import Agent, AgentError, AgentTrace
from armbench.agents.llm import LLMAgent, extract_program, prefetch
from armbench.agents.scripted import ScriptedAgent, detect_all
from armbench.llm import LLMParams, Provider, load_llm_params, make_provider
from armbench.paths import SKILLS_DIR
from armbench.skills import SkillLibrary

AGENT_IDS: Final[tuple[str, ...]] = ("A", "B", "B+S")
LLM_AGENT_IDS: Final[frozenset[str]] = frozenset({"B", "B+S"})
AGENTS: dict[str, Agent] = {ScriptedAgent.id: ScriptedAgent()}


def get_agent(
    agent_id: str,
    *,
    llm: LLMParams | None = None,
    provider: Provider | None = None,
    artifacts_dir: Path | None = None,
    skills_dir: Path = SKILLS_DIR,
) -> Agent:
    """``A`` needs nothing; ``B`` takes the LLM configuration (``configs/llm.yaml`` by default)
    and optionally a ready-made provider (the runner passes ``replay`` inside containers);
    ``B+S`` additionally loads the **frozen** skill library in ``skills_dir`` (D7: an unfrozen
    library is refused, so no run can grow it)."""
    if agent_id in AGENTS:
        return AGENTS[agent_id]
    if agent_id in LLM_AGENT_IDS:
        params = llm or load_llm_params()
        library = SkillLibrary.load(skills_dir, require_frozen=True) if agent_id == "B+S" else None
        return LLMAgent(
            provider or make_provider(params),
            params,
            agent_id=agent_id,
            artifacts_dir=artifacts_dir,
            library=library,
        )
    msg = f"unknown agent {agent_id!r}; known: {', '.join(AGENT_IDS)}"
    raise KeyError(msg)


__all__ = [
    "AGENTS",
    "AGENT_IDS",
    "LLM_AGENT_IDS",
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
