"""Agents that act on the primitives: the scripted baseline A and the LLM program agents
B (no memory), B+S (frozen skill library), C (energy block in the prompt) and C+S (both)."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from armbench.agents.base import Agent, AgentError, AgentTrace
from armbench.agents.llm import LLMAgent, extract_program, prefetch
from armbench.agents.scripted import ScriptedAgent, detect_all
from armbench.energy import EnergyReference
from armbench.llm import LLMParams, Provider, load_llm_params, make_provider
from armbench.paths import ENERGY_REF_DIR, SKILLS_DIR
from armbench.skills import SkillLibrary

AGENT_IDS: Final[tuple[str, ...]] = ("A", "B", "B+S", "C", "C+S")
LLM_AGENT_IDS: Final[frozenset[str]] = frozenset({"B", "B+S", "C", "C+S"})
SKILL_AGENT_IDS: Final[frozenset[str]] = frozenset({"B+S", "C+S"})
"""Configurations that load the frozen skill library (D7)."""
ENERGY_AGENT_IDS: Final[frozenset[str]] = frozenset({"C", "C+S"})
"""Configurations whose prompt carries the energy block with baseline A's Wh (D8)."""
AGENTS: dict[str, Agent] = {ScriptedAgent.id: ScriptedAgent()}


def get_agent(  # noqa: PLR0913 - keyword-only inputs, all optional
    agent_id: str,
    *,
    llm: LLMParams | None = None,
    provider: Provider | None = None,
    artifacts_dir: Path | None = None,
    skills_dir: Path = SKILLS_DIR,
    energy_ref_dir: Path = ENERGY_REF_DIR,
) -> Agent:
    """``A`` needs nothing; ``B`` takes the LLM configuration (``configs/llm.yaml`` by default)
    and optionally a ready-made provider (the runner passes ``replay`` inside containers);
    ``B+S`` and ``C+S`` additionally load the **frozen** skill library in ``skills_dir`` (D7: an
    unfrozen library is refused, so no run can grow it); ``C`` and ``C+S`` load the energy
    reference in ``energy_ref_dir`` (D8: baseline A's Wh per task, hashed into every trace)."""
    if agent_id in AGENTS:
        return AGENTS[agent_id]
    if agent_id in LLM_AGENT_IDS:
        params = llm or load_llm_params()
        library = (
            SkillLibrary.load(skills_dir, require_frozen=True)
            if agent_id in SKILL_AGENT_IDS
            else None
        )
        energy_ref = EnergyReference.load(energy_ref_dir) if agent_id in ENERGY_AGENT_IDS else None
        return LLMAgent(
            provider or make_provider(params),
            params,
            agent_id=agent_id,
            artifacts_dir=artifacts_dir,
            library=library,
            energy_ref=energy_ref,
        )
    msg = f"unknown agent {agent_id!r}; known: {', '.join(AGENT_IDS)}"
    raise KeyError(msg)


__all__ = [
    "AGENTS",
    "AGENT_IDS",
    "ENERGY_AGENT_IDS",
    "LLM_AGENT_IDS",
    "SKILL_AGENT_IDS",
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
