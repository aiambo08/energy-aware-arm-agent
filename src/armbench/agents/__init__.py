"""Agents that act on the primitives: the scripted baseline now, LLM agents from F6 on."""

from armbench.agents.base import Agent, AgentTrace
from armbench.agents.scripted import ScriptedAgent, detect_all

AGENTS: dict[str, Agent] = {ScriptedAgent.id: ScriptedAgent()}


def get_agent(agent_id: str) -> Agent:
    try:
        return AGENTS[agent_id]
    except KeyError:
        msg = f"unknown agent {agent_id!r}; known: {', '.join(AGENTS)}"
        raise KeyError(msg) from None


__all__ = ["AGENTS", "Agent", "AgentTrace", "ScriptedAgent", "detect_all", "get_agent"]
