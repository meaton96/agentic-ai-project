"""
Tool exposing the planning context to the planner agent: the goal, the
full agent catalog (already filtered to this run's capabilities — see
orchestrator/agent_registry.py::list_agent_summaries), the CURRENT run
state, and how many planning iterations remain. Bundled into one tool
call, same reasoning as every other bundled-evidence tool in this
pipeline: the planner has no real choice about what to inspect before
deciding — it always needs the full picture.
"""
from __future__ import annotations

from agentic_ml.agent_runtime import Tool


def make_planning_context_tool(
    goal: str, state_dict: dict, available_agents: list[dict],
    iteration: int, max_iterations: int,
) -> Tool:
    def handler() -> dict:
        return {
            "goal": goal,
            "current_state": state_dict,
            "available_agents": available_agents,
            "iteration": iteration,
            "max_iterations": max_iterations,
        }

    return Tool(
        name="get_planning_context",
        description=(
            "Get everything needed to decide the next action: the goal, the full catalog "
            "of available agents (each with a description, when_to_use, required_state "
            "preconditions, and arg schema), the current run state, and how many planning "
            "iterations remain. Call this once — you cannot see any of this otherwise."
        ),
        parameters={"type": "object", "properties": {}, "required": []},
        handler=handler,
    )
