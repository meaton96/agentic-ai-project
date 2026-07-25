"""
The planning agent: proposes WHICH catalog agent (agent_registry.py)
should run next, given the goal and current run state. It never
performs any work itself, and — critically — its proposal is not
trusted just because it parsed as JSON. orchestrator/dynamic_loop.py's
validate_plan() independently re-checks the proposed agent_id against
the real registry and the proposed preconditions against the real
RunStateSummary before anything executes. This function's `ok` field
means only "produced a well-formed, parseable proposal" — semantic
validity is a separate, deterministic step, the same separation
steps/verification_step.py and steps/intake_step.py already keep
between "did the LLM respond sensibly" and "is what it said true."
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Optional

from agentic_ml.agent_runtime import ToolCallingAgent
from agentic_ml.events import emit_event
from agentic_ml.mcp_facts.provider import LocalToolProvider, ToolProvider
from agentic_ml.model_client import ModelClient
from agentic_ml.prompt_loader import load_prompt, prompt_source, resolve_prompt_override_dir


@dataclass
class PlannerTurnResult:
    ok: bool  # True iff a well-formed proposal was parsed (NOT that it's valid to execute)
    proposal: Optional[dict]
    llm_raw_text: Optional[str]
    stopped_reason: str
    turns_used: int
    messages: list[dict]  # full conversation — see cli_common.make_transcript_writer


def run_planner_step(
    goal: str,
    state_dict: dict,
    available_agents: list[dict],
    iteration: int,
    max_iterations: int,
    client: ModelClient,
    model: Optional[str] = None,
    max_turns: int = 4,
    previous_error: Optional[str] = None,
    trace_fn: Optional[Callable[[dict], None]] = None,
    on_event: Optional[Callable[[dict], None]] = None,
    prompt_override_dir: Optional[str] = None,
    tool_provider: Optional[ToolProvider] = None,
) -> PlannerTurnResult:
    resolved_override_dir = resolve_prompt_override_dir(prompt_override_dir)
    prompt_src, prompt_path = prompt_source("planner", resolved_override_dir)
    system_prompt = load_prompt("planner", resolved_override_dir)
    emit_event(on_event, "planner", "prompt_loaded", {"source": prompt_src, "path": str(prompt_path)})
    emit_event(on_event, "planner", "agent_started", {"iteration": iteration, "previous_error": previous_error})

    provider = tool_provider or LocalToolProvider()
    tool = provider.make_planning_context_tool(goal, state_dict, available_agents, iteration, max_iterations)
    agent = ToolCallingAgent(
        model_client=client, tools=[tool], system_prompt=system_prompt,
        model=model, max_turns=max_turns,
    )
    user_message = (
        f"Your previous proposal was rejected: {previous_error}\n\n"
        "Propose a corrected next action."
        if previous_error else
        "Decide the next action."
    )
    result = agent.run(user_message, trace_fn=trace_fn)
    for entry in result.tool_call_log:
        emit_event(on_event, "planner", "tool_called", {"tool": entry["tool"], "result": entry["result"]})

    if result.final_text is None:
        emit_event(on_event, "planner", "planner_proposal",
                    {"ok": False, "proposal": None, "reason": "no final text"})
        return PlannerTurnResult(
            ok=False, proposal=None, llm_raw_text=None,
            stopped_reason=result.stopped_reason, turns_used=result.turns_used,
            messages=result.messages,
        )
    try:
        proposal = json.loads(result.final_text)
    except json.JSONDecodeError:
        emit_event(on_event, "planner", "planner_proposal",
                    {"ok": False, "proposal": None, "reason": "did not parse as JSON"})
        return PlannerTurnResult(
            ok=False, proposal=None, llm_raw_text=result.final_text,
            stopped_reason=result.stopped_reason, turns_used=result.turns_used,
            messages=result.messages,
        )
    emit_event(on_event, "planner", "planner_proposal", {"ok": True, "proposal": proposal})
    return PlannerTurnResult(
        ok=True, proposal=proposal, llm_raw_text=result.final_text,
        stopped_reason=result.stopped_reason, turns_used=result.turns_used,
        messages=result.messages,
    )
