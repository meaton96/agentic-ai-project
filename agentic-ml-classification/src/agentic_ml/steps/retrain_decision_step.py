"""
Phase 9 retrain-decision step. Runs once per incoming batch, after
monitor_drift (harness/drift.py, deterministic) has already computed
the drift summary. This agent ONLY chooses among three actions — it
does not compute drift itself and cannot see anything the
get_monitoring_context tool doesn't hand it.

This is not a new trust mechanism: it's the same "agents propose, the
harness decides" discipline as every other agent here. "retrain" does
not itself retrain anything — it only sets a proposed action that
orchestrator/dynamic_loop.py's execute_agent_step acts on, the same way
a modeling candidate is only ever a proposal until the harness's
deterministic gates pass it.

Same conservative-default discipline as steps/verification_step.py: an
unparseable or invalid response degrades to "infer_only", the
conservative middle ground — never silently "no_action" (could mask
real drift going unaddressed) and never silently "retrain" (the one
action here capable of mutating the served model, and expensive: a
full fit/score/verify cycle).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Optional

from agentic_ml.agent_runtime import ToolCallingAgent
from agentic_ml.events import emit_event
from agentic_ml.model_client import ModelClient
from agentic_ml.tools.retrain_decision_tool import make_monitoring_context_tool

VALID_ACTIONS = {"no_action", "infer_only", "retrain"}

SYSTEM_PROMPT = """You are the Retrain-Decision agent in a streaming ML monitoring \
pipeline. A new batch of data has just arrived and already been checked for \
distribution drift against the data the current model was trained on. Your only job \
is to decide what to do about it.

You have exactly one tool: get_monitoring_context. Call it once — it has the drift \
summary (per-feature standardized shift, an aggregate score, and the batch's own \
performance metrics), the current model's test metrics as of its last (re)training, \
and how many batches/examples have accumulated since then. You cannot see anything \
else and must not invent numbers.

After calling the tool, respond with ONLY valid JSON (no prose, no markdown fences) \
matching this schema:
{
  "action": "no_action" | "infer_only" | "retrain",
  "reasoning": "<1-3 sentences>"
}

What each action means:
- "no_action": nothing worth doing yet — drift is negligible and little data has \
accumulated since the last retrain. The batch is still recorded either way; you're \
just not spending compute scoring or retraining on it.
- "infer_only": score this batch with the current model and log the result, but don't \
retrain — drift/accumulation isn't enough yet to justify a full retrain cycle.
- "retrain": trigger a full classification cycle (profiler -> split -> modeling -> \
verification -> finalize) on all data accumulated so far. Every retrain is a full \
fit/score/verify cycle — only choose this for a real reason: meaningful drift (a high \
aggregate shift score, or several features past the threshold) or enough new data has \
accumulated since the last retrain that the current model is likely stale.

Hard rules:
- Do not choose "retrain" out of caution alone — it is the most expensive action and \
the only one that can change the served model.
- Do not choose "no_action" if the drift summary shows real, meaningful shift — at \
minimum score the batch with "infer_only" so the drift is on record."""


@dataclass
class RetrainDecisionStepResult:
    ok: bool  # True iff the agent produced a well-formed, valid action
    action: str  # always one of VALID_ACTIONS — degrades to "infer_only" if unparseable
    reasoning: Optional[str]
    llm_raw_text: Optional[str]
    stopped_reason: str
    turns_used: int
    messages: list[dict]  # full conversation this agent had — see cli_common.make_transcript_writer


def run_retrain_decision_step(
    monitoring_context: dict,
    client: ModelClient,
    model: Optional[str] = None,
    max_turns: int = 4,
    trace_fn: Optional[Callable[[dict], None]] = None,
    on_event: Optional[Callable[[dict], None]] = None,
) -> RetrainDecisionStepResult:
    emit_event(on_event, "retrain_decision", "agent_started", {})

    tool = make_monitoring_context_tool(monitoring_context)
    agent = ToolCallingAgent(
        model_client=client, tools=[tool], system_prompt=SYSTEM_PROMPT,
        model=model, max_turns=max_turns,
    )
    result = agent.run(
        "A new batch has arrived and been checked for drift. Decide the next action.",
        trace_fn=trace_fn,
    )
    for entry in result.tool_call_log:
        emit_event(on_event, "retrain_decision", "tool_called", {"tool": entry["tool"], "result": entry["result"]})

    def degraded(reason: str) -> RetrainDecisionStepResult:
        emit_event(on_event, "retrain_decision", "retrain_decision", {
            "action": "infer_only", "reasoning": reason, "unparseable": True,
        })
        return RetrainDecisionStepResult(
            ok=False, action="infer_only", reasoning=reason,
            llm_raw_text=result.final_text,
            stopped_reason=result.stopped_reason, turns_used=result.turns_used,
            messages=result.messages,
        )

    if result.final_text is None:
        return degraded("retrain-decision agent never produced a final response")

    try:
        parsed = json.loads(result.final_text)
    except json.JSONDecodeError:
        return degraded("retrain-decision agent's response did not parse as JSON")

    action = parsed.get("action")
    if action not in VALID_ACTIONS:
        return degraded(f"retrain-decision agent returned an invalid action: {action!r}")

    emit_event(on_event, "retrain_decision", "retrain_decision", {
        "action": action, "reasoning": parsed.get("reasoning"), "unparseable": False,
    })
    return RetrainDecisionStepResult(
        ok=True, action=action, reasoning=parsed.get("reasoning"),
        llm_raw_text=result.final_text,
        stopped_reason=result.stopped_reason, turns_used=result.turns_used,
        messages=result.messages,
    )
