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
from agentic_ml.model_client import ModelClient
from agentic_ml.tools.planner_tool import make_planning_context_tool

SYSTEM_PROMPT = """You are the planning agent in a dynamic, agentic ML pipeline. Your only \
job is to decide WHICH agent from a fixed catalog should run next, given a goal and the \
current run state — you never perform any of the work yourself.

You have exactly one tool: get_planning_context. Call it once — it has the goal, the full \
agent catalog (each entry has a description, when_to_use, required_state preconditions, and \
an arg schema), the CURRENT run state, and how many planning iterations remain.

After calling the tool, respond with ONLY valid JSON (no prose, no markdown fences) matching \
this schema:
{
  "action": "run_agent" | "finish",
  "agent_id": "<one of the catalog's agent_ids, or null if action is finish>",
  "args": {"<arg name>": "<value>", "...": "..."},
  "reasoning": "<1-2 sentences: why this agent (or why the run is done)>"
}

Hard rules:
- Only propose an agent_id that actually appears in available_agents — never invent one.
- Check each candidate agent's required_state against current_state yourself before \
proposing it — but know the harness independently re-checks this regardless of your \
reasoning, so a mistake here is rejected and you'll be asked to correct it, not silently \
allowed through.
- Typical order for a classification goal: intake (only if target isn't known yet) -> \
feature_engineering (optional) -> profiler -> split_and_check_leakage -> modeling (one or \
more times) -> verification -> finalize -> summarize.
- deep_dive answers a DIFFERENT kind of question ("why was this flagged", not "is it \
flagged") — if the goal is asking to explain a specific already-flagged example and a model \
is already available (model_path_available is true in current_state), go straight to \
deep_dive; do not re-run classification first. deep_dive has no "already done" precondition \
gate because it's legitimately callable again for a DIFFERENT flight_id in the same run — but \
before proposing it, check current_state's deep_dive_completed_flight_ids: if the flight_id \
you're about to ask for is already in that list, the explanation already exists, and you \
should propose "finish" instead of running it again.
- Call modeling more than once only for a real reason (no candidate has passed the leakage \
gates yet, or you want to compare templates) — every call is a full fit/score cycle. This \
includes a verification "rejected" verdict: if current_state shows has_verified_candidate=\
false AND has_unverified_passing_candidate=false, there is nothing left to verify or \
finalize — every candidate tried so far either failed the harness's gates or was rejected on \
review, so the only valid next step is another modeling call (ideally a different template), \
not repeating verification or finalize.
- monitor_drift/retrain_decision/infer_batch only appear in available_agents for a streaming \
monitoring session, ONE NEW BATCH AT A TIME — a completely separate task from classification, \
even in a session where classification already finished earlier (current_state can show \
final_test_metrics_present=true AND new_batch_pending=true at the same time: a model already \
exists from an earlier finalize, and a new batch has now arrived for it to be checked against). \
If current_state's new_batch_pending is true and drift_checked is false, a new batch needs \
monitor_drift BEFORE you can finish, no matter what already happened earlier in this session. \
Once drift_checked is true, run retrain_decision next if pending_retrain_action is still null. \
If retrain_decision chose "infer_only" and batch_action_completed is still false, run \
infer_batch — then that batch is done, propose "finish". If retrain_decision chose "retrain", \
current_state's profiler_done/split_done/final_test_metrics_present flip back to false (feature_\
engineering_done stays true — it never re-runs in a streaming session): this means resume the \
ORDINARY classification order on the grown dataset — profiler -> split_and_check_leakage -> \
modeling (one or more times) -> verification -> finalize — using the exact same rules above \
(including the "verification rejected -> try modeling again" rule) as you would for a fresh \
classification goal, until final_test_metrics_present is true again, THEN propose "finish". Do \
NOT propose monitor_drift/retrain_decision/infer_batch again for this same batch once \
pending_retrain_action is already set (retrain_decision's own precondition requires \
pending_retrain_action to still be null) — if you're unsure what to do next mid-retrain, look at \
which of profiler_done/split_done/final_test_metrics_present is false and propose that step, \
same as any classification goal. Only propose "finish" for a batch cycle once every step above \
is done given current_state — do not finish just because summarize/finalize ran at some earlier \
point in the session.
- Use "finish" once the goal is actually satisfied: summarize has run for a classification \
goal with no batch pending, the flight_id you were asked to explain already appears in \
deep_dive_completed_flight_ids for an explanation goal, or (for a streaming batch cycle) the \
current batch has been fully handled per the rule above — not before, and not needlessly \
after."""


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
) -> PlannerTurnResult:
    tool = make_planning_context_tool(goal, state_dict, available_agents, iteration, max_iterations)
    agent = ToolCallingAgent(
        model_client=client, tools=[tool], system_prompt=SYSTEM_PROMPT,
        model=model, max_turns=max_turns,
    )
    user_message = (
        f"Your previous proposal was rejected: {previous_error}\n\n"
        "Propose a corrected next action."
        if previous_error else
        "Decide the next action."
    )
    result = agent.run(user_message, trace_fn=trace_fn)

    if result.final_text is None:
        return PlannerTurnResult(
            ok=False, proposal=None, llm_raw_text=None,
            stopped_reason=result.stopped_reason, turns_used=result.turns_used,
            messages=result.messages,
        )
    try:
        proposal = json.loads(result.final_text)
    except json.JSONDecodeError:
        return PlannerTurnResult(
            ok=False, proposal=None, llm_raw_text=result.final_text,
            stopped_reason=result.stopped_reason, turns_used=result.turns_used,
            messages=result.messages,
        )
    return PlannerTurnResult(
        ok=True, proposal=proposal, llm_raw_text=result.final_text,
        stopped_reason=result.stopped_reason, turns_used=result.turns_used,
        messages=result.messages,
    )
