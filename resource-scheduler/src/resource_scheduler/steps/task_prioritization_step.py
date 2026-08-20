"""
Agent #2: Task Prioritization. Deterministic code
(environment/queue.py) computes the raw per-task signals; the LLM
combines them into a ranking and a final_score per task, using its own
judgment about the urgency/cost/availability tradeoff -- there is no
single "correct" ranking to gate against the way Load Monitor's flags
have one, so the gate here is structural (is this even a valid
permutation with complete scoring) rather than "does it match a
precomputed answer".

This is the first agent that hands its output directly to another agent
(Resource Allocation, agent #3, not yet built) via the Mailbox in
a2a/mailbox.py rather than only through a central orchestrator -- the
A2A path this project exists to test. The mailbox parameter is optional
so this step is fully testable/runnable standalone before agent #3
exists.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd

from resource_scheduler.a2a.mailbox import Mailbox
from resource_scheduler.agent_runtime import ToolCallingAgent
from resource_scheduler.environment.queue import is_ranking_score_consistent, validate_ranking_proposal
from resource_scheduler.events import emit_event
from resource_scheduler.model_client import ModelClient
from resource_scheduler.prompt_loader import load_prompt, prompt_source, resolve_prompt_override_dir
from resource_scheduler.tools.task_prioritization_tool import make_task_prioritization_tool


@dataclass
class TaskPrioritizationStepResult:
    ok: bool  # tool call succeeded and produced facts
    valid: bool  # proposal passed validate_ranking_proposal's hard structural gate
    validation_errors: list[str] = field(default_factory=list)
    score_inconsistent: bool = False  # soft flag: ranking order doesn't match descending final_score
    deterministic_facts: Optional[dict] = None
    llm_proposal: Optional[dict] = None
    llm_raw_text: Optional[str] = None
    sent_to_resource_allocation: bool = False
    stopped_reason: str = "completed"
    turns_used: int = 0
    messages: list[dict] = field(default_factory=list)


def run_task_prioritization_step(
    df: pd.DataFrame,
    variance_injected: dict[str, bool],
    client: ModelClient,
    queue_size: int = 8,
    snapshot_window: int = 200,
    model: Optional[str] = None,
    max_turns: int = 4,
    trace_fn: Optional[Callable[[dict], None]] = None,
    on_event: Optional[Callable[[dict], None]] = None,
    prompt_override_dir: Optional[str] = None,
    mailbox: Optional[Mailbox] = None,
) -> TaskPrioritizationStepResult:
    resolved_override_dir = resolve_prompt_override_dir(prompt_override_dir)
    prompt_src, prompt_path = prompt_source("task_prioritization", resolved_override_dir)
    system_prompt = load_prompt("task_prioritization", resolved_override_dir)
    emit_event(on_event, "task_prioritization", "prompt_loaded", {"source": prompt_src, "path": str(prompt_path)})
    emit_event(on_event, "task_prioritization", "agent_started", {"queue_size": queue_size})

    tool = make_task_prioritization_tool(df, variance_injected, queue_size=queue_size, snapshot_window=snapshot_window)
    agent = ToolCallingAgent(
        model_client=client, tools=[tool], system_prompt=system_prompt,
        model=model, max_turns=max_turns,
    )
    result = agent.run(
        "Rank the pending task queue and explain your scoring.",
        trace_fn=trace_fn,
    )

    deterministic_facts = None
    for entry in result.tool_call_log:
        emit_event(on_event, "task_prioritization", "tool_called", {"tool": entry["tool"], "result": entry["result"]})
        if entry["tool"] == "get_task_queue_profile" and deterministic_facts is None:
            deterministic_facts = entry["result"]

    llm_parsed = None
    if result.final_text:
        try:
            llm_parsed = json.loads(result.final_text)
        except json.JSONDecodeError:
            llm_parsed = None

    validation_errors: list[str] = []
    valid = False
    score_inconsistent = False
    if deterministic_facts is not None:
        pending_task_ids = [t["task_id"] for t in deterministic_facts["pending_tasks"]]
        if llm_parsed is None:
            validation_errors = ["LLM response was not valid JSON"]
        else:
            validation_errors = validate_ranking_proposal(pending_task_ids, llm_parsed)
            valid = not validation_errors
            if valid:
                score_inconsistent = not is_ranking_score_consistent(llm_parsed)

    sent = False
    if valid and mailbox is not None:
        mailbox.send(
            sender="task_prioritization",
            recipient="resource_allocation",
            message_type="task_ranking",
            payload={
                "ranked_task_ids": llm_parsed["ranked_task_ids"],
                "score_breakdown": llm_parsed["score_breakdown"],
                "reasoning": llm_parsed.get("reasoning", ""),
            },
        )
        sent = True
        emit_event(on_event, "task_prioritization", "a2a_sent", {"recipient": "resource_allocation"})

    emit_event(on_event, "task_prioritization", "task_prioritization_report", {
        "ok": deterministic_facts is not None,
        "valid": valid,
        "validation_errors": validation_errors,
        "score_inconsistent": score_inconsistent,
        "sent_to_resource_allocation": sent,
    })

    return TaskPrioritizationStepResult(
        ok=deterministic_facts is not None,
        valid=valid,
        validation_errors=validation_errors,
        score_inconsistent=score_inconsistent,
        deterministic_facts=deterministic_facts,
        llm_proposal=llm_parsed,
        llm_raw_text=result.final_text,
        sent_to_resource_allocation=sent,
        stopped_reason=result.stopped_reason,
        turns_used=result.turns_used,
        messages=result.messages,
    )
