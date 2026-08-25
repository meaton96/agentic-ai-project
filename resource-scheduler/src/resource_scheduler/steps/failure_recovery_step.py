"""
Agent #4: Failure Recovery. Deterministic code (environment/incidents.py)
detects incidents and computes which committed tasks are affected; the
LLM proposes where to reroute each one. Sends validated reroute
proposals to Resource Allocation's mailbox as a "reroute_request"
message -- the second A2A hop this project tests, re-using the exact
same constraint gate Resource Allocation already applies to its own
proposals (see steps/resource_allocation_step.py::run_reroute_validation_step)
rather than a separate one.

Short-circuits before ever calling an LLM when there's nothing to do:
no incidents detected, or incidents detected but nothing committed sits
on the affected machines. Calling a model to reroute an empty list
would just be wasted latency/cost for a guaranteed no-op response.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd

from resource_scheduler.a2a.mailbox import Mailbox
from resource_scheduler.agent_runtime import ToolCallingAgent
from resource_scheduler.environment.incidents import reroutes_avoid_source_machine, validate_recovery_proposal
from resource_scheduler.events import emit_event
from resource_scheduler.model_client import ModelClient
from resource_scheduler.prompt_loader import load_prompt, prompt_source, resolve_prompt_override_dir
from resource_scheduler.tools.failure_recovery_tool import build_incident_fact, make_failure_recovery_tool


@dataclass
class FailureRecoveryStepResult:
    ok: bool
    valid: bool = False
    validation_errors: list[str] = field(default_factory=list)
    reroute_avoids_source: bool = True  # soft flag; see environment.incidents.reroutes_avoid_source_machine
    incidents: list[dict] = field(default_factory=list)
    affected_tasks: list[dict] = field(default_factory=list)
    reroute_proposals: list[dict] = field(default_factory=list)
    sent_to_resource_allocation: bool = False
    deterministic_facts: Optional[dict] = None
    llm_proposal: Optional[dict] = None
    llm_raw_text: Optional[str] = None
    stopped_reason: str = "completed"
    turns_used: int = 0
    messages: list[dict] = field(default_factory=list)


def run_failure_recovery_step(
    df: pd.DataFrame,
    variance_injected: dict[str, bool],
    client: ModelClient,
    mailbox: Mailbox,
    committed_assignments: list[dict],
    before_row: int,
    after_row: int,
    snapshot_window: int = 200,
    model: Optional[str] = None,
    max_turns: int = 4,
    trace_fn: Optional[Callable[[dict], None]] = None,
    on_event: Optional[Callable[[dict], None]] = None,
    prompt_override_dir: Optional[str] = None,
) -> FailureRecoveryStepResult:
    deterministic_facts = build_incident_fact(
        df, variance_injected, committed_assignments, before_row, after_row, window=snapshot_window,
    )
    emit_event(on_event, "failure_recovery", "incident_scan", {
        "n_incidents": len(deterministic_facts["incidents"]),
        "n_affected": len(deterministic_facts["affected_tasks"]),
    })

    if not deterministic_facts["incidents"]:
        emit_event(on_event, "failure_recovery", "no_incidents_detected", {})
        return FailureRecoveryStepResult(
            ok=True, valid=True, deterministic_facts=deterministic_facts,
            stopped_reason="no_incidents_detected",
        )
    if not deterministic_facts["affected_tasks"]:
        emit_event(on_event, "failure_recovery", "no_affected_tasks", {})
        return FailureRecoveryStepResult(
            ok=True, valid=True, incidents=deterministic_facts["incidents"],
            deterministic_facts=deterministic_facts, stopped_reason="no_affected_tasks",
        )

    resolved_override_dir = resolve_prompt_override_dir(prompt_override_dir)
    prompt_src, prompt_path = prompt_source("failure_recovery", resolved_override_dir)
    system_prompt = load_prompt("failure_recovery", resolved_override_dir)
    emit_event(on_event, "failure_recovery", "prompt_loaded", {"source": prompt_src, "path": str(prompt_path)})

    tool = make_failure_recovery_tool(deterministic_facts)
    agent = ToolCallingAgent(
        model_client=client, tools=[tool], system_prompt=system_prompt,
        model=model, max_turns=max_turns,
    )
    result = agent.run(
        "Review the incident report and propose a reroute for every affected task.",
        trace_fn=trace_fn,
    )

    for entry in result.tool_call_log:
        emit_event(on_event, "failure_recovery", "tool_called", {"tool": entry["tool"], "result": entry["result"]})

    llm_parsed = None
    if result.final_text:
        try:
            llm_parsed = json.loads(result.final_text)
        except json.JSONDecodeError:
            llm_parsed = None

    affected_task_ids = [t["task_id"] for t in deterministic_facts["affected_tasks"]]
    validation_errors: list[str] = []
    valid = False
    reroutes: list[dict] = []
    reroute_avoids_source = True
    if llm_parsed is None:
        validation_errors = ["LLM response was not valid JSON"]
    else:
        validation_errors = validate_recovery_proposal(affected_task_ids, llm_parsed)
        valid = not validation_errors
        if valid:
            reroutes = llm_parsed.get("reroute_proposals", [])
            reroute_avoids_source = reroutes_avoid_source_machine(deterministic_facts["affected_tasks"], llm_parsed)

    sent = False
    if valid and reroutes:
        mailbox.send(
            sender="failure_recovery", recipient="resource_allocation",
            message_type="reroute_request",
            payload={"reroute_proposals": reroutes, "incidents": deterministic_facts["incidents"]},
        )
        sent = True
        emit_event(on_event, "failure_recovery", "a2a_sent", {"recipient": "resource_allocation"})

    emit_event(on_event, "failure_recovery", "failure_recovery_report", {
        "valid": valid, "n_reroutes": len(reroutes),
        "reroute_avoids_source": reroute_avoids_source, "sent_to_resource_allocation": sent,
    })

    return FailureRecoveryStepResult(
        ok=True,
        valid=valid,
        validation_errors=validation_errors,
        reroute_avoids_source=reroute_avoids_source,
        incidents=deterministic_facts["incidents"],
        affected_tasks=deterministic_facts["affected_tasks"],
        reroute_proposals=reroutes,
        sent_to_resource_allocation=sent,
        deterministic_facts=deterministic_facts,
        llm_proposal=llm_parsed,
        llm_raw_text=result.final_text,
        stopped_reason=result.stopped_reason,
        turns_used=result.turns_used,
        messages=result.messages,
    )
