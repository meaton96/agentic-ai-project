"""
Agent #3: Resource Allocation. First agent that reads from the A2A
mailbox (a2a/mailbox.py) instead of only being handed a dataframe by
its caller -- it consumes Task Prioritization's ranking directly, no
orchestrator in between.

The LLM proposes machine/slice assignments; environment/allocation.py's
check_constraints is the real gate, independent of anything the LLM
claims -- an assignment to a Maintenance machine or a saturated slice is
rejected by the environment regardless of the agent's stated rationale,
mirroring the "propose, harness decides" split the sibling ML pipeline
uses for its modeling candidates. Every proposed assignment produces a
traceable event either way (accepted or environment-rejected).

This file also holds run_reroute_validation_step, the receiving end of
the SECOND A2A hop: Failure Recovery -> Resource Allocation. That path
is deliberately NOT an LLM call -- Failure Recovery already did the
reasoning about where to reroute; Resource Allocation's role there is
purely to be the deterministic gate-keeper, reusing check_constraints
and build_allocation_events rather than a parallel implementation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd

from resource_scheduler.a2a.mailbox import Mailbox
from resource_scheduler.agent_runtime import ToolCallingAgent
from resource_scheduler.environment.allocation import (
    DEFAULT_SLICE_CAPACITY,
    build_allocation_events,
    check_constraints,
    compute_slice_load,
    identify_risky_assignments,
    validate_allocation_structure,
)
from resource_scheduler.environment.state import compute_snapshot
from resource_scheduler.events import emit_event
from resource_scheduler.model_client import ModelClient
from resource_scheduler.prompt_loader import load_prompt, prompt_source, resolve_prompt_override_dir
from resource_scheduler.tools.resource_allocation_tool import build_allocation_context_fact, make_resource_allocation_tool


@dataclass
class ResourceAllocationStepResult:
    ok: bool  # a ranking was available and deterministic facts were built
    valid: bool = False  # proposal passed validate_allocation_structure's hard structural gate
    validation_errors: list[str] = field(default_factory=list)
    accepted_assignments: list[dict] = field(default_factory=list)
    environment_rejected: list[dict] = field(default_factory=list)  # LLM proposed, environment vetoed
    agent_rejected: list[dict] = field(default_factory=list)  # LLM itself declined to assign
    events: list[dict] = field(default_factory=list)
    deterministic_facts: Optional[dict] = None
    llm_proposal: Optional[dict] = None
    llm_raw_text: Optional[str] = None
    ranking_source: Optional[str] = None
    sent_risky_decisions_to_human_oversight: bool = False
    stopped_reason: str = "completed"
    turns_used: int = 0
    messages: list[dict] = field(default_factory=list)


def run_resource_allocation_step(
    df: pd.DataFrame,
    variance_injected: dict[str, bool],
    client: ModelClient,
    mailbox: Mailbox,
    snapshot_window: int = 200,
    slice_capacity: int = DEFAULT_SLICE_CAPACITY,
    model: Optional[str] = None,
    max_turns: int = 4,
    trace_fn: Optional[Callable[[dict], None]] = None,
    on_event: Optional[Callable[[dict], None]] = None,
    prompt_override_dir: Optional[str] = None,
) -> ResourceAllocationStepResult:
    # Filtered by type: resource_allocation's inbox can also carry
    # reroute_request messages from Failure Recovery (see
    # run_reroute_validation_step below) -- popping unfiltered here
    # would silently discard those before they're ever handled.
    inbox = mailbox.inbox_for("resource_allocation", message_type="task_ranking")
    if not inbox:
        emit_event(on_event, "resource_allocation", "no_ranking_available", {})
        return ResourceAllocationStepResult(ok=False, stopped_reason="no_ranking_available")

    # MVP assumes one ranking in flight at a time; if several arrived,
    # only the most recent is actionable -- earlier ones are stale by
    # the time this step runs.
    ranking_message = inbox[-1]
    ranked_task_ids = ranking_message.payload["ranked_task_ids"]
    score_breakdown = ranking_message.payload.get("score_breakdown", {})
    emit_event(on_event, "resource_allocation", "ranking_received", {
        "sender": ranking_message.sender, "n_tasks": len(ranked_task_ids),
    })

    resolved_override_dir = resolve_prompt_override_dir(prompt_override_dir)
    prompt_src, prompt_path = prompt_source("resource_allocation", resolved_override_dir)
    system_prompt = load_prompt("resource_allocation", resolved_override_dir)
    emit_event(on_event, "resource_allocation", "prompt_loaded", {"source": prompt_src, "path": str(prompt_path)})

    deterministic_facts = build_allocation_context_fact(
        df, variance_injected, ranked_task_ids, score_breakdown,
        snapshot_window=snapshot_window, slice_capacity=slice_capacity,
    )
    tool = make_resource_allocation_tool(deterministic_facts)
    agent = ToolCallingAgent(
        model_client=client, tools=[tool], system_prompt=system_prompt,
        model=model, max_turns=max_turns,
    )
    result = agent.run(
        "Assign each ranked task to a machine and network slice, or reject it with a reason.",
        trace_fn=trace_fn,
    )

    for entry in result.tool_call_log:
        emit_event(on_event, "resource_allocation", "tool_called", {"tool": entry["tool"], "result": entry["result"]})

    llm_parsed = None
    if result.final_text:
        try:
            llm_parsed = json.loads(result.final_text)
        except json.JSONDecodeError:
            llm_parsed = None

    validation_errors: list[str] = []
    valid = False
    accepted: list[dict] = []
    environment_rejected: list[dict] = []
    agent_rejected: list[dict] = []
    events: list[dict] = []

    if llm_parsed is None:
        validation_errors = ["LLM response was not valid JSON"]
    else:
        validation_errors = validate_allocation_structure(ranked_task_ids, llm_parsed)
        valid = not validation_errors
        if valid:
            assignments = llm_parsed.get("assignments", [])
            agent_rejected = llm_parsed.get("rejected", [])

            machine_status = {m["machine_id"]: m["status"] for m in deterministic_facts["available_machines"]}
            slice_current_load = {s["slice_id"]: s["current_load"] for s in deterministic_facts["available_slices"]}
            violations = check_constraints(assignments, machine_status, slice_current_load, slice_capacity=slice_capacity)
            violations_by_task = {v["task_id"]: v for v in violations}

            for a in assignments:
                if a.get("task_id") in violations_by_task:
                    environment_rejected.append({"task_id": a.get("task_id"), "proposed": a,
                                                  "violation": violations_by_task[a["task_id"]]})
                else:
                    accepted.append(a)

            trigger_signal_by_task = {
                a.get("task_id"): f"final_score={(score_breakdown.get(a.get('task_id')) or {}).get('final_score')}"
                for a in assignments
            }
            events = build_allocation_events(assignments, violations_by_task, trigger_signal_by_task)

    sent_risky = False
    if accepted:
        machine_status_for_risk = {m["machine_id"]: m["status"] for m in deterministic_facts["available_machines"]}
        risky = identify_risky_assignments(accepted, machine_status_for_risk)
        if risky:
            mailbox.send(
                sender="resource_allocation", recipient="human_oversight",
                message_type="risky_decision",
                payload={"source": "resource_allocation", "risky_decisions": risky},
            )
            sent_risky = True
            emit_event(on_event, "resource_allocation", "a2a_sent", {"recipient": "human_oversight", "n_risky": len(risky)})

    emit_event(on_event, "resource_allocation", "resource_allocation_report", {
        "valid": valid,
        "n_accepted": len(accepted),
        "n_environment_rejected": len(environment_rejected),
        "n_agent_rejected": len(agent_rejected),
        "sent_risky_decisions_to_human_oversight": sent_risky,
    })

    return ResourceAllocationStepResult(
        ok=True,
        valid=valid,
        validation_errors=validation_errors,
        accepted_assignments=accepted,
        environment_rejected=environment_rejected,
        agent_rejected=agent_rejected,
        events=events,
        deterministic_facts=deterministic_facts,
        llm_proposal=llm_parsed,
        llm_raw_text=result.final_text,
        ranking_source=ranking_message.sender,
        sent_risky_decisions_to_human_oversight=sent_risky,
        stopped_reason=result.stopped_reason,
        turns_used=result.turns_used,
        messages=result.messages,
    )


@dataclass
class RerouteValidationResult:
    ok: bool
    accepted_reroutes: list[dict] = field(default_factory=list)
    environment_rejected: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    reroute_source: Optional[str] = None
    sent_risky_decisions_to_human_oversight: bool = False
    stopped_reason: str = "completed"


def run_reroute_validation_step(
    df: pd.DataFrame,
    variance_injected: dict[str, bool],
    mailbox: Mailbox,
    snapshot_window: int = 200,
    slice_capacity: int = DEFAULT_SLICE_CAPACITY,
    on_event: Optional[Callable[[dict], None]] = None,
) -> RerouteValidationResult:
    """Purely deterministic -- no LLM call, no ModelClient needed. Consumes
    a reroute_request message from Failure Recovery and re-validates each
    proposed reroute against the SAME check_constraints gate Resource
    Allocation's own LLM-driven proposals go through in
    run_resource_allocation_step above. Failure Recovery already did the
    reasoning about where to reroute; this step's only job is to be the
    deterministic gate-keeper, same role Resource Allocation already
    plays for its own agent's output -- not a second negotiation."""
    inbox = mailbox.inbox_for("resource_allocation", message_type="reroute_request")
    if not inbox:
        emit_event(on_event, "resource_allocation", "no_reroute_request_available", {})
        return RerouteValidationResult(ok=False, stopped_reason="no_reroute_request_available")

    message = inbox[-1]  # MVP assumes one reroute request in flight at a time
    reroute_proposals = message.payload["reroute_proposals"]
    emit_event(on_event, "resource_allocation", "reroute_request_received", {
        "sender": message.sender, "n_reroutes": len(reroute_proposals),
    })

    snapshot = compute_snapshot(df, variance_injected, window=snapshot_window)
    machine_status = {m["machine_id"]: m["status"] for m in snapshot["machines"]}
    # Same fix as tools/resource_allocation_tool.py's build_allocation_context_fact:
    # current_load must be scaled to the batch actually being decided
    # (here, the reroute proposals), not the broad monitoring snapshot_window
    # -- otherwise every reroute is rejected as "over capacity" regardless
    # of what's proposed, confirmed live.
    slice_current_load = compute_slice_load(df, window=max(len(reroute_proposals), 1))

    # Reshape into the assignment shape check_constraints/build_allocation_events
    # expect, carrying the original reasoning through as "rationale" so an
    # accepted reroute's event still records WHY, not just WHAT.
    as_assignments = [
        {
            "task_id": r["task_id"],
            "machine_id": r.get("new_machine_id"),
            "network_slice_id": r.get("new_network_slice_id"),
            "rationale": r.get("reasoning"),
        }
        for r in reroute_proposals
    ]
    violations = check_constraints(as_assignments, machine_status, slice_current_load, slice_capacity=slice_capacity)
    violations_by_task = {v["task_id"]: v for v in violations}

    accepted: list[dict] = []
    environment_rejected: list[dict] = []
    for a in as_assignments:
        if a["task_id"] in violations_by_task:
            environment_rejected.append({"task_id": a["task_id"], "proposed": a,
                                          "violation": violations_by_task[a["task_id"]]})
        else:
            accepted.append(a)

    trigger_signal_by_task = {a["task_id"]: "reroute_request (Failure Recovery, incident-triggered)" for a in as_assignments}
    events = build_allocation_events(as_assignments, violations_by_task, trigger_signal_by_task)

    sent_risky = False
    if accepted:
        # sender is the reroute's true origin (Failure Recovery, from the
        # original reroute_request message), not "resource_allocation" --
        # this code re-validates the reroute, but the decision to place
        # it here is Failure Recovery's, same attribution
        # build_allocation_events already gives it via trigger_signal_by_task.
        risky = identify_risky_assignments(accepted, machine_status)
        if risky:
            mailbox.send(
                sender=message.sender, recipient="human_oversight",
                message_type="risky_decision",
                payload={"source": message.sender, "risky_decisions": risky},
            )
            sent_risky = True
            emit_event(on_event, "resource_allocation", "a2a_sent", {"recipient": "human_oversight", "n_risky": len(risky)})

    emit_event(on_event, "resource_allocation", "reroute_validation_report", {
        "n_accepted": len(accepted), "n_environment_rejected": len(environment_rejected),
        "sent_risky_decisions_to_human_oversight": sent_risky,
    })

    return RerouteValidationResult(
        ok=True,
        accepted_reroutes=accepted,
        environment_rejected=environment_rejected,
        events=events,
        reroute_source=message.sender,
        sent_risky_decisions_to_human_oversight=sent_risky,
    )
