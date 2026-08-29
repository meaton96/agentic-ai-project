"""
Agent #5: Optimization. Deterministic code (environment/policy_evidence.py)
aggregates outcomes across recent runs; the LLM proposes parameter
adjustments based on that evidence. This agent NEVER applies its own
proposal -- steps/oversight_step.py's Human Oversight agent is the only
thing that can approve a change, and even then nothing in this MVP
actually writes the new value anywhere (that's a real "apply" mechanism
for a future iteration, out of scope here -- see the module docstring
in environment/policy_evidence.py for why this agent doesn't need the
task table at all, only run history).

Short-circuits with zero LLM calls when there's no run history to
reason about (n_runs_scanned == 0) -- there's nothing evidence-based to
propose from an empty history, and asking an LLM to invent a
recommendation from nothing would be exactly the ungrounded reasoning
this project's gates exist to prevent.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from resource_scheduler.a2a.mailbox import Mailbox
from resource_scheduler.agent_runtime import ToolCallingAgent
from resource_scheduler.environment.policy_evidence import validate_policy_proposal
from resource_scheduler.events import emit_event
from resource_scheduler.model_client import ModelClient
from resource_scheduler.prompt_loader import load_prompt, prompt_source, resolve_prompt_override_dir
from resource_scheduler.tools.optimization_tool import build_policy_evidence_fact, make_optimization_tool


@dataclass
class OptimizationStepResult:
    ok: bool  # evidence was available to reason about (n_runs_scanned > 0)
    valid: bool = False
    validation_errors: list[str] = field(default_factory=list)
    evidence: Optional[dict] = None
    llm_proposal: Optional[dict] = None
    llm_raw_text: Optional[str] = None
    sent_to_human_oversight: bool = False
    stopped_reason: str = "completed"
    turns_used: int = 0
    messages: list[dict] = field(default_factory=list)


def run_optimization_step(
    client: ModelClient,
    mailbox: Mailbox,
    n_runs: int = 10,
    runs_dir: Optional[Path] = None,
    model: Optional[str] = None,
    max_turns: int = 4,
    trace_fn: Optional[Callable[[dict], None]] = None,
    on_event: Optional[Callable[[dict], None]] = None,
    prompt_override_dir: Optional[str] = None,
) -> OptimizationStepResult:
    evidence = build_policy_evidence_fact(n_runs, runs_dir=runs_dir)
    emit_event(on_event, "optimization", "evidence_collected", {
        "n_runs_scanned": evidence["n_runs_scanned"],
        "n_run_dirs_considered": evidence["n_run_dirs_considered"],
    })

    if not evidence["n_runs_scanned"]:
        emit_event(on_event, "optimization", "no_run_history_available", {})
        return OptimizationStepResult(ok=False, evidence=evidence, stopped_reason="no_run_history_available")

    resolved_override_dir = resolve_prompt_override_dir(prompt_override_dir)
    prompt_src, prompt_path = prompt_source("optimization", resolved_override_dir)
    system_prompt = load_prompt("optimization", resolved_override_dir)
    emit_event(on_event, "optimization", "prompt_loaded", {"source": prompt_src, "path": str(prompt_path)})

    tool = make_optimization_tool(evidence)
    agent = ToolCallingAgent(
        model_client=client, tools=[tool], system_prompt=system_prompt,
        model=model, max_turns=max_turns,
    )
    result = agent.run(
        "Review recent run evidence and propose any policy adjustments, or none if unwarranted.",
        trace_fn=trace_fn,
    )

    for entry in result.tool_call_log:
        emit_event(on_event, "optimization", "tool_called", {"tool": entry["tool"], "result": entry["result"]})

    llm_parsed = None
    if result.final_text:
        try:
            llm_parsed = json.loads(result.final_text)
        except json.JSONDecodeError:
            llm_parsed = None

    validation_errors: list[str] = []
    valid = False
    if llm_parsed is None:
        validation_errors = ["LLM response was not valid JSON"]
    else:
        validation_errors = validate_policy_proposal(llm_parsed)
        valid = not validation_errors

    sent = False
    if valid:
        mailbox.send(
            sender="optimization", recipient="human_oversight",
            message_type="policy_update_proposal",
            payload={
                "policy_updates": llm_parsed["policy_updates"],
                "evidence": llm_parsed["evidence"],
                "recommend_apply": llm_parsed["recommend_apply"],
                # The actual deterministic evidence dict, not just the LLM's
                # prose claim about it -- Human Oversight must be able to
                # check the proposal against the real numbers, not just
                # trust the proposer's own characterization of them.
                "underlying_evidence": evidence,
            },
        )
        sent = True
        emit_event(on_event, "optimization", "a2a_sent", {"recipient": "human_oversight"})

    emit_event(on_event, "optimization", "optimization_report", {
        "valid": valid, "sent_to_human_oversight": sent,
    })

    return OptimizationStepResult(
        ok=True,
        valid=valid,
        validation_errors=validation_errors,
        evidence=evidence,
        llm_proposal=llm_parsed,
        llm_raw_text=result.final_text,
        sent_to_human_oversight=sent,
        stopped_reason=result.stopped_reason,
        turns_used=result.turns_used,
        messages=result.messages,
    )
