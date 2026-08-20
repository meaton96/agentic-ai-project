"""
Agent #6: Human Oversight. Deterministic code
(environment/oversight.py::build_oversight_review_bundle) assembles the
facts; the LLM's verdict is the point of this agent -- there is no
"correct" answer to check it against, unlike every other gate in this
project. parse_oversight_verdict enforces only one rule: an unparseable
or malformed response must degrade to "flagged", never silently pass as
"approved".

Reads Optimization's policy_update_proposal from its own mailbox
(message_type "policy_update_proposal") -- the third A2A hop, no
orchestrator in between.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Optional

from resource_scheduler.a2a.mailbox import Mailbox
from resource_scheduler.agent_runtime import ToolCallingAgent
from resource_scheduler.environment.oversight import build_oversight_review_bundle, parse_oversight_verdict
from resource_scheduler.events import emit_event
from resource_scheduler.model_client import ModelClient
from resource_scheduler.prompt_loader import load_prompt, prompt_source, resolve_prompt_override_dir
from resource_scheduler.tools.oversight_tool import make_oversight_tool


@dataclass
class OversightStepResult:
    ok: bool  # a proposal was available to review
    verdict: Optional[str] = None
    concerns: list[str] = field(default_factory=list)
    reasoning: str = ""
    review_bundle: Optional[dict] = None
    llm_raw_text: Optional[str] = None
    proposal_source: Optional[str] = None
    stopped_reason: str = "completed"
    turns_used: int = 0
    messages: list[dict] = field(default_factory=list)


def run_oversight_step(
    client: ModelClient,
    mailbox: Mailbox,
    model: Optional[str] = None,
    max_turns: int = 4,
    trace_fn: Optional[Callable[[dict], None]] = None,
    on_event: Optional[Callable[[dict], None]] = None,
    prompt_override_dir: Optional[str] = None,
) -> OversightStepResult:
    inbox = mailbox.inbox_for("human_oversight", message_type="policy_update_proposal")
    if not inbox:
        emit_event(on_event, "human_oversight", "no_proposal_available", {})
        return OversightStepResult(ok=False, stopped_reason="no_proposal_available")

    message = inbox[-1]  # MVP assumes one proposal in flight at a time
    proposal = message.payload
    emit_event(on_event, "human_oversight", "proposal_received", {"sender": message.sender})

    resolved_override_dir = resolve_prompt_override_dir(prompt_override_dir)
    prompt_src, prompt_path = prompt_source("human_oversight", resolved_override_dir)
    system_prompt = load_prompt("human_oversight", resolved_override_dir)
    emit_event(on_event, "human_oversight", "prompt_loaded", {"source": prompt_src, "path": str(prompt_path)})

    review_bundle = build_oversight_review_bundle(proposal, proposal.get("underlying_evidence") or {})
    tool = make_oversight_tool(review_bundle)
    agent = ToolCallingAgent(
        model_client=client, tools=[tool], system_prompt=system_prompt,
        model=model, max_turns=max_turns,
    )
    result = agent.run(
        "Review the pending policy update proposal and issue your verdict.",
        trace_fn=trace_fn,
    )

    for entry in result.tool_call_log:
        emit_event(on_event, "human_oversight", "tool_called", {"tool": entry["tool"], "result": entry["result"]})

    llm_parsed = None
    if result.final_text:
        try:
            llm_parsed = json.loads(result.final_text)
        except json.JSONDecodeError:
            llm_parsed = None

    verdict = parse_oversight_verdict(llm_parsed)

    emit_event(on_event, "human_oversight", "oversight_report", {
        "verdict": verdict["verdict"], "n_concerns": len(verdict["concerns"]),
    })

    return OversightStepResult(
        ok=True,
        verdict=verdict["verdict"],
        concerns=verdict["concerns"],
        reasoning=verdict["reasoning"],
        review_bundle=review_bundle,
        llm_raw_text=result.final_text,
        proposal_source=message.sender,
        stopped_reason=result.stopped_reason,
        turns_used=result.turns_used,
        messages=result.messages,
    )
