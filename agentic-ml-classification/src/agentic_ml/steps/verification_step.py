"""
Phase 4 verification step. Runs only on a candidate that has ALREADY
passed both of modeling_step.py's deterministic leakage gates
(label_permutation_test + check_suspicious_feature_correlation) — this
is a second, independent audit pass, not a first line of defense.

Design constraint, mirroring every other agent in this pipeline: the
harness's deterministic gates are the actual enforcement. This agent
can only make the outcome MORE conservative than "the gates passed" —
"flagged" (proceed, but the concern is recorded for a human) or
"rejected" (block promotion, same practical effect as a failed gate).
It can never grant approval to a candidate that failed a deterministic
check, because it is never shown one.
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

VALID_VERDICTS = {"approved", "flagged", "rejected"}


@dataclass
class VerificationStepResult:
    ok: bool  # True iff the agent produced a well-formed, parseable verdict
    verdict: Optional[str]  # "approved" | "flagged" | "rejected" | None if unparseable
    concerns: list[str]
    reasoning: Optional[str]
    llm_raw_text: Optional[str]
    stopped_reason: str
    turns_used: int
    messages: list[dict]  # full conversation this agent had — see cli_common.make_transcript_writer


def run_verification_step(
    bundle: dict,
    client: ModelClient,
    model: Optional[str] = None,
    max_turns: int = 4,
    trace_fn: Optional[Callable[[dict], None]] = None,
    on_event: Optional[Callable[[dict], None]] = None,
    prompt_override_dir: Optional[str] = None,
    tool_provider: Optional[ToolProvider] = None,
) -> VerificationStepResult:
    resolved_override_dir = resolve_prompt_override_dir(prompt_override_dir)
    prompt_src, prompt_path = prompt_source("verification", resolved_override_dir)
    system_prompt = load_prompt("verification", resolved_override_dir)
    emit_event(on_event, "verification", "prompt_loaded", {"source": prompt_src, "path": str(prompt_path)})
    emit_event(on_event, "verification", "agent_started", {"candidate_id": bundle.get("candidate_id")})

    provider = tool_provider or LocalToolProvider()
    tool = provider.make_review_bundle_tool(bundle)
    agent = ToolCallingAgent(
        model_client=client, tools=[tool], system_prompt=system_prompt,
        model=model, max_turns=max_turns,
    )
    result = agent.run(
        f"Review candidate {bundle.get('candidate_id')!r} and give your verdict.",
        trace_fn=trace_fn,
    )
    for entry in result.tool_call_log:
        emit_event(on_event, "verification", "tool_called", {"tool": entry["tool"], "result": entry["result"]})

    def unparseable(reason: str) -> VerificationStepResult:
        # Defensive default: a formatting glitch from the LLM shouldn't silently
        # promote a candidate (that would be treating unparseable as "approved"),
        # nor should it hard-block a candidate that already passed two deterministic
        # gates over an LLM output-formatting issue. "flagged" is the conservative
        # middle ground — proceed, but make sure a human sees why.
        emit_event(on_event, "verification", "verification_verdict", {
            "candidate_id": bundle.get("candidate_id"), "verdict": "flagged", "concerns": [reason],
            "reasoning": None, "unparseable": True,
        })
        return VerificationStepResult(
            ok=False, verdict="flagged", concerns=[reason], reasoning=None,
            llm_raw_text=result.final_text,
            stopped_reason=result.stopped_reason, turns_used=result.turns_used,
            messages=result.messages,
        )

    if result.final_text is None:
        return unparseable("verification agent never produced a final verdict")

    try:
        parsed = json.loads(result.final_text)
    except json.JSONDecodeError:
        return unparseable("verification agent's response did not parse as JSON")

    verdict = parsed.get("verdict")
    if verdict not in VALID_VERDICTS:
        return unparseable(f"verification agent returned an invalid verdict: {verdict!r}")

    emit_event(on_event, "verification", "verification_verdict", {
        "candidate_id": bundle.get("candidate_id"), "verdict": verdict,
        "concerns": parsed.get("concerns") or [], "reasoning": parsed.get("reasoning"),
        "unparseable": False,
    })
    return VerificationStepResult(
        ok=True,
        verdict=verdict,
        concerns=parsed.get("concerns") or [],
        reasoning=parsed.get("reasoning"),
        llm_raw_text=result.final_text,
        stopped_reason=result.stopped_reason,
        turns_used=result.turns_used,
        messages=result.messages,
    )
