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
from agentic_ml.model_client import ModelClient
from agentic_ml.tools.verification_tool import make_review_bundle_tool

VALID_VERDICTS = {"approved", "flagged", "rejected"}

SYSTEM_PROMPT = """You are the Verification agent in a deterministic ML evaluation \
pipeline. You review ONE candidate that has ALREADY passed the harness's automated \
gates (a sandbox-built pipeline, a label-permutation leakage test, and a feature-\
correlation leakage check scoped to its selected columns) — this is a second, \
independent audit, not a first line of defense.

You have exactly one tool: get_candidate_review_bundle. Call it once — it has \
everything you need: the template used, the agent's config and explanation, \
validation metrics with confidence intervals, both leakage check results, and \
relevant dataset facts from the profiler. You cannot see anything else and must \
not invent numbers.

After calling the tool, respond with ONLY valid JSON (no prose, no markdown fences) \
matching this schema:
{
  "verdict": "approved" | "flagged" | "rejected",
  "concerns": ["<short bullet>", "..."],
  "reasoning": "<2-4 sentences>"
}

What to check:
- Does the stated explanation actually match the template and config chosen? \
(e.g. the explanation gives a reason unrelated to what the config actually does)
- Do the validation metrics look plausible for this dataset, given its imbalance \
and leakage risk flags? A suspiciously perfect score (e.g. > 0.99 ROC-AUC with no \
obvious reason the dataset would be that separable) is worth a "flagged" verdict \
even though it already passed both automated leakage gates — those gates catch \
specific known failure modes, not "this looks too good to be true."
- Does the config use any column the profiler's leakage_risk_flags mention?

Hard rules:
- Use "rejected" only for a genuine, specific concern you can name — not vague \
unease. A rejected candidate is blocked from promotion entirely.
- Use "flagged" for a real but non-blocking concern worth a human's attention.
- Use "approved" if you find nothing worth flagging.
- You cannot approve or unblock a candidate that failed an automated gate — you \
are never shown one, so this situation should not arise."""


@dataclass
class VerificationStepResult:
    ok: bool  # True iff the agent produced a well-formed, parseable verdict
    verdict: Optional[str]  # "approved" | "flagged" | "rejected" | None if unparseable
    concerns: list[str]
    reasoning: Optional[str]
    llm_raw_text: Optional[str]
    stopped_reason: str
    turns_used: int


def run_verification_step(
    bundle: dict,
    client: ModelClient,
    model: Optional[str] = None,
    max_turns: int = 4,
    trace_fn: Optional[Callable[[dict], None]] = None,
) -> VerificationStepResult:
    tool = make_review_bundle_tool(bundle)
    agent = ToolCallingAgent(
        model_client=client, tools=[tool], system_prompt=SYSTEM_PROMPT,
        model=model, max_turns=max_turns,
    )
    result = agent.run(
        f"Review candidate {bundle.get('candidate_id')!r} and give your verdict.",
        trace_fn=trace_fn,
    )

    def unparseable(reason: str) -> VerificationStepResult:
        # Defensive default: a formatting glitch from the LLM shouldn't silently
        # promote a candidate (that would be treating unparseable as "approved"),
        # nor should it hard-block a candidate that already passed two deterministic
        # gates over an LLM output-formatting issue. "flagged" is the conservative
        # middle ground — proceed, but make sure a human sees why.
        return VerificationStepResult(
            ok=False, verdict="flagged", concerns=[reason], reasoning=None,
            llm_raw_text=result.final_text,
            stopped_reason=result.stopped_reason, turns_used=result.turns_used,
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

    return VerificationStepResult(
        ok=True,
        verdict=verdict,
        concerns=parsed.get("concerns") or [],
        reasoning=parsed.get("reasoning"),
        llm_raw_text=result.final_text,
        stopped_reason=result.stopped_reason,
        turns_used=result.turns_used,
    )
