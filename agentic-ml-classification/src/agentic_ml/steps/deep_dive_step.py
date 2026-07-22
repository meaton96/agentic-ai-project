"""
Deep-dive agent: explains why one already-flagged flight was flagged.
This answers a different analysis question than the rest of this
pipeline ("why", not "is it") — it runs on-demand against one flight
that a completed classification run already flagged, not as a
mandatory step in that run.

Same trust-boundary pattern as every other agent here: one bundled tool
(tools/deep_dive_tool.py) exposes deterministic evidence the agent
cannot alter, and the agent's only job is to synthesize a hedged,
plain-language hypothesis from it — never to compute a number itself.
Mirrors steps/verification_step.py's structure, including the
conservative default on anything unparseable: a malformed LLM response
degrades to a deterministic template built from the same evidence,
rather than failing the whole deep-dive.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd

from agentic_ml.agent_runtime import ToolCallingAgent
from agentic_ml.events import emit_event
from agentic_ml.model_client import ModelClient
from agentic_ml.prompt_loader import load_prompt, prompt_source, resolve_prompt_override_dir
from agentic_ml.tools.deep_dive_tool import gather_deep_dive_evidence, make_deep_dive_evidence_tool

VALID_CONFIDENCE = {"high", "medium", "low"}


@dataclass
class DeepDiveStepResult:
    ok: bool  # True iff the agent produced a well-formed, parseable hypothesis
    hypothesis: Optional[str]
    agrees_with_localization: Optional[bool]
    confidence: Optional[str]
    used_template_fallback: bool
    evidence: Optional[dict]
    llm_raw_text: Optional[str]
    stopped_reason: str
    turns_used: int
    messages: list[dict]  # full conversation this agent had — see cli_common.make_transcript_writer


def _template_result(evidence: dict) -> dict:
    """Deterministic fallback hypothesis, built from the same evidence
    the LLM would have seen. Used when the LLM is unavailable or its
    response doesn't parse — the conservative default this repo uses
    everywhere rather than failing outright."""
    p = evidence["p_maintenance"]
    loc = evidence["localization"]
    attr = [a["channel"] for a in evidence["attribution_top"][:3] if a["prob_drop"] > 0.001]

    if not loc:
        return {
            "hypothesis": (
                f"Flagged (p={p:.2f}). Model leaned on {', '.join(attr) or 'no single channel'}; "
                f"no cross-cylinder imbalance localized in the raw signal."
            ),
            "agrees_with_localization": None,
            "confidence": "low",
        }

    f = loc[0]
    agree = f["channel"] in attr
    hypothesis = (
        f"Flagged (p={p:.2f}). {f['channel']} (cylinder {f['cylinder']}) ran "
        f"{f['deviation']:+.0f}° vs its siblings, worst during {f['worst_phase']} "
        f"(~{f['worst_segment_start_s']}s)"
        + (", corroborated by the other temp group" if f.get("corroborated_by_other_group") else "")
        + (f"; model attribution agrees ({f['channel']} among top drivers)."
           if agree else
           f"; note the model's top drivers ({', '.join(attr) or 'none'}) do not center on "
           f"this cylinder, so treat as a lead, not a confirmed cause.")
    )
    return {"hypothesis": hypothesis, "agrees_with_localization": agree, "confidence": "low"}


def run_deep_dive_step(
    flight_df: pd.DataFrame,
    feature_row,
    pipeline,
    feature_columns: list[str],
    background: pd.Series,
    client: ModelClient,
    model: Optional[str] = None,
    max_turns: int = 4,
    sample_hz: float = 1.0,
    loc_thresholds: Optional[dict] = None,
    trace_fn: Optional[Callable[[dict], None]] = None,
    on_event: Optional[Callable[[dict], None]] = None,
    prompt_override_dir: Optional[str] = None,
) -> DeepDiveStepResult:
    resolved_override_dir = resolve_prompt_override_dir(prompt_override_dir)
    prompt_src, prompt_path = prompt_source("deep_dive", resolved_override_dir)
    system_prompt = load_prompt("deep_dive", resolved_override_dir)
    emit_event(on_event, "deep_dive", "prompt_loaded", {"source": prompt_src, "path": str(prompt_path)})
    emit_event(on_event, "deep_dive", "agent_started", {})

    tool = make_deep_dive_evidence_tool(
        flight_df, feature_row, pipeline, feature_columns, background,
        sample_hz=sample_hz, loc_thresholds=loc_thresholds,
    )
    agent = ToolCallingAgent(
        model_client=client, tools=[tool], system_prompt=system_prompt,
        model=model, max_turns=max_turns,
    )
    result = agent.run(
        "Explain why this flight was flagged.", trace_fn=trace_fn,
    )

    evidence = None
    for entry in result.tool_call_log:
        emit_event(on_event, "deep_dive", "tool_called", {"tool": entry["tool"], "result": entry["result"]})
        if entry["tool"] == "get_flight_deep_dive_evidence":
            evidence = entry["result"]
            break
    # tool always runs deterministically regardless of the LLM's behavior after
    # it — compute it directly if the agent never actually called the tool, so
    # a template fallback is still possible.
    if evidence is None:
        evidence = gather_deep_dive_evidence(
            flight_df, feature_row, pipeline, feature_columns, background,
            sample_hz=sample_hz, loc_thresholds=loc_thresholds,
        )

    def fallback(reason: str) -> DeepDiveStepResult:
        template = _template_result(evidence)
        emit_event(on_event, "deep_dive", "deep_dive_hypothesis", {
            "hypothesis": template["hypothesis"], "confidence": template["confidence"],
            "used_template_fallback": True, "reason": reason,
        })
        return DeepDiveStepResult(
            ok=False, hypothesis=template["hypothesis"],
            agrees_with_localization=template["agrees_with_localization"],
            confidence=template["confidence"], used_template_fallback=True,
            evidence=evidence, llm_raw_text=result.final_text,
            stopped_reason=result.stopped_reason, turns_used=result.turns_used,
            messages=result.messages,
        )

    if result.final_text is None:
        return fallback("deep-dive agent never produced a final hypothesis")

    try:
        parsed = json.loads(result.final_text)
    except json.JSONDecodeError:
        return fallback("deep-dive agent's response did not parse as JSON")

    hypothesis = parsed.get("hypothesis")
    confidence = parsed.get("confidence")
    if not hypothesis or confidence not in VALID_CONFIDENCE:
        return fallback("deep-dive agent's response was missing a hypothesis or a valid confidence")

    emit_event(on_event, "deep_dive", "deep_dive_hypothesis", {
        "hypothesis": hypothesis, "confidence": confidence, "used_template_fallback": False,
    })
    return DeepDiveStepResult(
        ok=True, hypothesis=hypothesis,
        agrees_with_localization=parsed.get("agrees_with_localization"),
        confidence=confidence, used_template_fallback=False,
        evidence=evidence, llm_raw_text=result.final_text,
        stopped_reason=result.stopped_reason, turns_used=result.turns_used,
        messages=result.messages,
    )
