"""
Phase 2 profiler step, extracted from scripts/run_profiler_agent.py so
scripts/run_orchestrator.py drives the exact same logic instead of
duplicating it. Deterministic code (harness/profiler.py) computes every
fact; the LLM only narrates and recommends.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd

from agentic_ml.agent_runtime import ToolCallingAgent
from agentic_ml.events import emit_event
from agentic_ml.model_client import ModelClient
from agentic_ml.tools.profiler_tool import make_profiler_tool

SYSTEM_PROMPT = """You are the Profiler agent in a deterministic ML evaluation \
pipeline. You have exactly one tool: get_dataset_profile. Call it once to get \
factual information about the dataset — you cannot compute these facts yourself \
and must not guess or invent numbers.

After calling the tool, respond with ONLY valid JSON (no prose, no markdown \
fences) matching this schema:
{
  "summary": "<2-4 sentence plain-language summary of the dataset>",
  "recommended_split_strategy": "<copy exactly from the tool output>",
  "key_risks": ["<short bullet>", "..."],
  "recommended_next_steps": ["<short bullet>", "..."]
}

Do not contradict the tool's recommended_split_strategy or leakage_risk_flags — \
your job is to explain and contextualize them, not override them. If the tool \
output already lists leakage_risk_flags, they must appear in key_risks."""


@dataclass
class ProfilerStepResult:
    ok: bool
    deterministic_report: Optional[dict]
    llm_narrative: Optional[dict]
    llm_raw_text: Optional[str]
    stopped_reason: str
    turns_used: int
    messages: list[dict]  # full conversation this agent had — see cli_common.make_transcript_writer


def run_profiler_step(
    df: pd.DataFrame,
    target_column: str,
    client: ModelClient,
    model: Optional[str] = None,
    max_turns: int = 4,
    trace_fn: Optional[Callable[[dict], None]] = None,
    on_event: Optional[Callable[[dict], None]] = None,
) -> ProfilerStepResult:
    emit_event(on_event, "profiler", "agent_started", {"target_column": target_column})

    tool = make_profiler_tool(df, target_column)
    agent = ToolCallingAgent(
        model_client=client, tools=[tool], system_prompt=SYSTEM_PROMPT,
        model=model, max_turns=max_turns,
    )
    result = agent.run(
        "Profile this dataset and give your summary and recommendations.",
        trace_fn=trace_fn,
    )

    deterministic_report = None
    for entry in result.tool_call_log:
        emit_event(on_event, "profiler", "tool_called", {"tool": entry["tool"], "result": entry["result"]})
        if entry["tool"] == "get_dataset_profile" and deterministic_report is None:
            deterministic_report = entry["result"]

    llm_parsed = None
    if result.final_text:
        try:
            llm_parsed = json.loads(result.final_text)
        except json.JSONDecodeError:
            llm_parsed = None

    emit_event(on_event, "profiler", "profiler_report", {
        "ok": deterministic_report is not None,
        "recommended_split_strategy": (deterministic_report or {}).get("recommended_split_strategy"),
        "is_imbalanced": (deterministic_report or {}).get("is_imbalanced"),
        "leakage_risk_flags": (deterministic_report or {}).get("leakage_risk_flags"),
    })

    return ProfilerStepResult(
        ok=deterministic_report is not None,
        deterministic_report=deterministic_report,
        llm_narrative=llm_parsed,
        llm_raw_text=result.final_text,
        stopped_reason=result.stopped_reason,
        turns_used=result.turns_used,
        messages=result.messages,
    )
