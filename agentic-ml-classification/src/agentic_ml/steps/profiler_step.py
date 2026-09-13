"""
Phase 2 profiler step, extracted from scripts/run_profiler_agent.py so
scripts/run_orchestrator.py drives the exact same logic instead of
duplicating it. Deterministic code (harness/profiler.py) computes every
fact; the LLM only narrates and recommends.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Callable, Optional

import pandas as pd

from agentic_ml.agent_runtime import ToolCallingAgent
from agentic_ml.events import emit_event
from agentic_ml.mcp_facts.provider import LocalToolProvider, ToolProvider
from agentic_ml.model_client import ModelClient
from agentic_ml.prompt_loader import load_prompt, prompt_source, resolve_prompt_override_dir


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
    group_column: Optional[str] = None,
    time_column: Optional[str] = None,
    model: Optional[str] = None,
    max_turns: int = 4,
    trace_fn: Optional[Callable[[dict], None]] = None,
    on_event: Optional[Callable[[dict], None]] = None,
    prompt_override_dir: Optional[str] = None,
    tool_provider: Optional[ToolProvider] = None,
) -> ProfilerStepResult:
    resolved_override_dir = resolve_prompt_override_dir(prompt_override_dir)
    prompt_src, prompt_path = prompt_source("profiler", resolved_override_dir)
    system_prompt = load_prompt("profiler", resolved_override_dir)
    emit_event(on_event, "profiler", "prompt_loaded", {"source": prompt_src, "path": str(prompt_path)})
    emit_event(on_event, "profiler", "agent_started", {"target_column": target_column})

    provider = tool_provider or LocalToolProvider()
    tool = provider.make_profiler_tool(df, target_column, group_column, time_column)
    agent = ToolCallingAgent(
        model_client=client, tools=[tool], system_prompt=system_prompt,
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

    decision = record_profiler_narrative(deterministic_report, result.final_text, on_event=on_event)
    return replace(
        decision, stopped_reason=result.stopped_reason, turns_used=result.turns_used, messages=result.messages,
    )


def record_profiler_narrative(
    deterministic_report: Optional[dict],
    final_text: Optional[str],
    on_event: Optional[Callable[[dict], None]] = None,
) -> ProfilerStepResult:
    """The deterministic half of the profiler step. The narrative has no
    decision impact — it's parsed best-effort and carried for the record;
    `ok` depends only on the harness-computed report existing. Shared by
    run_profiler_step and gate_adapters.profiler_and_split_decide."""
    llm_parsed = None
    if final_text:
        try:
            llm_parsed = json.loads(final_text)
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
        llm_raw_text=final_text,
        stopped_reason="",
        turns_used=0,
        messages=[],
    )
