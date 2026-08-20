"""
Agent #1: Load Monitor. Deterministic code (environment/state.py)
computes every fact and every flag; the LLM only narrates. Mirrors the
shape of agentic_ml.steps.profiler_step.run_profiler_step in the sibling
agentic-ml-classification project: one tool, one call, structured JSON
response, deterministic report is authoritative regardless of what the
LLM says.

This is a read-only agent -- it never proposes an action, never mutates
environment state, and has no negotiation partner. It's step 1 of the
build order because everything downstream (Task Prioritization, Resource
Allocation, ...) needs a trustworthy read of current state first.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd

from resource_scheduler.agent_runtime import ToolCallingAgent
from resource_scheduler.events import emit_event
from resource_scheduler.model_client import ModelClient
from resource_scheduler.prompt_loader import load_prompt, prompt_source, resolve_prompt_override_dir
from resource_scheduler.tools.load_monitor_tool import make_load_monitor_tool


@dataclass
class LoadMonitorStepResult:
    ok: bool
    deterministic_report: Optional[dict]
    llm_narrative: Optional[dict]
    llm_raw_text: Optional[str]
    flag_mismatch: bool  # True if the agent's reported flags differ from the deterministic ones
    stopped_reason: str
    turns_used: int
    messages: list[dict]  # full conversation -- see cli_common.make_transcript_writer


def _flags_match(reported: object, authoritative: list[dict]) -> bool:
    """Order-independent comparison on (scope, id, severity) triples --
    the fields that actually matter for "did the agent narrate the
    right thing", not full dict equality (which would also fail on
    harmless float formatting differences in value/threshold)."""
    if not isinstance(reported, list):
        return False
    try:
        reported_keys = {(f["scope"], f["id"], f["severity"]) for f in reported}
    except (TypeError, KeyError):
        return False
    authoritative_keys = {(f["scope"], f["id"], f["severity"]) for f in authoritative}
    return reported_keys == authoritative_keys


def run_load_monitor_step(
    df: pd.DataFrame,
    variance_injected: dict[str, bool],
    client: ModelClient,
    window: int = 200,
    model: Optional[str] = None,
    max_turns: int = 4,
    trace_fn: Optional[Callable[[dict], None]] = None,
    on_event: Optional[Callable[[dict], None]] = None,
    prompt_override_dir: Optional[str] = None,
) -> LoadMonitorStepResult:
    resolved_override_dir = resolve_prompt_override_dir(prompt_override_dir)
    prompt_src, prompt_path = prompt_source("load_monitor", resolved_override_dir)
    system_prompt = load_prompt("load_monitor", resolved_override_dir)
    emit_event(on_event, "load_monitor", "prompt_loaded", {"source": prompt_src, "path": str(prompt_path)})
    emit_event(on_event, "load_monitor", "agent_started", {"window": window})

    tool = make_load_monitor_tool(df, variance_injected, window=window)
    agent = ToolCallingAgent(
        model_client=client, tools=[tool], system_prompt=system_prompt,
        model=model, max_turns=max_turns,
    )
    result = agent.run(
        "Check current resource state and report any flags.",
        trace_fn=trace_fn,
    )

    deterministic_report = None
    for entry in result.tool_call_log:
        emit_event(on_event, "load_monitor", "tool_called", {"tool": entry["tool"], "result": entry["result"]})
        if entry["tool"] == "get_resource_snapshot" and deterministic_report is None:
            deterministic_report = entry["result"]

    llm_parsed = None
    if result.final_text:
        try:
            llm_parsed = json.loads(result.final_text)
        except json.JSONDecodeError:
            llm_parsed = None

    flag_mismatch = False
    if deterministic_report is not None:
        reported_flags = (llm_parsed or {}).get("flags")
        flag_mismatch = not _flags_match(reported_flags, deterministic_report["flags"])

    emit_event(on_event, "load_monitor", "load_monitor_report", {
        "ok": deterministic_report is not None,
        "n_flags": len(deterministic_report["flags"]) if deterministic_report else None,
        "flag_mismatch": flag_mismatch,
    })

    return LoadMonitorStepResult(
        ok=deterministic_report is not None,
        deterministic_report=deterministic_report,
        llm_narrative=llm_parsed,
        llm_raw_text=result.final_text,
        flag_mismatch=flag_mismatch,
        stopped_reason=result.stopped_reason,
        turns_used=result.turns_used,
        messages=result.messages,
    )
