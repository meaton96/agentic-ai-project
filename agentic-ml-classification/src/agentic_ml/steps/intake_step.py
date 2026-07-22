"""
Phase 5 intake step: infer a DatasetSpec (target/task/id/group/time
columns) from raw schema facts plus an optional natural-language goal,
for use when the caller doesn't already know the target column. The
agent only ever sees pre-target column facts
(harness/intake.py::raw_schema_summary) — it never sees fitted metrics,
and its proposal is re-validated (harness/intake.py::
validate_dataset_spec_proposal) before anything downstream ever runs.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd

from agentic_ml.agent_runtime import ToolCallingAgent
from agentic_ml.events import emit_event
from agentic_ml.harness.intake import validate_dataset_spec_proposal
from agentic_ml.model_client import ModelClient
from agentic_ml.prompt_loader import load_prompt, prompt_source, resolve_prompt_override_dir
from agentic_ml.tools.intake_tool import make_raw_schema_tool


@dataclass
class IntakeStepResult:
    ok: bool
    dataset_spec_proposal: Optional[dict]
    validation_errors: list[str]
    raw_schema: Optional[dict]
    llm_raw_text: Optional[str]
    stopped_reason: str
    turns_used: int
    messages: list[dict]  # full conversation this agent had — see cli_common.make_transcript_writer


def run_intake_step(
    df: pd.DataFrame,
    goal_text: str,
    client: ModelClient,
    model: Optional[str] = None,
    max_turns: int = 4,
    trace_fn: Optional[Callable[[dict], None]] = None,
    on_event: Optional[Callable[[dict], None]] = None,
    prompt_override_dir: Optional[str] = None,
) -> IntakeStepResult:
    resolved_override_dir = resolve_prompt_override_dir(prompt_override_dir)
    prompt_src, prompt_path = prompt_source("intake", resolved_override_dir)
    system_prompt = load_prompt("intake", resolved_override_dir)
    emit_event(on_event, "intake", "prompt_loaded", {"source": prompt_src, "path": str(prompt_path)})
    emit_event(on_event, "intake", "agent_started", {"goal": goal_text})

    tool = make_raw_schema_tool(df)
    agent = ToolCallingAgent(
        model_client=client, tools=[tool], system_prompt=system_prompt,
        model=model, max_turns=max_turns,
    )

    user_message = (
        f"Goal: {goal_text}\n\nPropose a DatasetSpec for this dataset."
        if goal_text.strip()
        else "No goal was given. Propose the most plausible DatasetSpec for this dataset."
    )
    result = agent.run(user_message, trace_fn=trace_fn)

    raw_schema = None
    for entry in result.tool_call_log:
        if entry["tool"] == "get_raw_schema":
            raw_schema = entry["result"]
        emit_event(on_event, "intake", "tool_called", {"tool": entry["tool"], "result": entry["result"]})

    if result.final_text is None:
        emit_event(on_event, "intake", "proposal_rejected",
                    {"errors": ["agent never produced a proposal"]})
        return IntakeStepResult(
            ok=False, dataset_spec_proposal=None,
            validation_errors=["agent never produced a proposal"],
            raw_schema=raw_schema, llm_raw_text=None,
            stopped_reason=result.stopped_reason, turns_used=result.turns_used,
            messages=result.messages,
        )

    try:
        proposal = json.loads(result.final_text)
    except json.JSONDecodeError:
        emit_event(on_event, "intake", "proposal_rejected",
                    {"errors": ["agent's final response did not parse as JSON"]})
        return IntakeStepResult(
            ok=False, dataset_spec_proposal=None,
            validation_errors=["agent's final response did not parse as JSON"],
            raw_schema=raw_schema, llm_raw_text=result.final_text,
            stopped_reason=result.stopped_reason, turns_used=result.turns_used,
            messages=result.messages,
        )

    errors = validate_dataset_spec_proposal(df, proposal)
    emit_event(
        on_event, "intake", "proposal_validated" if not errors else "proposal_rejected",
        {"dataset_spec_proposal": proposal, "errors": errors},
    )
    return IntakeStepResult(
        ok=len(errors) == 0, dataset_spec_proposal=proposal, validation_errors=errors,
        raw_schema=raw_schema, llm_raw_text=result.final_text,
        stopped_reason=result.stopped_reason, turns_used=result.turns_used,
        messages=result.messages,
    )
