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
from agentic_ml.harness.intake import validate_dataset_spec_proposal
from agentic_ml.model_client import ModelClient
from agentic_ml.tools.intake_tool import make_raw_schema_tool

SYSTEM_PROMPT = """You are the Intake agent in a deterministic ML evaluation \
pipeline. Your only job is to propose a DatasetSpec: which column is the \
prediction target, and which columns are identifiers, group keys, or \
timestamps that must be excluded from modeling features.

You have exactly one tool: get_raw_schema. Call it once — you cannot see \
column names, dtypes, or values otherwise and must not invent them.

This pipeline supports classification, binary or multiclass: the target \
column must have between 2 and 20 distinct non-null values.

After calling the tool, respond with ONLY valid JSON (no prose, no markdown \
fences) matching this schema:
{
  "target_column": "<column name>",
  "task": "binary_classification or multiclass_classification",
  "id_columns": ["<column name>", "..."],
  "group_column": "<column name or null>",
  "time_column": "<column name or null>",
  "positive_label": "<value or null — for binary targets, the class that means the positive/event outcome; null for multiclass>",
  "reasoning": "<1-3 sentences: why this column is the target, and why any group/time/id columns were chosen>"
}

Hard rules:
- target_column must be a real column name from get_raw_schema's output, \
with somewhere between 2 and 20 unique values reported (you cannot compute \
this exactly yourself, but avoid picking an obviously continuous or \
high-cardinality column).
- id_columns should include any column whose name/values look like an \
identifier (name_suggests_id) and any column that is clearly not \
predictive (e.g. free-text names).
- group_column should be set only if some column's values plausibly repeat \
per real-world entity (customer, patient, machine, session).
- time_column should be set only if some column looks like a timestamp \
(looks_like_datetime).
- If the stated goal names a specific outcome, prefer the column that \
matches it; otherwise pick the most plausible outcome column from the \
schema alone."""


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
) -> IntakeStepResult:
    tool = make_raw_schema_tool(df)
    agent = ToolCallingAgent(
        model_client=client, tools=[tool], system_prompt=SYSTEM_PROMPT,
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

    if result.final_text is None:
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
        return IntakeStepResult(
            ok=False, dataset_spec_proposal=None,
            validation_errors=["agent's final response did not parse as JSON"],
            raw_schema=raw_schema, llm_raw_text=result.final_text,
            stopped_reason=result.stopped_reason, turns_used=result.turns_used,
            messages=result.messages,
        )

    errors = validate_dataset_spec_proposal(df, proposal)
    return IntakeStepResult(
        ok=len(errors) == 0, dataset_spec_proposal=proposal, validation_errors=errors,
        raw_schema=raw_schema, llm_raw_text=result.final_text,
        stopped_reason=result.stopped_reason, turns_used=result.turns_used,
        messages=result.messages,
    )
