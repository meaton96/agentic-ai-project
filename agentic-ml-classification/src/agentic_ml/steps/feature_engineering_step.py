"""
Feature engineering step: the agent proposes structural changes to the
feature set — columns to drop entirely, and new derived columns built
from a vetted, deterministic operation catalog
(harness/feature_engineering.py) — before the profiler and modeling
agent ever see the data. Every operation in that catalog is stateless
and row-wise (no fitted statistics), which is what makes it safe to
apply before the train/val/test split even exists, the same way the
profiler's own descriptive facts are computed dataset-wide.

What this agent does NOT decide: imputation values, scaling, encoding
— those stay inside the modeling templates' ColumnTransformer, fit
only on the training fold, unchanged by this step.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd

from agentic_ml.agent_runtime import ToolCallingAgent
from agentic_ml.events import emit_event
from agentic_ml.harness.feature_engineering import apply_feature_op, validate_feature_proposal
from agentic_ml.mcp_facts.provider import LocalToolProvider, ToolProvider
from agentic_ml.model_client import ModelClient
from agentic_ml.prompt_loader import load_prompt, prompt_source, resolve_prompt_override_dir


@dataclass
class FeatureEngineeringStepResult:
    ok: bool
    df: Optional[pd.DataFrame]  # augmented dataframe, only present if ok
    drop_columns: list[str]
    new_columns: list[str]
    applied_ops: list[dict]
    explanation: Optional[str]
    errors: list[str] = field(default_factory=list)
    stopped_reason: str = ""
    turns_used: int = 0
    messages: list[dict] = field(default_factory=list)


def run_feature_engineering_step(
    df: pd.DataFrame,
    target_column: str,
    client: ModelClient,
    group_column: Optional[str] = None,
    time_column: Optional[str] = None,
    model: Optional[str] = None,
    max_turns: int = 6,
    trace_fn: Optional[Callable[[dict], None]] = None,
    on_event: Optional[Callable[[dict], None]] = None,
    prompt_override_dir: Optional[str] = None,
    tool_provider: Optional[ToolProvider] = None,
    previous_error: Optional[str] = None,
) -> FeatureEngineeringStepResult:
    resolved_override_dir = resolve_prompt_override_dir(prompt_override_dir)
    prompt_src, prompt_path = prompt_source("feature_engineering", resolved_override_dir)
    system_prompt = load_prompt("feature_engineering", resolved_override_dir)
    emit_event(on_event, "feature_engineering", "prompt_loaded", {"source": prompt_src, "path": str(prompt_path)})
    emit_event(on_event, "feature_engineering", "agent_started",
               {"target_column": target_column, "previous_error": previous_error})

    provider = tool_provider or LocalToolProvider()
    tools = [
        provider.make_profiler_tool(df, target_column, group_column, time_column),
        provider.make_list_feature_ops_tool(),
    ]
    # Every call here is otherwise a fresh conversation with no memory of a
    # prior rejected proposal — at temperature=0.0 (ModelClient's default)
    # a rejected-but-plausible proposal can repeat identically forever
    # (real incident: 13 identical rejections). previous_error alone isn't
    # enough to fix that: if the rejection reason is the same every time
    # (as it was here), the whole prompt is byte-identical to the prior
    # attempt, and temperature=0.0 GUARANTEES the same output again
    # regardless of what the feedback says. A nonzero temperature on retry
    # gives the same feedback an actual chance to land differently — this
    # is a mitigation, not a guarantee; dynamic_loop.py's bounded-retry
    # fallback to a deterministic zero-changes result is what actually
    # guarantees this step can't get the run stuck.
    agent = ToolCallingAgent(
        model_client=client, tools=tools, system_prompt=system_prompt,
        model=model, max_turns=max_turns, temperature=0.4 if previous_error else 0.0,
    )
    user_message = (
        f"Your previous proposal was rejected: {previous_error}\n\n"
        "Propose a corrected drop_columns/derived_features for this dataset. "
        "If you don't have a clear alternative, it is valid and expected to "
        "propose no changes at all: drop_columns: [] and derived_features: []."
        if previous_error else
        "Propose drop_columns and derived_features for this dataset."
    )
    result = agent.run(user_message, trace_fn=trace_fn)

    profile_report = None
    for entry in result.tool_call_log:
        if entry["tool"] == "get_dataset_profile":
            profile_report = entry["result"]
        emit_event(on_event, "feature_engineering", "tool_called",
                    {"tool": entry["tool"], "result": entry["result"]})

    def fail(errors: list[str]) -> FeatureEngineeringStepResult:
        emit_event(on_event, "feature_engineering", "proposal_rejected", {"errors": errors})
        return FeatureEngineeringStepResult(
            ok=False, df=None, drop_columns=[], new_columns=[], applied_ops=[],
            explanation=None, errors=errors,
            stopped_reason=result.stopped_reason, turns_used=result.turns_used,
            messages=result.messages,
        )

    if result.final_text is None:
        return fail(["agent never produced a final proposal"])

    try:
        proposal = json.loads(result.final_text)
    except json.JSONDecodeError:
        return fail([f"agent's final response did not parse as JSON: {result.final_text[:500]}"])

    if profile_report is None:
        return fail(["profiler was never called — cannot validate proposed columns"])

    errors = validate_feature_proposal(profile_report, target_column, group_column, time_column, proposal)
    if errors:
        return fail(errors)

    drop_columns = proposal.get("drop_columns") or []
    derived_features = proposal.get("derived_features") or []
    explanation = proposal.get("explanation", "")

    new_df = df.copy()
    new_columns: list[str] = []
    applied_ops: list[dict] = []
    for feat in derived_features:
        op_id = feat["op_id"]
        params = feat.get("params") or {}
        try:
            new_df, names = apply_feature_op(new_df, op_id, params)
        except Exception as e:
            return fail([f"failed to apply feature op '{op_id}': {type(e).__name__}: {e}"])
        new_columns.extend(names)
        applied_ops.append({"op_id": op_id, "params": params, "new_columns": names})

    emit_event(on_event, "feature_engineering", "proposal_validated", {
        "drop_columns": drop_columns, "new_columns": new_columns, "applied_ops": applied_ops,
    })
    return FeatureEngineeringStepResult(
        ok=True, df=new_df, drop_columns=drop_columns, new_columns=new_columns,
        applied_ops=applied_ops, explanation=explanation, errors=[],
        stopped_reason=result.stopped_reason, turns_used=result.turns_used,
        messages=result.messages,
    )
