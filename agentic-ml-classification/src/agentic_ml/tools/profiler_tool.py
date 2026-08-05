"""
Tool exposing the deterministic profiler to an agent. The agent can
call this to get facts, but cannot alter what it computes — the tool
handler just calls agentic_ml.harness.profiler.profile_dataset()
directly and returns its output, annotated with the declared column
roles (build_profile_fact below).
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from agentic_ml.agent_runtime import Tool
from agentic_ml.harness.profiler import profile_dataset


def build_profile_fact(
    df: pd.DataFrame,
    target_column: str,
    group_column: Optional[str] = None,
    time_column: Optional[str] = None,
) -> dict:
    """profile_dataset()'s report plus the DECLARED column roles the
    profiler itself cannot know. profile_dataset is pure data-derived
    facts; which column intake (or the CLI) declared as group/time is
    run context, not a property of the dataframe — but agents are told
    'never include a declared group/time column' and previously had no
    way to see which columns those were. Real incident: after
    'date_diff' stopped being (mis)flagged is_likely_datetime, nothing
    in the profile marked it at all, and a modeling agent proposed it
    as a feature 8 runs in a row — each one correctly rejected by
    _validate_candidate_columns, burning the whole iteration budget.

    excluded_columns mirrors exactly what modeling_step.py's
    _validate_candidate_columns enforces (target + declared group/time
    + likely-ID columns): one authoritative do-not-use list, so agents
    don't have to reassemble it from scattered flags.
    """
    fact = profile_dataset(df, target_column=target_column).to_dict()
    fact["declared_group_column"] = group_column
    fact["declared_time_column"] = time_column
    fact["excluded_columns"] = sorted(
        {c for c in (target_column, group_column, time_column) if c}
        | set(fact.get("likely_id_columns") or [])
    )
    return fact


def make_profiler_tool(
    df: pd.DataFrame,
    target_column: str,
    group_column: Optional[str] = None,
    time_column: Optional[str] = None,
) -> Tool:
    """
    Binds a specific (already-loaded) dataframe + target column into a
    Tool the agent can call. The agent never receives a raw file path —
    only this pre-loaded, harness-controlled dataframe is profiled.
    """

    def handler() -> dict:
        return build_profile_fact(df, target_column, group_column, time_column)

    return Tool(
        name="get_dataset_profile",
        description=(
            "Get a deterministic profile of the loaded dataset: column types, "
            "missingness, cardinality, likely ID/group/datetime columns, target "
            "class distribution and imbalance, a recommended split strategy, "
            "any leakage risk flags, and excluded_columns — the authoritative "
            "list of columns (target, declared group/time, likely IDs) that "
            "must never be used as features. Call this once at the start of "
            "your analysis — it takes no arguments."
        ),
        parameters={"type": "object", "properties": {}, "required": []},
        handler=handler,
    )
