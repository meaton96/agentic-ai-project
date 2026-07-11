"""
Tool exposing the deterministic profiler to an agent. The agent can
call this to get facts, but cannot alter what it computes — the tool
handler just calls agentic_ml.harness.profiler.profile_dataset()
directly and returns its output as-is.
"""
from __future__ import annotations

import pandas as pd

from agentic_ml.agent_runtime import Tool
from agentic_ml.harness.profiler import profile_dataset


def make_profiler_tool(df: pd.DataFrame, target_column: str) -> Tool:
    """
    Binds a specific (already-loaded) dataframe + target column into a
    Tool the agent can call. The agent never receives a raw file path —
    only this pre-loaded, harness-controlled dataframe is profiled.
    """

    def handler() -> dict:
        report = profile_dataset(df, target_column=target_column)
        return report.to_dict()

    return Tool(
        name="get_dataset_profile",
        description=(
            "Get a deterministic profile of the loaded dataset: column types, "
            "missingness, cardinality, likely ID/group/datetime columns, target "
            "class distribution and imbalance, a recommended split strategy, "
            "and any leakage risk flags. Call this once at the start of your "
            "analysis — it takes no arguments."
        ),
        parameters={"type": "object", "properties": {}, "required": []},
        handler=handler,
    )
