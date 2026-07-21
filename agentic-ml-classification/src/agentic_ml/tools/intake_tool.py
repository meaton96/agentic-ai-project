"""
Tool exposing pre-target dataset facts to the intake agent. The agent
can call this to see column names/dtypes/samples, but cannot alter
what it computes — the handler just calls
agentic_ml.harness.intake.raw_schema_summary() and returns it as-is,
mirroring tools/profiler_tool.py's pattern.
"""
from __future__ import annotations

import pandas as pd

from agentic_ml.agent_runtime import Tool
from agentic_ml.harness.intake import raw_schema_summary


def make_raw_schema_tool(df: pd.DataFrame) -> Tool:
    def handler() -> dict:
        return raw_schema_summary(df)

    return Tool(
        name="get_raw_schema",
        description=(
            "Get the dataset's raw column facts, computed with no target column "
            "assumed yet: names, dtypes, missingness, cardinality, sample values, "
            "and name-based hints for id/group/datetime columns. Call this once "
            "before proposing a dataset spec — you cannot see the actual data "
            "otherwise and must not invent column names or values."
        ),
        parameters={"type": "object", "properties": {}, "required": []},
        handler=handler,
    )
