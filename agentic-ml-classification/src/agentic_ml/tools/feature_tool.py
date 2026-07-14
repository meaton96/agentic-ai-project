"""
Tool exposing the feature-engineering operation catalog to the
feature-engineering agent. Mirrors tools/template_tool.py's pattern:
the agent can see what's available, but cannot alter what it computes.
"""
from __future__ import annotations

from agentic_ml.agent_runtime import Tool
from agentic_ml.harness.feature_engineering import list_feature_ops


def make_list_feature_ops_tool() -> Tool:
    def handler() -> dict:
        return {"feature_ops": list_feature_ops()}

    return Tool(
        name="list_feature_ops",
        description=(
            "Get the list of available feature-engineering operations: each "
            "op's id, what it does, required/optional parameters, and what "
            "column dtype it needs. Call this once, then propose derived "
            "features using ONLY these op_ids — you cannot write "
            "transformation code yourself."
        ),
        parameters={"type": "object", "properties": {}, "required": []},
        handler=handler,
    )
