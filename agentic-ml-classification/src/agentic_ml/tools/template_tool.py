"""
Tool exposing the recipe template registry to the modeling agent. The
agent can see what templates exist and their config contracts, but
cannot alter the templates themselves — the handler just calls
agentic_ml.templates.registry.list_template_summaries() and returns it
as-is, mirroring tools/profiler_tool.py's pattern.
"""
from __future__ import annotations

from agentic_ml.agent_runtime import Tool
from agentic_ml.templates.registry import list_template_summaries


def make_list_templates_tool() -> Tool:
    def handler() -> dict:
        return {"templates": list_template_summaries()}

    return Tool(
        name="list_templates",
        description=(
            "Get the list of available recipe templates: each template's id, "
            "what it does, when to use it, and its config contract (required "
            "and optional keys). Call this once, then propose a candidate by "
            "picking exactly one template_id and filling in its config. Do not "
            "invent a template_id that isn't in this list."
        ),
        parameters={"type": "object", "properties": {}, "required": []},
        handler=handler,
    )
