"""
Tool exposing one batch's monitoring context to the retrain-decision
agent. The agent can call this to see the drift summary, the current
model's last-training metrics, and how much has accumulated since —
but cannot alter what it computes; the handler just returns the
pre-assembled context (see orchestrator/dynamic_loop.py's
retrain_decision branch) as-is, mirroring every other tool in this
pipeline.
"""
from __future__ import annotations

from agentic_ml.agent_runtime import Tool


def make_monitoring_context_tool(monitoring_context: dict) -> Tool:
    def handler() -> dict:
        return monitoring_context

    return Tool(
        name="get_monitoring_context",
        description=(
            "Get everything needed to decide what to do about the newly arrived batch: "
            "the drift summary (per-feature standardized shift, an aggregate score, and "
            "the batch's own performance metrics), the current model's test metrics as "
            "of its last (re)training, and how many batches/examples have accumulated "
            "since then. Call this once — you cannot see any of this otherwise and must "
            "not invent it."
        ),
        parameters={"type": "object", "properties": {}, "required": []},
        handler=handler,
    )
