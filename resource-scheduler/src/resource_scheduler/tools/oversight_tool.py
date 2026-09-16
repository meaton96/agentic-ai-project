"""
Tool exposing the deterministic review bundle to the Human Oversight
agent. Same shape as every other tool in this project.
"""
from __future__ import annotations

from resource_scheduler.agent_runtime import Tool


def make_oversight_tool(review_bundle: dict) -> Tool:
    def handler() -> dict:
        return review_bundle

    return Tool(
        name="get_policy_review_bundle",
        description=(
            "Get the Optimization agent's proposed policy_updates, its own "
            "stated evidence and recommend_apply flag, and the full "
            "underlying_evidence those were based on. Call this once — it "
            "takes no arguments."
        ),
        parameters={"type": "object", "properties": {}, "required": []},
        handler=handler,
    )


def make_decision_oversight_tool(review_bundle: dict) -> Tool:
    def handler() -> dict:
        return review_bundle

    return Tool(
        name="get_decision_review_bundle",
        description=(
            "Get a batch of risky scheduling decisions flagged by Resource "
            "Allocation or Failure Recovery — each one already passed the "
            "deterministic constraint gate and has been committed; it's "
            "flagged here only because it targets a machine currently "
            "Overloaded. Call this once — it takes no arguments."
        ),
        parameters={"type": "object", "properties": {}, "required": []},
        handler=handler,
    )
