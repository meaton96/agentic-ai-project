"""
Tool exposing a single candidate's review bundle to the verification
agent. The agent can call this to see everything about the candidate
under review, but cannot alter what it computes — the handler just
returns the pre-assembled bundle (see harness/verification.py) as-is,
mirroring the other tools' pattern.
"""
from __future__ import annotations

from agentic_ml.agent_runtime import Tool


def make_review_bundle_tool(bundle: dict) -> Tool:
    def handler() -> dict:
        return bundle

    return Tool(
        name="get_candidate_review_bundle",
        description=(
            "Get everything about the candidate under review: the template used "
            "and when it's meant to be used, the agent's config and explanation, "
            "validation metrics with confidence intervals, both leakage check "
            "results, and relevant dataset facts from the profiler. Call this "
            "once — you cannot see any of this otherwise and must not invent it."
        ),
        parameters={"type": "object", "properties": {}, "required": []},
        handler=handler,
    )
