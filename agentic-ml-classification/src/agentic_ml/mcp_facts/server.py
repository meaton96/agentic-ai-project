"""
FastMCP server exposing harness-computed facts as MCP tools — the
standardized replacement for the in-process Tool closures in
tools/*.py, one seam of this pipeline's "agents propose, harness
decides" architecture (see CLAUDE.md invariant #1).

This process never computes anything a real run's harness code
doesn't already compute, and it never touches raw data, dataframes,
or fitted pipelines. Two tools (list_templates, list_feature_ops) are
static and stateless, so they're recomputed live from the same
registries tools/template_tool.py and tools/feature_tool.py already
call. Every other tool reads a JSON fact under
runs/<run_id>/facts/<name>.json that McpToolProvider (provider.py)
already wrote, using the exact same harness calls the corresponding
tools/*.py handler makes — this server is a read-only view over that
directory, nothing more.

`enabled_tools` in config is the configurable surface: a tool absent
from it is never registered, so one deployment can serve a restricted
subset (e.g. omit deep-dive evidence) with no code changes.
"""
from __future__ import annotations

from typing import Optional

from mcp.server.fastmcp import FastMCP

from agentic_ml.harness.feature_engineering import list_feature_ops
from agentic_ml.mcp_facts.fact_store import read_fact
from agentic_ml.templates.registry import list_template_summaries

# MCP tool name -> fact name written by McpToolProvider via fact_store.write_fact
_RUN_SCOPED_FACT_TOOLS: dict[str, tuple[str, str]] = {
    "get_raw_schema": (
        "raw_schema",
        "Get the dataset's raw column facts for this run, computed with no "
        "target column assumed yet: names, dtypes, missingness, cardinality, "
        "sample values, and name-based hints for id/group/datetime columns.",
    ),
    "get_dataset_profile": (
        "dataset_profile",
        "Get the deterministic profile of this run's loaded dataset: column "
        "types, missingness, cardinality, likely ID/group/datetime columns, "
        "target class distribution and imbalance, a recommended split "
        "strategy, and any leakage risk flags.",
    ),
    "get_candidate_review_bundle": (
        "review_bundle",
        "Get everything about the candidate under review for this run: the "
        "template used and when it's meant to be used, the agent's config and "
        "explanation, validation metrics with confidence intervals, both "
        "leakage check results, and relevant dataset facts from the profiler.",
    ),
    "get_monitoring_context": (
        "monitoring_context",
        "Get this run's monitoring context for the newly arrived batch: the "
        "drift summary, the current model's test metrics as of its last "
        "(re)training, and how much has accumulated since then.",
    ),
    "get_planning_context": (
        "planning_context",
        "Get this run's planning context: the goal, the full catalog of "
        "available agents, the current run state, and how many planning "
        "iterations remain.",
    ),
}

ALL_TOOL_NAMES: tuple[str, ...] = tuple(_RUN_SCOPED_FACT_TOOLS) + (
    "list_templates", "list_feature_ops", "get_flight_deep_dive_evidence",
)


def build_server(config: Optional[dict] = None) -> FastMCP:
    config = config or {}
    enabled = set(config.get("enabled_tools", ALL_TOOL_NAMES))
    unknown = enabled - set(ALL_TOOL_NAMES)
    if unknown:
        raise ValueError(f"Unknown tool(s) in enabled_tools config: {sorted(unknown)}")

    server = FastMCP(
        name=config.get("name", "agentic-ml-facts"),
        host=config.get("host", "127.0.0.1"),
        port=config.get("port", 8765),
    )

    for tool_name, (fact_name, description) in _RUN_SCOPED_FACT_TOOLS.items():
        if tool_name not in enabled:
            continue

        def make_handler(fact_name: str):
            def handler(run_id: str) -> dict:
                return read_fact(run_id, fact_name)
            return handler

        server.tool(name=tool_name, description=description)(make_handler(fact_name))

    if "list_templates" in enabled:
        @server.tool(name="list_templates", description=(
            "Get the list of available recipe templates: each template's id, "
            "what it does, when to use it, and its config contract (required "
            "and optional keys)."
        ))
        def _list_templates() -> dict:
            return {"templates": list_template_summaries()}

    if "list_feature_ops" in enabled:
        @server.tool(name="list_feature_ops", description=(
            "Get the list of available feature-engineering operations: each "
            "op's id, what it does, required/optional parameters, and what "
            "column dtype it needs."
        ))
        def _list_feature_ops() -> dict:
            return {"feature_ops": list_feature_ops()}

    if "get_flight_deep_dive_evidence" in enabled:
        @server.tool(name="get_flight_deep_dive_evidence", description=(
            "Get the full deep-dive evidence for one flagged flight in this "
            "run: the model's predicted maintenance probability, which sensor "
            "channels drove that prediction (occlusion attribution), "
            "flight-phase segmentation, and independent raw-signal "
            "cross-cylinder imbalance findings."
        ))
        def _get_flight_deep_dive_evidence(run_id: str, flight_id: str) -> dict:
            return read_fact(run_id, f"deep_dive_evidence_{flight_id}")

    return server
