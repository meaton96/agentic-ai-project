"""
FastMCP server exposing gate_adapters.py's prepare_* facts as MCP tools
-- the standardized replacement for the in-process Tool closures in
tools/*.py, once a stage's LLM call moves out of an in-process
ToolCallingAgent and into a real sandbox AgentSpec agent step (see
gate_adapters.py's module docstring).

This process never computes anything a real run's gate code doesn't
already compute, and it never touches raw dataframes or file paths --
every tool here is a read-only view over runs/<run_id>/facts/<name>.json,
written by a prepare_* gate via fact_store.write_fact.

This module only builds the server; it never runs one. The agentic-ml-facts
MCP server (the agentic-mcp deployment) registers these tools next to
agentic_ml's and serves them over HTTP with its own bearer token. Agents
reach them at https://agentsandbox.gccis.rit.edu/agentic-ml-facts/mcp.
That server mounts the owner's scratch volume, where the prepare_* gates
write facts (see paths.py for GATE_SCRATCH_DIR).

`enabled_tools` in config is the configurable surface: a tool absent from
it is never registered, so one deployment can serve a restricted subset
with no code changes.
"""
from __future__ import annotations

from typing import Iterable, Optional

from mcp.server.fastmcp import FastMCP

from resource_scheduler.mcp_facts.fact_store import read_fact_or_default

# MCP tool name -> fact name written via fact_store.write_fact
_RUN_SCOPED_FACT_TOOLS: dict[str, tuple[str, str]] = {
    "get_resource_snapshot": (
        "resource_snapshot",
        "Get a deterministic snapshot of current machine and network-slice "
        "state for this run: per-machine status/queue depth/utilization, "
        "per-slice latency/capacity/URLLC score, the warning/critical "
        "thresholds used to judge them, and the authoritative list of flags "
        "already raised against those thresholds. You must not invent your "
        "own severity judgments; report exactly the flags this tool returns.",
    ),
    "get_task_queue_profile": (
        "task_queue_profile",
        "Get this run's current pending task queue with three pre-computed "
        "raw signals per task -- urgency_signal, energy_cost_proxy, "
        "availability_bonus -- plus per-machine status. You must base your "
        "scoring on these signals; do not invent your own task IDs or raw "
        "values.",
    ),
    "get_allocation_context": (
        "allocation_context",
        "Get this run's ranked task queue (already priority-ordered -- "
        "respect that order) plus every available machine's status and "
        "every network slice's current load/capacity. Only use "
        "machine_id/network_slice_id values that appear in "
        "available_machines/available_slices; do not invent ids.",
    ),
    "get_incident_report": (
        "incident_report",
        "Get this run's current incident report: which machines just "
        "transitioned into a fault status (Maintenance or Overloaded), "
        "which currently committed tasks are affected by each one, and "
        "every machine's current status. Do not invent incidents or task "
        "IDs not present here.",
    ),
    "get_policy_evidence": (
        "policy_evidence",
        "Get aggregated outcomes across this run's recent-run history: "
        "ranking validity/consistency rates, allocation acceptance/"
        "rejection counts, reroute acceptance counts. Do not invent "
        "statistics beyond what this returns.",
    ),
    "get_policy_review_bundle": (
        "policy_review_bundle",
        "Get the Optimization agent's proposed policy_updates for this "
        "run, its own stated evidence and recommend_apply flag, and the "
        "full underlying_evidence those were based on.",
    ),
    "get_decision_review_bundle": (
        "decision_review_bundle",
        "Get a batch of risky scheduling decisions flagged for this run by "
        "Resource Allocation or Failure Recovery -- each one already "
        "passed the deterministic constraint gate and has been committed; "
        "it's flagged here only because it targets a machine currently "
        "Overloaded.",
    ),
}

ALL_TOOL_NAMES: tuple[str, ...] = tuple(_RUN_SCOPED_FACT_TOOLS)

# fact name -> the MCP tool that serves it (inverse of _RUN_SCOPED_FACT_TOOLS)
FACT_TOOL_NAMES: dict[str, str] = {fact: tool for tool, (fact, _) in _RUN_SCOPED_FACT_TOOLS.items()}


def register_tools(server: FastMCP, enabled_tools: Optional[Iterable[str]] = None) -> None:
    """Add this package's fact tools to `server`, which may be another
    package's server (agentic-mcp adds them to agentic_ml's).
    enabled_tools: the tool names to register; None for all of them."""
    enabled = set(ALL_TOOL_NAMES if enabled_tools is None else enabled_tools)
    unknown = enabled - set(ALL_TOOL_NAMES)
    if unknown:
        raise ValueError(f"Unknown tool(s) in enabled_tools config: {sorted(unknown)}")

    for tool_name, (fact_name, description) in _RUN_SCOPED_FACT_TOOLS.items():
        if tool_name not in enabled:
            continue

        def make_handler(fact_name: str):
            def handler(run_id: str) -> dict:
                return read_fact_or_default(run_id, fact_name)
            return handler

        server.tool(name=tool_name, description=description)(make_handler(fact_name))


def build_server(config: Optional[dict] = None) -> FastMCP:
    config = config or {}
    server = FastMCP(
        name=config.get("name", "resource-scheduler-facts"),
        host=config.get("host", "127.0.0.1"),
        port=config.get("port", 8766),
    )
    register_tools(server, config.get("enabled_tools"))
    return server
