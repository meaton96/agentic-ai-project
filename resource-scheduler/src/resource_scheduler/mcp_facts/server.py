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

Unlike agentic_ml's mcp_facts/server.py, there is no LocalToolProvider/
McpToolProvider split and no HTTP/bearer-auth path -- resource_scheduler's
sandbox port only ever serves this over stdio, spawned fresh per agent
run by Strands' MCPClient (an AgentSpec.mcp_servers stdio binding), torn
down when that run ends. There is no persistent, concurrently-shared
listener to secure, so that whole layer (agentic_ml's
AUTH_TOKEN_ENV/build_http_app/BearerTokenMiddleware) would be
unused generality here. Revisit only if a persistent, concurrently-shared
fact server becomes necessary (e.g. driving a live pipeline from the
continuous optimization loop) -- see spec/sandbox-port-spec.md §5.

`enabled_tools` in config is still the configurable surface: a tool
absent from it is never registered, so one deployment can serve a
restricted subset with no code changes -- kept for parity with
agentic_ml's server.py even though nothing here needs it yet.

Run standalone (what an AgentSpec's stdio mcp_servers binding actually
invokes): `python -m resource_scheduler.mcp_facts.server`
"""
from __future__ import annotations

from typing import Optional

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


def build_server(config: Optional[dict] = None) -> FastMCP:
    config = config or {}
    enabled = set(config.get("enabled_tools", ALL_TOOL_NAMES))
    unknown = enabled - set(ALL_TOOL_NAMES)
    if unknown:
        raise ValueError(f"Unknown tool(s) in enabled_tools config: {sorted(unknown)}")

    server = FastMCP(
        name=config.get("name", "resource-scheduler-facts"),
        host=config.get("host", "127.0.0.1"),
        port=config.get("port", 8766),
    )

    for tool_name, (fact_name, description) in _RUN_SCOPED_FACT_TOOLS.items():
        if tool_name not in enabled:
            continue

        def make_handler(fact_name: str):
            def handler(run_id: str) -> dict:
                return read_fact_or_default(run_id, fact_name)
            return handler

        server.tool(name=tool_name, description=description)(make_handler(fact_name))

    return server


if __name__ == "__main__":
    build_server().run(transport="stdio")
