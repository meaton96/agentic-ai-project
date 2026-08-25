"""
Tool exposing the deterministic resource snapshot to the Load Monitor
agent. The agent can call this to get facts, but cannot alter what it
computes -- the tool handler just calls
environment.state.compute_snapshot/compute_thresholds/compute_flags
directly and returns their output. Mirrors
agentic_ml.tools.profiler_tool's build_profile_fact / make_profiler_tool
pair in the sibling agentic-ml-classification project.
"""
from __future__ import annotations

import pandas as pd

from resource_scheduler.agent_runtime import Tool
from resource_scheduler.environment.state import compute_flags, compute_snapshot, compute_thresholds


def build_resource_snapshot_fact(
    df: pd.DataFrame,
    variance_injected: dict[str, bool],
    window: int = 200,
) -> dict:
    """Snapshot + thresholds + the authoritative flags, bundled into one
    fact so the agent never has to (and never gets to) compute severity
    itself -- it only narrates flags this function already decided."""
    snapshot = compute_snapshot(df, variance_injected, window=window)
    thresholds = compute_thresholds(df)
    flags = compute_flags(snapshot, thresholds)
    return {
        "snapshot": snapshot,
        "thresholds": thresholds,
        "flags": flags,
    }


def make_load_monitor_tool(
    df: pd.DataFrame,
    variance_injected: dict[str, bool],
    window: int = 200,
) -> Tool:
    """Binds an already-loaded task table into a Tool the agent can call.
    The agent never receives a raw file path -- only this pre-loaded,
    environment-controlled dataframe is snapshotted."""

    def handler() -> dict:
        return build_resource_snapshot_fact(df, variance_injected, window=window)

    return Tool(
        name="get_resource_snapshot",
        description=(
            "Get a deterministic snapshot of current machine and network-slice "
            "state: per-machine status/queue depth/utilization, per-slice "
            "latency/capacity/URLLC score, the warning/critical thresholds used "
            "to judge them, and the authoritative list of flags already raised "
            "against those thresholds. Call this once -- it takes no arguments. "
            "You must not invent your own severity judgments; report exactly "
            "the flags this tool returns."
        ),
        parameters={"type": "object", "properties": {}, "required": []},
        handler=handler,
    )
