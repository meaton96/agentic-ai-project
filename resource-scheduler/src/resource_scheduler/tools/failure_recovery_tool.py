"""
Tool exposing the deterministic incident report to the Failure Recovery
agent. Like resource_allocation_tool.py, the fact depends on caller
context (committed_assignments, the before/after replay cutoffs) that
isn't purely a function of the loaded dataframe, so the step function
builds the fact once and this module wraps it in a Tool.
"""
from __future__ import annotations

import pandas as pd

from resource_scheduler.agent_runtime import Tool
from resource_scheduler.environment.incidents import affected_tasks, diff_snapshots, snapshot_at


def build_incident_fact(
    df: pd.DataFrame,
    variance_injected: dict[str, bool],
    committed_assignments: list[dict],
    before_row: int,
    after_row: int,
    window: int = 200,
) -> dict:
    previous = snapshot_at(df, variance_injected, before_row, window=window)
    current = snapshot_at(df, variance_injected, after_row, window=window)
    incidents = diff_snapshots(previous, current)
    affected = affected_tasks(committed_assignments, incidents)
    return {
        "incidents": incidents,
        "affected_tasks": affected,
        "current_machine_status": {m["machine_id"]: m["status"] for m in current["machines"]},
        "synthetic_variance_injected": variance_injected,
    }


def make_failure_recovery_tool(context_fact: dict) -> Tool:
    def handler() -> dict:
        return context_fact

    return Tool(
        name="get_incident_report",
        description=(
            "Get the current incident report: which machines just transitioned "
            "into a fault status (Maintenance or Overloaded), which currently "
            "committed tasks are affected by each one, and every machine's "
            "current status. Call this once — it takes no arguments. Do not "
            "invent incidents or task IDs not present here."
        ),
        parameters={"type": "object", "properties": {}, "required": []},
        handler=handler,
    )
