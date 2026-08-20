"""
Tool exposing the deterministic pending-queue profile to the Task
Prioritization agent. Mirrors tools/load_monitor_tool.py's shape: the
handler only calls environment functions and returns their output --
the agent proposes a ranking, it never computes the raw signals itself.
"""
from __future__ import annotations

import pandas as pd

from resource_scheduler.agent_runtime import Tool
from resource_scheduler.environment.queue import compute_availability_bonus, compute_energy_cost_proxy, compute_pending_queue
from resource_scheduler.environment.state import compute_snapshot


def build_task_queue_fact(
    df: pd.DataFrame,
    variance_injected: dict[str, bool],
    queue_size: int = 8,
    snapshot_window: int = 200,
) -> dict:
    snapshot = compute_snapshot(df, variance_injected, window=snapshot_window)
    pending_tasks = compute_pending_queue(df, queue_size=queue_size)
    energy_cost = compute_energy_cost_proxy(pending_tasks)
    availability = compute_availability_bonus(pending_tasks, snapshot["machines"])

    for t in pending_tasks:
        t["urgency_signal"] = 1.0 if t["needs_reallocation"] else 0.2
        t["energy_cost_proxy"] = energy_cost[t["task_id"]]
        t["availability_bonus"] = availability[t["task_id"]]

    return {
        "pending_tasks": pending_tasks,
        "machine_status": {m["machine_id"]: m["status"] for m in snapshot["machines"]},
        "synthetic_variance_injected": variance_injected,
    }


def make_task_prioritization_tool(
    df: pd.DataFrame,
    variance_injected: dict[str, bool],
    queue_size: int = 8,
    snapshot_window: int = 200,
) -> Tool:
    def handler() -> dict:
        return build_task_queue_fact(df, variance_injected, queue_size, snapshot_window)

    return Tool(
        name="get_task_queue_profile",
        description=(
            "Get the current pending task queue with three pre-computed raw "
            "signals per task -- urgency_signal, energy_cost_proxy, "
            "availability_bonus -- plus per-machine status. Call this once -- "
            "it takes no arguments. You must base your scoring on these signals; "
            "do not invent your own task IDs or raw values."
        ),
        parameters={"type": "object", "properties": {}, "required": []},
        handler=handler,
    )
