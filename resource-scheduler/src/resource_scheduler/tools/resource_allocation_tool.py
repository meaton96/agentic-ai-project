"""
Tool exposing the deterministic allocation context to the Resource
Allocation agent. Unlike load_monitor_tool/task_prioritization_tool,
the fact this binds isn't purely a function of the loaded dataframe --
it also depends on the ranking Task Prioritization sent over the A2A
mailbox, so the step function builds the fact once (see
build_allocation_context_fact) and this module just wraps that
already-built dict in a Tool, rather than recomputing it per call.
"""
from __future__ import annotations

import pandas as pd

from resource_scheduler.agent_runtime import Tool
from resource_scheduler.environment.allocation import DEFAULT_SLICE_CAPACITY, build_task_lookup, compute_slice_load
from resource_scheduler.environment.state import compute_snapshot


def build_allocation_context_fact(
    df: pd.DataFrame,
    variance_injected: dict[str, bool],
    ranked_task_ids: list[str],
    score_breakdown: dict,
    snapshot_window: int = 200,
    slice_capacity: int = DEFAULT_SLICE_CAPACITY,
) -> dict:
    snapshot = compute_snapshot(df, variance_injected, window=snapshot_window)
    machine_status = {m["machine_id"]: m["status"] for m in snapshot["machines"]}
    # current_load must be on the SAME scale as slice_capacity (a small
    # per-batch cap -- see environment/allocation.py's DEFAULT_SLICE_CAPACITY
    # docstring). Using the full snapshot_window here was a real bug,
    # confirmed live: counting task-slice associations across 200 historical
    # rows gives ~65-70 per slice regardless of what's actually being
    # proposed, so every batch was rejected as "over capacity" no matter
    # what the agent chose -- the agent (correctly, given what it was shown)
    # refused to assign anything. Scoped instead to a window matching the
    # batch actually being decided (len(ranked_task_ids)), so current_load
    # sits in a range comparable to slice_capacity and the gate can
    # meaningfully accept some batches and reject others.
    slice_current_load = compute_slice_load(df, window=max(len(ranked_task_ids), 1))
    task_lookup = build_task_lookup(df, ranked_task_ids)

    ranked_tasks = []
    for task_id in ranked_task_ids:
        attrs = task_lookup.get(task_id, {})
        ranked_tasks.append({
            "task_id": task_id,
            "task_type": attrs.get("task_type"),
            "final_score": (score_breakdown.get(task_id) or {}).get("final_score"),
            "previously_recorded_machine_id": attrs.get("previously_recorded_machine_id"),
            "previously_recorded_network_slice_id": attrs.get("previously_recorded_network_slice_id"),
        })

    available_machines = [{"machine_id": mid, "status": status} for mid, status in sorted(machine_status.items())]
    all_slice_ids = sorted(set(slice_current_load.keys()) | {s["slice_id"] for s in snapshot["slices"]})
    available_slices = [
        {"slice_id": sid, "current_load": slice_current_load.get(sid, 0), "capacity": slice_capacity}
        for sid in all_slice_ids
    ]

    return {
        "ranked_tasks": ranked_tasks,
        "available_machines": available_machines,
        "available_slices": available_slices,
        "synthetic_variance_injected": variance_injected,
    }


def make_resource_allocation_tool(context_fact: dict) -> Tool:
    def handler() -> dict:
        return context_fact

    return Tool(
        name="get_allocation_context",
        description=(
            "Get the ranked task queue (already priority-ordered — respect that "
            "order) plus every available machine's status and every network "
            "slice's current load/capacity. Call this once — it takes no "
            "arguments. Only use machine_id/network_slice_id values that appear "
            "in available_machines/available_slices; do not invent ids."
        ),
        parameters={"type": "object", "properties": {}, "required": []},
        handler=handler,
    )
