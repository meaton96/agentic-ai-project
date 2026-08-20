"""
Pending-queue facts and ranking-proposal validation for the Task
Prioritization agent. Split out from state.py (which owns Load
Monitor's snapshot/threshold/flag concerns) the same way harness/
in the sibling agentic-ml-classification project splits splits.py,
leakage.py, metrics.py, profiler.py, and intake.py into one file per
concern rather than one monolithic module.

Honesty note, same spirit as state.py's synthetic-variance handling:
this dataset has no explicit deadline or energy-cost column, so
urgency/energy/availability are PROXIES derived from real columns
(Reallocation, Execution_Time, Machine_Status), not ground-truth
priority data. That's declared here, not silently assumed -- see each
function's docstring for exactly what it stands in for.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

# Placeholder domain assumption, not fact: how much scheduling headroom
# each machine status implies right now. Replace with a real
# capacity/SLA model once one exists -- this is deliberately a small,
# visible lookup table rather than logic buried in a formula, so it's
# easy to find and revise.
MACHINE_AVAILABILITY_BONUS = {"Idle": 1.0, "Active": 0.6, "Overloaded": 0.1, "Maintenance": 0.0}


def compute_pending_queue(df: pd.DataFrame, queue_size: int = 8) -> list[dict]:
    """The most recent `queue_size` rows (by row order, same recency
    proxy compute_snapshot uses) as "tasks currently needing a
    scheduling decision". Kept deliberately small -- a real accepted
    lesson from the sibling ML pipeline is that asking an LLM to
    enumerate many items in one structured response risks truncation at
    the output token limit (see agent_runtime.py's max_tokens history).

    Was 15, lowered to 8 after a second, DIFFERENT failure mode showed
    up live against the RIT GenAI endpoint: raising max_tokens (see
    agent_runtime.py) fixed the truncation problem but gave a
    reasoning-heavy model (qwen3) enough runway to generate long enough
    to exceed the gateway's own fixed timeout -- confirmed by
    reproducing a 504 Gateway Time-out 4/4 times at queue_size=15 across
    two different models, then succeeding immediately at queue_size=5.
    8 is a middle ground, not a proven boundary -- if timeouts return,
    lower this further before assuming something else is wrong."""
    recent = df.tail(queue_size)
    tasks = []
    for _, row in recent.iterrows():
        tasks.append({
            "task_id": row["Task_ID"],
            "machine_id": row["Machine_ID"],
            "network_slice_id": row["Network_Slice_ID"],
            "task_type": row["Task_Type"],
            "execution_time": float(row["Execution_Time"]),
            "needs_reallocation": bool(row["Reallocation"] == "Yes"),
        })
    return tasks


def compute_energy_cost_proxy(pending_tasks: list[dict]) -> dict[str, float]:
    """Min-max normalized Execution_Time within the queue: 0 (cheapest)
    to 1 (most expensive). A queue where every task has identical
    Execution_Time (e.g. --no-inject-variance against the current CSV)
    has no real spread to normalize against -- every task gets a
    neutral 0.5 rather than an arbitrary 0 or a div-by-zero, so a flat
    baseline doesn't masquerade as "everything is cheapest"."""
    if not pending_tasks:
        return {}
    values = [t["execution_time"] for t in pending_tasks]
    lo, hi = min(values), max(values)
    spread = hi - lo
    result = {}
    for t in pending_tasks:
        result[t["task_id"]] = 0.5 if spread == 0 else round((t["execution_time"] - lo) / spread, 4)
    return result


def compute_availability_bonus(pending_tasks: list[dict], machine_snapshot: list[dict]) -> dict[str, float]:
    """machine_snapshot: the `machines` list from
    environment.state.compute_snapshot(). Looks up each task's assigned
    machine's status through MACHINE_AVAILABILITY_BONUS -- a task queued
    against an Idle machine is easier to schedule right now than one
    against a machine already Overloaded or in Maintenance. An unknown
    machine (not in the snapshot) gets a neutral-low 0.3, never a
    fabricated confident value."""
    status_by_machine = {m["machine_id"]: m["status"] for m in machine_snapshot}
    result = {}
    for t in pending_tasks:
        status = status_by_machine.get(t["machine_id"])
        result[t["task_id"]] = MACHINE_AVAILABILITY_BONUS.get(status, 0.3)
    return result


def validate_ranking_proposal(pending_task_ids: list[str], proposal: object) -> list[str]:
    """Hard structural gate on the agent's proposal -- run before
    anything downstream (an eventual Resource Allocation agent) is
    allowed to trust ranked_task_ids. Mirrors
    agentic_ml.harness.intake.validate_dataset_spec_proposal's shape:
    pure function, dataset facts in, error strings out, never raises."""
    errors: list[str] = []
    if not isinstance(proposal, dict):
        return ["proposal is not a JSON object"]

    ranked = proposal.get("ranked_task_ids")
    if not isinstance(ranked, list):
        errors.append("ranked_task_ids is missing or not a list")
    elif sorted(ranked) != sorted(pending_task_ids) or len(ranked) != len(set(ranked)):
        errors.append("ranked_task_ids must be exactly one permutation of the pending task ids "
                       "(no missing, duplicate, or invented ids)")

    breakdown = proposal.get("score_breakdown")
    if not isinstance(breakdown, dict):
        errors.append("score_breakdown is missing or not an object")
    else:
        missing = set(pending_task_ids) - set(breakdown.keys())
        extra = set(breakdown.keys()) - set(pending_task_ids)
        if missing:
            errors.append(f"score_breakdown is missing entries for: {sorted(missing)}")
        if extra:
            errors.append(f"score_breakdown has entries for unknown task ids: {sorted(extra)}")
        for task_id, entry in breakdown.items():
            if not isinstance(entry, dict) or "final_score" not in entry:
                errors.append(f"score_breakdown[{task_id!r}] is missing final_score")

    return errors


def is_ranking_score_consistent(proposal: dict) -> bool:
    """Soft check, only meaningful once validate_ranking_proposal has
    already passed: does ranked_task_ids actually run in descending
    final_score order, per the agent's own numbers? A violation doesn't
    invalidate the proposal (the agent may have deliberately broken ties
    non-numerically) but is worth surfacing -- same soft-flag spirit as
    load_monitor_step.py's flag_mismatch."""
    ranked = proposal.get("ranked_task_ids") or []
    breakdown = proposal.get("score_breakdown") or {}
    scores = []
    for task_id in ranked:
        entry = breakdown.get(task_id)
        if not isinstance(entry, dict) or "final_score" not in entry:
            return False
        scores.append(entry["final_score"])
    return all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
