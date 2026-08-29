"""
Deterministic allocation constraints and traceable-event construction
for the Resource Allocation agent (agent #3) -- the first agent that
reads from the A2A mailbox (Task Prioritization's ranking) rather than
only being invoked directly by a CLI script.

Placeholder domain assumption, declared not hidden: this dataset has no
declared per-slice bandwidth/QoS capacity, so DEFAULT_SLICE_CAPACITY is
a flat cap on concurrent tasks per slice within the snapshot window --
the same kind of stand-in MACHINE_AVAILABILITY_BONUS already is in
queue.py. Replace with a real capacity model once one exists.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

DEFAULT_SLICE_CAPACITY = 8  # max concurrent tasks a slice can carry before it's considered saturated
BLOCKED_MACHINE_STATUSES = {"Maintenance"}  # machines in these statuses cannot receive new assignments


def compute_slice_load(df: pd.DataFrame, window: int = 200) -> dict[str, int]:
    """Count of tasks per network slice within the most recent `window`
    rows -- the same recency window Load Monitor's snapshot uses, so
    "current load" means the same thing across agents."""
    recent = df.tail(window)
    return recent.groupby("Network_Slice_ID").size().to_dict()


def build_task_lookup(df: pd.DataFrame, task_ids: list[str]) -> dict[str, dict]:
    """Task-type and historical machine/slice association for a specific
    set of task ids. Resource Allocation only receives ranked_task_ids
    over the A2A mailbox (not full task records), so this joins back
    against the task table by Task_ID. The historical machine/slice a
    task was recorded against is informational context, not a
    constraint -- the agent is free to place it anywhere valid; that's
    the actual scheduling decision."""
    subset = df[df["Task_ID"].isin(task_ids)]
    lookup = {}
    for _, row in subset.iterrows():
        lookup[row["Task_ID"]] = {
            "task_type": row["Task_Type"],
            "previously_recorded_machine_id": row["Machine_ID"],
            "previously_recorded_network_slice_id": row["Network_Slice_ID"],
        }
    return lookup


def validate_allocation_structure(task_ids: list[str], proposal: object) -> list[str]:
    """Hard structural gate, same shape as
    environment.queue.validate_ranking_proposal: every ranked task must
    appear exactly once across assignments+rejected combined -- no
    missing, no duplicate, no invented ids. Does NOT check constraint
    validity -- that's check_constraints' job, run separately so a
    structurally valid but constraint-violating proposal stays
    distinguishable from a malformed one."""
    errors: list[str] = []
    if not isinstance(proposal, dict):
        return ["proposal is not a JSON object"]

    assignments = proposal.get("assignments")
    rejected = proposal.get("rejected")
    if not isinstance(assignments, list):
        errors.append("assignments is missing or not a list")
        assignments = []
    if not isinstance(rejected, list):
        errors.append("rejected is missing or not a list")
        rejected = []

    seen_ids = []
    for a in assignments:
        if not isinstance(a, dict) or "task_id" not in a:
            errors.append("an assignment entry is missing task_id")
            continue
        seen_ids.append(a["task_id"])
        for field_name in ("machine_id", "network_slice_id", "rationale"):
            if field_name not in a:
                errors.append(f"assignment for {a['task_id']!r} is missing {field_name}")
    for r in rejected:
        if not isinstance(r, dict) or "task_id" not in r:
            errors.append("a rejected entry is missing task_id")
            continue
        seen_ids.append(r["task_id"])
        if "reason" not in r:
            errors.append(f"rejected entry for {r['task_id']!r} is missing reason")

    if sorted(seen_ids) != sorted(task_ids) or len(seen_ids) != len(set(seen_ids)):
        errors.append("assignments + rejected together must cover exactly the ranked task ids, each exactly once")

    return errors


def check_constraints(
    assignments: list[dict],
    machine_status: dict[str, str],
    slice_current_load: dict[str, int],
    slice_capacity: int = DEFAULT_SLICE_CAPACITY,
) -> list[dict]:
    """The actual gate: independent of what the agent claims, checks
    every proposed assignment against (a) the machine not being in a
    blocked status, (b) the slice not being at or over capacity once
    this assignment -- and any earlier assignment in this same batch
    targeting the same slice -- is counted. Returns one violation dict
    per failing assignment; an assignment with no entry here passed.

    Capacity is checked cumulatively across the whole batch (a running
    local copy of slice_current_load, not the static pre-batch
    snapshot) -- otherwise ten proposed assignments to the same
    near-full slice could each individually "pass" against the same
    stale count, oversubscribing it exactly the way this check exists
    to prevent."""
    violations = []
    running_load = dict(slice_current_load)
    for a in assignments:
        task_id = a.get("task_id")
        machine_id = a.get("machine_id")
        slice_id = a.get("network_slice_id")

        status = machine_status.get(machine_id)
        if status in BLOCKED_MACHINE_STATUSES:
            violations.append({
                "task_id": task_id, "violation": "machine_in_maintenance",
                "detail": f"machine {machine_id} is in Maintenance",
            })
            continue

        current = running_load.get(slice_id, 0)
        if current >= slice_capacity:
            violations.append({
                "task_id": task_id, "violation": "slice_capacity_exceeded",
                "detail": f"slice {slice_id} at {current}/{slice_capacity} before this assignment",
            })
            continue

        running_load[slice_id] = current + 1
    return violations


@dataclass
class AllocationEvent:
    task_id: str
    agent: str
    trigger_signal: str
    decision: dict
    constraint_checks_passed: bool
    reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id, "agent": self.agent,
            "trigger_signal": self.trigger_signal, "decision": self.decision,
            "constraint_checks_passed": self.constraint_checks_passed, "reason": self.reason,
        }


def build_allocation_events(
    assignments: list[dict],
    violations_by_task: dict[str, dict],
    trigger_signal_by_task: dict[str, str],
) -> list[dict]:
    """One traceable event per proposed assignment: which agent fired,
    what signal triggered it, and what was assigned -- the explicit
    auditability requirement from the Use Case II spec.
    trigger_signal_by_task is supplied by the caller rather than
    computed here, since what counts as "the signal that caused this"
    differs by path: a fresh ranking's trigger is its final_score from
    Task Prioritization; a Failure Recovery reroute's trigger is the
    incident that displaced it. Neither should be hardcoded into this
    shared function -- see steps/resource_allocation_step.py's two
    callers for each shape."""
    events = []
    for a in assignments:
        task_id = a.get("task_id")
        violation = violations_by_task.get(task_id)
        events.append(AllocationEvent(
            task_id=task_id,
            agent="resource_allocation",
            trigger_signal=trigger_signal_by_task.get(task_id, "unknown"),
            decision={"machine_id": a.get("machine_id"), "network_slice_id": a.get("network_slice_id")},
            constraint_checks_passed=violation is None,
            reason=violation["detail"] if violation else a.get("rationale"),
        ).to_dict())
    return events
