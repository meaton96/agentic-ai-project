"""
Incident detection and reroute-proposal validation for the Failure
Recovery agent (agent #4) -- the second A2A hop this project tests:
Failure Recovery -> Resource Allocation, re-using the exact same
constraint gate (environment/allocation.py::check_constraints) Resource
Allocation already enforces for its own proposals, rather than a
separate one.

This dataset has no live stream, so "before" and "after" are two
snapshots of the same static table taken at different `as_of_row`
cutoffs -- the closest thing available to watching machine status
evolve over time, same replay-window convention environment/state.py's
Load Monitor snapshot already uses. A machine's status flip between the
two windows may be genuine drift in the underlying data rather than an
actual live fault -- flagged here, not hidden, the same honesty
environment/state.py already applies to synthetic variance injection.
"""
from __future__ import annotations

import pandas as pd

from resource_scheduler.environment.state import compute_snapshot

# A machine transitioning INTO one of these states is what counts as a
# fault worth recovering from. Maintenance is a hard placement block
# (see allocation.py::BLOCKED_MACHINE_STATUSES); Overloaded is included
# here too because a task already committed to a now-overloaded machine
# is a real disruption risk even though Overloaded alone isn't a hard
# block on NEW placements.
FAULT_STATUSES = {"Maintenance", "Overloaded"}


def snapshot_at(df: pd.DataFrame, variance_injected: dict[str, bool], as_of_row: int, window: int = 200) -> dict:
    """compute_snapshot() as if only the first `as_of_row` rows of the
    table had been observed -- lets two snapshots be compared at
    different points in the replay."""
    truncated = df.iloc[:as_of_row]
    return compute_snapshot(truncated, variance_injected, window=window)


def diff_snapshots(previous: dict, current: dict) -> list[dict]:
    """Machines that transitioned INTO a FAULT_STATUS between the two
    snapshots. A machine already in a fault status in BOTH snapshots is
    not a new incident -- it was already known; only a fresh transition
    is reported."""
    prev_status = {m["machine_id"]: m["status"] for m in previous["machines"]}
    curr_status = {m["machine_id"]: m["status"] for m in current["machines"]}
    incidents = []
    for machine_id, status in curr_status.items():
        prev = prev_status.get(machine_id)
        if status in FAULT_STATUSES and prev not in FAULT_STATUSES:
            incidents.append({
                "machine_id": machine_id, "previous_status": prev, "current_status": status,
                "detected_issue": f"transitioned from {prev} to {status}",
            })
    return incidents


def affected_tasks(committed_assignments: list[dict], incidents: list[dict]) -> list[dict]:
    """Which of the currently committed assignments sit on a machine
    that just had an incident. committed_assignments is whatever the
    caller currently considers "assigned" -- typically Resource
    Allocation's own accepted_assignments from a prior step, passed in
    directly rather than replayed through the mailbox (Failure Recovery
    needs to know current state, not receive a proposal to validate)."""
    incident_machines = {i["machine_id"]: i for i in incidents}
    affected = []
    for a in committed_assignments:
        machine_id = a.get("machine_id")
        if machine_id in incident_machines:
            affected.append({
                "task_id": a.get("task_id"),
                "machine_id": machine_id,
                "network_slice_id": a.get("network_slice_id"),
                "incident": incident_machines[machine_id]["detected_issue"],
            })
    return affected


def validate_recovery_proposal(affected_task_ids: list[str], proposal: object) -> list[str]:
    """Hard structural gate, same shape as
    environment.queue.validate_ranking_proposal /
    environment.allocation.validate_allocation_structure: every affected
    task must get exactly one reroute proposal, no missing, no
    duplicate, no invented ids. Does not itself check whether a reroute
    is a VALID placement -- that's Resource Allocation's job, via the
    same check_constraints gate it already applies to its own
    proposals, once this message reaches its mailbox."""
    errors: list[str] = []
    if not isinstance(proposal, dict):
        return ["proposal is not a JSON object"]

    incidents = proposal.get("incidents")
    reroutes = proposal.get("reroute_proposals")
    if not isinstance(incidents, list):
        errors.append("incidents is missing or not a list")
    if not isinstance(reroutes, list):
        errors.append("reroute_proposals is missing or not a list")
        reroutes = []

    seen_ids = []
    for r in reroutes:
        if not isinstance(r, dict) or "task_id" not in r:
            errors.append("a reroute_proposals entry is missing task_id")
            continue
        seen_ids.append(r["task_id"])
        for field_name in ("new_machine_id", "new_network_slice_id", "reasoning"):
            if field_name not in r:
                errors.append(f"reroute for {r['task_id']!r} is missing {field_name}")

    if sorted(seen_ids) != sorted(affected_task_ids) or len(seen_ids) != len(set(seen_ids)):
        errors.append("reroute_proposals must cover exactly the affected task ids, each exactly once")

    return errors


def reroutes_avoid_source_machine(affected: list[dict], proposal: dict) -> bool:
    """Soft check: a reroute that targets the SAME machine a task was
    just displaced from doesn't actually fix anything. This matters
    specifically because Overloaded (unlike Maintenance) is not a hard
    constraint block in check_constraints, so that gate alone would not
    catch a reroute that lands right back on the same overloaded
    machine. True if every reroute proposes a different machine_id than
    the task's current (incident) machine."""
    source_by_task = {t["task_id"]: t["machine_id"] for t in affected}
    for r in (proposal.get("reroute_proposals") or []):
        task_id = r.get("task_id")
        if r.get("new_machine_id") == source_by_task.get(task_id):
            return False
    return True
