"""
Tests for environment/incidents.py -- no LLM involved. Covers snapshot
diffing, affected-task computation, and the validation functions
steps/failure_recovery_step.py gates on. Mirrors the hand-built-input
style of test_allocation.py/test_task_queue.py for the pure functions,
and exercises snapshot_at against the real dataset for the one function
that actually reads the dataframe.
"""
from pathlib import Path

from resource_scheduler.environment.incidents import (
    FAULT_STATUSES,
    affected_tasks,
    diff_snapshots,
    reroutes_avoid_source_machine,
    snapshot_at,
    validate_recovery_proposal,
)
from resource_scheduler.environment.state import load_task_table

REAL_DATASET = Path(__file__).resolve().parents[1] / "datasets" / "raw" / "industrial_scheduling_dataset.csv"


def snap(machine_statuses: dict[str, str]) -> dict:
    """Minimal snapshot-shaped dict -- diff_snapshots only reads machines."""
    return {"machines": [{"machine_id": mid, "status": status} for mid, status in machine_statuses.items()]}


def test_diff_snapshots_detects_fresh_transition_into_fault():
    previous = snap({"M01": "Active", "M02": "Idle"})
    current = snap({"M01": "Maintenance", "M02": "Idle"})
    incidents = diff_snapshots(previous, current)
    assert len(incidents) == 1
    assert incidents[0]["machine_id"] == "M01"
    assert incidents[0]["previous_status"] == "Active"
    assert incidents[0]["current_status"] == "Maintenance"


def test_diff_snapshots_ignores_machine_already_faulted_in_both():
    previous = snap({"M01": "Overloaded"})
    current = snap({"M01": "Maintenance"})  # still faulted, just a different fault -- not a fresh incident
    assert diff_snapshots(previous, current) == []


def test_diff_snapshots_no_change_no_incident():
    previous = snap({"M01": "Active"})
    current = snap({"M01": "Active"})
    assert diff_snapshots(previous, current) == []


def test_diff_snapshots_recovery_is_not_an_incident():
    """A machine going FROM a fault status back to normal is not itself
    an incident -- only transitions INTO FAULT_STATUSES are reported."""
    previous = snap({"M01": "Maintenance"})
    current = snap({"M01": "Active"})
    assert diff_snapshots(previous, current) == []


def test_affected_tasks_matches_committed_assignments_on_incident_machine():
    committed = [
        {"task_id": "T1", "machine_id": "M01", "network_slice_id": "NS_1"},
        {"task_id": "T2", "machine_id": "M02", "network_slice_id": "NS_2"},
    ]
    incidents = [{"machine_id": "M01", "previous_status": "Active", "current_status": "Maintenance",
                  "detected_issue": "transitioned from Active to Maintenance"}]
    affected = affected_tasks(committed, incidents)
    assert len(affected) == 1
    assert affected[0]["task_id"] == "T1"
    assert affected[0]["incident"] == "transitioned from Active to Maintenance"


def test_affected_tasks_empty_when_no_incidents():
    committed = [{"task_id": "T1", "machine_id": "M01", "network_slice_id": "NS_1"}]
    assert affected_tasks(committed, []) == []


def test_snapshot_at_truncates_before_computing():
    df, variance_injected = load_task_table(REAL_DATASET, inject_variance=True, seed=2)
    full_len = len(df)
    snapshot = snapshot_at(df, variance_injected, as_of_row=100, window=200)
    # window=200 requested but only 100 rows existed "as of" that cutoff --
    # compute_snapshot's tail(window) on a 100-row frame is just all 100 rows.
    assert snapshot["window_rows"] == 100
    assert full_len > 100  # sanity: the real file is bigger than the cutoff used


RECOVERY_AFFECTED_IDS = ["T1", "T2"]


def valid_recovery_proposal():
    return {
        "incidents": [{"machine_id": "M01", "previous_status": "Active", "current_status": "Maintenance",
                       "detected_issue": "transitioned from Active to Maintenance"}],
        "reroute_proposals": [
            {"task_id": "T1", "new_machine_id": "M02", "new_network_slice_id": "NS_2", "reasoning": "M02 is idle"},
            {"task_id": "T2", "new_machine_id": "M03", "new_network_slice_id": "NS_1", "reasoning": "M03 has room"},
        ],
    }


def test_validate_recovery_proposal_accepts_valid_proposal():
    assert validate_recovery_proposal(RECOVERY_AFFECTED_IDS, valid_recovery_proposal()) == []


def test_validate_recovery_proposal_rejects_non_dict():
    assert validate_recovery_proposal(RECOVERY_AFFECTED_IDS, "nope")


def test_validate_recovery_proposal_rejects_missing_task():
    proposal = valid_recovery_proposal()
    proposal["reroute_proposals"] = proposal["reroute_proposals"][:1]  # T2 dropped
    errors = validate_recovery_proposal(RECOVERY_AFFECTED_IDS, proposal)
    assert any("exactly the affected task ids" in e for e in errors)


def test_validate_recovery_proposal_rejects_duplicate_task():
    proposal = valid_recovery_proposal()
    proposal["reroute_proposals"].append(dict(proposal["reroute_proposals"][0]))
    errors = validate_recovery_proposal(RECOVERY_AFFECTED_IDS, proposal)
    assert any("exactly the affected task ids" in e for e in errors)


def test_validate_recovery_proposal_rejects_incomplete_reroute():
    proposal = valid_recovery_proposal()
    del proposal["reroute_proposals"][0]["reasoning"]
    errors = validate_recovery_proposal(RECOVERY_AFFECTED_IDS, proposal)
    assert any("missing reasoning" in e for e in errors)


def test_reroutes_avoid_source_machine_true_when_all_differ():
    affected = [{"task_id": "T1", "machine_id": "M01"}, {"task_id": "T2", "machine_id": "M01"}]
    assert reroutes_avoid_source_machine(affected, valid_recovery_proposal())


def test_reroutes_avoid_source_machine_false_when_reroute_targets_same_machine():
    affected = [{"task_id": "T1", "machine_id": "M02"}]  # same as T1's proposed new_machine_id below
    proposal = {"reroute_proposals": [{"task_id": "T1", "new_machine_id": "M02"}]}
    assert not reroutes_avoid_source_machine(affected, proposal)


def test_fault_statuses_contains_maintenance_and_overloaded():
    assert FAULT_STATUSES == {"Maintenance", "Overloaded"}
