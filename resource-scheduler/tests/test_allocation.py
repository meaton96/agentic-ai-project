"""
Tests for environment/allocation.py -- no LLM involved. Covers the
structural gate, the constraint gate (the part that actually matters:
an agent's claimed rationale never overrides it), and the traceable
event construction.
"""
from pathlib import Path

import pandas as pd
import pytest

from resource_scheduler.environment.allocation import (
    RISKY_MACHINE_STATUSES,
    build_allocation_events,
    build_task_lookup,
    check_constraints,
    compute_slice_load,
    identify_risky_assignments,
    validate_allocation_structure,
)
from resource_scheduler.environment.state import load_task_table

REAL_DATASET = Path(__file__).resolve().parents[1] / "datasets" / "raw" / "industrial_scheduling_dataset.csv"
TASK_IDS = ["T1", "T2", "T3"]


def valid_proposal():
    return {
        "assignments": [
            {"task_id": "T1", "machine_id": "M01", "network_slice_id": "NS_1", "rationale": "M01 is idle"},
            {"task_id": "T2", "machine_id": "M02", "network_slice_id": "NS_2", "rationale": "load balance"},
        ],
        "rejected": [{"task_id": "T3", "reason": "no capacity anywhere reasonable"}],
    }


def test_validate_allocation_structure_accepts_valid_proposal():
    assert validate_allocation_structure(TASK_IDS, valid_proposal()) == []


def test_validate_allocation_structure_rejects_non_dict():
    assert validate_allocation_structure(TASK_IDS, "nope")


def test_validate_allocation_structure_rejects_missing_task():
    proposal = valid_proposal()
    proposal["rejected"] = []  # T3 now covered by neither list
    errors = validate_allocation_structure(TASK_IDS, proposal)
    assert any("exactly the ranked task ids" in e for e in errors)


def test_validate_allocation_structure_rejects_duplicate_task():
    proposal = valid_proposal()
    proposal["assignments"].append({"task_id": "T1", "machine_id": "M03", "network_slice_id": "NS_1", "rationale": "dup"})
    errors = validate_allocation_structure(TASK_IDS, proposal)
    assert any("exactly the ranked task ids" in e for e in errors)


def test_validate_allocation_structure_rejects_incomplete_assignment():
    proposal = valid_proposal()
    del proposal["assignments"][0]["rationale"]
    errors = validate_allocation_structure(TASK_IDS, proposal)
    assert any("missing rationale" in e for e in errors)


def test_validate_allocation_structure_rejects_incomplete_rejection():
    proposal = valid_proposal()
    del proposal["rejected"][0]["reason"]
    errors = validate_allocation_structure(TASK_IDS, proposal)
    assert any("missing reason" in e for e in errors)


def test_check_constraints_blocks_maintenance_machine():
    assignments = [{"task_id": "T1", "machine_id": "M01", "network_slice_id": "NS_1"}]
    machine_status = {"M01": "Maintenance"}
    violations = check_constraints(assignments, machine_status, {}, slice_capacity=8)
    assert len(violations) == 1
    assert violations[0]["violation"] == "machine_in_maintenance"


def test_check_constraints_allows_active_machine_under_capacity():
    assignments = [{"task_id": "T1", "machine_id": "M01", "network_slice_id": "NS_1"}]
    machine_status = {"M01": "Active"}
    violations = check_constraints(assignments, machine_status, {"NS_1": 0}, slice_capacity=8)
    assert violations == []


def test_check_constraints_blocks_over_capacity_slice():
    assignments = [{"task_id": "T1", "machine_id": "M01", "network_slice_id": "NS_1"}]
    machine_status = {"M01": "Active"}
    violations = check_constraints(assignments, machine_status, {"NS_1": 8}, slice_capacity=8)
    assert len(violations) == 1
    assert violations[0]["violation"] == "slice_capacity_exceeded"


def test_check_constraints_is_cumulative_within_one_batch():
    """Regression-shaped test for the exact failure mode the cumulative
    running_load exists to prevent: several assignments to the same
    near-full slice in one batch must not all individually pass against
    the same stale pre-batch count."""
    assignments = [
        {"task_id": "T1", "machine_id": "M01", "network_slice_id": "NS_1"},
        {"task_id": "T2", "machine_id": "M02", "network_slice_id": "NS_1"},
        {"task_id": "T3", "machine_id": "M03", "network_slice_id": "NS_1"},
    ]
    machine_status = {"M01": "Active", "M02": "Active", "M03": "Active"}
    violations = check_constraints(assignments, machine_status, {"NS_1": 6}, slice_capacity=8)
    # slots available: 8-6=2 -> T1 and T2 fit, T3 pushes it over
    assert [v["task_id"] for v in violations] == ["T3"]


def test_check_constraints_unknown_machine_is_not_blocked_by_status_alone():
    """An unknown machine_id has no status entry -- machine_status.get()
    returns None, which is not in BLOCKED_MACHINE_STATUSES, so this
    passes the machine check (an invented machine_id is instead caught
    upstream by prompt instructions / a future stricter check, not
    silently treated as Maintenance)."""
    assignments = [{"task_id": "T1", "machine_id": "M99", "network_slice_id": "NS_1"}]
    violations = check_constraints(assignments, {}, {"NS_1": 0}, slice_capacity=8)
    assert violations == []


def test_build_allocation_events_marks_violations_and_passes():
    assignments = [
        {"task_id": "T1", "machine_id": "M01", "network_slice_id": "NS_1", "rationale": "fine"},
        {"task_id": "T2", "machine_id": "M02", "network_slice_id": "NS_1", "rationale": "also fine"},
    ]
    violations_by_task = {"T2": {"task_id": "T2", "violation": "slice_capacity_exceeded", "detail": "full"}}
    trigger_signal_by_task = {"T1": "final_score=0.9", "T2": "final_score=0.5"}
    events = build_allocation_events(assignments, violations_by_task, trigger_signal_by_task)
    assert len(events) == 2
    by_task = {e["task_id"]: e for e in events}
    assert by_task["T1"]["constraint_checks_passed"] is True
    assert by_task["T1"]["reason"] == "fine"
    assert by_task["T1"]["trigger_signal"] == "final_score=0.9"
    assert by_task["T2"]["constraint_checks_passed"] is False
    assert by_task["T2"]["reason"] == "full"


def test_compute_slice_load_and_task_lookup_against_real_dataset():
    df, _ = load_task_table(REAL_DATASET, inject_variance=True, seed=5)
    load = compute_slice_load(df, window=50)
    assert set(load.keys()) <= {"NS_1", "NS_2", "NS_3"}
    assert sum(load.values()) == 50

    some_ids = df.tail(50)["Task_ID"].tolist()[:3]
    lookup = build_task_lookup(df, some_ids)
    assert set(lookup.keys()) == set(some_ids)
    for entry in lookup.values():
        assert "task_type" in entry
        assert "previously_recorded_machine_id" in entry


def test_identify_risky_assignments_flags_overloaded_machine():
    assignments = [
        {"task_id": "T1", "machine_id": "M01", "network_slice_id": "NS_1", "rationale": "fine"},
        {"task_id": "T2", "machine_id": "M02", "network_slice_id": "NS_2", "rationale": "risky but needed"},
    ]
    machine_status = {"M01": "Active", "M02": "Overloaded"}
    risky = identify_risky_assignments(assignments, machine_status)
    assert len(risky) == 1
    assert risky[0]["task_id"] == "T2"
    assert risky[0]["rationale"] == "risky but needed"
    assert risky[0]["risk_reason"] == "target machine M02 is Overloaded"


def test_identify_risky_assignments_does_not_flag_maintenance():
    """Maintenance is a hard block in check_constraints -- an assignment
    to a Maintenance machine should never reach identify_risky_assignments
    at all (it would already be environment-rejected), so this isn't in
    RISKY_MACHINE_STATUSES. Confirms the two concepts stay separate."""
    assert "Maintenance" not in RISKY_MACHINE_STATUSES
    assignments = [{"task_id": "T1", "machine_id": "M01", "network_slice_id": "NS_1"}]
    risky = identify_risky_assignments(assignments, {"M01": "Maintenance"})
    assert risky == []


def test_identify_risky_assignments_none_flagged_when_all_healthy():
    assignments = [{"task_id": "T1", "machine_id": "M01", "network_slice_id": "NS_1"}]
    assert identify_risky_assignments(assignments, {"M01": "Active"}) == []


def test_identify_risky_assignments_unknown_machine_not_flagged():
    """No fabricated risk for a machine that isn't in machine_status at
    all -- absence of data is not itself a risk signal."""
    assignments = [{"task_id": "T1", "machine_id": "M99", "network_slice_id": "NS_1"}]
    assert identify_risky_assignments(assignments, {}) == []
