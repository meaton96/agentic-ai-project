"""
Tests for environment/queue.py -- no LLM involved. Covers the raw
signal computations Task Prioritization's tool exposes, and the
validation functions steps/task_prioritization_step.py gates on.
"""
from pathlib import Path

import pandas as pd
import pytest

from resource_scheduler.environment.queue import (
    compute_availability_bonus,
    compute_energy_cost_proxy,
    compute_pending_queue,
    is_ranking_score_consistent,
    validate_ranking_proposal,
)
from resource_scheduler.environment.state import compute_snapshot, load_task_table

REAL_DATASET = Path(__file__).resolve().parents[1] / "datasets" / "raw" / "industrial_scheduling_dataset.csv"


@pytest.fixture
def loaded_real_table():
    return load_task_table(REAL_DATASET, inject_variance=True, seed=3)


def test_compute_pending_queue_respects_queue_size(loaded_real_table):
    df, _ = loaded_real_table
    queue = compute_pending_queue(df, queue_size=5)
    assert len(queue) == 5
    ids = [t["task_id"] for t in queue]
    assert len(set(ids)) == 5  # no duplicates -- distinct rows


def test_compute_pending_queue_needs_reallocation_matches_source(loaded_real_table):
    df, _ = loaded_real_table
    queue = compute_pending_queue(df, queue_size=10)
    expected = dict(zip(df.tail(10)["Task_ID"], df.tail(10)["Reallocation"] == "Yes"))
    for t in queue:
        assert t["needs_reallocation"] == expected[t["task_id"]]


def test_compute_energy_cost_proxy_zero_spread_is_neutral():
    tasks = [
        {"task_id": "T1", "execution_time": 10.0},
        {"task_id": "T2", "execution_time": 10.0},
    ]
    proxy = compute_energy_cost_proxy(tasks)
    assert proxy == {"T1": 0.5, "T2": 0.5}


def test_compute_energy_cost_proxy_min_max_scaling():
    tasks = [
        {"task_id": "cheap", "execution_time": 0.0},
        {"task_id": "mid", "execution_time": 5.0},
        {"task_id": "expensive", "execution_time": 10.0},
    ]
    proxy = compute_energy_cost_proxy(tasks)
    assert proxy["cheap"] == 0.0
    assert proxy["expensive"] == 1.0
    assert proxy["mid"] == 0.5


def test_compute_energy_cost_proxy_empty_queue():
    assert compute_energy_cost_proxy([]) == {}


def test_compute_availability_bonus_uses_machine_status():
    tasks = [{"task_id": "T1", "machine_id": "M01"}, {"task_id": "T2", "machine_id": "M02"}]
    machine_snapshot = [
        {"machine_id": "M01", "status": "Idle"},
        {"machine_id": "M02", "status": "Overloaded"},
    ]
    bonus = compute_availability_bonus(tasks, machine_snapshot)
    assert bonus["T1"] == 1.0
    assert bonus["T2"] == 0.1


def test_compute_availability_bonus_unknown_machine_is_neutral_low():
    tasks = [{"task_id": "T1", "machine_id": "M99"}]
    bonus = compute_availability_bonus(tasks, [])
    assert bonus["T1"] == 0.3


PENDING_IDS = ["T1", "T2", "T3"]


def valid_proposal():
    return {
        "ranked_task_ids": ["T2", "T1", "T3"],
        "score_breakdown": {
            "T1": {"urgency": 0.2, "energy_cost": 0.5, "availability_bonus": 0.6, "final_score": 0.5},
            "T2": {"urgency": 1.0, "energy_cost": 0.2, "availability_bonus": 1.0, "final_score": 0.9},
            "T3": {"urgency": 0.2, "energy_cost": 0.8, "availability_bonus": 0.1, "final_score": 0.1},
        },
        "reasoning": "T2 is urgent and cheap; T3 is neither.",
    }


def test_validate_ranking_proposal_accepts_valid_permutation():
    assert validate_ranking_proposal(PENDING_IDS, valid_proposal()) == []


def test_validate_ranking_proposal_rejects_non_dict():
    errors = validate_ranking_proposal(PENDING_IDS, "not a dict")
    assert errors


def test_validate_ranking_proposal_rejects_missing_task():
    proposal = valid_proposal()
    proposal["ranked_task_ids"] = ["T1", "T2"]  # T3 dropped
    errors = validate_ranking_proposal(PENDING_IDS, proposal)
    assert any("permutation" in e for e in errors)


def test_validate_ranking_proposal_rejects_duplicate_task():
    proposal = valid_proposal()
    proposal["ranked_task_ids"] = ["T1", "T1", "T3"]
    errors = validate_ranking_proposal(PENDING_IDS, proposal)
    assert any("permutation" in e for e in errors)


def test_validate_ranking_proposal_rejects_invented_task():
    proposal = valid_proposal()
    proposal["ranked_task_ids"] = ["T1", "T2", "T99"]
    errors = validate_ranking_proposal(PENDING_IDS, proposal)
    assert any("permutation" in e for e in errors)


def test_validate_ranking_proposal_rejects_incomplete_breakdown():
    proposal = valid_proposal()
    del proposal["score_breakdown"]["T3"]
    errors = validate_ranking_proposal(PENDING_IDS, proposal)
    assert any("missing entries" in e for e in errors)


def test_validate_ranking_proposal_rejects_extra_breakdown_entry():
    proposal = valid_proposal()
    proposal["score_breakdown"]["T99"] = {"final_score": 0.5}
    errors = validate_ranking_proposal(PENDING_IDS, proposal)
    assert any("unknown task ids" in e for e in errors)


def test_is_ranking_score_consistent_true_for_descending_scores():
    assert is_ranking_score_consistent(valid_proposal())


def test_is_ranking_score_consistent_false_for_out_of_order_scores():
    proposal = valid_proposal()
    proposal["ranked_task_ids"] = ["T1", "T2", "T3"]  # T1's score (0.5) < T2's (0.9)
    assert not is_ranking_score_consistent(proposal)


def test_is_ranking_score_consistent_false_when_breakdown_entry_missing():
    proposal = {"ranked_task_ids": ["T1", "T2"], "score_breakdown": {"T1": {"final_score": 0.5}}}
    assert not is_ranking_score_consistent(proposal)
