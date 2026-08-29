"""
Tests for run_reroute_validation_step -- unlike the other steps, this
one is pure Python (no LLM call, no ModelClient), so it can be tested
directly rather than only its deterministic dependencies. This is the
receiving end of the second A2A hop (Failure Recovery -> Resource
Allocation).
"""
from pathlib import Path

from resource_scheduler.a2a.mailbox import Mailbox
from resource_scheduler.environment.state import load_task_table
from resource_scheduler.steps.resource_allocation_step import run_reroute_validation_step

REAL_DATASET = Path(__file__).resolve().parents[1] / "datasets" / "raw" / "industrial_scheduling_dataset.csv"


def test_no_reroute_request_available_returns_ok_false():
    df, variance_injected = load_task_table(REAL_DATASET, inject_variance=True, seed=9)
    mailbox = Mailbox()
    result = run_reroute_validation_step(df, variance_injected, mailbox)
    assert result.ok is False
    assert result.stopped_reason == "no_reroute_request_available"


def test_valid_reroute_is_accepted_and_produces_an_event():
    df, variance_injected = load_task_table(REAL_DATASET, inject_variance=True, seed=9)
    mailbox = Mailbox()
    # Force the last row's machine to a known-good status rather than
    # trusting whatever the real data's mode happens to be -- otherwise
    # this test's "valid" assumption could silently be wrong if that
    # machine's real recent status happened to be Maintenance/saturated.
    df = df.copy()
    df.loc[df.index[-1], "Machine_Status"] = "Active"
    machine_id = df["Machine_ID"].iloc[-1]
    slice_id = df["Network_Slice_ID"].iloc[-1]
    mailbox.send(
        sender="failure_recovery", recipient="resource_allocation", message_type="reroute_request",
        payload={
            "reroute_proposals": [{"task_id": "T_TEST", "new_machine_id": machine_id,
                                    "new_network_slice_id": slice_id, "reasoning": "test reroute"}],
            "incidents": [],
        },
    )
    result = run_reroute_validation_step(df, variance_injected, mailbox, snapshot_window=1, slice_capacity=10_000)
    assert result.ok is True
    assert result.reroute_source == "failure_recovery"
    assert len(result.accepted_reroutes) == 1
    assert result.accepted_reroutes[0]["task_id"] == "T_TEST"
    assert len(result.events) == 1
    assert result.events[0]["constraint_checks_passed"] is True
    assert result.events[0]["reason"] == "test reroute"


def test_reroute_to_maintenance_machine_is_environment_rejected():
    df, variance_injected = load_task_table(REAL_DATASET, inject_variance=True, seed=9)
    mailbox = Mailbox()
    # Force a machine into Maintenance in the tail window this step will read.
    df = df.copy()
    df.loc[df.index[-1], "Machine_Status"] = "Maintenance"
    faulty_machine = df["Machine_ID"].iloc[-1]

    mailbox.send(
        sender="failure_recovery", recipient="resource_allocation", message_type="reroute_request",
        payload={
            "reroute_proposals": [{"task_id": "T_TEST", "new_machine_id": faulty_machine,
                                    "new_network_slice_id": "NS_1", "reasoning": "bad reroute on purpose"}],
            "incidents": [],
        },
    )
    result = run_reroute_validation_step(df, variance_injected, mailbox, snapshot_window=1)
    assert result.ok is True
    assert result.accepted_reroutes == []
    assert len(result.environment_rejected) == 1
    assert result.environment_rejected[0]["violation"]["violation"] == "machine_in_maintenance"
    assert result.events[0]["constraint_checks_passed"] is False


def test_reroute_request_inbox_does_not_consume_task_ranking_messages():
    """Confirms the message_type filter from the mailbox fix actually
    protects this step: a task_ranking message sitting in the same
    inbox must be left alone."""
    df, variance_injected = load_task_table(REAL_DATASET, inject_variance=True, seed=9)
    mailbox = Mailbox()
    mailbox.send(sender="task_prioritization", recipient="resource_allocation",
                  message_type="task_ranking", payload={"ranked_task_ids": ["T1"]})
    result = run_reroute_validation_step(df, variance_injected, mailbox)
    assert result.ok is False  # no reroute_request present, task_ranking doesn't count
    remaining = mailbox.peek("resource_allocation")
    assert len(remaining) == 1
    assert remaining[0].message_type == "task_ranking"
