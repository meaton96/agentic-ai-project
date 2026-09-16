#!/usr/bin/env python3
"""
Agent #4: run the Failure Recovery agent, then close the loop by
running the deterministic reroute-validation step on Resource
Allocation's side (the second A2A hop: Failure Recovery ->
Resource Allocation, re-validated against the same constraint gate,
no second LLM call).

By default this chains from real Task Prioritization + Resource
Allocation runs (agents #2 and #3) so "committed_assignments" reflects
an actual prior decision. Pass --synthetic-committed to skip both and
inject a placeholder committed-assignment list instead, for testing
Failure Recovery in isolation.

Usage:
    set -a; source .env; set +a
    python scripts/run_failure_recovery_agent.py \\
        --data datasets/raw/industrial_scheduling_dataset.csv \\
        --model rit-qwen3-8b
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from resource_scheduler.a2a.mailbox import Mailbox
from resource_scheduler.cli_common import make_retry_logger, make_run_dir, make_tracer, make_transcript_writer, resolve_model_endpoint
from resource_scheduler.environment.state import load_task_table
from resource_scheduler.model_client import ModelClient
from resource_scheduler.steps.failure_recovery_step import run_failure_recovery_step
from resource_scheduler.steps.resource_allocation_step import run_reroute_validation_step, run_resource_allocation_step
from resource_scheduler.steps.task_prioritization_step import run_task_prioritization_step


def synthetic_committed_assignments(df, queue_size: int) -> list[dict]:
    """Placeholder committed assignments -- NOT a stand-in for a real
    Resource Allocation decision, only for exercising Failure Recovery's
    incident/affected-task logic in isolation. Assigns each task to its
    own historically-recorded machine/slice, which is enough to test
    whether an incident on that machine correctly flags it as affected."""
    recent = df.tail(queue_size)
    return [
        {"task_id": row["Task_ID"], "machine_id": row["Machine_ID"], "network_slice_id": row["Network_Slice_ID"]}
        for _, row in recent.iterrows()
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--queue-size", type=int, default=8)
    parser.add_argument("--snapshot-window", type=int, default=200)
    parser.add_argument("--slice-capacity", type=int, default=8)
    parser.add_argument("--before-row", type=int, default=None, help="default: len(df) - 50")
    parser.add_argument("--after-row", type=int, default=None, help="default: len(df)")
    parser.add_argument("--no-inject-variance", action="store_true")
    parser.add_argument("--synthetic-committed", action="store_true",
                         help="skip Task Prioritization + Resource Allocation; inject placeholder committed assignments")
    parser.add_argument("--model", default=None)
    parser.add_argument("--use-gateway", action="store_true")
    parser.add_argument("--use-local", action="store_true")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    run_id, run_dir = make_run_dir(args.run_id)
    trace = make_tracer(run_dir / "trace.jsonl")
    write_transcript = make_transcript_writer(run_dir)

    df, variance_injected = load_task_table(args.data, inject_variance=not args.no_inject_variance)
    print(f"Loaded task table: {len(df)} rows, synthetic_variance_injected={variance_injected}")
    trace("task_table_loaded", n_rows=len(df), variance_injected=variance_injected)

    before_row = args.before_row if args.before_row is not None else max(len(df) - 50, 1)
    after_row = args.after_row if args.after_row is not None else len(df)

    base_url, api_key, default_model = resolve_model_endpoint(
        args.use_gateway, args.model, "qwen3:8b", "rit-qwen3-8b",
        use_local=args.use_local,
    )
    client = ModelClient(base_url=base_url, api_key=api_key, default_model=default_model, on_retry=make_retry_logger())
    mailbox = Mailbox(on_event=lambda e: trace(e["type"], **e["payload"]))

    if args.synthetic_committed:
        committed_assignments = synthetic_committed_assignments(df, args.queue_size)
        print(f"Injected {len(committed_assignments)} synthetic committed assignments "
              "(Task Prioritization + Resource Allocation skipped).")
    else:
        tp_result = run_task_prioritization_step(
            df, variance_injected, client,
            queue_size=args.queue_size, snapshot_window=args.snapshot_window,
            trace_fn=lambda record: trace(**record), mailbox=mailbox,
        )
        write_transcript("task_prioritization", tp_result.messages)
        print(f"Task Prioritization: ok={tp_result.ok} valid={tp_result.valid}")
        if not tp_result.sent_to_resource_allocation:
            print("No valid ranking -- exiting. Use --synthetic-committed to test Failure Recovery in isolation.")
            return

        ra_result = run_resource_allocation_step(
            df, variance_injected, client, mailbox,
            snapshot_window=args.snapshot_window, slice_capacity=args.slice_capacity,
            trace_fn=lambda record: trace(**record),
        )
        write_transcript("resource_allocation", ra_result.messages)
        print(f"Resource Allocation: valid={ra_result.valid} accepted={len(ra_result.accepted_assignments)}")
        committed_assignments = ra_result.accepted_assignments
        if not committed_assignments:
            print("No accepted assignments -- exiting. Use --synthetic-committed to test Failure Recovery in isolation.")
            return

    fr_result = run_failure_recovery_step(
        df, variance_injected, client, mailbox, committed_assignments,
        before_row=before_row, after_row=after_row, snapshot_window=args.snapshot_window,
        trace_fn=lambda record: trace(**record),
    )
    write_transcript("failure_recovery", fr_result.messages)
    print(f"Failure Recovery: stopped_reason={fr_result.stopped_reason} "
          f"n_incidents={len(fr_result.incidents)} n_affected={len(fr_result.affected_tasks)} "
          f"valid={fr_result.valid} reroute_avoids_source={fr_result.reroute_avoids_source} "
          f"sent_to_resource_allocation={fr_result.sent_to_resource_allocation}")

    reroute_result = None
    if fr_result.sent_to_resource_allocation:
        reroute_result = run_reroute_validation_step(
            df, variance_injected, mailbox,
            snapshot_window=args.snapshot_window, slice_capacity=args.slice_capacity,
        )
        print(f"Reroute validation: accepted={len(reroute_result.accepted_reroutes)} "
              f"environment_rejected={len(reroute_result.environment_rejected)}")

    report_path = run_dir / "failure_recovery_report.json"
    report_path.write_text(json.dumps({
        "before_row": before_row, "after_row": after_row,
        "incidents": fr_result.incidents,
        "affected_tasks": fr_result.affected_tasks,
        "valid": fr_result.valid,
        "validation_errors": fr_result.validation_errors,
        "reroute_avoids_source": fr_result.reroute_avoids_source,
        "reroute_proposals": fr_result.reroute_proposals,
        "reroute_validation": None if reroute_result is None else {
            "accepted_reroutes": reroute_result.accepted_reroutes,
            "environment_rejected": reroute_result.environment_rejected,
            "events": reroute_result.events,
        },
        "stopped_reason": fr_result.stopped_reason,
    }, indent=2, default=str))
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
