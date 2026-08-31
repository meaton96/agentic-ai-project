#!/usr/bin/env python3
"""
Agent #3: run the Resource Allocation agent. By default this chains
from a real Task Prioritization run (agent #2) so the mailbox has a
real ranking to consume -- the actual A2A path. Pass
--synthetic-ranking to skip agent #2 entirely and inject a deterministic
placeholder ranking instead, for testing Resource Allocation in
isolation (agent #2's LLM call may be slow/unavailable/still being
iterated on; this keeps agent #3 independently testable, same
one-at-a-time philosophy as agent #1/#2).

Usage:
    set -a; source .env; set +a
    python scripts/run_resource_allocation_agent.py \\
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
from resource_scheduler.environment.queue import compute_pending_queue
from resource_scheduler.environment.state import load_task_table
from resource_scheduler.model_client import ModelClient
from resource_scheduler.steps.resource_allocation_step import run_resource_allocation_step
from resource_scheduler.steps.task_prioritization_step import run_task_prioritization_step


def inject_synthetic_ranking(mailbox: Mailbox, df, queue_size: int) -> None:
    """Deterministic placeholder ranking (original queue order, evenly
    decreasing scores) -- NOT a stand-in for Task Prioritization's
    actual reasoning, only for exercising Resource Allocation's
    mailbox-consumption and constraint-gating logic in isolation."""
    tasks = compute_pending_queue(df, queue_size=queue_size)
    ranked_task_ids = [t["task_id"] for t in tasks]
    score_breakdown = {
        tid: {"urgency": None, "energy_cost": None, "availability_bonus": None,
              "final_score": round(1.0 - i * (1.0 / max(len(ranked_task_ids), 1)), 4)}
        for i, tid in enumerate(ranked_task_ids)
    }
    mailbox.send(
        sender="synthetic_ranking", recipient="resource_allocation",
        message_type="task_ranking",
        payload={"ranked_task_ids": ranked_task_ids, "score_breakdown": score_breakdown,
                 "reasoning": "synthetic placeholder ranking, not a real Task Prioritization proposal"},
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--queue-size", type=int, default=8)
    parser.add_argument("--snapshot-window", type=int, default=200)
    parser.add_argument("--slice-capacity", type=int, default=8)
    parser.add_argument("--no-inject-variance", action="store_true")
    parser.add_argument("--synthetic-ranking", action="store_true",
                         help="skip Task Prioritization entirely; inject a deterministic placeholder ranking")
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

    base_url, api_key, default_model = resolve_model_endpoint(
        args.use_gateway, args.model, "qwen3:8b", "rit-qwen3-8b",
        use_local=args.use_local,
    )
    client = ModelClient(base_url=base_url, api_key=api_key, default_model=default_model, on_retry=make_retry_logger())
    mailbox = Mailbox(on_event=lambda e: trace(e["type"], **e["payload"]))

    if args.synthetic_ranking:
        inject_synthetic_ranking(mailbox, df, args.queue_size)
        print("Injected synthetic ranking (Task Prioritization skipped).")
    else:
        tp_result = run_task_prioritization_step(
            df, variance_injected, client,
            queue_size=args.queue_size, snapshot_window=args.snapshot_window,
            trace_fn=lambda record: trace(**record), mailbox=mailbox,
        )
        write_transcript("task_prioritization", tp_result.messages)
        print(f"Task Prioritization: ok={tp_result.ok} valid={tp_result.valid} "
              f"sent_to_resource_allocation={tp_result.sent_to_resource_allocation}")
        if not tp_result.sent_to_resource_allocation:
            print("No valid ranking to allocate against -- exiting. "
                  "Use --synthetic-ranking to test Resource Allocation in isolation.")
            return

    ra_result = run_resource_allocation_step(
        df, variance_injected, client, mailbox,
        snapshot_window=args.snapshot_window, slice_capacity=args.slice_capacity,
        trace_fn=lambda record: trace(**record),
    )
    write_transcript("resource_allocation", ra_result.messages)

    report_path = run_dir / "resource_allocation_report.json"
    report_path.write_text(json.dumps({
        "ok": ra_result.ok,
        "valid": ra_result.valid,
        "validation_errors": ra_result.validation_errors,
        "accepted_assignments": ra_result.accepted_assignments,
        "environment_rejected": ra_result.environment_rejected,
        "agent_rejected": ra_result.agent_rejected,
        "events": ra_result.events,
        "ranking_source": ra_result.ranking_source,
        "stopped_reason": ra_result.stopped_reason,
        "turns_used": ra_result.turns_used,
    }, indent=2, default=str))

    print(f"ok={ra_result.ok} valid={ra_result.valid} "
          f"accepted={len(ra_result.accepted_assignments)} "
          f"environment_rejected={len(ra_result.environment_rejected)} "
          f"agent_rejected={len(ra_result.agent_rejected)}")
    if ra_result.validation_errors:
        print("validation_errors:", ra_result.validation_errors)
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
