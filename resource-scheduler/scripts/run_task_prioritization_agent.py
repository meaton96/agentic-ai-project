#!/usr/bin/env python3
"""
Agent #2: run the Task Prioritization agent standalone against the task
table. Deterministic code (environment/queue.py) computes the raw
per-task signals; the LLM combines them into a ranking. On a valid
proposal, the ranking is sent via the Mailbox to "resource_allocation"
-- the first A2A hop in this project, even though agent #3 doesn't
exist yet to consume it (its inbox is inspectable below).

Usage:
    set -a; source .env; set +a
    python scripts/run_task_prioritization_agent.py \\
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
from resource_scheduler.steps.task_prioritization_step import run_task_prioritization_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--queue-size", type=int, default=8)
    parser.add_argument("--snapshot-window", type=int, default=200)
    parser.add_argument("--no-inject-variance", action="store_true")
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
    result = run_task_prioritization_step(
        df, variance_injected, client,
        queue_size=args.queue_size, snapshot_window=args.snapshot_window,
        trace_fn=lambda record: trace(**record), mailbox=mailbox,
    )
    write_transcript("task_prioritization", result.messages)

    report_path = run_dir / "task_prioritization_report.json"
    report_path.write_text(json.dumps({
        "ok": result.ok,
        "valid": result.valid,
        "validation_errors": result.validation_errors,
        "score_inconsistent": result.score_inconsistent,
        "deterministic_facts": result.deterministic_facts,
        "llm_proposal": result.llm_proposal,
        "sent_to_resource_allocation": result.sent_to_resource_allocation,
        "stopped_reason": result.stopped_reason,
        "turns_used": result.turns_used,
    }, indent=2, default=str))

    print(f"ok={result.ok} valid={result.valid} score_inconsistent={result.score_inconsistent} "
          f"sent_to_resource_allocation={result.sent_to_resource_allocation}")
    if result.validation_errors:
        print("validation_errors:", result.validation_errors)
    inbox = mailbox.peek("resource_allocation")
    print(f"resource_allocation inbox (unconsumed, agent #3 not built yet): {len(inbox)} message(s)")
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
