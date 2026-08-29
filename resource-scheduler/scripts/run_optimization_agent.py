#!/usr/bin/env python3
"""
Agent #5: run the Optimization agent standalone. Unlike every other
agent script in this project, this one takes no --data argument -- it
reads existing run history under runs/ rather than the task table
directly (see environment/policy_evidence.py). On a valid proposal, it
sends a policy_update_proposal to Human Oversight's mailbox -- the
third A2A hop.

Usage:
    set -a; source .env; set +a
    python scripts/run_optimization_agent.py --model rit-qwen3-8b
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from resource_scheduler.a2a.mailbox import Mailbox
from resource_scheduler.cli_common import make_run_dir, make_tracer, make_transcript_writer, resolve_model_endpoint
from resource_scheduler.model_client import ModelClient
from resource_scheduler.steps.optimization_step import run_optimization_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-runs", type=int, default=10, help="how many recent run dirs to aggregate over")
    parser.add_argument("--model", default=None)
    parser.add_argument("--use-gateway", action="store_true")
    parser.add_argument("--use-local", action="store_true")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    run_id, run_dir = make_run_dir(args.run_id)
    trace = make_tracer(run_dir / "trace.jsonl")
    write_transcript = make_transcript_writer(run_dir)

    base_url, api_key, default_model = resolve_model_endpoint(
        args.use_gateway, args.model, "qwen3:8b", "rit-qwen3-8b",
        use_local=args.use_local,
    )
    client = ModelClient(base_url=base_url, api_key=api_key, default_model=default_model)
    mailbox = Mailbox(on_event=lambda e: trace(e["type"], **e["payload"]))

    result = run_optimization_step(client, mailbox, n_runs=args.n_runs, trace_fn=lambda record: trace(**record))
    write_transcript("optimization", result.messages)

    if not result.ok:
        print(f"stopped_reason={result.stopped_reason} — no run history to reason about yet. "
              "Run agents #2-4 at least once first.")
        return

    report_path = run_dir / "optimization_report.json"
    report_path.write_text(json.dumps({
        "ok": result.ok,
        "valid": result.valid,
        "validation_errors": result.validation_errors,
        "evidence": result.evidence,
        "llm_proposal": result.llm_proposal,
        "sent_to_human_oversight": result.sent_to_human_oversight,
        "stopped_reason": result.stopped_reason,
        "turns_used": result.turns_used,
    }, indent=2, default=str))

    print(f"ok={result.ok} valid={result.valid} sent_to_human_oversight={result.sent_to_human_oversight}")
    if result.validation_errors:
        print("validation_errors:", result.validation_errors)
    inbox = mailbox.peek("human_oversight")
    print(f"human_oversight inbox (unconsumed): {len(inbox)} message(s)")
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
