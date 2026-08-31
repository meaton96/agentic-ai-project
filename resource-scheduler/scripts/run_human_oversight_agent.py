#!/usr/bin/env python3
"""
Agent #6: run the Human Oversight agent. By default this chains from a
real Optimization run (agent #5) so there's an actual proposal to
review. Pass --synthetic-proposal to skip agent #5 and inject a
placeholder proposal instead, for testing Human Oversight in isolation.

Usage:
    set -a; source .env; set +a
    python scripts/run_human_oversight_agent.py --model rit-qwen3-8b
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from resource_scheduler.a2a.mailbox import Mailbox
from resource_scheduler.cli_common import make_retry_logger, make_run_dir, make_tracer, make_transcript_writer, resolve_model_endpoint
from resource_scheduler.model_client import ModelClient
from resource_scheduler.steps.optimization_step import run_optimization_step
from resource_scheduler.steps.oversight_step import run_oversight_step


def inject_synthetic_proposal(mailbox: Mailbox) -> None:
    """Deterministic placeholder proposal -- NOT a stand-in for a real
    Optimization decision, only for exercising Human Oversight's
    mailbox-consumption and review logic in isolation."""
    mailbox.send(
        sender="synthetic_proposal", recipient="human_oversight",
        message_type="policy_update_proposal",
        payload={
            "policy_updates": {"slice_capacity": 12},
            "evidence": "synthetic placeholder proposal, not a real Optimization decision",
            "recommend_apply": True,
            "underlying_evidence": {
                "n_runs_scanned": 1, "n_run_dirs_considered": 1,
                "allocation_acceptance_rate": 0.5, "allocation_accepted": 5,
                "allocation_environment_rejected": 5, "allocation_agent_rejected": 0,
                "ranking_valid_rate": 1.0, "ranking_score_inconsistent_rate": 0.0,
                "reroute_acceptance_rate": None, "reroute_accepted": 0, "reroute_environment_rejected": 0,
            },
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-runs", type=int, default=10, help="passed through to Optimization if it runs")
    parser.add_argument("--synthetic-proposal", action="store_true",
                         help="skip Optimization entirely; inject a placeholder proposal")
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
    client = ModelClient(base_url=base_url, api_key=api_key, default_model=default_model, on_retry=make_retry_logger())
    mailbox = Mailbox(on_event=lambda e: trace(e["type"], **e["payload"]))

    if args.synthetic_proposal:
        inject_synthetic_proposal(mailbox)
        print("Injected synthetic policy update proposal (Optimization skipped).")
    else:
        opt_result = run_optimization_step(client, mailbox, n_runs=args.n_runs, trace_fn=lambda record: trace(**record))
        write_transcript("optimization", opt_result.messages)
        print(f"Optimization: ok={opt_result.ok} valid={opt_result.valid} "
              f"sent_to_human_oversight={opt_result.sent_to_human_oversight}")
        if not opt_result.sent_to_human_oversight:
            print("No valid proposal to review -- exiting. "
                  "Use --synthetic-proposal to test Human Oversight in isolation.")
            return

    ho_result = run_oversight_step(client, mailbox, trace_fn=lambda record: trace(**record))
    write_transcript("human_oversight", ho_result.messages)

    report_path = run_dir / "human_oversight_report.json"
    report_path.write_text(json.dumps({
        "ok": ho_result.ok,
        "verdict": ho_result.verdict,
        "concerns": ho_result.concerns,
        "reasoning": ho_result.reasoning,
        "review_bundle": ho_result.review_bundle,
        "proposal_source": ho_result.proposal_source,
        "stopped_reason": ho_result.stopped_reason,
    }, indent=2, default=str))

    print(f"ok={ho_result.ok} verdict={ho_result.verdict} concerns={ho_result.concerns}")
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
