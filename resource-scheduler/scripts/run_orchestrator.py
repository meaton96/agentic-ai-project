#!/usr/bin/env python3
"""
The Orchestrator: runs all six resource-scheduler agents in sequence
against one task table, chaining them through the same A2A mailbox
their standalone scripts already use. Mirrors
agentic-ml-classification/scripts/run_orchestrator.py's shape (run-dir
setup, a fail() helper, a master report dict, per-phase report files,
a final plain-text LLM-narrated summary) -- see that file for the
original pattern this one borrows from.

One deliberate departure from that pattern: the ML pipeline's
orchestrator sys.exit(1)s on almost any imperfect outcome (failed
intake, no candidate passing its gates) because nothing downstream CAN
proceed without a valid target column or a passing model. This
project's six agents were built the opposite way -- every one already
degrades gracefully on its own (no_ranking_available,
no_incidents_detected, no_run_history_available). So this orchestrator
only calls fail() for genuine structural breakage (the data file
doesn't exist / can't be loaded, a step never got far enough to call
its tool at all); an individual agent's invalid or imperfect LLM output
is recorded in the report and the chain continues, since every later
step is already designed to handle "the thing upstream of me didn't
happen" cleanly.

Sequence: Load Monitor (informational) -> Task Prioritization (sends a
ranking) -> Resource Allocation (consumes it, applies the constraint
gate) -> Failure Recovery (uses Resource Allocation's own accepted
assignments as "currently committed", may send a reroute_request) ->
reroute validation (deterministic, only if Failure Recovery sent one) ->
Optimization (reads run history, INCLUDING this run's own report files,
already written to disk by the time it runs) -> Human Oversight
(reviews Optimization's proposal, if any).

Usage:
    set -a; source .env; set +a
    python scripts/run_orchestrator.py \\
        --data datasets/raw/industrial_scheduling_dataset.csv \\
        --model rit-qwen3-8b
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from resource_scheduler.a2a.mailbox import Mailbox
from resource_scheduler.cli_common import make_retry_logger, make_run_dir, make_tracer, make_transcript_writer, resolve_model_endpoint
from resource_scheduler.environment.state import load_task_table
from resource_scheduler.events import make_event_emitter, make_event_logger
from resource_scheduler.model_client import ModelClient
from resource_scheduler.steps.failure_recovery_step import run_failure_recovery_step
from resource_scheduler.steps.load_monitor_step import run_load_monitor_step
from resource_scheduler.steps.optimization_step import run_optimization_step
from resource_scheduler.steps.oversight_step import run_oversight_step
from resource_scheduler.steps.resource_allocation_step import run_reroute_validation_step, run_resource_allocation_step
from resource_scheduler.steps.task_prioritization_step import run_task_prioritization_step


def main(on_event: Optional[Callable[[dict], None]] = None, prompt_override_dir: Optional[str] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--queue-size", type=int, default=8)
    parser.add_argument("--snapshot-window", type=int, default=200,
                         help="rows treated as 'current' state for every agent's snapshot")
    parser.add_argument("--slice-capacity", type=int, default=8)
    parser.add_argument("--before-row", type=int, default=None,
                         help="Failure Recovery's 'before' replay cutoff; default: len(df) - 50")
    parser.add_argument("--after-row", type=int, default=None,
                         help="Failure Recovery's 'after' replay cutoff; default: len(df)")
    parser.add_argument("--n-runs", type=int, default=10, help="Optimization: how many recent run dirs to aggregate over")
    parser.add_argument("--no-inject-variance", action="store_true")
    parser.add_argument("--skip-optimization", action="store_true")
    parser.add_argument("--skip-oversight", action="store_true", help="implies --skip-optimization has no effect; "
                         "skips Human Oversight even if Optimization sent a proposal")
    parser.add_argument("--model", default=None)
    parser.add_argument("--use-gateway", action="store_true")
    parser.add_argument("--use-local", action="store_true")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--prompt-override-dir", default=None)
    args = parser.parse_args()

    effective_prompt_override_dir = (
        prompt_override_dir if prompt_override_dir is not None else args.prompt_override_dir
    )

    run_id, run_dir = make_run_dir(args.run_id)
    # Two separate event streams, same split the ML pipeline's orchestrator
    # uses: trace.jsonl (agent_runtime.py's per-model-call trace, passed as
    # trace_fn to each step below) and events.jsonl (step-level lifecycle
    # events -- prompt_loaded, agent_started, tool_called, each agent's
    # final report event, AND every mailbox send, all in one place). `emit`
    # is a single callable satisfying both emit_event()'s and Mailbox's
    # on_event contract (each calls it with one {"phase","type","payload"}
    # dict), so the same object wires into every step call AND the mailbox
    # directly -- this is events.py's make_event_emitter/make_event_logger,
    # unused until now (built alongside agent #1, never actually called).
    trace = make_tracer(run_dir / "trace.jsonl")
    write_transcript = make_transcript_writer(run_dir)
    emit = make_event_emitter(run_id, external_on_event=on_event, persist_fn=make_event_logger(run_dir))

    base_url, api_key, default_model = resolve_model_endpoint(
        args.use_gateway, args.model, "qwen3:8b", "rit-qwen3-8b",
        use_local=args.use_local,
    )
    client = ModelClient(base_url=base_url, api_key=api_key, default_model=default_model, on_retry=make_retry_logger())
    mailbox = Mailbox(on_event=emit)

    report: dict = {"run_id": run_id, "model": default_model, "status": "in_progress", "issues": []}

    def write_report():
        (run_dir / "orchestrator_report.json").write_text(json.dumps(report, indent=2, default=str))

    def fail(status: str, **extra):
        report["status"] = status
        report.update(extra)
        write_report()
        print(f"\nABORTING: {status}")
        sys.exit(1)

    # --- 0. Load the task table (shared by agents #1-4) ---
    try:
        df, variance_injected = load_task_table(args.data, inject_variance=not args.no_inject_variance)
    except (FileNotFoundError, ValueError) as e:
        fail("failed_data_load", error=str(e))
        return  # unreachable after sys.exit, satisfies type checkers
    print(f"Loaded task table: {len(df)} rows, synthetic_variance_injected={variance_injected}")
    report["data"] = {"path": args.data, "n_rows": len(df), "synthetic_variance_injected": variance_injected}

    # --- 1. Load Monitor (read-only, informational) ---
    print("\nRunning Load Monitor agent...")
    lm_result = run_load_monitor_step(
        df, variance_injected, client, window=args.snapshot_window, model=default_model,
        trace_fn=lambda record: trace(**record), on_event=emit, prompt_override_dir=effective_prompt_override_dir,
    )
    write_transcript("load_monitor", lm_result.messages)
    (run_dir / "load_monitor_report.json").write_text(json.dumps({
        "ok": lm_result.ok,
        "deterministic_report": lm_result.deterministic_report,
        "llm_narrative": lm_result.llm_narrative,
        "flag_mismatch": lm_result.flag_mismatch,
        "stopped_reason": lm_result.stopped_reason,
    }, indent=2, default=str))
    if not lm_result.ok:
        fail("failed_load_monitor", error="Load Monitor never called get_resource_snapshot")
        return
    n_flags = len(lm_result.deterministic_report["flags"])
    print(f"  {n_flags} flag(s) raised" + ("; flag_mismatch (agent's narration disagreed)" if lm_result.flag_mismatch else ""))
    if lm_result.flag_mismatch:
        report["issues"].append("load_monitor: agent's reported flags did not match the deterministic ones")
    report["load_monitor"] = {"n_flags": n_flags, "flag_mismatch": lm_result.flag_mismatch}

    # --- 2. Task Prioritization (first A2A send) ---
    print("\nRunning Task Prioritization agent...")
    tp_result = run_task_prioritization_step(
        df, variance_injected, client, queue_size=args.queue_size, snapshot_window=args.snapshot_window,
        model=default_model, trace_fn=lambda record: trace(**record), on_event=emit, mailbox=mailbox,
        prompt_override_dir=effective_prompt_override_dir,
    )
    write_transcript("task_prioritization", tp_result.messages)
    (run_dir / "task_prioritization_report.json").write_text(json.dumps({
        "ok": tp_result.ok, "valid": tp_result.valid, "validation_errors": tp_result.validation_errors,
        "score_inconsistent": tp_result.score_inconsistent,
        "deterministic_facts": tp_result.deterministic_facts, "llm_proposal": tp_result.llm_proposal,
        "sent_to_resource_allocation": tp_result.sent_to_resource_allocation,
    }, indent=2, default=str))
    if not tp_result.ok:
        fail("failed_task_prioritization", error="Task Prioritization never called get_task_queue_profile")
        return
    print(f"  valid={tp_result.valid} sent_to_resource_allocation={tp_result.sent_to_resource_allocation}")
    if not tp_result.valid:
        report["issues"].append(f"task_prioritization: invalid proposal — {tp_result.validation_errors}")
    if tp_result.score_inconsistent:
        report["issues"].append("task_prioritization: ranking order was inconsistent with its own scores")
    report["task_prioritization"] = {
        "valid": tp_result.valid, "score_inconsistent": tp_result.score_inconsistent,
        "sent_to_resource_allocation": tp_result.sent_to_resource_allocation,
    }

    # --- 3. Resource Allocation (consumes the ranking, applies the constraint gate) ---
    print("\nRunning Resource Allocation agent...")
    ra_result = run_resource_allocation_step(
        df, variance_injected, client, mailbox, snapshot_window=args.snapshot_window,
        slice_capacity=args.slice_capacity, model=default_model, trace_fn=lambda record: trace(**record), on_event=emit,
        prompt_override_dir=effective_prompt_override_dir,
    )
    write_transcript("resource_allocation", ra_result.messages)
    (run_dir / "resource_allocation_report.json").write_text(json.dumps({
        "ok": ra_result.ok, "valid": ra_result.valid, "validation_errors": ra_result.validation_errors,
        "accepted_assignments": ra_result.accepted_assignments,
        "environment_rejected": ra_result.environment_rejected, "agent_rejected": ra_result.agent_rejected,
        "events": ra_result.events, "ranking_source": ra_result.ranking_source,
        "stopped_reason": ra_result.stopped_reason,
    }, indent=2, default=str))
    print(f"  stopped_reason={ra_result.stopped_reason} valid={ra_result.valid} "
          f"accepted={len(ra_result.accepted_assignments)} environment_rejected={len(ra_result.environment_rejected)}")
    if ra_result.stopped_reason == "no_ranking_available":
        report["issues"].append("resource_allocation: no ranking was available to allocate against")
    elif not ra_result.valid:
        report["issues"].append(f"resource_allocation: invalid proposal — {ra_result.validation_errors}")
    elif ra_result.environment_rejected:
        report["issues"].append(
            f"resource_allocation: {len(ra_result.environment_rejected)} assignment(s) environment-rejected"
        )
    report["resource_allocation"] = {
        "valid": ra_result.valid, "n_accepted": len(ra_result.accepted_assignments),
        "n_environment_rejected": len(ra_result.environment_rejected),
        "n_agent_rejected": len(ra_result.agent_rejected), "stopped_reason": ra_result.stopped_reason,
    }

    # --- 4. Failure Recovery (uses Resource Allocation's own accepted
    # assignments as "currently committed"; may send a reroute_request) ---
    print("\nRunning Failure Recovery agent...")
    before_row = args.before_row if args.before_row is not None else max(len(df) - 50, 1)
    after_row = args.after_row if args.after_row is not None else len(df)
    fr_result = run_failure_recovery_step(
        df, variance_injected, client, mailbox, ra_result.accepted_assignments,
        before_row=before_row, after_row=after_row, snapshot_window=args.snapshot_window,
        model=default_model, trace_fn=lambda record: trace(**record), on_event=emit,
        prompt_override_dir=effective_prompt_override_dir,
    )
    write_transcript("failure_recovery", fr_result.messages)

    reroute_result = None
    if fr_result.sent_to_resource_allocation:
        print("  reroute_request sent — running reroute validation...")
        reroute_result = run_reroute_validation_step(
            df, variance_injected, mailbox, snapshot_window=args.snapshot_window,
            slice_capacity=args.slice_capacity, on_event=emit,
        )
        print(f"    accepted={len(reroute_result.accepted_reroutes)} "
              f"environment_rejected={len(reroute_result.environment_rejected)}")
        if reroute_result.environment_rejected:
            report["issues"].append(
                f"failure_recovery: {len(reroute_result.environment_rejected)} reroute(s) environment-rejected"
            )

    (run_dir / "failure_recovery_report.json").write_text(json.dumps({
        "before_row": before_row, "after_row": after_row,
        "incidents": fr_result.incidents, "affected_tasks": fr_result.affected_tasks,
        "valid": fr_result.valid, "validation_errors": fr_result.validation_errors,
        "reroute_avoids_source": fr_result.reroute_avoids_source,
        "reroute_proposals": fr_result.reroute_proposals,
        "reroute_validation": None if reroute_result is None else {
            "accepted_reroutes": reroute_result.accepted_reroutes,
            "environment_rejected": reroute_result.environment_rejected,
            "events": reroute_result.events,
        },
        "stopped_reason": fr_result.stopped_reason,
    }, indent=2, default=str))
    print(f"  stopped_reason={fr_result.stopped_reason} "
          f"n_incidents={len(fr_result.incidents)} n_affected={len(fr_result.affected_tasks)}")
    if fr_result.incidents and not fr_result.reroute_avoids_source:
        report["issues"].append("failure_recovery: at least one reroute targeted the same machine it was displaced from")
    report["failure_recovery"] = {
        "n_incidents": len(fr_result.incidents), "n_affected_tasks": len(fr_result.affected_tasks),
        "stopped_reason": fr_result.stopped_reason,
        "reroute_accepted": None if reroute_result is None else len(reroute_result.accepted_reroutes),
        "reroute_environment_rejected": None if reroute_result is None else len(reroute_result.environment_rejected),
    }

    # --- 5. Optimization (reads run history -- INCLUDING this run's own
    # report files above, already written to disk by this point) ---
    # opt_sent_to_human_oversight is tracked as a plain variable (not read
    # off opt_result, which doesn't exist at all when Optimization is
    # skipped) so step 6 below is a sibling of this block, not nested
    # inside its else -- it always runs and always sets
    # report["human_oversight"] to something, regardless of which flags
    # were passed, instead of that key silently going missing whenever
    # --skip-optimization made step 6 unreachable.
    opt_sent_to_human_oversight = False
    if args.skip_optimization:
        print("\nSkipping Optimization (--skip-optimization).")
        report["optimization"] = {"skipped": True}
    else:
        print("\nRunning Optimization agent...")
        opt_result = run_optimization_step(
            client, mailbox, n_runs=args.n_runs, model=default_model, trace_fn=lambda record: trace(**record), on_event=emit,
            prompt_override_dir=effective_prompt_override_dir,
        )
        write_transcript("optimization", opt_result.messages)
        if opt_result.ok:
            (run_dir / "optimization_report.json").write_text(json.dumps({
                "ok": opt_result.ok, "valid": opt_result.valid, "validation_errors": opt_result.validation_errors,
                "evidence": opt_result.evidence, "llm_proposal": opt_result.llm_proposal,
                "sent_to_human_oversight": opt_result.sent_to_human_oversight,
            }, indent=2, default=str))
        print(f"  ok={opt_result.ok} valid={opt_result.valid} "
              f"sent_to_human_oversight={opt_result.sent_to_human_oversight}")
        if opt_result.ok and not opt_result.valid:
            report["issues"].append(f"optimization: invalid proposal — {opt_result.validation_errors}")
        report["optimization"] = {
            "ok": opt_result.ok, "valid": opt_result.valid,
            "sent_to_human_oversight": opt_result.sent_to_human_oversight,
        }
        opt_sent_to_human_oversight = opt_result.sent_to_human_oversight

    # --- 6. Human Oversight (reviews Optimization's proposal, if any) ---
    if args.skip_oversight:
        print("\nSkipping Human Oversight (--skip-oversight).")
        report["human_oversight"] = {"skipped": True}
    elif not opt_sent_to_human_oversight:
        print("\nSkipping Human Oversight: no proposal was sent.")
        report["human_oversight"] = {"skipped": True, "reason": "no proposal available"}
    else:
        print("\nRunning Human Oversight agent...")
        ho_result = run_oversight_step(
            client, mailbox, model=default_model, trace_fn=lambda record: trace(**record), on_event=emit,
            prompt_override_dir=effective_prompt_override_dir,
        )
        write_transcript("human_oversight", ho_result.messages)
        (run_dir / "human_oversight_report.json").write_text(json.dumps({
            "ok": ho_result.ok, "verdict": ho_result.verdict, "concerns": ho_result.concerns,
            "reasoning": ho_result.reasoning, "review_bundle": ho_result.review_bundle,
        }, indent=2, default=str))
        print(f"  verdict={ho_result.verdict}")
        if ho_result.verdict in ("rejected", "flagged"):
            report["issues"].append(f"human_oversight: verdict={ho_result.verdict} — {ho_result.concerns}")
        report["human_oversight"] = {"verdict": ho_result.verdict, "concerns": ho_result.concerns}

    # --- 7. Narrated final summary: plain text only, no tools, no decisions ---
    report["status"] = "completed"
    summary_messages = [
        {"role": "system", "content": (
            "You are the summary step of a deterministic multi-agent resource "
            "scheduler. Respond with 4-6 sentences of plain prose only (no JSON, "
            "no markdown fences) summarizing the given facts for a non-technical "
            "stakeholder. Do not invent any numbers not present in the input. If "
            "the issues list is non-empty, mention the most important ones as "
            "caveats."
        )},
        {"role": "user", "content": json.dumps({
            k: v for k, v in report.items() if k not in ("run_id", "model")
        }, indent=2, default=str)},
    ]
    summary_response = client.call(summary_messages, model=default_model, max_tokens=400)
    print("\n--- Final summary ---")
    print(summary_response.text)
    report["final_summary"] = summary_response.text
    write_transcript("orchestrator_summary", summary_messages + [{"role": "assistant", "content": summary_response.text}])

    write_report()
    print(f"\nOrchestrator report written to {run_dir / 'orchestrator_report.json'}")
    if report["issues"]:
        print(f"{len(report['issues'])} issue(s) flagged — see report['issues'].")
    print("DONE.")


if __name__ == "__main__":
    main()
