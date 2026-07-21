#!/usr/bin/env python3
"""
Phase 9: streaming ingestion + drift-triggered retraining. The outer
driver around the dynamic orchestrator (orchestrator/dynamic_loop.py) —
NOT a new trust mechanism and not a new agent loop of its own. A
long-running streaming session is not one giant run_dynamic_loop call:
this script pushes one new batch into a persisted (ctx, state) pair,
then calls run_dynamic_loop UNCHANGED with a small max_iterations
(default 5) so the planner decides what to do with THAT batch —
"finish" means "done handling this batch," not "session over."

harness/streaming.py simulates arrival order (a replay of historical
data, not a live feed) by grouping an already-featurized flight-level
table by plane_id and shuffling group order deterministically — a
single aircraft's flights never straddle a batch boundary. Every
incoming batch is always a RAW slice of that source table — there is
no mechanism to replay feature_engineering's derived-feature ops onto a
streaming arrival — so feature_engineering is pre-seeded as already
done (never runs, cold start or any later retrain) to keep the column
schema monitor_drift/infer_batch depend on stable for the whole
session; the same reasoning evaluate_dynamic_orchestrator.py's
task-routing scenario applies via --skip-feature-engineering. See
orchestrator/dynamic_loop.py's retrain_decision branch for why this
also means feature_engineering_done is deliberately excluded from that
branch's classification-phase reset.

Usage:
    set -a; source .env; set +a
    python scripts/run_streaming_monitor.py \\
        --data datasets/processed/ngafid_c28_flights_smoke.csv \\
        --target before_after --group-column plane_id \\
        --id-columns id,date_diff,split \\
        --n-initial-planes 20 --batch-size-planes 5 \\
        --use-local
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agentic_ml.cli_common import make_run_dir, make_tracer, make_transcript_writer, resolve_model_endpoint
from agentic_ml.harness.dataset import read_dataframe
from agentic_ml.harness.streaming import simulate_batches
from agentic_ml.model_client import ModelClient
from agentic_ml.orchestrator.dynamic_loop import load_raw_hash, run_dynamic_loop
from agentic_ml.orchestrator.run_state import DynamicRunContext, RunStateSummary


def make_streaming_logger(run_dir: Path):
    log_path = run_dir / "streaming_log.jsonl"

    def log(entry: dict) -> None:
        with open(log_path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    return log, log_path


def push_batch(ctx: DynamicRunContext, state: RunStateSummary, batch_df) -> None:
    """Starts a fresh per-batch cycle: whatever the previous batch decided
    (no_action/infer_only/retrain) is done and shouldn't leak into this
    one's gating."""
    ctx.pending_batch_df = batch_df
    state.new_batch_pending = True
    state.drift_checked = False
    state.drift_summary = None
    state.pending_retrain_action = None
    state.batch_action_completed = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="full, already-featurized flight-level table")
    parser.add_argument("--target", required=True)
    parser.add_argument("--group-column", default="plane_id")
    parser.add_argument("--id-column", default="id", help="row identifier, used to order rows "
                         "deterministically within each simulated batch")
    parser.add_argument("--id-columns", default=None, help="comma-separated non-predictive "
                         "columns (passed through to the classification DatasetSpec)")
    parser.add_argument("--time-column", default=None)
    parser.add_argument("--strategy", default=None,
                         choices=["random", "stratified", "group", "time", "group_time"])
    parser.add_argument("--n-initial-groups", type=int, default=20, help="number of groups "
                         "(e.g. planes) in the cold-start pool")
    parser.add_argument("--batch-size-groups", type=int, default=5, help="number of groups per "
                         "simulated incoming batch")
    parser.add_argument("--max-batches", type=int, default=None, help="stop after this many "
                         "batches (default: process every batch simulate_batches produces)")
    parser.add_argument("--interval", type=float, default=0.0, help="real seconds to sleep "
                         "between batches — 0 (default) for a fast, reproducible replay")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metrics", default="roc_auc,pr_auc,f1,accuracy")
    parser.add_argument("--model", default=None)
    parser.add_argument("--verification-model", default=None)
    parser.add_argument("--use-gateway", action="store_true")
    parser.add_argument("--use-local", action="store_true")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--max-iterations-cold-start", type=int, default=15, help="planner-turn "
                         "cap for the cold-start classification run (same default as "
                         "run_dynamic_orchestrator.py)")
    parser.add_argument("--max-iterations-per-batch", type=int, default=5, help="planner-turn "
                         "cap per incoming batch — small, since a batch cycle is at most "
                         "monitor_drift -> retrain_decision -> (infer_batch | full retrain)")
    args = parser.parse_args()

    run_id, run_dir = make_run_dir(args.run_id)
    trace = make_tracer(run_dir / "trace.jsonl")
    write_transcript = make_transcript_writer(run_dir)
    log_batch, streaming_log_path = make_streaming_logger(run_dir)

    base_url, api_key, default_model = resolve_model_endpoint(
        args.use_gateway, args.model, "qwen3-coder:30b", "rit-qwen3-coder-30b",
        use_local=args.use_local,
    )
    client = ModelClient(base_url=base_url, api_key=api_key, default_model=default_model)
    _, _, verification_model = resolve_model_endpoint(
        args.use_gateway, args.verification_model, "gemma4:latest", "rit-gemma4-latest",
        use_local=args.use_local,
    )

    full_df = read_dataframe(args.data)
    initial_df, batches = simulate_batches(
        full_df, group_column=args.group_column, id_column=args.id_column,
        n_initial_groups=args.n_initial_groups, batch_size_groups=args.batch_size_groups,
        seed=args.seed,
    )
    if args.max_batches is not None:
        batches = batches[: args.max_batches]
    print(f"Cold-start pool: {len(initial_df)} rows. {len(batches)} batches to replay "
          f"({sum(len(b) for b in batches)} rows total).")

    batches_dir = run_dir / "batches"
    batches_dir.mkdir(parents=True, exist_ok=True)
    initial_csv = batches_dir / "initial.csv"
    initial_df.to_csv(initial_csv, index=False)

    id_columns = [c.strip() for c in (args.id_columns or "").split(",") if c.strip()]
    metric_names = args.metrics.split(",")
    # Deliberately a plain classification goal, not "...and keep the model
    # current as new data arrives" — a real local-model evaluation
    # (scripts/evaluate_streaming_monitor.py) showed that phrasing tempts
    # the planner into hallucinating a monitoring agent (e.g. "drift_check")
    # during COLD START, before any batch is pending and before the
    # "streaming" capability even grants monitor_drift/retrain_decision/
    # infer_batch a place in its catalog. The catalog + current_state
    # (available_agents, new_batch_pending, drift_checked, ...) are what
    # actually gate correct behavior every run_dynamic_loop call, cold
    # start or batch cycle alike — the goal text is just context, and
    # over-describing the session tempted a real model into ignoring its
    # own catalog.
    goal_text = f"predict {args.target}"

    ctx = DynamicRunContext(
        data_path=str(initial_csv), goal=goal_text, seed=args.seed,
        target_column=args.target, group_column=args.group_column, time_column=args.time_column,
        id_columns=id_columns, strategy_override=args.strategy, metric_names=metric_names,
        run_id=run_id,
    )
    state = RunStateSummary(goal=goal_text)
    state.target_known = True
    state.target_column = args.target
    # see the module docstring: feature_engineering never runs in a
    # streaming session — pre-seeding this the same way --target seeds
    # target_known keeps it out of the planner's catalog from turn one.
    state.feature_engineering_done = True
    load_raw_hash(ctx)

    print(f"Starting cold-start classification run (model={default_model}, "
          f"verification_model={verification_model})")
    cold_start_result = run_dynamic_loop(
        ctx, state, client, model=default_model, verification_model=verification_model,
        max_iterations=args.max_iterations_cold_start,
        trace_fn=lambda record: trace(**record), write_transcript=write_transcript,
    )
    print(f"Cold-start status: {cold_start_result.status}")
    if cold_start_result.status != "success" or ctx.model_path is None:
        print("Cold-start classification did not produce a model — aborting before streaming.")
        sys.exit(1)
    print(f"Model v{ctx.model_version} ready. Test metrics: "
          + ", ".join(f"{m}={r['value']:.4f}" for m, r in (ctx.final_test_metrics or {}).items()))

    for batch_index, batch_df in enumerate(batches):
        if args.interval:
            time.sleep(args.interval)

        push_batch(ctx, state, batch_df)
        model_version_before = ctx.model_version
        batch_result = run_dynamic_loop(
            ctx, state, client, model=default_model, verification_model=verification_model,
            max_iterations=args.max_iterations_per_batch,
            trace_fn=lambda record: trace(**record), write_transcript=write_transcript,
        )
        retrained = ctx.model_version != model_version_before

        log_entry = {
            "batch_index": batch_index,
            "n_examples": len(batch_df),
            "status": batch_result.status,
            "drift_summary": state.drift_summary,
            "decision": state.pending_retrain_action,
            "model_version_before": model_version_before,
            "model_version_after": ctx.model_version,
            "retrained": retrained,
            "test_metrics": ctx.final_test_metrics if retrained else None,
            "batch_inference_log_tail": ctx.batch_inference_log[-1:] if not retrained else None,
        }
        log_batch(log_entry)
        print(f"  [batch {batch_index:3d}] n={len(batch_df):3d}  decision={state.pending_retrain_action!s:10s}  "
              f"model_version={ctx.model_version}  status={batch_result.status}")

        ctx.pending_batch_df = None

    report = {
        "run_id": run_id, "model": default_model, "verification_model": verification_model,
        "n_initial_groups": args.n_initial_groups, "batch_size_groups": args.batch_size_groups,
        "n_batches_processed": len(batches),
        "final_model_version": ctx.model_version,
        "model_history": ctx.model_history,
        "final_state": state.to_planner_dict(),
        "streaming_log_path": str(streaming_log_path),
    }
    out_path = run_dir / "streaming_monitor_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nProcessed {len(batches)} batches. Final model version: {ctx.model_version}.")
    print(f"Report written to {out_path}")
    print(f"Per-batch audit trail: {streaming_log_path}")
    print("SUCCESS.")


if __name__ == "__main__":
    main()
