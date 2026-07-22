#!/usr/bin/env python3
"""
The dynamic orchestrator: an agent catalog (agentic_ml.orchestrator.
agent_registry) plus a planning agent (agentic_ml.steps.planner_step)
that decides which catalog agent to run next, given the goal and the
current run state — instead of run_orchestrator.py's fixed sequence.
Every proposed step is independently re-validated against the real
registry and real run state before it executes
(agentic_ml.orchestrator.dynamic_loop.validate_plan) — same "agents
propose, harness decides" discipline as every other agent here, applied
to control flow instead of just content.

run_orchestrator.py (the static pipeline) is untouched and remains the
baseline this is evaluated against — see
scripts/evaluate_dynamic_orchestrator.py.

Usage:
    set -a; source .env; set +a

    # ordinary classification goal, same shape as run_orchestrator.py:
    python scripts/run_dynamic_orchestrator.py \\
        --data datasets/raw/train.csv \\
        --goal "predict whether a passenger survived"

    # explain-only task against an existing model bundle — the planner
    # should route straight to deep_dive, skipping classification:
    python scripts/run_dynamic_orchestrator.py \\
        --data datasets/processed/ngafid_c28_flights_smoke.csv \\
        --goal "explain why flight 5 was flagged for maintenance" \\
        --existing-model artifacts/models/some_run_model.joblib \\
        --raw-csv datasets/raw/C28.csv \\
        --features-csv datasets/processed/ngafid_c28_flights_smoke.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import joblib

from agentic_ml.cli_common import make_run_dir, make_tracer, make_transcript_writer, resolve_model_endpoint
from agentic_ml.events import make_event_emitter, make_event_logger
from agentic_ml.harness.dataset import read_dataframe
from agentic_ml.model_client import ModelClient
from agentic_ml.orchestrator.dynamic_loop import load_raw_hash, run_dynamic_loop
from agentic_ml.orchestrator.run_state import DynamicRunContext, RunStateSummary


def main(on_event: Optional[Callable[[dict], None]] = None, prompt_override_dir: Optional[str] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--goal", default="")
    parser.add_argument("--target", default=None)
    parser.add_argument("--group-column", default=None)
    parser.add_argument("--time-column", default=None)
    parser.add_argument("--id-columns", default=None)
    parser.add_argument("--strategy", default=None,
                         choices=["random", "stratified", "group", "time", "group_time"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metrics", default="roc_auc,pr_auc,f1,accuracy")
    parser.add_argument("--model", default=None)
    parser.add_argument("--verification-model", default=None)
    parser.add_argument("--use-gateway", action="store_true")
    parser.add_argument("--use-local", action="store_true", help="call a local OpenAI-"
                         "compatible server instead of RIT/the gateway — takes precedence "
                         "over --use-gateway if both are given")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--max-iterations", type=int, default=15, help="hard cap on planner "
                         "turns, in case the planner never proposes 'finish'")
    parser.add_argument("--existing-model", default=None, help="path to a joblib bundle "
                         "from a prior run's finalize step — pre-seeds model availability "
                         "so an explain-only goal can route straight to deep_dive without "
                         "re-running classification")
    parser.add_argument("--raw-csv", default=None, help="raw long-format sensor CSV, "
                         "enables the deep_dive agent (needs --features-csv too)")
    parser.add_argument("--features-csv", default=None, help="engineered flight-level "
                         "table, enables the deep_dive agent (needs --raw-csv too)")
    parser.add_argument("--prompt-override-dir", default=None, help="directory of per-agent "
                         "<agent_name>.md system-prompt overrides (falls back to the shipped "
                         "default for any agent not present there); defaults to the "
                         "AGENTIC_ML_PROMPT_OVERRIDE_DIR env var if not given")
    args = parser.parse_args()

    # the function argument (e.g. a future server calling main() directly) wins
    # over the CLI flag if both are given; either may still be None, in which
    # case each step falls back to AGENTIC_ML_PROMPT_OVERRIDE_DIR itself.
    effective_prompt_override_dir = (
        prompt_override_dir if prompt_override_dir is not None else args.prompt_override_dir
    )

    run_id, run_dir = make_run_dir(args.run_id)
    trace = make_tracer(run_dir / "trace.jsonl")
    write_transcript = make_transcript_writer(run_dir)
    emit = make_event_emitter(run_id, external_on_event=on_event, persist_fn=make_event_logger(run_dir))

    base_url, api_key, default_model = resolve_model_endpoint(
        args.use_gateway, args.model, "qwen3-coder:30b", "rit-qwen3-coder-30b",
        use_local=args.use_local,
    )
    client = ModelClient(base_url=base_url, api_key=api_key, default_model=default_model)
    _, _, verification_model = resolve_model_endpoint(
        args.use_gateway, args.verification_model, "gemma4:latest", "rit-gemma4-latest",
        use_local=args.use_local,
    )

    id_columns = [c.strip() for c in (args.id_columns or "").split(",") if c.strip()]
    metric_names = args.metrics.split(",")

    goal_text = args.goal or (f"predict {args.target}" if args.target else "")

    ctx = DynamicRunContext(
        data_path=args.data, goal=goal_text, seed=args.seed,
        target_column=args.target, group_column=args.group_column, time_column=args.time_column,
        id_columns=id_columns, strategy_override=args.strategy, metric_names=metric_names,
        raw_csv=args.raw_csv, features_csv=args.features_csv, run_id=run_id,
    )
    ctx.raw_df = read_dataframe(args.data)

    state = RunStateSummary(goal=goal_text)

    if args.target:
        state.target_known = True
        state.target_column = args.target
        load_raw_hash(ctx)

    if args.existing_model:
        bundle = joblib.load(args.existing_model)
        ctx.model_path = args.existing_model
        ctx.final_pipeline = bundle["model"]
        ctx.model_feature_columns = bundle["feature_columns"]
        ctx.model_background = bundle["background"]
        state.model_path_available = True
        # A pre-seeded model means this run isn't doing fresh classification
        # (that's what produced the model in the first place, in an earlier
        # run) — so intake is never relevant here, whether or not --target
        # was also given. Without this, an explain-only goal against a
        # feature table with no plausible target column (e.g. an engineered
        # aviation table) would leave intake looking like a legitimate next
        # step, and it would burn iterations failing over and over instead
        # of the planner going straight to deep_dive.
        state.target_known = True
        print(f"Pre-seeded model bundle from {args.existing_model} "
              f"({len(ctx.model_feature_columns)} feature columns)")

    print(f"Starting dynamic orchestrator (model={default_model}, "
          f"verification_model={verification_model}, max_iterations={args.max_iterations})")
    print(f"Goal: {args.goal or '(none given)'}")

    result = run_dynamic_loop(
        ctx, state, client, model=default_model, verification_model=verification_model,
        max_iterations=args.max_iterations,
        trace_fn=lambda record: trace(**record), write_transcript=write_transcript,
        on_event=emit, prompt_override_dir=effective_prompt_override_dir,
    )

    for entry in result.history:
        tag = "FINISH" if entry.get("finished") else (
            entry.get("agent_id", "?") if entry.get("executed") else "REJECTED"
        )
        ok_str = "" if not entry.get("executed") else ("OK" if entry.get("ok") else "FAILED")
        print(f"  [{entry['iteration']:2d}] {tag:24s} {ok_str}  "
              f"{'; '.join(entry.get('validation_errors') or entry.get('errors') or [])}")

    report = {
        "run_id": run_id, "model": default_model, "verification_model": verification_model,
        "goal": args.goal, "status": result.status,
        "final_state": result.state.to_planner_dict(),
        "history": result.history,
        "final_test_metrics": ctx.final_test_metrics,
        "model_path": ctx.model_path,
        "summary": ctx.summary_text,
        "deep_dive_results": {
            fid: {
                "hypothesis": r.hypothesis, "confidence": r.confidence,
                "used_template_fallback": r.used_template_fallback,
                "evidence": r.evidence,
            }
            for fid, r in ctx.deep_dive_results.items()
        },
    }
    out_path = run_dir / "dynamic_orchestrator_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nStatus: {result.status}")
    if ctx.final_test_metrics:
        print("Final test-set metrics:")
        for m, r in ctx.final_test_metrics.items():
            print(f"  {m}: {r['value']:.4f}  (95% CI [{r['ci_low']:.4f}, {r['ci_high']:.4f}])")
    if ctx.summary_text:
        print(f"\nSummary: {ctx.summary_text}")
    for fid, r in ctx.deep_dive_results.items():
        print(f"\nDeep-dive (flight {fid}): {r.hypothesis}")
    print(f"\nReport written to {out_path}")

    if result.status != "success":
        sys.exit(1)
    print("SUCCESS.")


if __name__ == "__main__":
    main()
