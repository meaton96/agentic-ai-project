#!/usr/bin/env python3
"""
Feature Engineering agent, standalone: the agent proposes columns to
drop and derived features to add (from a vetted operation catalog) —
it never writes transformation code. This script is a thin CLI wrapper
around agentic_ml.steps.feature_engineering_step.run_feature_engineering_step,
the same function scripts/run_orchestrator.py calls as part of the
full loop (there, right after intake and before the profiler, so the
profiler's facts reflect the augmented column set).

Usage:
    set -a; source .env; set +a
    python scripts/run_feature_engineering_agent.py \\
        --data datasets/raw/train.csv \\
        --target Survived
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agentic_ml.cli_common import make_run_dir, make_tracer, make_transcript_writer, resolve_model_endpoint
from agentic_ml.harness.dataset import DatasetSpec, load_dataset
from agentic_ml.model_client import ModelClient
from agentic_ml.steps.feature_engineering_step import run_feature_engineering_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--group-column", default=None)
    parser.add_argument("--time-column", default=None)
    parser.add_argument("--model", default=None, help="model id or gateway model_name; "
                         "defaults to RIT_DEFAULT_MODEL env var")
    parser.add_argument("--use-gateway", action="store_true")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    run_id, run_dir = make_run_dir(args.run_id)
    trace = make_tracer(run_dir / "trace.jsonl")
    write_transcript = make_transcript_writer(run_dir)

    spec = DatasetSpec(path=args.data, target_column=args.target)
    loaded = load_dataset(spec)
    print(f"Loaded dataset: {len(loaded.df)} rows, {len(loaded.df.columns)} columns, "
          f"data_hash={loaded.data_hash[:16]}...")

    base_url, api_key, default_model = resolve_model_endpoint(
        args.use_gateway, args.model, "qwen3-coder:30b", "rit-qwen3-coder-30b",
    )
    client = ModelClient(base_url=base_url, api_key=api_key, default_model=default_model)

    print(f"Running FeatureEngineeringAgent (model={default_model})...")
    step_result = run_feature_engineering_step(
        loaded.df, args.target, client,
        group_column=args.group_column, time_column=args.time_column,
        model=default_model, trace_fn=lambda record: trace(**record),
    )
    print(f"Agent stopped: {step_result.stopped_reason} (turns_used={step_result.turns_used})")
    transcript_path = write_transcript("feature_engineering", step_result.messages)
    print(f"Transcript written to {transcript_path}")

    output = {
        "run_id": run_id,
        "model": default_model,
        "ok": step_result.ok,
        "drop_columns": step_result.drop_columns,
        "new_columns": step_result.new_columns,
        "applied_ops": step_result.applied_ops,
        "explanation": step_result.explanation,
        "errors": step_result.errors,
        "transcript": str(transcript_path),
    }
    out_path = run_dir / "feature_engineering_report.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nFull report written to {out_path}")

    if not step_result.ok:
        print("FAILED:")
        for e in step_result.errors:
            print(f"  - {e}")
        sys.exit(1)

    print(f"\ndrop_columns: {step_result.drop_columns}")
    print(f"new_columns: {step_result.new_columns}")
    print(f"explanation: {step_result.explanation}")

    preview_path = run_dir / "engineered_dataset_preview.csv"
    step_result.df.head(20).to_csv(preview_path, index=False)
    print(f"\nPreview of engineered dataset (first 20 rows) written to {preview_path}")
    print("SUCCESS.")


if __name__ == "__main__":
    main()
