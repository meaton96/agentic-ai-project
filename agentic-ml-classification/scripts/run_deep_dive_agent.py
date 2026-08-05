#!/usr/bin/env python3
"""
Deep-dive agent: explains why one already-flagged flight was flagged.
Answers a different analysis question than the rest of this pipeline —
"why", not "is it" — so it runs on-demand against a specific flight from
a completed classification run, not as a step inside that run.

Needs three things a prior orchestrator run produces: the persisted
model bundle (run_orchestrator.py now saves one to
artifacts/models/<run_id>_model.joblib after a successful run), the
engineered flight-level feature table (from
scripts/featurize_ngafid_flights.py), and the raw long-format sensor CSV
(to pull that one flight's per-timestep trace for phase segmentation and
cross-cylinder localization).

This script is a thin CLI wrapper around
agentic_ml.steps.deep_dive_step.run_deep_dive_step — same pattern as
scripts/run_profiler_agent.py.

Usage:
    set -a; source .env; set +a
    python scripts/run_deep_dive_agent.py \\
        --model artifacts/models/run_abc123_model.joblib \\
        --raw-csv datasets/raw/C28.csv \\
        --features-csv datasets/processed/ngafid_c28_flights.csv \\
        --flight-id 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import joblib
import pandas as pd

from agentic_ml.cli_common import make_run_dir, make_tracer, make_transcript_writer, resolve_model_endpoint
from agentic_ml.domain.aviation.ngafid_config import NGAFID_SENSORS
from agentic_ml.harness.timeseries_features import extract_single_flight_raw
from agentic_ml.model_client import ModelClient
from agentic_ml.steps.deep_dive_step import run_deep_dive_step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="path to a joblib bundle written by "
                         "run_orchestrator.py: {model, feature_columns, background}")
    parser.add_argument("--raw-csv", required=True, help="the raw long-format sensor CSV "
                         "this flight's id came from")
    parser.add_argument("--features-csv", required=True, help="the engineered flight-level "
                         "table (from featurize_ngafid_flights.py) to pull this flight's "
                         "feature row from")
    parser.add_argument("--flight-id", required=True)
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--sample-hz", type=float, default=1.0)
    parser.add_argument("--model-id", default=None, help="LLM model id or gateway "
                         "model_name; defaults to RIT_DEFAULT_MODEL env var")
    parser.add_argument("--use-gateway", action="store_true")
    parser.add_argument("--use-local", action="store_true", help="call a local OpenAI-"
                         "compatible server (LOCAL_MODEL_BASE_URL) instead of RIT/the "
                         "gateway — takes precedence over --use-gateway if both are given")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    run_id, run_dir = make_run_dir(args.run_id)
    trace = make_tracer(run_dir / "trace.jsonl")
    write_transcript = make_transcript_writer(run_dir)

    bundle = joblib.load(args.model)
    pipeline, feature_columns, background = bundle["model"], bundle["feature_columns"], bundle["background"]
    print(f"Loaded model bundle: {len(feature_columns)} feature columns")

    features = pd.read_csv(args.features_csv)
    match = features[features[args.id_column].astype(str) == str(args.flight_id)]
    if match.empty:
        print(f"FAILED: flight id {args.flight_id!r} not found in {args.features_csv}")
        sys.exit(1)
    feature_row = match.iloc[0]
    print(f"Found feature row for flight {args.flight_id!r}")

    print(f"Streaming raw trace for flight {args.flight_id!r} from {args.raw_csv} "
          f"(this can take a while on a multi-GB file)...")
    flight_df = extract_single_flight_raw(
        args.raw_csv, args.flight_id, NGAFID_SENSORS, id_column=args.id_column,
    )
    print(f"Got {len(flight_df)} raw timesteps")

    base_url, api_key, default_model = resolve_model_endpoint(
        args.use_gateway, args.model_id, "qwen3-coder:30b", "rit-qwen3-coder-30b",
        use_local=args.use_local,
    )
    client = ModelClient(base_url=base_url, api_key=api_key, default_model=default_model)

    print(f"Running DeepDiveAgent (model={default_model})...")
    step_result = run_deep_dive_step(
        flight_df, feature_row, pipeline, feature_columns, background, client,
        model=default_model, sample_hz=args.sample_hz,
        trace_fn=lambda record: trace(**record),
    )
    print(f"Agent stopped: {step_result.stopped_reason} (turns_used={step_result.turns_used})")
    transcript_path = write_transcript("deep_dive", step_result.messages)
    print(f"Transcript written to {transcript_path}")

    # deterministic evidence is always saved, regardless of whether the LLM's
    # hypothesis parsed cleanly — the facts don't depend on the LLM
    if step_result.evidence:
        (run_dir / "deep_dive_evidence.json").write_text(json.dumps(step_result.evidence, indent=2, default=str))
        print("\n--- Deterministic evidence (tool output, not LLM-generated) ---")
        print(f"  p_maintenance: {step_result.evidence['p_maintenance']}")
        print(f"  segmentation: {step_result.evidence['segmentation']}")
        print(f"  top attribution channels: "
              f"{[a['channel'] for a in step_result.evidence['attribution_top'][:3]]}")
        print(f"  n_localized: {step_result.evidence['n_localized']}")

    print(f"\n--- Hypothesis{' (template fallback, LLM unavailable/unparseable)' if step_result.used_template_fallback else ''} ---")
    print(step_result.hypothesis)
    print(f"  agrees_with_localization: {step_result.agrees_with_localization}")
    print(f"  confidence: {step_result.confidence}")

    output = {
        "run_id": run_id, "flight_id": args.flight_id, "model": default_model,
        "evidence": step_result.evidence, "hypothesis": step_result.hypothesis,
        "agrees_with_localization": step_result.agrees_with_localization,
        "confidence": step_result.confidence,
        "used_template_fallback": step_result.used_template_fallback,
        "llm_raw_text": step_result.llm_raw_text,
        "stopped_reason": step_result.stopped_reason,
        "turns_used": step_result.turns_used,
        "transcript": str(transcript_path),
    }
    out_path = run_dir / "deep_dive_report.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nFull report written to {out_path}")

    if step_result.evidence is None:
        print("FAILED: deep-dive agent never gathered evidence.")
        sys.exit(1)
    if step_result.used_template_fallback:
        print("PARTIAL: evidence is solid, but the LLM hypothesis fell back to the "
              "deterministic template. Check the raw text above.")
        sys.exit(2)
    print("SUCCESS.")


if __name__ == "__main__":
    main()
