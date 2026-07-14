#!/usr/bin/env python3
"""
Phase 2: run the ProfilerAgent. Deterministic code (profile_dataset)
computes every fact; the LLM only narrates and recommends. The agent
gets exactly one tool and is instructed to call it once.

This script is a thin CLI wrapper around
agentic_ml.steps.profiler_step.run_profiler_step — the same function
scripts/run_orchestrator.py calls internally as part of the full loop.

Usage:
    set -a; source .env; set +a
    python scripts/run_profiler_agent.py \\
        --data datasets/raw/train.csv \\
        --target Survived \\
        --model rit-qwen3-8b   # or a raw model id if hitting RIT directly
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
from agentic_ml.steps.profiler_step import run_profiler_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--model", default=None, help="model id or gateway model_name; "
                         "defaults to RIT_DEFAULT_MODEL env var")
    parser.add_argument("--use-gateway", action="store_true",
                         help="call through the LiteLLM gateway instead of RIT directly")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    run_id, run_dir = make_run_dir(args.run_id)
    trace = make_tracer(run_dir / "trace.jsonl")
    write_transcript = make_transcript_writer(run_dir)

    spec = DatasetSpec(path=args.data, target_column=args.target)
    loaded = load_dataset(spec)
    print(f"Loaded dataset: {len(loaded.df)} rows, data_hash={loaded.data_hash[:16]}...")
    trace("dataset_loaded", data_hash=loaded.data_hash, n_rows=len(loaded.df))

    base_url, api_key, default_model = resolve_model_endpoint(
        args.use_gateway, args.model, "qwen3:8b", "rit-qwen3-8b",
    )
    client = ModelClient(base_url=base_url, api_key=api_key, default_model=default_model)

    print(f"Running ProfilerAgent (model={default_model})...")
    step_result = run_profiler_step(
        loaded.df, args.target, client, model=default_model, max_turns=4,
        trace_fn=lambda record: trace(**record),
    )
    print(f"Agent stopped: {step_result.stopped_reason} (turns_used={step_result.turns_used})")
    transcript_path = write_transcript("profiler", step_result.messages)
    print(f"Transcript written to {transcript_path}")

    # deterministic tool output is always saved, regardless of whether the
    # LLM's narrative parses cleanly — the facts don't depend on the LLM
    if step_result.deterministic_report:
        dr = step_result.deterministic_report
        (run_dir / "profiler_report_deterministic.json").write_text(json.dumps(dr, indent=2))
        print("\n--- Deterministic facts (tool output, not LLM-generated) ---")
        print(f"  recommended_split_strategy: {dr['recommended_split_strategy']}")
        print(f"  likely_id_columns: {dr['likely_id_columns']}")
        print(f"  likely_group_columns: {dr['likely_group_columns']}")
        print(f"  likely_datetime_columns: {dr['likely_datetime_columns']}")
        print(f"  is_imbalanced: {dr['is_imbalanced']} (ratio={dr['class_imbalance_ratio']})")
        print(f"  leakage_risk_flags: {dr['leakage_risk_flags']}")

    print("\n--- LLM narrative/recommendation ---")
    if step_result.llm_narrative:
        print(json.dumps(step_result.llm_narrative, indent=2))
    else:
        print("WARNING: LLM output did not parse as JSON. Raw text:")
        print(step_result.llm_raw_text)

    output = {
        "run_id": run_id,
        "model": default_model,
        "deterministic_report": step_result.deterministic_report,
        "llm_narrative": step_result.llm_narrative,
        "llm_raw_text": step_result.llm_raw_text,
        "stopped_reason": step_result.stopped_reason,
        "turns_used": step_result.turns_used,
        "transcript": str(transcript_path),
    }
    out_path = run_dir / "profiler_report.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nFull report written to {out_path}")

    if step_result.deterministic_report is None:
        print("FAILED: agent never called get_dataset_profile.")
        sys.exit(1)
    if step_result.llm_narrative is None:
        print("PARTIAL: deterministic facts are solid, but LLM narrative did not "
              "parse as valid JSON. Check the raw text above.")
        sys.exit(2)
    print("SUCCESS.")


if __name__ == "__main__":
    main()
