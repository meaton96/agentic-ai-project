#!/usr/bin/env python3
"""
Agent #1: run the Load Monitor agent standalone against the task table.
Deterministic code (environment/state.py) computes every fact and flag;
the LLM only narrates. Mirrors
agentic-ml-classification/scripts/run_profiler_agent.py's shape.

Usage:
    set -a; source .env; set +a
    python scripts/run_load_monitor_agent.py \\
        --data datasets/raw/industrial_scheduling_dataset.csv \\
        --model rit-qwen3-8b   # or a raw model id if hitting RIT directly
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from resource_scheduler.cli_common import make_run_dir, make_tracer, make_transcript_writer, resolve_model_endpoint
from resource_scheduler.environment.state import load_task_table
from resource_scheduler.model_client import ModelClient
from resource_scheduler.steps.load_monitor_step import run_load_monitor_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--window", type=int, default=200, help="rows treated as 'current' state")
    parser.add_argument("--no-inject-variance", action="store_true",
                         help="disable synthetic jitter on constant columns (use once a real live feed exists)")
    parser.add_argument("--model", default=None, help="model id or gateway model_name; "
                         "defaults to RIT_DEFAULT_MODEL env var")
    parser.add_argument("--use-gateway", action="store_true",
                         help="call through the LiteLLM gateway instead of RIT directly")
    parser.add_argument("--use-local", action="store_true", help="call a local OpenAI-"
                         "compatible server (LOCAL_MODEL_BASE_URL) instead of RIT/the "
                         "gateway -- takes precedence over --use-gateway if both are given")
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
    client = ModelClient(base_url=base_url, api_key=api_key, default_model=default_model)

    result = run_load_monitor_step(
        df, variance_injected, client, window=args.window, trace_fn=lambda record: trace(**record),
    )
    write_transcript("load_monitor", result.messages)

    report_path = run_dir / "load_monitor_report.json"
    report_path.write_text(json.dumps({
        "ok": result.ok,
        "deterministic_report": result.deterministic_report,
        "llm_narrative": result.llm_narrative,
        "flag_mismatch": result.flag_mismatch,
        "stopped_reason": result.stopped_reason,
        "turns_used": result.turns_used,
    }, indent=2, default=str))

    print(f"ok={result.ok} flags={len(result.deterministic_report['flags']) if result.deterministic_report else None} "
          f"flag_mismatch={result.flag_mismatch}")
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
