#!/usr/bin/env python3
"""
Phase 2: run the ProfilerAgent. Deterministic code (profile_dataset)
computes every fact; the LLM only narrates and recommends. The agent
gets exactly one tool and is instructed to call it once.

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
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import os
from agentic_ml.model_client import ModelClient
from agentic_ml.agent_runtime import ToolCallingAgent
from agentic_ml.tools.profiler_tool import make_profiler_tool
from agentic_ml.harness.dataset import DatasetSpec, load_dataset


SYSTEM_PROMPT = """You are the Profiler agent in a deterministic ML evaluation \
pipeline. You have exactly one tool: get_dataset_profile. Call it once to get \
factual information about the dataset — you cannot compute these facts yourself \
and must not guess or invent numbers.

After calling the tool, respond with ONLY valid JSON (no prose, no markdown \
fences) matching this schema:
{
  "summary": "<2-4 sentence plain-language summary of the dataset>",
  "recommended_split_strategy": "<copy exactly from the tool output>",
  "key_risks": ["<short bullet>", "..."],
  "recommended_next_steps": ["<short bullet>", "..."]
}

Do not contradict the tool's recommended_split_strategy or leakage_risk_flags — \
your job is to explain and contextualize them, not override them. If the tool \
output already lists leakage_risk_flags, they must appear in key_risks."""


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

    run_id = args.run_id or f"run_{uuid.uuid4().hex[:8]}"
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    trace_path = run_dir / "trace.jsonl"

    def trace(event: str, **fields):
        with open(trace_path, "a") as f:
            f.write(json.dumps({"ts": time.time(), "event": event, **fields}) + "\n")

    spec = DatasetSpec(path=args.data, target_column=args.target)
    loaded = load_dataset(spec)
    print(f"Loaded dataset: {len(loaded.df)} rows, data_hash={loaded.data_hash[:16]}...")
    trace("dataset_loaded", data_hash=loaded.data_hash, n_rows=len(loaded.df))

    if args.use_gateway:
        base_url = os.environ.get("MODEL_GATEWAY_BASE_URL", "http://localhost:4000/v1")
        api_key = os.environ.get("LITELLM_MASTER_KEY", "")
        default_model = args.model or "rit-qwen3-8b"
    else:
        base_url = os.environ["RIT_BASE_URL"]
        api_key = os.environ["RIT_API_KEY"]
        default_model = args.model or os.environ.get("RIT_DEFAULT_MODEL", "qwen3:8b")

    client = ModelClient(base_url=base_url, api_key=api_key, default_model=default_model)
    tool = make_profiler_tool(loaded.df, args.target)

    agent = ToolCallingAgent(
        model_client=client,
        tools=[tool],
        system_prompt=SYSTEM_PROMPT,
        model=default_model,
        max_turns=4,
    )

    print(f"Running ProfilerAgent (model={default_model})...")
    result = agent.run(
        "Profile this dataset and give your summary and recommendations.",
        trace_fn=lambda record: trace(**record),
    )

    print(f"Agent stopped: {result.stopped_reason} (turns_used={result.turns_used})")

    # deterministic tool output is always saved, regardless of whether the
    # LLM's narrative parses cleanly — the facts don't depend on the LLM
    deterministic_report = None
    for entry in result.tool_call_log:
        if entry["tool"] == "get_dataset_profile":
            deterministic_report = entry["result"]
            break

    if deterministic_report:
        (run_dir / "profiler_report_deterministic.json").write_text(
            json.dumps(deterministic_report, indent=2)
        )
        print("\n--- Deterministic facts (tool output, not LLM-generated) ---")
        print(f"  recommended_split_strategy: {deterministic_report['recommended_split_strategy']}")
        print(f"  likely_id_columns: {deterministic_report['likely_id_columns']}")
        print(f"  likely_group_columns: {deterministic_report['likely_group_columns']}")
        print(f"  likely_datetime_columns: {deterministic_report['likely_datetime_columns']}")
        print(f"  is_imbalanced: {deterministic_report['is_imbalanced']} "
              f"(ratio={deterministic_report['class_imbalance_ratio']})")
        print(f"  leakage_risk_flags: {deterministic_report['leakage_risk_flags']}")

    llm_parsed = None
    if result.final_text:
        try:
            llm_parsed = json.loads(result.final_text)
        except json.JSONDecodeError:
            llm_parsed = None

    print("\n--- LLM narrative/recommendation ---")
    if llm_parsed:
        print(json.dumps(llm_parsed, indent=2))
    else:
        print("WARNING: LLM output did not parse as JSON. Raw text:")
        print(result.final_text)

    output = {
        "run_id": run_id,
        "model": default_model,
        "deterministic_report": deterministic_report,
        "llm_narrative": llm_parsed,
        "llm_raw_text": result.final_text,
        "stopped_reason": result.stopped_reason,
        "turns_used": result.turns_used,
    }
    out_path = run_dir / "profiler_report.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nFull report written to {out_path}")

    if deterministic_report is None:
        print("FAILED: agent never called get_dataset_profile.")
        sys.exit(1)
    if llm_parsed is None:
        print("PARTIAL: deterministic facts are solid, but LLM narrative did not "
              "parse as valid JSON. Check the raw text above.")
        sys.exit(2)
    print("SUCCESS.")


if __name__ == "__main__":
    main()