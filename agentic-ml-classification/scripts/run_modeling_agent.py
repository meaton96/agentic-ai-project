#!/usr/bin/env python3
"""
Phase 3: run the ModelingAgent. The agent picks one verified recipe
template (src/agentic_ml/templates/) and fills in its config "holes"
(feature column lists, a few hyperparameters) — it never writes
pipeline code from scratch. The harness then does everything the agent
is not trusted to do: validate the proposed columns against the
profiler's facts, static-check + sandbox-build the template source,
fit/score on the harness-owned split, run a label-permutation leakage
gate, and only then append to the leaderboard.

This script is a thin CLI wrapper around
agentic_ml.steps.modeling_step.run_modeling_step — the same function
scripts/run_orchestrator.py calls (possibly multiple times, to try
several candidates) as part of the full loop.

Usage:
    set -a; source .env; set +a
    python scripts/run_modeling_agent.py \\
        --data datasets/raw/train.csv \\
        --target Survived \\
        --model rit-qwen3-coder-30b   # or a raw model id if hitting RIT directly
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agentic_ml.cli_common import make_run_dir, make_tracer, resolve_model_endpoint
from agentic_ml.harness.dataset import DatasetSpec, load_dataset, write_dataset_spec
from agentic_ml.harness.leaderboard import append_leaderboard_entry
from agentic_ml.harness.leakage import run_all_split_leakage_checks
from agentic_ml.harness.splits import make_split
from agentic_ml.model_client import ModelClient
from agentic_ml.steps.modeling_step import run_modeling_step
from agentic_ml.templates.registry import get_template


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--strategy", default="stratified",
                         choices=["random", "stratified", "group", "time", "group_time"])
    parser.add_argument("--group-column", default=None)
    parser.add_argument("--time-column", default=None)
    parser.add_argument("--id-columns", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metrics", default="roc_auc,pr_auc,f1,accuracy")
    parser.add_argument("--model", default=None, help="model id or gateway model_name; "
                         "defaults to RIT_DEFAULT_MODEL env var")
    parser.add_argument("--use-gateway", action="store_true")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--max-turns", type=int, default=6)
    args = parser.parse_args()

    run_id, run_dir = make_run_dir(args.run_id)
    trace = make_tracer(run_dir / "trace.jsonl")

    id_columns = [c.strip() for c in args.id_columns.split(",") if c.strip()]
    spec = DatasetSpec(
        path=args.data, target_column=args.target, group_column=args.group_column,
        time_column=args.time_column, id_columns=id_columns,
    )
    write_dataset_spec(spec, run_dir / "dataset_spec.json")

    loaded = load_dataset(spec)
    trace("dataset_loaded", data_hash=loaded.data_hash, n_rows=len(loaded.df))
    print(f"Loaded dataset: {len(loaded.df)} rows, data_hash={loaded.data_hash[:16]}...")

    manifest = make_split(
        df=loaded.df, target_column=args.target, data_hash=loaded.data_hash,
        strategy=args.strategy, seed=args.seed,
        group_column=args.group_column, time_column=args.time_column,
    )
    manifest.write(run_dir / "split_manifest.json")
    print(f"Split ({args.strategy}): train={len(manifest.train_idx)} "
          f"val={len(manifest.val_idx)} test={len(manifest.test_idx)}")

    leakage_checks = run_all_split_leakage_checks(
        df=loaded.df, group_column=args.group_column, time_column=args.time_column,
        train_idx=manifest.train_idx, val_idx=manifest.val_idx, test_idx=manifest.test_idx,
        strategy=args.strategy,
    )
    (run_dir / "leakage_checks.json").write_text(
        json.dumps([c.to_dict() for c in leakage_checks], indent=2)
    )
    for check in leakage_checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"  leakage check [{status}] {check.check_name}: {check.detail}")
    if not all(c.passed for c in leakage_checks):
        print("\nABORTING: split-level leakage check failed.")
        sys.exit(1)

    base_url, api_key, default_model = resolve_model_endpoint(
        args.use_gateway, args.model, "qwen3-coder:30b", "rit-qwen3-coder-30b",
    )
    client = ModelClient(base_url=base_url, api_key=api_key, default_model=default_model)

    print(f"\nRunning ModelingAgent (model={default_model})...")
    step_result = run_modeling_step(
        full_df=loaded.df, X=loaded.X, y=loaded.y, target_column=args.target,
        group_column=args.group_column, time_column=args.time_column,
        train_idx=manifest.train_idx, val_idx=manifest.val_idx,
        client=client, model=default_model, max_turns=args.max_turns,
        metric_names=args.metrics.split(","), seed=args.seed,
        trace_fn=lambda record: trace(**record),
    )
    print(f"Agent stopped: {step_result.stopped_reason} (turns_used={step_result.turns_used})")

    if step_result.candidate_id:
        candidate_dir = run_dir / "candidates" / step_result.candidate_id
        candidate_dir.mkdir(parents=True, exist_ok=True)
        if step_result.template_id:
            (candidate_dir / "candidate.py").write_text(get_template(step_result.template_id).read_source())
        if step_result.config:
            (candidate_dir / "candidate_config.json").write_text(json.dumps(step_result.config, indent=2))
        (candidate_dir / "explanation.md").write_text(
            f"# {step_result.candidate_id}\n\ntemplate: {step_result.template_id}\n\n"
            f"{step_result.explanation or ''}\n"
        )
        print(f"\nCandidate proposal: {step_result.candidate_id} (template={step_result.template_id})")
        if step_result.explanation:
            print(f"  {step_result.explanation}")

        if step_result.metrics:
            print("\nValidation metrics:")
            for m, r in step_result.metrics.items():
                print(f"  {m}: {r['value']:.4f}  (95% CI [{r['ci_low']:.4f}, {r['ci_high']:.4f}])")

        if step_result.label_permutation_check:
            lp = step_result.label_permutation_check
            print(f"\n  leakage check [{'PASS' if lp['passed'] else 'FAIL'}] {lp['check']}: {lp['detail']}")

        evaluation = {
            "candidate_id": step_result.candidate_id,
            "template_id": step_result.template_id,
            "metrics": step_result.metrics,
            "label_permutation_check": step_result.label_permutation_check,
            "errors": step_result.errors,
        }
        (candidate_dir / "evaluation.json").write_text(json.dumps(evaluation, indent=2))

    if not step_result.ok:
        print("\nFAILED:")
        for e in step_result.errors:
            print(f"  - {e}")
        sys.exit(1)

    leaderboard_path = Path("artifacts/reports/leaderboard.jsonl")
    entry = {
        "run_id": run_id,
        "candidate": step_result.candidate_id,
        "template_id": step_result.template_id,
        "source": "modeling_agent",
        "model": default_model,
        "split": "validation",
        "strategy": args.strategy,
        "data_hash": loaded.data_hash,
        "seed": args.seed,
        "metrics": step_result.metrics,
    }
    append_leaderboard_entry(leaderboard_path, entry)
    print(f"\nLeaderboard appended: {leaderboard_path}")
    print(f"Candidate artifacts: {run_dir / 'candidates' / step_result.candidate_id}")
    print("\nNOTE: test split was never touched. Only validation metrics were computed.")
    print("SUCCESS.")


if __name__ == "__main__":
    main()
