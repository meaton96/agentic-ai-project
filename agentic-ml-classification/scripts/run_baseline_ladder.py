#!/usr/bin/env python3
"""
Phase 1 entry point: evaluate the baseline ladder on a dataset with
ZERO agent/LLM involvement. This proves the harness works standalone,
independent of any model connectivity questions.

Usage:
    python scripts/run_baseline_ladder.py \\
        --data datasets/raw/synthetic_churn.csv \\
        --target churned \\
        --group-column customer_id \\
        --time-column signup_day \\
        --strategy group_time
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agentic_ml.cli_common import make_run_dir
from agentic_ml.harness.dataset import DatasetSpec, load_dataset, write_dataset_spec
from agentic_ml.harness.splits import make_split
from agentic_ml.harness.leakage import run_all_split_leakage_checks
from agentic_ml.harness.baseline_ladder import ColumnProfile, build_baseline_pipelines, fit_and_predict
from agentic_ml.harness.metrics import compute_metrics
from agentic_ml.harness.leaderboard import append_leaderboard_entry
from agentic_ml.paths import leaderboard_path as resolve_leaderboard_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--strategy", default="stratified",
                         choices=["random", "stratified", "group", "time", "group_time"])
    parser.add_argument("--group-column", default=None)
    parser.add_argument("--time-column", default=None)
    parser.add_argument("--id-columns", default="",
                         help="comma-separated columns to exclude entirely from modeling "
                              "(identifiers, free text, columns with too much missingness, etc.)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metrics", default="roc_auc,pr_auc,f1,accuracy")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    run_id, run_dir = make_run_dir(args.run_id)

    trace_path = run_dir / "trace.jsonl"

    def trace(event: str, **fields):
        with open(trace_path, "a") as f:
            f.write(json.dumps({"ts": time.time(), "event": event, **fields}) + "\n")

    id_columns = [c.strip() for c in args.id_columns.split(",") if c.strip()]

    spec = DatasetSpec(
        path=args.data,
        target_column=args.target,
        group_column=args.group_column,
        time_column=args.time_column,
        id_columns=id_columns,
    )
    write_dataset_spec(spec, run_dir / "dataset_spec.json")
    trace("dataset_spec_written")

    loaded = load_dataset(spec)
    trace("dataset_loaded", data_hash=loaded.data_hash, n_rows=len(loaded.df))
    print(f"Loaded dataset: {len(loaded.df)} rows, data_hash={loaded.data_hash[:16]}...")

    manifest = make_split(
        df=loaded.df,
        target_column=args.target,
        data_hash=loaded.data_hash,
        strategy=args.strategy,
        seed=args.seed,
        group_column=args.group_column,
        time_column=args.time_column,
    )
    manifest.write(run_dir / "split_manifest.json")
    trace("split_created", strategy=args.strategy,
          n_train=len(manifest.train_idx), n_val=len(manifest.val_idx), n_test=len(manifest.test_idx))
    print(f"Split ({args.strategy}): train={len(manifest.train_idx)} "
          f"val={len(manifest.val_idx)} test={len(manifest.test_idx)}")
    print(f"Target distribution: {manifest.target_distribution}")

    leakage_checks = run_all_split_leakage_checks(
        df=loaded.df,
        group_column=args.group_column,
        time_column=args.time_column,
        train_idx=manifest.train_idx,
        val_idx=manifest.val_idx,
        test_idx=manifest.test_idx,
        strategy=args.strategy,
    )
    (run_dir / "leakage_checks.json").write_text(
        json.dumps([c.to_dict() for c in leakage_checks], indent=2)
    )
    for check in leakage_checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"  leakage check [{status}] {check.check_name}: {check.detail}")
    trace("leakage_checks_complete", all_passed=all(c.passed for c in leakage_checks))

    if not all(c.passed for c in leakage_checks):
        print("\nABORTING: leakage check failed. Baseline ladder will not run "
              "against a compromised split.")
        sys.exit(1)

    X = loaded.X
    y = loaded.y
    train_idx, val_idx, test_idx = manifest.train_idx, manifest.val_idx, manifest.test_idx

    # exclude declared group/time columns from model features (they're
    # metadata for splitting, not predictive signal by default)
    exclude_cols = [c for c in (args.group_column, args.time_column) if c]
    X_model = X.drop(columns=[c for c in exclude_cols if c in X.columns])

    profile = ColumnProfile.infer(X_model)
    print(f"\nColumn profile: {len(profile.numeric_cols)} numeric, "
          f"{len(profile.categorical_cols)} categorical")

    pipelines = build_baseline_pipelines(profile, seed=args.seed)
    metric_names = args.metrics.split(",")

    print(f"\nRunning baseline ladder ({len(pipelines)} candidates) on VALIDATION split:")
    print(f"{'candidate':<24}" + "".join(f"{m:>12}" for m in metric_names))

    leaderboard_path = resolve_leaderboard_path()

    for name, pipeline in pipelines.items():
        start = time.time()
        y_pred, y_proba = fit_and_predict(
            pipeline,
            X_model.iloc[train_idx], y.iloc[train_idx],
            X_model.iloc[val_idx],
        )
        elapsed = round(time.time() - start, 3)

        results = compute_metrics(
            y.iloc[val_idx].values, y_pred, y_proba, metric_names, n_bootstrap=200, seed=args.seed
        )
        row = f"{name:<24}" + "".join(f"{results[m].value:>12.4f}" for m in metric_names)
        print(row)

        entry = {
            "run_id": run_id,
            "candidate": name,
            "split": "validation",
            "strategy": args.strategy,
            "data_hash": loaded.data_hash,
            "seed": args.seed,
            "elapsed_seconds": elapsed,
            "metrics": {m: results[m].to_dict() for m in metric_names},
        }
        append_leaderboard_entry(leaderboard_path, entry)
        trace("candidate_evaluated", candidate=name, elapsed_seconds=elapsed)

    print(f"\nLeaderboard appended: {leaderboard_path}")
    print(f"Run artifacts: {run_dir}")
    print("\nNOTE: test split was never touched. Only validation metrics were computed.")


if __name__ == "__main__":
    main()