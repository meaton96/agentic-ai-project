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
import os
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from agentic_ml.model_client import ModelClient
from agentic_ml.agent_runtime import ToolCallingAgent
from agentic_ml.tools.profiler_tool import make_profiler_tool
from agentic_ml.tools.template_tool import make_list_templates_tool
from agentic_ml.templates.registry import get_template, validate_config
from agentic_ml.harness.dataset import DatasetSpec, load_dataset, write_dataset_spec
from agentic_ml.harness.splits import make_split
from agentic_ml.harness.leakage import run_all_split_leakage_checks, label_permutation_test
from agentic_ml.harness.sandbox import run_candidate_build
from agentic_ml.harness.metrics import compute_metrics
from agentic_ml.harness.leaderboard import append_leaderboard_entry


SYSTEM_PROMPT = """You are the Modeling agent in a deterministic ML evaluation \
pipeline. You have two tools:

  get_dataset_profile — factual column types, cardinality, missingness, \
likely id/group/datetime columns, target imbalance, and leakage risk flags. \
You cannot compute these facts yourself and must not guess or invent them.

  list_templates — the available recipe templates, what each does, when to \
use it, and its config contract.

Call both tools exactly once, in either order. Then propose exactly ONE \
candidate by responding with ONLY valid JSON (no prose, no markdown fences) \
matching this schema:
{
  "candidate_id": "<short_snake_case_id>",
  "template_id": "<copy exactly from list_templates>",
  "config": {
    "numeric_cols": ["<...>"],
    "categorical_cols": ["<...>"],
    "...": "<any other keys the chosen template's config contract lists>"
  },
  "explanation": "<2-4 sentences: why this template, why these columns, \
why any non-default hyperparameters>"
}

Hard rules:
- numeric_cols and categorical_cols must be built ONLY from columns the \
profiler reported. Never invent a column name.
- Never include the target column, or any column the profiler flagged as \
is_likely_id, is_likely_datetime, or a declared group/time column.
- A column may appear in at most one of numeric_cols / categorical_cols.
- Pick the template whose when_to_use best matches what the profiler found \
(cardinality, imbalance, categorical dtypes) — don't default to the same \
template every time.
- Do not include a metric, score, or fitted result. You do not compute those; \
the harness does, after your proposal."""


def _validate_candidate_spec_shape(obj) -> list[str]:
    errors = []
    if not isinstance(obj, dict):
        return ["top-level response is not a JSON object"]
    for key in ("candidate_id", "template_id", "config"):
        if key not in obj:
            errors.append(f"missing required key: '{key}'")
    if "candidate_id" in obj and not isinstance(obj["candidate_id"], str):
        errors.append("'candidate_id' must be a string")
    if "template_id" in obj and not isinstance(obj["template_id"], str):
        errors.append("'template_id' must be a string")
    if "config" in obj and not isinstance(obj["config"], dict):
        errors.append("'config' must be an object")
    return errors


def _validate_candidate_columns(
    profile_report: dict, target_column: str, group_column: str | None,
    time_column: str | None, config: dict,
) -> list[str]:
    errors = []
    known_cols = {c["name"]: c for c in profile_report["columns"]}
    disallowed = {target_column}
    if group_column:
        disallowed.add(group_column)
    if time_column:
        disallowed.add(time_column)

    for key in ("numeric_cols", "categorical_cols"):
        for col in config.get(key, []):
            if col not in known_cols:
                errors.append(f"{key} references unknown column '{col}'")
                continue
            if col in disallowed:
                errors.append(f"{key} includes disallowed column '{col}' (target/group/time)")
            elif known_cols[col]["is_likely_id"]:
                errors.append(f"{key} includes a likely-ID column '{col}'")

    overlap = set(config.get("numeric_cols", [])) & set(config.get("categorical_cols", []))
    if overlap:
        errors.append(f"columns listed in both numeric_cols and categorical_cols: {sorted(overlap)}")

    if not config.get("numeric_cols") and not config.get("categorical_cols"):
        errors.append("config declares no feature columns at all")

    return errors


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

    run_id = args.run_id or f"run_{uuid.uuid4().hex[:8]}"
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    trace_path = run_dir / "trace.jsonl"

    def trace(event: str, **fields):
        with open(trace_path, "a") as f:
            f.write(json.dumps({"ts": time.time(), "event": event, **fields}) + "\n")

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

    if args.use_gateway:
        base_url = os.environ.get("MODEL_GATEWAY_BASE_URL", "http://localhost:4000/v1")
        api_key = os.environ.get("LITELLM_MASTER_KEY", "")
        default_model = args.model or "rit-qwen3-coder-30b"
    else:
        base_url = os.environ["RIT_BASE_URL"]
        api_key = os.environ["RIT_API_KEY"]
        default_model = args.model or os.environ.get("RIT_DEFAULT_MODEL", "qwen3-coder:30b")

    client = ModelClient(base_url=base_url, api_key=api_key, default_model=default_model)
    tools = [make_profiler_tool(loaded.df, args.target), make_list_templates_tool()]

    agent = ToolCallingAgent(
        model_client=client, tools=tools, system_prompt=SYSTEM_PROMPT,
        model=default_model, max_turns=args.max_turns,
    )

    print(f"\nRunning ModelingAgent (model={default_model})...")
    result = agent.run(
        "Propose one modeling candidate for this dataset.",
        trace_fn=lambda record: trace(**record),
    )
    print(f"Agent stopped: {result.stopped_reason} (turns_used={result.turns_used})")

    profile_report = None
    for entry in result.tool_call_log:
        if entry["tool"] == "get_dataset_profile":
            profile_report = entry["result"]

    if result.final_text is None:
        print("FAILED: agent never produced a final candidate proposal.")
        sys.exit(1)

    try:
        candidate = json.loads(result.final_text)
    except json.JSONDecodeError:
        print("FAILED: agent's final response did not parse as JSON:")
        print(result.final_text)
        sys.exit(1)

    shape_errors = _validate_candidate_spec_shape(candidate)
    if shape_errors:
        print("FAILED: candidate proposal failed shape validation:")
        for e in shape_errors:
            print(f"  - {e}")
        sys.exit(1)

    template_id = candidate["template_id"]
    config = dict(candidate["config"])
    # seed is harness-controlled, never agent-controlled — overwrite whatever
    # the agent proposed so evaluation stays reproducible under our seed.
    config["seed"] = args.seed

    try:
        template_errors = validate_config(template_id, config)
    except KeyError as e:
        print(f"FAILED: {e}")
        sys.exit(1)

    column_errors = _validate_candidate_columns(
        profile_report, args.target, args.group_column, args.time_column, config,
    ) if profile_report else ["profiler was never called — cannot validate proposed columns"]

    all_errors = template_errors + column_errors
    if all_errors:
        print("FAILED: candidate config failed validation:")
        for e in all_errors:
            print(f"  - {e}")
        sys.exit(1)

    candidate_id = candidate["candidate_id"]
    template = get_template(template_id)
    candidate_dir = run_dir / "candidates" / candidate_id
    candidate_dir.mkdir(parents=True, exist_ok=True)
    (candidate_dir / "candidate.py").write_text(template.read_source())
    (candidate_dir / "candidate_config.json").write_text(json.dumps(config, indent=2))
    (candidate_dir / "explanation.md").write_text(
        f"# {candidate_id}\n\ntemplate: {template_id}\n\n{candidate.get('explanation', '')}\n"
    )
    print(f"\nCandidate proposal: {candidate_id} (template={template_id})")
    print(f"  {candidate.get('explanation', '')}")

    pipeline, build_error = run_candidate_build(template.read_source(), config, timeout_seconds=60)
    (candidate_dir / "build_result.json").write_text(
        json.dumps({"ok": build_error is None, "error": build_error}, indent=2)
    )
    if build_error:
        print(f"\nFAILED: sandbox build error: {build_error}")
        sys.exit(1)

    X = loaded.X
    y = loaded.y
    train_idx, val_idx = manifest.train_idx, manifest.val_idx

    pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
    y_pred = pipeline.predict(X.iloc[val_idx])
    proba = pipeline.predict_proba(X.iloc[val_idx])
    y_proba = proba[:, 1] if proba.shape[1] == 2 else proba.max(axis=1)

    metric_names = args.metrics.split(",")
    results = compute_metrics(
        y.iloc[val_idx].values, y_pred, y_proba, metric_names, n_bootstrap=200, seed=args.seed,
    )
    print("\nValidation metrics:")
    for m in metric_names:
        r = results[m]
        print(f"  {m}: {r.value:.4f}  (95% CI [{r.ci_low:.4f}, {r.ci_high:.4f}])")

    def fit_and_score(X_tr, y_tr, X_va, y_va) -> float:
        candidate_pipeline = clone(pipeline)
        candidate_pipeline.fit(X_tr, y_tr)
        p = candidate_pipeline.predict_proba(X_va)[:, 1]
        return roc_auc_score(y_va, p)

    permutation_check = label_permutation_test(
        fit_and_score, X.iloc[train_idx], y.iloc[train_idx], X.iloc[val_idx], y.iloc[val_idx],
        metric_name="roc_auc", seed=args.seed,
    )
    print(f"\n  leakage check [{'PASS' if permutation_check.passed else 'FAIL'}] "
          f"{permutation_check.check_name}: {permutation_check.detail}")

    evaluation = {
        "candidate_id": candidate_id,
        "template_id": template_id,
        "metrics": {m: results[m].to_dict() for m in metric_names},
        "label_permutation_check": permutation_check.to_dict(),
    }
    (candidate_dir / "evaluation.json").write_text(json.dumps(evaluation, indent=2))

    if not permutation_check.passed:
        print("\nABORTING: candidate pipeline failed the label-permutation leakage "
              "check. Not promoting to the leaderboard.")
        sys.exit(1)

    leaderboard_path = Path("artifacts/reports/leaderboard.jsonl")
    entry = {
        "run_id": run_id,
        "candidate": candidate_id,
        "template_id": template_id,
        "source": "modeling_agent",
        "model": default_model,
        "split": "validation",
        "strategy": args.strategy,
        "data_hash": loaded.data_hash,
        "seed": args.seed,
        "metrics": {m: results[m].to_dict() for m in metric_names},
    }
    append_leaderboard_entry(leaderboard_path, entry)
    print(f"\nLeaderboard appended: {leaderboard_path}")
    print(f"Candidate artifacts: {candidate_dir}")
    print("\nNOTE: test split was never touched. Only validation metrics were computed.")
    print("SUCCESS.")


if __name__ == "__main__":
    main()
