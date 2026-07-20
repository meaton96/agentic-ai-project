#!/usr/bin/env python3
"""
Phase 5: the Orchestrator. This is the first entry point that runs the
full "prompt (or just a dataset) -> agent pipeline -> finished
classification" loop: intake (skippable via --target) -> feature
engineering (proposes columns to drop + stateless derived features
from a vetted catalog, skippable via --skip-feature-engineering) ->
profiler (drives the split strategy, sees the augmented column set) ->
up to --max-candidates modeling-agent proposals, each independently
gated by two deterministic leakage checks -> select-and-verify (best-
first, VerificationAgent reviews each and can only veto, never approve/
unblock one that failed a gate; rejected candidates fall back to the
next-best) -> refit the accepted candidate on train+val -> one locked
test-set evaluation -> a short LLM-narrated plain-text summary.

scripts/run_profiler_agent.py and scripts/run_modeling_agent.py remain
useful for driving a single phase in isolation; this script drives the
same underlying step functions (agentic_ml.steps.*) end to end.

Usage:
    set -a; source .env; set +a

    # natural-language goal, dataset only otherwise:
    python scripts/run_orchestrator.py \\
        --data datasets/raw/train.csv \\
        --goal "predict whether a passenger survived"

    # dataset only, no goal, no --target — intake infers a target from
    # the schema alone:
    python scripts/run_orchestrator.py --data datasets/raw/train.csv

    # explicit --target skips intake entirely, same as the standalone scripts:
    python scripts/run_orchestrator.py --data datasets/raw/train.csv --target Survived
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import joblib
from sklearn.base import clone

from agentic_ml.cli_common import make_run_dir, make_tracer, make_transcript_writer, resolve_model_endpoint
from agentic_ml.harness.attribution import compute_background
from agentic_ml.harness.dataset import DatasetSpec, LoadedDataset, load_dataset, read_dataframe, write_dataset_spec
from agentic_ml.harness.leaderboard import append_leaderboard_entry
from agentic_ml.harness.leakage import run_all_split_leakage_checks
from agentic_ml.harness.metrics import compute_metrics
from agentic_ml.harness.splits import make_split, resolve_split_columns
from agentic_ml.harness.verification import build_review_bundle
from agentic_ml.model_client import ModelClient
from agentic_ml.steps.feature_engineering_step import run_feature_engineering_step
from agentic_ml.steps.intake_step import run_intake_step
from agentic_ml.steps.modeling_step import run_modeling_step
from agentic_ml.steps.profiler_step import run_profiler_step
from agentic_ml.steps.verification_step import run_verification_step
from agentic_ml.templates.registry import get_template


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--goal", default="", help="natural-language description of the "
                         "prediction goal; optional, only used when --target is not given")
    parser.add_argument("--target", default=None, help="skip intake and use this target "
                         "column directly")
    parser.add_argument("--group-column", default=None)
    parser.add_argument("--time-column", default=None)
    parser.add_argument("--id-columns", default=None, help="comma-separated; overrides "
                         "intake's proposal if given")
    parser.add_argument("--strategy", default=None,
                         choices=["random", "stratified", "group", "time", "group_time"],
                         help="overrides the profiler's recommended split strategy if given")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metrics", default="roc_auc,pr_auc,f1,accuracy")
    parser.add_argument("--max-candidates", type=int, default=2)
    parser.add_argument("--model", default=None)
    parser.add_argument("--verification-model", default=None, help="model id or gateway "
                         "model_name for the VerificationAgent; defaults to a reasoning-"
                         "oriented model distinct from the modeling agent's default")
    parser.add_argument("--use-gateway", action="store_true")
    parser.add_argument("--use-local", action="store_true", help="call a local OpenAI-"
                         "compatible server (LOCAL_MODEL_BASE_URL, default "
                         "http://localhost:8000/v1) instead of RIT/the gateway — "
                         "takes precedence over --use-gateway if both are given")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--skip-feature-engineering", action="store_true",
                         help="skip the feature-engineering step entirely")
    args = parser.parse_args()

    run_id, run_dir = make_run_dir(args.run_id)
    trace = make_tracer(run_dir / "trace.jsonl")
    write_transcript = make_transcript_writer(run_dir)

    base_url, api_key, default_model = resolve_model_endpoint(
        args.use_gateway, args.model, "qwen3-coder:30b", "rit-qwen3-coder-30b",
        use_local=args.use_local,
    )
    client = ModelClient(base_url=base_url, api_key=api_key, default_model=default_model)

    report: dict = {"run_id": run_id, "model": default_model, "status": "in_progress"}

    def write_report():
        (run_dir / "orchestrator_report.json").write_text(json.dumps(report, indent=2, default=str))

    # --- 1. Intake: figure out the DatasetSpec, unless --target was given ---
    if args.target is None:
        print(f"No --target given; running IntakeAgent (goal={args.goal!r})...")
        raw_df = read_dataframe(args.data)

        intake_result = run_intake_step(
            raw_df, args.goal, client, model=default_model,
            trace_fn=lambda record: trace(**record),
        )
        intake_transcript_path = write_transcript("intake", intake_result.messages)
        (run_dir / "intake_report.json").write_text(json.dumps({
            "dataset_spec_proposal": intake_result.dataset_spec_proposal,
            "validation_errors": intake_result.validation_errors,
            "raw_schema": intake_result.raw_schema,
            "llm_raw_text": intake_result.llm_raw_text,
            "transcript": str(intake_transcript_path),
        }, indent=2, default=str))

        if not intake_result.ok:
            print("FAILED: IntakeAgent's proposal did not validate:")
            for e in intake_result.validation_errors:
                print(f"  - {e}")
            report["status"] = "failed_intake"
            report["errors"] = intake_result.validation_errors
            write_report()
            sys.exit(1)

        proposal = intake_result.dataset_spec_proposal
        target_column = proposal["target_column"]
        group_column = args.group_column or proposal.get("group_column")
        time_column = args.time_column or proposal.get("time_column")
        id_columns = (
            [c.strip() for c in args.id_columns.split(",") if c.strip()]
            if args.id_columns is not None else (proposal.get("id_columns") or [])
        )
        print(f"IntakeAgent proposed target_column={target_column!r}, "
              f"group_column={group_column!r}, time_column={time_column!r}, "
              f"id_columns={id_columns}")
        print(f"  reasoning: {proposal.get('reasoning', '')}")
        report["intake"] = {
            "target_column": target_column, "group_column": group_column,
            "time_column": time_column, "id_columns": id_columns,
            "reasoning": proposal.get("reasoning", ""),
            "transcript": str(intake_transcript_path),
        }
    else:
        target_column = args.target
        group_column = args.group_column
        time_column = args.time_column
        id_columns = [c.strip() for c in (args.id_columns or "").split(",") if c.strip()]
        report["intake"] = {"skipped": True, "reason": "--target was given explicitly"}

    # --- 2. Load the raw dataset, then Feature Engineering (proposes columns
    # to drop + stateless derived features from a vetted catalog — see
    # harness/feature_engineering.py for why these are safe to apply before
    # the split even exists). Skippable via --skip-feature-engineering. ---
    raw_spec = DatasetSpec(
        path=args.data, target_column=target_column, group_column=group_column,
        time_column=time_column, id_columns=id_columns,
    )
    raw_loaded = load_dataset(raw_spec)
    print(f"\nLoaded dataset: {len(raw_loaded.df)} rows, {len(raw_loaded.df.columns)} columns, "
          f"data_hash={raw_loaded.data_hash[:16]}...")

    if args.skip_feature_engineering:
        print("Skipping FeatureEngineeringAgent (--skip-feature-engineering).")
        report["feature_engineering"] = {"skipped": True}
        final_id_columns = id_columns
        engineered_df = raw_loaded.df
    else:
        print("Running FeatureEngineeringAgent...")
        fe_result = run_feature_engineering_step(
            raw_loaded.df, target_column, client,
            group_column=group_column, time_column=time_column,
            model=default_model, trace_fn=lambda record: trace(**record),
        )
        fe_transcript_path = write_transcript("feature_engineering", fe_result.messages)
        (run_dir / "feature_engineering_report.json").write_text(json.dumps({
            "ok": fe_result.ok, "drop_columns": fe_result.drop_columns,
            "new_columns": fe_result.new_columns, "applied_ops": fe_result.applied_ops,
            "explanation": fe_result.explanation, "errors": fe_result.errors,
            "transcript": str(fe_transcript_path),
        }, indent=2, default=str))

        if not fe_result.ok:
            print("FAILED: FeatureEngineeringAgent's proposal did not validate:")
            for e in fe_result.errors:
                print(f"  - {e}")
            report["status"] = "failed_feature_engineering"
            report["errors"] = fe_result.errors
            write_report()
            sys.exit(1)

        print(f"  drop_columns: {fe_result.drop_columns}")
        print(f"  new_columns: {fe_result.new_columns}")
        print(f"  reasoning: {fe_result.explanation}")
        report["feature_engineering"] = {
            "drop_columns": fe_result.drop_columns, "new_columns": fe_result.new_columns,
            "applied_ops": fe_result.applied_ops, "explanation": fe_result.explanation,
            "transcript": str(fe_transcript_path),
        }
        final_id_columns = sorted(set(id_columns) | set(fe_result.drop_columns))
        engineered_df = fe_result.df

    # --- 3. Build the final DatasetSpec/LoadedDataset over the (possibly
    # augmented) dataframe, then run the Profiler agent (drives split
    # strategy unless overridden) — its facts reflect the final column set. ---
    spec = DatasetSpec(
        path=args.data, target_column=target_column, group_column=group_column,
        time_column=time_column, id_columns=final_id_columns,
    )
    write_dataset_spec(spec, run_dir / "dataset_spec.json")
    # data_hash still traces back to the original raw file — the feature
    # engineering step's applied_ops (logged above) are what make the
    # augmented columns reproducible from that same raw data, so a second
    # hash isn't needed.
    loaded = LoadedDataset(df=engineered_df, spec=spec, data_hash=raw_loaded.data_hash)
    if len(loaded.df.columns) != len(raw_loaded.df.columns):
        print(f"Dataset after feature engineering: {len(loaded.df.columns)} columns "
              f"(was {len(raw_loaded.df.columns)})")

    print("Running ProfilerAgent...")
    profiler_result = run_profiler_step(
        loaded.df, target_column, client, model=default_model,
        trace_fn=lambda record: trace(**record),
    )
    profiler_transcript_path = write_transcript("profiler", profiler_result.messages)
    (run_dir / "profiler_report.json").write_text(json.dumps({
        "deterministic_report": profiler_result.deterministic_report,
        "llm_narrative": profiler_result.llm_narrative,
        "llm_raw_text": profiler_result.llm_raw_text,
        "transcript": str(profiler_transcript_path),
    }, indent=2, default=str))

    if not profiler_result.ok:
        print("FAILED: ProfilerAgent never called get_dataset_profile.")
        report["status"] = "failed_profiler"
        write_report()
        sys.exit(1)

    strategy = args.strategy or profiler_result.deterministic_report["recommended_split_strategy"]
    print(f"Split strategy: {strategy} "
          f"({'explicit override' if args.strategy else 'profiler recommendation'})")

    # The profiler's recommendation is derived from its own heuristic column
    # detection, independent of what intake actually declared as
    # group_column/time_column — reconcile them before make_split() would
    # otherwise raise a bare ValueError deep in the call stack.
    group_column, time_column, split_column_notes = resolve_split_columns(
        strategy, group_column, time_column, profiler_result.deterministic_report,
    )
    for note in split_column_notes:
        print(f"NOTE: {note}")

    report["profiler"] = {
        "recommended_split_strategy": profiler_result.deterministic_report["recommended_split_strategy"],
        "is_imbalanced": profiler_result.deterministic_report["is_imbalanced"],
        "leakage_risk_flags": profiler_result.deterministic_report["leakage_risk_flags"],
        "narrative": profiler_result.llm_narrative,
        "strategy_used": strategy,
        "group_column_used": group_column,
        "time_column_used": time_column,
        "split_column_notes": split_column_notes,
        "transcript": str(profiler_transcript_path),
    }

    # --- 3. Split + split-level leakage checks ---
    manifest = make_split(
        df=loaded.df, target_column=target_column, data_hash=loaded.data_hash,
        strategy=strategy, seed=args.seed, group_column=group_column, time_column=time_column,
    )
    manifest.write(run_dir / "split_manifest.json")
    print(f"Split ({strategy}): train={len(manifest.train_idx)} "
          f"val={len(manifest.val_idx)} test={len(manifest.test_idx)}")

    leakage_checks = run_all_split_leakage_checks(
        df=loaded.df, group_column=group_column, time_column=time_column,
        train_idx=manifest.train_idx, val_idx=manifest.val_idx, test_idx=manifest.test_idx,
        strategy=strategy,
    )
    (run_dir / "leakage_checks.json").write_text(
        json.dumps([c.to_dict() for c in leakage_checks], indent=2)
    )
    for check in leakage_checks:
        print(f"  leakage check [{'PASS' if check.passed else 'FAIL'}] {check.check_name}: {check.detail}")
    if not all(c.passed for c in leakage_checks):
        print("\nABORTING: split-level leakage check failed.")
        report["status"] = "failed_split_leakage"
        write_report()
        sys.exit(1)

    # --- 4. Modeling: try up to max_candidates, each independently gated by
    # the harness's two deterministic leakage checks (label-permutation +
    # feature-correlation, scoped to the candidate's own selected columns) ---
    metric_names = args.metrics.split(",")
    primary_metric = metric_names[0]
    candidates_report = []
    passing_candidates = []
    tried_template_ids: list[str] = []

    for i in range(args.max_candidates):
        print(f"\nModelingAgent candidate {i + 1}/{args.max_candidates}...")
        step_result = run_modeling_step(
            full_df=loaded.df, X=loaded.X, y=loaded.y, target_column=target_column,
            group_column=group_column, time_column=time_column,
            train_idx=manifest.train_idx, val_idx=manifest.val_idx,
            client=client, model=default_model, metric_names=metric_names, seed=args.seed,
            already_tried_template_ids=tried_template_ids,
            trace_fn=lambda record: trace(**record),
        )
        if step_result.template_id:
            tried_template_ids.append(step_result.template_id)

        modeling_transcript_path = write_transcript("modeling", step_result.messages)

        candidate_summary = {
            "candidate_id": step_result.candidate_id, "template_id": step_result.template_id,
            "config": step_result.config, "explanation": step_result.explanation,
            "metrics": step_result.metrics,
            "label_permutation_check": step_result.label_permutation_check,
            "feature_correlation_check": step_result.feature_correlation_check,
            "errors": step_result.errors, "passed_gate": step_result.ok,
            "verification": None,  # filled in during step 4.5, if this candidate gets reviewed
            "transcript": str(modeling_transcript_path),
        }
        candidates_report.append(candidate_summary)

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
            (candidate_dir / "evaluation.json").write_text(json.dumps(candidate_summary, indent=2, default=str))

        if not step_result.ok:
            print(f"  candidate failed harness gates: {step_result.errors}")
            continue

        print(f"  {step_result.candidate_id} ({step_result.template_id}): "
              f"{primary_metric}={step_result.metrics[primary_metric]['value']:.4f}  gates PASSED")
        passing_candidates.append(step_result)

    report["candidates"] = candidates_report

    if not passing_candidates:
        print("\nABORTING: no candidate passed the harness's deterministic leakage gates. "
              "Test set was never touched.")
        report["status"] = "no_candidate_passed"
        write_report()
        sys.exit(1)

    # --- 4.5. Select + verify, best-first. The VerificationAgent reviews each
    # gate-passing candidate in validation-metric order and can only veto
    # (flag/reject) — never approve one that already failed a gate, since it
    # never sees those. A rejection falls back to the next-best candidate. ---
    _, _, verification_model = resolve_model_endpoint(
        args.use_gateway, args.verification_model, "gemma4:latest", "rit-gemma4-latest",
        use_local=args.use_local,
    )
    ranked = sorted(passing_candidates, key=lambda r: r.metrics[primary_metric]["value"], reverse=True)
    leaderboard_path = Path("artifacts/reports/leaderboard.jsonl")

    best = None
    selected_verification = None
    for candidate in ranked:
        print(f"\nVerificationAgent reviewing {candidate.candidate_id} "
              f"(model={verification_model})...")
        template = get_template(candidate.template_id)
        bundle = build_review_bundle(
            candidate_id=candidate.candidate_id, template_id=candidate.template_id,
            template_description=template.description, template_when_to_use=template.when_to_use,
            config=candidate.config, explanation=candidate.explanation,
            metrics=candidate.metrics, label_permutation_check=candidate.label_permutation_check,
            feature_correlation_check=candidate.feature_correlation_check,
            profiler_report=profiler_result.deterministic_report,
        )
        v_result = run_verification_step(
            bundle, client, model=verification_model, trace_fn=lambda record: trace(**record),
        )
        print(f"  verdict: {v_result.verdict}"
              + (f"  concerns: {v_result.concerns}" if v_result.concerns else ""))
        verification_transcript_path = write_transcript("verification", v_result.messages)

        for entry in candidates_report:
            if entry["candidate_id"] == candidate.candidate_id:
                entry["verification"] = {
                    "verdict": v_result.verdict, "concerns": v_result.concerns,
                    "reasoning": v_result.reasoning,
                    "transcript": str(verification_transcript_path),
                }

        append_leaderboard_entry(leaderboard_path, {
            "run_id": run_id, "candidate": candidate.candidate_id, "template_id": candidate.template_id,
            "source": "orchestrator", "model": default_model, "split": "validation", "strategy": strategy,
            "data_hash": loaded.data_hash, "seed": args.seed, "metrics": candidate.metrics,
            "verification_verdict": v_result.verdict,
        })

        if v_result.verdict == "rejected":
            print("  REJECTED by VerificationAgent — trying the next-best candidate.")
            continue

        best = candidate
        selected_verification = v_result
        break

    if best is None:
        print("\nABORTING: every gate-passing candidate was rejected by the VerificationAgent. "
              "Test set was never touched.")
        report["status"] = "no_candidate_passed_verification"
        write_report()
        sys.exit(1)

    print(f"\nSelected candidate: {best.candidate_id} ({best.template_id}) "
          f"— {primary_metric}={best.metrics[primary_metric]['value']:.4f} on validation "
          f"(verification: {selected_verification.verdict})")
    report["selected_candidate"] = {
        "candidate_id": best.candidate_id, "template_id": best.template_id,
        "validation_metrics": best.metrics,
        "verification_verdict": selected_verification.verdict,
        "verification_concerns": selected_verification.concerns,
    }

    # --- 5. Final, one-time test-set evaluation: refit the selected candidate
    # on train+val combined, then touch the test set exactly once ---
    train_and_val_idx = sorted(manifest.train_idx + manifest.val_idx)
    final_pipeline = clone(best.pipeline)
    final_pipeline.fit(loaded.X.iloc[train_and_val_idx], loaded.y.iloc[train_and_val_idx])
    y_pred = final_pipeline.predict(loaded.X.iloc[manifest.test_idx])
    proba = final_pipeline.predict_proba(loaded.X.iloc[manifest.test_idx])
    test_results = compute_metrics(
        loaded.y.iloc[manifest.test_idx].values, y_pred, proba, metric_names,
        n_bootstrap=200, seed=args.seed,
    )
    test_metrics = {m: test_results[m].to_dict() for m in metric_names}
    print("\nFinal test-set metrics (touched exactly once):")
    for m in metric_names:
        r = test_metrics[m]
        print(f"  {m}: {r['value']:.4f}  (95% CI [{r['ci_low']:.4f}, {r['ci_high']:.4f}])")

    report["final_test_metrics"] = test_metrics
    report["status"] = "success"

    # --- 5.5. Persist the accepted model as a bundle the deep-dive agent
    # (scripts/run_deep_dive_agent.py) can load later, against a specific
    # flagged example — this is the only place the fitted pipeline is ever
    # saved. Deep-dive's occlusion attribution (harness/attribution.py) is
    # binary-classification-only for now, so only persist for a binary
    # target; the background reference is scoped to the negative-class
    # ("healthy") cohort of train+val, matching what "occlude toward the
    # background" is meant to mean. ---
    if loaded.y.iloc[train_and_val_idx].nunique() == 2:
        background = compute_background(
            loaded.X.iloc[train_and_val_idx], list(loaded.X.columns),
            normal_mask=(loaded.y.iloc[train_and_val_idx] == 0),
        )
        model_path = Path("artifacts/models") / f"{run_id}_model.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": final_pipeline, "feature_columns": list(loaded.X.columns), "background": background},
            model_path,
        )
        report["model_artifact"] = str(model_path)
        print(f"Model bundle saved to {model_path} (for scripts/run_deep_dive_agent.py)")
    else:
        print("Skipping model bundle persistence: deep-dive attribution is binary-only "
              "and this target has more than 2 classes.")

    # --- 6. Narrated final summary: plain text only, no tools, no decisions ---
    summary_messages = [
        {"role": "system", "content": (
            "You are the Analyst step of a deterministic ML pipeline. Respond "
            "with 4-6 sentences of plain prose only (no JSON, no markdown "
            "fences) summarizing the given facts for a non-technical "
            "stakeholder. Do not invent any numbers not present in the input."
        )},
        {"role": "user", "content": json.dumps({
            "goal": args.goal or "(none given — target inferred from data alone)",
            "target_column": target_column,
            "n_candidates_tried": len(candidates_report),
            "n_candidates_passed_leakage_gate": len(passing_candidates),
            "selected_candidate": report["selected_candidate"],
            "final_test_metrics": test_metrics,
            "note_for_summary": (
                "If verification_concerns is non-empty, mention it briefly as a "
                "caveat for human review." if report["selected_candidate"]["verification_concerns"]
                else None
            ),
        }, indent=2)},
    ]
    summary_response = client.call(summary_messages, model=default_model, max_tokens=400)
    print("\n--- Final summary ---")
    print(summary_response.text)
    report["final_summary"] = summary_response.text
    report["final_summary_transcript"] = str(write_transcript(
        "analyst_summary",
        summary_messages + [{"role": "assistant", "content": summary_response.text}],
    ))

    write_report()
    print(f"\nOrchestrator report written to {run_dir / 'orchestrator_report.json'}")
    print("SUCCESS.")


if __name__ == "__main__":
    main()
