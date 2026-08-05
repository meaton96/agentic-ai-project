"""
The dynamic orchestrator's engine: validate_plan() (deterministic plan
validation — the control-flow analog of every other agent's content
validation in this pipeline), execute_agent_step() (a dispatch table
mapping a validated agent_id to the same, already-tested step functions
the static orchestrator uses), and run_dynamic_loop() (the outer
propose -> validate -> execute -> update-state loop).

Nothing here duplicates agent logic — every branch of execute_agent_step
calls an existing steps/*_step.py function unchanged. This module is
routing and bookkeeping, not a new agent.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Optional

import joblib
import pandas as pd

from agentic_ml.domain.aviation.ngafid_config import (
    NGAFID_EXTRA_COLUMNS, NGAFID_GROUP_COLUMN, NGAFID_ID_COLUMN, NGAFID_LABEL_COLUMN,
    NGAFID_LABEL_MAP, NGAFID_SENSORS,
)
from agentic_ml.events import emit_event
from agentic_ml.harness.dataset import DatasetSpec, LoadedDataset, hash_dataframe, load_dataset
from agentic_ml.harness.drift import compute_drift_report
from agentic_ml.harness.leaderboard import append_leaderboard_entry
from agentic_ml.harness.metrics import compute_metrics
from agentic_ml.harness.timeseries_features import (
    build_flight_feature_table_streaming, extract_single_flight_raw,
)
from agentic_ml.harness.verification import build_review_bundle
from agentic_ml.mcp_facts.provider import ToolProvider
from agentic_ml.model_client import ModelClient
from agentic_ml.orchestrator.agent_registry import AGENTS, get_agent, list_agent_summaries
from agentic_ml.orchestrator.run_state import CandidateSummary, DynamicRunContext, RunStateSummary
from agentic_ml.paths import datasets_root, leaderboard_path
from agentic_ml.steps.deep_dive_step import run_deep_dive_step
from agentic_ml.steps.feature_engineering_step import run_feature_engineering_step
from agentic_ml.steps.finalize_step import run_finalize_step
from agentic_ml.steps.intake_step import run_intake_step
from agentic_ml.steps.modeling_step import run_modeling_step
from agentic_ml.steps.planner_step import run_planner_step
from agentic_ml.steps.profiler_step import run_profiler_step
from agentic_ml.steps.retrain_decision_step import run_retrain_decision_step
from agentic_ml.steps.split_step import run_split_step
from agentic_ml.steps.verification_step import run_verification_step
from agentic_ml.templates.registry import get_template

# feature_engineering explicitly allows proposing zero changes ("nothing
# worth doing" is a legitimate outcome, not a skipped step — see its
# AgentSpec description). That makes it the one content-proposing agent
# with a safe, always-valid deterministic fallback — after this many
# rejected attempts (real incident: 13 identical rejections, all
# proposing to drop the same harness-protected column), take that
# fallback ourselves rather than keep trusting the model to find it.
MAX_FEATURE_ENGINEERING_ATTEMPTS = 3


def normalize_proposal(proposal):
    """Repairs one specific, harmless schema confusion observed from a
    real model during evaluation: putting the agent id directly in
    'action' (e.g. {"action": "finalize"}) instead of {"action":
    "run_agent", "agent_id": "finalize"}. This does NOT relax what's
    allowed to execute — the normalized proposal still goes through the
    exact same validate_plan() checks afterward (unknown agent ids,
    unmet preconditions, etc. are still rejected exactly as before);
    it only canonicalizes an unambiguous formatting mistake before that
    check runs, since burning all of a run's retries on a pure format
    slip (confirmed happening against a real local model — see
    scripts/evaluate_dynamic_orchestrator.py) wastes the run for no
    safety benefit. Applies whether or not 'agent_id' was also set —
    a real model was observed setting both action='finalize' AND
    agent_id='finalize' redundantly, not just omitting agent_id."""
    if not isinstance(proposal, dict):
        return proposal
    action = proposal.get("action")
    if action in AGENTS and action not in ("run_agent", "finish"):
        return {**proposal, "action": "run_agent", "agent_id": action}
    return proposal


def validate_plan(proposal, state: RunStateSummary, capabilities: set) -> list[str]:
    """Deterministic re-check of the planner's proposal against the
    REAL registry and REAL run state — never trusts the planner's own
    claim that a precondition holds. This is the control-flow analog of
    e.g. harness/intake.py::validate_dataset_spec_proposal: an agent's
    output is a proposal, not a fact, until this passes."""
    if not isinstance(proposal, dict):
        return ["proposal is not a JSON object"]

    action = proposal.get("action")
    if action not in ("run_agent", "finish"):
        return [f"'action' must be 'run_agent' or 'finish', got {action!r}"]
    if action == "finish":
        return []

    agent_id = proposal.get("agent_id")
    available_ids = {a["agent_id"] for a in list_agent_summaries(capabilities)}
    if agent_id not in available_ids:
        return [f"agent_id {agent_id!r} is not a known/available agent for this run "
                f"(available: {sorted(available_ids)})"]

    spec = get_agent(agent_id)
    state_dict = state.to_planner_dict()
    errors = []
    for key, required_val in spec.required_state.items():
        actual = state_dict.get(key)
        # bool preconditions check truthiness (the original, still-dominant case);
        # anything else (e.g. required_val=None, or a specific string like
        # "infer_only" — see agent_registry.py's retrain_decision/infer_batch
        # entries) checks exact equality instead. isinstance check comes first
        # since bool is a subtype of int and would otherwise compare oddly.
        satisfied = bool(actual) == required_val if isinstance(required_val, bool) else actual == required_val
        if not satisfied:
            errors.append(
                f"precondition failed for '{agent_id}': requires {key}={required_val!r}, "
                f"actual {key}={actual!r}"
            )

    args = proposal.get("args")
    if args is None:
        args = {}
    elif not isinstance(args, dict):
        errors.append("'args' must be an object")
        args = {}
    for arg_name, arg_spec in spec.arg_schema.items():
        if arg_spec.get("required") and not args.get(arg_name):
            errors.append(f"agent '{agent_id}' requires arg '{arg_name}'")

    return errors


def load_raw_hash(ctx: DynamicRunContext) -> None:
    """Computes the content hash + a spec-loaded df now that
    target_column is known (either from intake, or pre-seeded via
    --target). Mirrors run_orchestrator.py's own two-stage load: the
    raw file is read once for intake (no target needed), then re-read
    through DatasetSpec/load_dataset once a target exists, purely to
    get the same content-hashing guarantee every other entry point has."""
    raw_spec = DatasetSpec(
        path=ctx.data_path, target_column=ctx.target_column,
        group_column=ctx.group_column, time_column=ctx.time_column,
        id_columns=ctx.id_columns,
    )
    raw_loaded = load_dataset(raw_spec)
    ctx.raw_df = raw_loaded.df
    ctx.engineered_df = raw_loaded.df
    ctx.raw_data_hash = raw_loaded.data_hash


def _previous_error_for(agent_id: str, state: RunStateSummary) -> Optional[str]:
    """The prior rejection reason to feed back into a fresh retry of the
    SAME content-proposing agent (feature_engineering, modeling) —
    derived from state.last_action, which already records exactly this
    ("<agent_id>: FAILED - <reason>") without needing a new dedicated
    tracking field. Without this, a rejected-but-plausible proposal can
    repeat identically forever at temperature=0.0, since nothing tells
    the agent WHY it was rejected — a real incident (feature_engineering
    rejected 14 times proposing to drop a column intake had misclassified
    as the time_column) is what surfaced this gap. Returns None unless
    the IMMEDIATELY preceding action was this same agent failing — a
    successful run, a different agent, or no history yet all correctly
    get no feedback to react to."""
    prefix = f"{agent_id}: FAILED - "
    if state.last_action and state.last_action.startswith(prefix):
        return state.last_action[len(prefix):]
    return None


def execute_agent_step(
    agent_id: str, args: dict, ctx: DynamicRunContext, state: RunStateSummary,
    client: ModelClient, model: Optional[str], verification_model: Optional[str],
    trace_fn: Optional[Callable[[dict], None]], write_transcript: Callable[[str, list[dict]], object],
    on_event: Optional[Callable[[dict], None]] = None,
    prompt_override_dir: Optional[str] = None,
    tool_provider: Optional[ToolProvider] = None,
) -> tuple[bool, list[str]]:
    """Executes one already-validated agent step, mutating ctx/state in
    place. Returns (ok, errors). ok=False means this attempt didn't
    succeed (e.g. a modeling candidate failed its leakage gates) — that
    is a legitimate, expected outcome the run continues from, not a
    reason to abort the whole loop; only validate_plan() rejections
    (checked before this is ever called) and unhandled exceptions are
    the "something is wrong" cases."""
    try:
        if agent_id == "featurize_timeseries":
            # Rolls up via the streaming/chunked engine, straight from
            # ctx.data_path — deliberately NEVER read_dataframe(), which is
            # exactly the unguarded full-load path this whole mechanism
            # exists to route around for a multi-GB raw file. Config is
            # hardcoded to NGAFID for now (see PROJECT_OVERVIEW.md /
            # README.md for why: a pluggable per-dataset-family config is
            # explicitly future work, not needed to unblock this dataset).
            table = build_flight_feature_table_streaming(
                ctx.data_path, sensor_columns=NGAFID_SENSORS,
                id_column=NGAFID_ID_COLUMN, group_column=NGAFID_GROUP_COLUMN,
                label_column=NGAFID_LABEL_COLUMN, label_map=NGAFID_LABEL_MAP,
                extra_columns=NGAFID_EXTRA_COLUMNS, max_flights=ctx.featurize_max_flights,
            )
            out_path = datasets_root() / "processed" / f"{ctx.run_id}_featurized.csv"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            table.to_csv(out_path, index=False)

            # raw_csv/features_csv populated the same way --raw-csv/
            # --features-csv would be for a pre-existing model — this is
            # what keeps deep_dive's extract_single_flight_raw(ctx.raw_csv, ...)
            # working later in this same run without any change on its end.
            ctx.raw_csv = ctx.data_path
            ctx.features_csv = str(out_path)
            # Equivalent to what load_raw_hash() would produce, but from the
            # table already in memory rather than a redundant disk re-read —
            # this is what lets the --target-given path skip load_raw_hash()
            # entirely once this branch has run (see run_dynamic_orchestrator.py).
            ctx.raw_df = table
            ctx.engineered_df = table
            ctx.raw_data_hash = hash_dataframe(table)

            summary = {
                "n_examples": len(table),
                "n_groups": int(table[NGAFID_GROUP_COLUMN].nunique())
                            if NGAFID_GROUP_COLUMN in table.columns else None,
                "label_balance": table[NGAFID_LABEL_COLUMN].value_counts().to_dict()
                                  if NGAFID_LABEL_COLUMN in table.columns else {},
                "output_path": str(out_path),
            }
            state.featurization_done = True
            state.featurization_summary = summary
            emit_event(on_event, "featurize_timeseries", "featurization_completed", summary)
            return True, []

        if agent_id == "intake":
            result = run_intake_step(ctx.raw_df, ctx.goal, client, model=model, trace_fn=trace_fn,
                                      on_event=on_event, prompt_override_dir=prompt_override_dir,
                                      tool_provider=tool_provider)
            write_transcript("intake", result.messages)
            if not result.ok:
                return False, result.validation_errors or ["intake agent failed"]
            proposal = result.dataset_spec_proposal
            ctx.target_column = proposal["target_column"]
            ctx.group_column = proposal.get("group_column")
            ctx.time_column = proposal.get("time_column")
            ctx.id_columns = proposal.get("id_columns") or []
            # Skip when featurize_timeseries already populated raw_df/
            # engineered_df/raw_data_hash directly from its in-memory
            # rolled-up table — load_raw_hash() re-reads ctx.data_path from
            # disk, which for a long-format run is the ORIGINAL raw file,
            # and would silently clobber the rolled-up table with it.
            if not state.featurization_done:
                load_raw_hash(ctx)
            state.target_known = True
            state.target_column = ctx.target_column
            return True, []

        if agent_id == "feature_engineering":
            ctx.feature_engineering_attempts += 1
            if ctx.feature_engineering_attempts > MAX_FEATURE_ENGINEERING_ATTEMPTS:
                # Deterministic conservative degradation — the same
                # discipline every other agent here already follows for
                # unparseable/repeatedly-failing LLM output (verification ->
                # flagged, retrain_decision -> infer_only, deep_dive ->
                # template): after enough rejected attempts, take the
                # SAME "zero changes" outcome this agent is already allowed
                # to propose itself, rather than trust the model to
                # eventually land on it. previous_error + a bumped retry
                # temperature (feature_engineering_step.py) make that more
                # likely, but aren't a guarantee — this is what actually
                # guarantees a run can't get stuck here, proven necessary
                # by a real incident (13 identical rejections in a row).
                emit_event(on_event, "feature_engineering", "proposal_degraded", {
                    "attempts": ctx.feature_engineering_attempts, "last_error": state.last_action,
                })
                state.feature_engineering_done = True
                return True, []

            result = run_feature_engineering_step(
                ctx.engineered_df, ctx.target_column, client,
                group_column=ctx.group_column, time_column=ctx.time_column,
                model=model, trace_fn=trace_fn, on_event=on_event,
                prompt_override_dir=prompt_override_dir, tool_provider=tool_provider,
                previous_error=_previous_error_for("feature_engineering", state),
            )
            write_transcript("feature_engineering", result.messages)
            if not result.ok:
                return False, result.errors or ["feature engineering agent failed"]
            ctx.engineered_df = result.df
            ctx.final_id_columns = sorted(set(ctx.id_columns) | set(result.drop_columns))
            state.feature_engineering_done = True
            return True, []

        if agent_id == "profiler":
            result = run_profiler_step(ctx.engineered_df, ctx.target_column, client,
                                       group_column=ctx.group_column, time_column=ctx.time_column,
                                       model=model, trace_fn=trace_fn, on_event=on_event,
                                       prompt_override_dir=prompt_override_dir, tool_provider=tool_provider)
            write_transcript("profiler", result.messages)
            if not result.ok:
                return False, ["profiler agent never called get_dataset_profile"]
            ctx.profiler_report = result.deterministic_report
            state.profiler_done = True
            state.recommended_split_strategy = result.deterministic_report["recommended_split_strategy"]
            final_id_columns = ctx.final_id_columns or ctx.id_columns
            spec = DatasetSpec(
                path=ctx.data_path, target_column=ctx.target_column,
                group_column=ctx.group_column, time_column=ctx.time_column,
                id_columns=final_id_columns,
            )
            ctx.loaded = LoadedDataset(df=ctx.engineered_df, spec=spec, data_hash=ctx.raw_data_hash)
            return True, []

        if agent_id == "split_and_check_leakage":
            result = run_split_step(
                ctx.loaded.df, ctx.target_column, ctx.loaded.data_hash, ctx.profiler_report,
                ctx.group_column, ctx.time_column, ctx.seed, strategy_override=ctx.strategy_override,
                on_event=on_event,
            )
            ctx.manifest = result.manifest
            ctx.strategy_used = result.strategy_used
            ctx.group_column = result.group_column
            ctx.time_column = result.time_column
            # A failed split is NOT "done" — same convention as every other
            # step's _done flag. Marking it done unconditionally created a
            # structural dead end: split_and_check_leakage requires
            # split_done=False (can't re-run) while modeling requires
            # split_leakage_passed=True (can't proceed), so after one gate
            # failure NO catalog agent was proposable and the planner
            # burned its retries into planner_failed (a real NGAFID run
            # died exactly this way). Leaving it False keeps the planner
            # free to retry or finish; the leakage gates themselves are
            # unchanged.
            state.split_done = result.ok
            state.split_leakage_passed = result.ok
            if not result.ok:
                return False, result.errors
            return True, []

        if agent_id == "modeling":
            result = run_modeling_step(
                full_df=ctx.loaded.df, X=ctx.loaded.X, y=ctx.loaded.y, target_column=ctx.target_column,
                group_column=ctx.group_column, time_column=ctx.time_column,
                train_idx=ctx.manifest.train_idx, val_idx=ctx.manifest.val_idx,
                client=client, model=model, metric_names=ctx.metric_names, seed=ctx.seed,
                already_tried_template_ids=ctx.tried_template_ids, trace_fn=trace_fn, on_event=on_event,
                prompt_override_dir=prompt_override_dir, tool_provider=tool_provider,
                previous_error=_previous_error_for("modeling", state),
            )
            write_transcript("modeling", result.messages)
            if result.template_id:
                ctx.tried_template_ids.append(result.template_id)
            if result.candidate_id:
                ctx.modeling_results[result.candidate_id] = result
                primary = ctx.metric_names[0]
                metric_value = result.metrics[primary]["value"] if result.metrics else None
                state.candidates.append(CandidateSummary(
                    candidate_id=result.candidate_id, template_id=result.template_id or "?",
                    metric_name=primary, metric_value=metric_value, passed_gate=result.ok,
                ))
            if not result.ok:
                return False, result.errors or ["modeling candidate failed harness gates"]
            return True, []

        if agent_id == "verification":
            candidate_id = args.get("candidate_id") or state.best_unverified_candidate_id()
            if not candidate_id or candidate_id not in ctx.modeling_results:
                return False, [f"no such unverified passing candidate: {candidate_id!r}"]
            candidate = ctx.modeling_results[candidate_id]
            template = get_template(candidate.template_id)
            bundle = build_review_bundle(
                candidate_id=candidate.candidate_id, template_id=candidate.template_id,
                template_description=template.description, template_when_to_use=template.when_to_use,
                config=candidate.config, explanation=candidate.explanation,
                metrics=candidate.metrics, label_permutation_check=candidate.label_permutation_check,
                feature_correlation_check=candidate.feature_correlation_check,
                profiler_report=ctx.profiler_report,
            )
            result = run_verification_step(bundle, client, model=verification_model, trace_fn=trace_fn,
                                            on_event=on_event, prompt_override_dir=prompt_override_dir,
                                            tool_provider=tool_provider)
            write_transcript("verification", result.messages)
            for c in state.candidates:
                if c.candidate_id == candidate_id:
                    c.verification_verdict = result.verdict
            append_leaderboard_entry(leaderboard_path(), {
                "run_id": ctx.run_id, "candidate": candidate_id, "template_id": candidate.template_id,
                "source": "dynamic_orchestrator", "model": model, "split": "validation",
                "strategy": ctx.strategy_used, "data_hash": ctx.loaded.data_hash, "seed": ctx.seed,
                "metrics": candidate.metrics, "verification_verdict": result.verdict,
            })
            return True, [] if result.verdict != "rejected" else [f"candidate {candidate_id} rejected"]

        if agent_id == "finalize":
            candidate_id = state.best_verified_candidate_id()
            if not candidate_id:
                return False, ["no verified (approved/flagged) candidate available to finalize"]
            candidate = ctx.modeling_results[candidate_id]
            result = run_finalize_step(
                ctx.loaded.X, ctx.loaded.y, ctx.manifest, candidate.pipeline,
                ctx.metric_names, ctx.seed, ctx.run_id, on_event=on_event,
            )
            ctx.final_pipeline = result.pipeline
            ctx.final_test_metrics = result.test_metrics
            ctx.model_path = result.model_path
            ctx.model_feature_columns = result.feature_columns
            state.final_test_metrics_present = True
            state.model_path_available = result.model_path is not None

            # Phase 9 bookkeeping — harmless no-op shape for ordinary (non-
            # streaming) runs: accumulated_df just becomes a copy of
            # engineered_df and model_history gets one entry nobody reads.
            # First finalize (cold start, or any plain classification run)
            # establishes accumulated_df; a later retrain's finalize call
            # finds it already pointed at the grown dataset (see the
            # retrain_decision branch below) and just records against it.
            if ctx.accumulated_df is None:
                ctx.accumulated_df = ctx.engineered_df
            ctx.last_train_accumulated_len = len(ctx.accumulated_df)
            ctx.model_version += 1
            ctx.model_history.append({
                "version": ctx.model_version, "model_path": result.model_path,
                "test_metrics": ctx.final_test_metrics,
                "n_training_examples": ctx.last_train_accumulated_len,
            })
            state.model_version = ctx.model_version
            return True, []

        # --- Phase 9: streaming ingestion + drift-triggered retraining ---
        # Both new_batch_pending and drift_checked (required_state for the
        # two agents below) are only ever True inside a run driven by
        # scripts/run_streaming_monitor.py — ordinary dynamic-orchestrator
        # runs never set them, and these agents are also invisible to those
        # runs' planners (requires_capability="streaming", granted only
        # once ctx.accumulated_df exists — see DynamicRunContext.capabilities()).

        if agent_id == "monitor_drift":
            if ctx.accumulated_df is None or ctx.pending_batch_df is None:
                return False, ["monitor_drift requires an accumulated baseline and a pending batch"]
            baseline_df = ctx.accumulated_df.iloc[: ctx.last_train_accumulated_len]
            feature_columns = ctx.model_feature_columns or list(ctx.loaded.X.columns)
            batch_y = (
                ctx.pending_batch_df[ctx.target_column]
                if ctx.target_column in ctx.pending_batch_df.columns else None
            )
            report = compute_drift_report(
                baseline_df, ctx.pending_batch_df, feature_columns,
                pipeline=ctx.final_pipeline, batch_y=batch_y, metric_names=ctx.metric_names,
            )
            ctx.drift_report = report
            state.drift_checked = True
            state.drift_summary = report
            state.last_action = f"monitor_drift: mean_abs_shift={report['mean_abs_shift']}"
            emit_event(on_event, "monitor_drift", "drift_report", report)
            return True, []

        if agent_id == "retrain_decision":
            ctx.n_batches_since_last_train += 1
            n_new_examples = len(ctx.pending_batch_df) if ctx.pending_batch_df is not None else 0
            n_examples_since_last_train = (
                (len(ctx.accumulated_df) - ctx.last_train_accumulated_len) + n_new_examples
            )
            monitoring_context = {
                "drift_summary": state.drift_summary,
                "current_model_test_metrics": ctx.final_test_metrics,
                "model_version": ctx.model_version,
                "n_batches_since_last_retrain": ctx.n_batches_since_last_train,
                "n_examples_accumulated_since_last_retrain": n_examples_since_last_train,
            }
            result = run_retrain_decision_step(monitoring_context, client, model=model, trace_fn=trace_fn,
                                                on_event=on_event, tool_provider=tool_provider)
            write_transcript("retrain_decision", result.messages)
            state.pending_retrain_action = result.action
            state.last_action = f"retrain_decision: {result.action} ({result.reasoning or 'n/a'})"

            # The batch has now been "seen" regardless of which action was
            # chosen — fold it into the accumulated pool so a later retrain
            # (or the next batch's drift comparison) reflects everything
            # observed, not just the newest arrival.
            if ctx.pending_batch_df is not None:
                ctx.accumulated_df = pd.concat(
                    [ctx.accumulated_df, ctx.pending_batch_df], ignore_index=True,
                )
            state.n_batches_processed += 1

            if result.action == "retrain":
                ctx.n_batches_since_last_train = 0
                ctx.engineered_df = ctx.accumulated_df
                # Resets the classification-phase state a fresh cycle needs to
                # redo — profiler/split/modeling/verification/finalize are the
                # SAME catalog entries Phase 8 already has, completely
                # unchanged; this reset is what lets the same planner walk
                # them again on the grown dataset. split_leakage_passed is
                # reset alongside split_done (required for correctness:
                # without it, "modeling" — gated only on
                # split_leakage_passed=True — would stay validly proposable
                # using the OLD split/manifest the instant split_done flips
                # to False, before split_and_check_leakage has re-run on the
                # grown data).
                #
                # Deliberately NOT resetting feature_engineering_done: every
                # incoming batch (ctx.pending_batch_df) is always a RAW slice
                # of the source table — batches are never run through
                # feature_engineering (there's no mechanism to replay
                # arbitrary derived-feature ops onto a streaming arrival).
                # monitor_drift/infer_batch index into pending_batch_df using
                # ctx.model_feature_columns, so the feature/column schema has
                # to stay identical for the WHOLE session. If feature
                # engineering were allowed to run again here, it could
                # legitimately choose a DIFFERENT column set than cold start
                # did, and the very next batch's monitor_drift/infer_batch
                # call would KeyError on columns that only exist in
                # engineered_df, not in the raw batch. Real-model evaluation
                # (scripts/evaluate_streaming_monitor.py) hit exactly this
                # before feature_engineering_done was excluded from this
                # reset — see run_streaming_monitor.py, which also pre-seeds
                # feature_engineering_done=True before cold start so it never
                # runs even once in a streaming session.
                state.profiler_done = False
                state.split_done = False
                state.split_leakage_passed = False
                state.final_test_metrics_present = False
                state.candidates = []
                ctx.modeling_results = {}
                ctx.tried_template_ids = []

            return True, []

        if agent_id == "infer_batch":
            if ctx.pending_batch_df is None or ctx.final_pipeline is None:
                return False, ["infer_batch requires a pending batch and an existing model"]
            feature_columns = ctx.model_feature_columns or list(ctx.loaded.X.columns)
            X_batch = ctx.pending_batch_df[feature_columns]
            y_pred = ctx.final_pipeline.predict(X_batch)
            y_proba = ctx.final_pipeline.predict_proba(X_batch)
            batch_metrics = None
            if ctx.target_column in ctx.pending_batch_df.columns:
                y_batch = ctx.pending_batch_df[ctx.target_column]
                metric_results = compute_metrics(
                    y_batch.values, y_pred, y_proba, ctx.metric_names, n_bootstrap=200, seed=ctx.seed,
                )
                batch_metrics = {m: metric_results[m].to_dict() for m in ctx.metric_names}
            ctx.batch_inference_log.append({
                "batch_index": state.n_batches_processed,
                "n_examples": len(ctx.pending_batch_df),
                "model_version": ctx.model_version,
                "metrics": batch_metrics,
            })
            state.batch_action_completed = True
            state.last_action = f"infer_batch: scored {len(ctx.pending_batch_df)} examples"
            emit_event(on_event, "infer_batch", "batch_inference_completed", {
                "n_examples": len(ctx.pending_batch_df), "metrics": batch_metrics,
            })
            return True, []

        if agent_id == "deep_dive":
            flight_id = args.get("flight_id")
            if not flight_id:
                return False, ["deep_dive requires a 'flight_id' arg"]
            if not (ctx.raw_csv and ctx.features_csv):
                return False, ["deep_dive requires --raw-csv and --features-csv for this run"]
            if not ctx.model_path:
                return False, ["no model bundle available — run finalize first, or pass --existing-model"]
            bundle = joblib.load(ctx.model_path)
            pipeline = bundle["model"]
            feature_columns = bundle["feature_columns"]
            background = bundle["background"]
            features_df = pd.read_csv(ctx.features_csv)
            match = features_df[features_df["id"].astype(str) == str(flight_id)]
            if match.empty:
                return False, [f"flight_id {flight_id!r} not found in {ctx.features_csv}"]
            feature_row = match.iloc[0]
            flight_df = extract_single_flight_raw(ctx.raw_csv, flight_id, NGAFID_SENSORS, id_column="id")
            result = run_deep_dive_step(
                flight_df, feature_row, pipeline, feature_columns, background, client,
                model=model, trace_fn=trace_fn, on_event=on_event,
                prompt_override_dir=prompt_override_dir, tool_provider=tool_provider,
                flight_id=str(flight_id),
            )
            write_transcript("deep_dive", result.messages)
            ctx.deep_dive_results[str(flight_id)] = result
            if str(flight_id) not in state.deep_dive_completed_flight_ids:
                state.deep_dive_completed_flight_ids.append(str(flight_id))
            return True, []

        if agent_id == "summarize":
            emit_event(on_event, "summarize", "agent_started", {})
            summary_messages = [
                {"role": "system", "content": (
                    "You are the Analyst step of a dynamic, agentic ML pipeline. Respond "
                    "with 4-6 sentences of plain prose only (no JSON, no markdown fences) "
                    "summarizing the given facts for a non-technical stakeholder. Do not "
                    "invent any numbers not present in the input."
                )},
                {"role": "user", "content": json.dumps({
                    "goal": ctx.goal, "target_column": ctx.target_column,
                    "n_candidates_tried": len(state.candidates),
                    "final_test_metrics": ctx.final_test_metrics,
                }, indent=2)},
            ]
            response = client.call(summary_messages, model=model, max_tokens=400)
            ctx.summary_text = response.text
            write_transcript("summarize", summary_messages + [{"role": "assistant", "content": response.text}])
            state.summary_present = True
            emit_event(on_event, "summarize", "summary_produced", {"summary": response.text})
            return True, []

        return False, [f"no dispatcher registered for agent_id {agent_id!r}"]
    except Exception as e:
        return False, [f"{type(e).__name__}: {e}"]


@dataclass
class DynamicLoopResult:
    status: str  # "success" | "max_iterations_reached" | "planner_failed"
    state: RunStateSummary
    ctx: DynamicRunContext
    history: list[dict] = field(default_factory=list)


def run_dynamic_loop(
    ctx: DynamicRunContext,
    state: RunStateSummary,
    client: ModelClient,
    model: Optional[str] = None,
    verification_model: Optional[str] = None,
    max_iterations: int = 15,
    max_retries_per_iteration: int = 2,
    trace_fn: Optional[Callable[[dict], None]] = None,
    write_transcript: Optional[Callable[[str, list[dict]], object]] = None,
    on_event: Optional[Callable[[dict], None]] = None,
    prompt_override_dir: Optional[str] = None,
    tool_provider: Optional[ToolProvider] = None,
) -> DynamicLoopResult:
    write_transcript = write_transcript or (lambda name, messages: None)
    history: list[dict] = []
    capabilities = ctx.capabilities()

    emit_event(on_event, "run", "run_started", {"goal": ctx.goal, "max_iterations": max_iterations})

    for iteration in range(max_iterations):
        state.iteration = iteration
        available_agents = list_agent_summaries(capabilities)

        proposal = None
        previous_error = None
        attempts: list[dict] = []  # every rejected proposal this iteration, not just the final one
        for _attempt in range(max_retries_per_iteration + 1):
            state_dict = state.to_planner_dict()
            planner_result = run_planner_step(
                ctx.goal, state_dict, available_agents, iteration, max_iterations,
                client, model=model, previous_error=previous_error, trace_fn=trace_fn, on_event=on_event,
                prompt_override_dir=prompt_override_dir, tool_provider=tool_provider,
            )
            write_transcript("planner", planner_result.messages)
            if not planner_result.ok:
                previous_error = "your response did not parse as the required JSON schema"
                attempts.append({"proposal": planner_result.llm_raw_text, "errors": [previous_error]})
                emit_event(on_event, "planner", "planner_proposal_rejected",
                            {"iteration": iteration, "proposal": planner_result.llm_raw_text,
                             "errors": [previous_error]})
                continue
            normalized_proposal = normalize_proposal(planner_result.proposal)
            plan_errors = validate_plan(normalized_proposal, state, capabilities)
            if not plan_errors:
                proposal = normalized_proposal
                emit_event(on_event, "planner", "planner_proposal_accepted",
                            {"iteration": iteration, "proposal": proposal})
                break
            previous_error = "; ".join(plan_errors)
            attempts.append({"proposal": normalized_proposal, "errors": plan_errors})
            emit_event(on_event, "planner", "planner_proposal_rejected",
                        {"iteration": iteration, "proposal": normalized_proposal, "errors": plan_errors})
        else:
            history.append({
                "iteration": iteration, "proposal": None, "attempts": attempts,
                "validation_errors": [previous_error], "executed": False,
            })
            emit_event(on_event, "run", "run_failed", {"status": "planner_failed", "iteration": iteration})
            return DynamicLoopResult(status="planner_failed", state=state, ctx=ctx, history=history)

        if proposal["action"] == "finish":
            history.append({
                "iteration": iteration, "proposal": proposal, "attempts": attempts,
                "validation_errors": [], "executed": False, "finished": True,
            })
            emit_event(on_event, "run", "run_completed", {"status": "success", "iteration": iteration})
            return DynamicLoopResult(status="success", state=state, ctx=ctx, history=history)

        agent_id = proposal["agent_id"]
        step_args = proposal.get("args") or {}
        ok, errors = execute_agent_step(
            agent_id, step_args, ctx, state, client, model, verification_model,
            trace_fn, write_transcript, on_event=on_event, prompt_override_dir=prompt_override_dir,
            tool_provider=tool_provider,
        )
        state.last_action = f"{agent_id}: {'OK' if ok else 'FAILED - ' + '; '.join(errors)}"
        history.append({
            "iteration": iteration, "proposal": proposal, "attempts": attempts,
            "validation_errors": [], "executed": True, "agent_id": agent_id, "ok": ok, "errors": errors,
        })

    emit_event(on_event, "run", "run_failed", {"status": "max_iterations_reached"})
    return DynamicLoopResult(status="max_iterations_reached", state=state, ctx=ctx, history=history)
