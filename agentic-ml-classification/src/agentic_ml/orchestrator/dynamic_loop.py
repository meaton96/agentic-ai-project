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
from pathlib import Path
from typing import Callable, Optional

import joblib
import pandas as pd

from agentic_ml.harness.dataset import DatasetSpec, LoadedDataset, load_dataset
from agentic_ml.harness.leaderboard import append_leaderboard_entry
from agentic_ml.harness.timeseries_features import extract_single_flight_raw
from agentic_ml.harness.verification import build_review_bundle
from agentic_ml.model_client import ModelClient
from agentic_ml.orchestrator.agent_registry import AGENTS, get_agent, list_agent_summaries
from agentic_ml.orchestrator.run_state import CandidateSummary, DynamicRunContext, RunStateSummary
from agentic_ml.steps.deep_dive_step import run_deep_dive_step
from agentic_ml.steps.feature_engineering_step import run_feature_engineering_step
from agentic_ml.steps.finalize_step import run_finalize_step
from agentic_ml.steps.intake_step import run_intake_step
from agentic_ml.steps.modeling_step import run_modeling_step
from agentic_ml.steps.planner_step import run_planner_step
from agentic_ml.steps.profiler_step import run_profiler_step
from agentic_ml.steps.split_step import run_split_step
from agentic_ml.steps.verification_step import run_verification_step
from agentic_ml.templates.registry import get_template

# Duplicated in scripts/featurize_ngafid_flights.py and
# scripts/run_deep_dive_agent.py — same pre-existing tradeoff as there:
# a plain 22-name column list isn't worth a shared module.
NGAFID_SENSORS = [
    "volt1", "volt2", "amp1", "amp2", "FQtyL", "FQtyR", "E1 FFlow",
    "E1 OilT", "E1 OilP", "E1 RPM", "E1 CHT1", "E1 CHT2", "E1 CHT3",
    "E1 CHT4", "E1 EGT1", "E1 EGT2", "E1 EGT3", "E1 EGT4", "OAT",
    "IAS", "VSpd", "NormAc", "AltMSL",
]


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
        if bool(actual) != required_val:
            errors.append(
                f"precondition failed for '{agent_id}': requires {key}={required_val}, "
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


def execute_agent_step(
    agent_id: str, args: dict, ctx: DynamicRunContext, state: RunStateSummary,
    client: ModelClient, model: Optional[str], verification_model: Optional[str],
    trace_fn: Optional[Callable[[dict], None]], write_transcript: Callable[[str, list[dict]], object],
) -> tuple[bool, list[str]]:
    """Executes one already-validated agent step, mutating ctx/state in
    place. Returns (ok, errors). ok=False means this attempt didn't
    succeed (e.g. a modeling candidate failed its leakage gates) — that
    is a legitimate, expected outcome the run continues from, not a
    reason to abort the whole loop; only validate_plan() rejections
    (checked before this is ever called) and unhandled exceptions are
    the "something is wrong" cases."""
    try:
        if agent_id == "intake":
            result = run_intake_step(ctx.raw_df, ctx.goal, client, model=model, trace_fn=trace_fn)
            write_transcript("intake", result.messages)
            if not result.ok:
                return False, result.validation_errors or ["intake agent failed"]
            proposal = result.dataset_spec_proposal
            ctx.target_column = proposal["target_column"]
            ctx.group_column = proposal.get("group_column")
            ctx.time_column = proposal.get("time_column")
            ctx.id_columns = proposal.get("id_columns") or []
            load_raw_hash(ctx)
            state.target_known = True
            state.target_column = ctx.target_column
            return True, []

        if agent_id == "feature_engineering":
            result = run_feature_engineering_step(
                ctx.engineered_df, ctx.target_column, client,
                group_column=ctx.group_column, time_column=ctx.time_column,
                model=model, trace_fn=trace_fn,
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
                                       model=model, trace_fn=trace_fn)
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
            )
            ctx.manifest = result.manifest
            ctx.strategy_used = result.strategy_used
            ctx.group_column = result.group_column
            ctx.time_column = result.time_column
            state.split_done = True
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
                already_tried_template_ids=ctx.tried_template_ids, trace_fn=trace_fn,
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
            result = run_verification_step(bundle, client, model=verification_model, trace_fn=trace_fn)
            write_transcript("verification", result.messages)
            for c in state.candidates:
                if c.candidate_id == candidate_id:
                    c.verification_verdict = result.verdict
            append_leaderboard_entry(Path("artifacts/reports/leaderboard.jsonl"), {
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
                ctx.metric_names, ctx.seed, ctx.run_id,
            )
            ctx.final_pipeline = result.pipeline
            ctx.final_test_metrics = result.test_metrics
            ctx.model_path = result.model_path
            ctx.model_feature_columns = result.feature_columns
            state.final_test_metrics_present = True
            state.model_path_available = result.model_path is not None
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
                model=model, trace_fn=trace_fn,
            )
            write_transcript("deep_dive", result.messages)
            ctx.deep_dive_results[str(flight_id)] = result
            if str(flight_id) not in state.deep_dive_completed_flight_ids:
                state.deep_dive_completed_flight_ids.append(str(flight_id))
            return True, []

        if agent_id == "summarize":
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
) -> DynamicLoopResult:
    write_transcript = write_transcript or (lambda name, messages: None)
    history: list[dict] = []
    capabilities = ctx.capabilities()

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
                client, model=model, previous_error=previous_error, trace_fn=trace_fn,
            )
            write_transcript("planner", planner_result.messages)
            if not planner_result.ok:
                previous_error = "your response did not parse as the required JSON schema"
                attempts.append({"proposal": planner_result.llm_raw_text, "errors": [previous_error]})
                continue
            normalized_proposal = normalize_proposal(planner_result.proposal)
            plan_errors = validate_plan(normalized_proposal, state, capabilities)
            if not plan_errors:
                proposal = normalized_proposal
                break
            previous_error = "; ".join(plan_errors)
            attempts.append({"proposal": normalized_proposal, "errors": plan_errors})
        else:
            history.append({
                "iteration": iteration, "proposal": None, "attempts": attempts,
                "validation_errors": [previous_error], "executed": False,
            })
            return DynamicLoopResult(status="planner_failed", state=state, ctx=ctx, history=history)

        if proposal["action"] == "finish":
            history.append({
                "iteration": iteration, "proposal": proposal, "attempts": attempts,
                "validation_errors": [], "executed": False, "finished": True,
            })
            return DynamicLoopResult(status="success", state=state, ctx=ctx, history=history)

        agent_id = proposal["agent_id"]
        step_args = proposal.get("args") or {}
        ok, errors = execute_agent_step(
            agent_id, step_args, ctx, state, client, model, verification_model,
            trace_fn, write_transcript,
        )
        state.last_action = f"{agent_id}: {'OK' if ok else 'FAILED - ' + '; '.join(errors)}"
        history.append({
            "iteration": iteration, "proposal": proposal, "attempts": attempts,
            "validation_errors": [], "executed": True, "agent_id": agent_id, "ok": ok, "errors": errors,
        })

    return DynamicLoopResult(status="max_iterations_reached", state=state, ctx=ctx, history=history)
