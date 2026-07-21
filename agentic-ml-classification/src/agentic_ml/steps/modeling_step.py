"""
Phase 3 modeling step, extracted from scripts/run_modeling_agent.py so
scripts/run_orchestrator.py can call it multiple times (to try several
candidates before picking a winner) without duplicating the validation/
sandbox/scoring logic. The harness never trusts the agent's proposal
blindly: every proposed column is re-validated against the profiler's
facts, the template source is static-checked + sandbox-built with the
agent's config, the resulting pipeline is fit/scored on the caller-
supplied train/val split, and TWO independent leakage gates must both
pass before ok=True — that, not just "no errors", is what callers
should treat as "safe to use":

  1. label_permutation_test — catches pipeline-level leakage (e.g. a
     preprocessing step fit on data outside its proper fold).
  2. check_suspicious_feature_correlation, re-run scoped to exactly the
     columns THIS candidate selected — catches raw feature-level
     leakage (a near-copy of the target) among the agent's actual
     choices, which is a stronger check than the profiler's dataset-
     wide pass since the profiler doesn't know which columns any given
     candidate will pick. See priors/general/leakage_rules.md Rule 4:
     these two checks are complementary, not redundant, and both must
     run on every candidate.

A candidate that passes both gates still isn't automatically promoted
— see steps/verification_step.py for the second-opinion LLM audit that
runs on gate-passing candidates before promotion.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd
from sklearn.base import clone

from agentic_ml.agent_runtime import ToolCallingAgent
from agentic_ml.events import emit_event
from agentic_ml.harness.leakage import check_suspicious_feature_correlation, label_permutation_test
from agentic_ml.harness.sandbox import run_candidate_build
from agentic_ml.harness.metrics import compute_metrics, roc_auc_any
from agentic_ml.model_client import ModelClient
from agentic_ml.templates.registry import get_template, validate_config
from agentic_ml.tools.profiler_tool import make_profiler_tool
from agentic_ml.tools.template_tool import make_list_templates_tool

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
    profile_report: dict, target_column: str, group_column: Optional[str],
    time_column: Optional[str], config: dict,
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


@dataclass
class ModelingStepResult:
    ok: bool
    candidate_id: Optional[str]
    template_id: Optional[str]
    config: Optional[dict]
    explanation: Optional[str]
    pipeline: Optional[object]  # unfitted-then-fit-on-train pipeline, only present if ok
    metrics: Optional[dict]
    label_permutation_check: Optional[dict]
    feature_correlation_check: Optional[dict]
    errors: list[str] = field(default_factory=list)
    stopped_reason: str = ""
    turns_used: int = 0
    messages: list[dict] = field(default_factory=list)  # full conversation — see cli_common.make_transcript_writer


def run_modeling_step(
    full_df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    target_column: str,
    group_column: Optional[str],
    time_column: Optional[str],
    train_idx: list[int],
    val_idx: list[int],
    client: ModelClient,
    model: Optional[str] = None,
    max_turns: int = 6,
    metric_names: Optional[list[str]] = None,
    seed: int = 42,
    already_tried_template_ids: Optional[list[str]] = None,
    trace_fn: Optional[Callable[[dict], None]] = None,
    on_event: Optional[Callable[[dict], None]] = None,
) -> ModelingStepResult:
    if metric_names is None:
        metric_names = ["roc_auc", "pr_auc", "f1", "accuracy"]

    emit_event(on_event, "modeling", "agent_started", {"already_tried_template_ids": already_tried_template_ids or []})

    system_prompt = SYSTEM_PROMPT
    if already_tried_template_ids:
        system_prompt += (
            f"\n\nTemplates already tried in this run: {already_tried_template_ids}. "
            "Prefer proposing a different template_id unless you have a strong reason not to."
        )

    tools = [make_profiler_tool(full_df, target_column), make_list_templates_tool()]
    agent = ToolCallingAgent(
        model_client=client, tools=tools, system_prompt=system_prompt,
        model=model, max_turns=max_turns,
    )
    result = agent.run("Propose one modeling candidate for this dataset.", trace_fn=trace_fn)

    profile_report = None
    for entry in result.tool_call_log:
        if entry["tool"] == "get_dataset_profile":
            profile_report = entry["result"]
        emit_event(on_event, "modeling", "tool_called", {"tool": entry["tool"], "result": entry["result"]})

    def fail(errors: list[str], **extra) -> ModelingStepResult:
        emit_event(on_event, "modeling", "candidate_rejected", {
            "candidate_id": extra.get("candidate_id"), "template_id": extra.get("template_id"),
            "errors": errors,
        })
        return ModelingStepResult(
            ok=False,
            candidate_id=extra.get("candidate_id"), template_id=extra.get("template_id"),
            config=extra.get("config"), explanation=extra.get("explanation"),
            pipeline=None, metrics=None, label_permutation_check=None,
            feature_correlation_check=None, errors=errors,
            stopped_reason=result.stopped_reason, turns_used=result.turns_used,
            messages=result.messages,
        )

    if result.final_text is None:
        return fail(["agent never produced a final candidate proposal"])

    try:
        candidate = json.loads(result.final_text)
    except json.JSONDecodeError:
        return fail([f"agent's final response did not parse as JSON: {result.final_text[:500]}"])

    shape_errors = _validate_candidate_spec_shape(candidate)
    if shape_errors:
        return fail(shape_errors)

    template_id = candidate["template_id"]
    config = dict(candidate["config"])
    # seed is harness-controlled, never agent-controlled — overwrite whatever
    # the agent proposed so evaluation stays reproducible under our seed.
    config["seed"] = seed
    explanation = candidate.get("explanation", "")
    candidate_id = candidate["candidate_id"]

    try:
        template_errors = validate_config(template_id, config)
    except KeyError as e:
        return fail([str(e)], candidate_id=candidate_id, template_id=template_id,
                    config=config, explanation=explanation)

    column_errors = (
        _validate_candidate_columns(profile_report, target_column, group_column, time_column, config)
        if profile_report else ["profiler was never called — cannot validate proposed columns"]
    )
    all_errors = template_errors + column_errors
    if all_errors:
        return fail(all_errors, candidate_id=candidate_id, template_id=template_id,
                    config=config, explanation=explanation)

    template = get_template(template_id)
    pipeline, build_error = run_candidate_build(template.read_source(), config, timeout_seconds=60)
    if build_error:
        return fail([f"sandbox build error: {build_error}"], candidate_id=candidate_id,
                    template_id=template_id, config=config, explanation=explanation)

    pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
    y_pred = pipeline.predict(X.iloc[val_idx])
    proba = pipeline.predict_proba(X.iloc[val_idx])

    results = compute_metrics(
        y.iloc[val_idx].values, y_pred, proba, metric_names, n_bootstrap=200, seed=seed,
    )
    metrics_dict = {m: results[m].to_dict() for m in metric_names}

    def fit_and_score(X_tr, y_tr, X_va, y_va) -> float:
        candidate_pipeline = clone(pipeline)
        candidate_pipeline.fit(X_tr, y_tr)
        p = candidate_pipeline.predict_proba(X_va)
        return roc_auc_any(y_va, p)

    permutation_check = label_permutation_test(
        fit_and_score, X.iloc[train_idx], y.iloc[train_idx], X.iloc[val_idx], y.iloc[val_idx],
        metric_name="roc_auc", seed=seed,
    )
    emit_event(on_event, "modeling", "leakage_gate_result",
                {"candidate_id": candidate_id, **permutation_check.to_dict()})

    selected_cols = [c for c in config.get("numeric_cols", []) + config.get("categorical_cols", []) if c in X.columns]
    correlation_check = check_suspicious_feature_correlation(
        X[selected_cols].iloc[train_idx], y.iloc[train_idx],
    )
    emit_event(on_event, "modeling", "leakage_gate_result",
                {"candidate_id": candidate_id, **correlation_check.to_dict()})

    passed_both_gates = permutation_check.passed and correlation_check.passed
    gate_errors = []
    if not permutation_check.passed:
        gate_errors.append("failed label_permutation_test leakage gate")
    if not correlation_check.passed:
        gate_errors.append("failed feature-correlation leakage gate")

    emit_event(on_event, "modeling", "candidate_scored", {
        "candidate_id": candidate_id, "template_id": template_id, "metrics": metrics_dict,
        "passed_gate": passed_both_gates,
    })

    return ModelingStepResult(
        ok=passed_both_gates,
        candidate_id=candidate_id, template_id=template_id, config=config, explanation=explanation,
        pipeline=pipeline, metrics=metrics_dict,
        label_permutation_check=permutation_check.to_dict(),
        feature_correlation_check=correlation_check.to_dict(),
        errors=gate_errors,
        stopped_reason=result.stopped_reason, turns_used=result.turns_used,
        messages=result.messages,
    )
