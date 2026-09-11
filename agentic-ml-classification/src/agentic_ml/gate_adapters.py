"""agent-sandbox GateStep wrappers for agentic_ml's static orchestrator
(agentic-ml-classification/scripts/run_orchestrator.py), ported stage by
stage per docs/phase3-gates-handoff.md. Each function below wraps one or
more real agentic_ml.steps.*_step / harness.* calls, unmodified, and is
resolvable as a GateStep.gate path: "agentic_ml_static:run_intake" etc.

Data flow: every gate reads a JSON "manifest" (paths + small JSON-safe
fields) from the prior step's output and writes its own manifest for the
next step to read — never a bare DataFrame/fitted pipeline in `outputs`,
since GateStepResult.output is a plain string. The one place a fitted
sklearn Pipeline needs to cross a stage boundary (modeling -> finalize) it
goes through joblib, referenced by path in the manifest — same pattern
agentic_ml's own artifacts/models/*.joblib already uses.

`run_dir` is created once, by run_intake, via agentic_ml's own
cli_common.make_run_dir() — every later stage reads it from the manifest
chain rather than recomputing it, so agentic_ml keeps owning its own
filesystem conventions (CLAUDE.md invariant #9). AGENTIC_ML_DATA_ROOT must
be set (see repo-root .env) so that run_dir lands under
agentic-ml-classification/ instead of agent-sandbox/runs/, which
RunManager.list_runs() scans for agent runs.

Every gate below builds one `on_event` (via `make_event_emitter`/
`make_event_logger`, same as run_orchestrator.py's own CLI path) and passes
it into every `*_step`/`run_split_step` call it makes, appending to that
run's single `events.jsonl` — this is what makes `prompt_loaded`,
`tool_called`, `leakage_gate_result`, `candidate_scored`,
`candidate_rejected`, and `verification_verdict` events show up there at
all. Before 2026-09-08 only run_intake ever wired this (its one
`network_fetch` event, via resolve_dataset_path's own callback, not a step
function's on_event) — every other gate silently dropped this data, making
a `no_candidate` result from agent-sandbox's pipeline UI undiagnosable
without reproducing the run by hand through this module's own CLI sibling.

v1 scope: no --target/--skip-feature-engineering equivalents (intake and
feature_engineering always run), no natural-language --goal (the seed
task is just a bare CSV path, intake infers everything from schema alone).

Requires running under a Python with `sandbox_core`, `agentic_ml`, and
agentic_ml's own deps (pandas, scikit-learn, joblib, lightgbm, xgboost)
importable together — e.g. the repo-root .venv, which has both
editable-installed. See docs/phase3-testing-guide.md.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone

from agentic_ml.cli_common import make_run_dir, resolve_model_endpoint
from agentic_ml.events import emit_event, make_event_emitter, make_event_logger
from agentic_ml.harness.attribution import compute_background
from agentic_ml.harness.dataset import (
    DatasetSpec,
    LoadedDataset,
    load_dataset,
    read_dataframe,
    resolve_dataset_path,
    write_dataset_spec,
)
from agentic_ml.harness.metrics import compute_metrics
from agentic_ml.harness.splits import SplitManifest
from agentic_ml.harness.verification import build_review_bundle
from agentic_ml.model_client import ModelClient
from agentic_ml.steps.feature_engineering_step import run_feature_engineering_step
from agentic_ml.steps.intake_step import run_intake_step
from agentic_ml.steps.modeling_step import run_modeling_step
from agentic_ml.steps.profiler_step import run_profiler_step
from agentic_ml.steps.split_step import run_split_step
from agentic_ml.steps.verification_step import run_verification_step
from agentic_ml.templates.registry import get_template

# Same defaults run_orchestrator.py itself uses (lines 116, 340, 395),
# except _DEFAULT_MAX_CANDIDATES (see its own comment below) — unlike
# run_orchestrator.py's CLI, this module has no per-call override for it
# (no `--max-candidates` equivalent), so its one hardcoded value has to
# work for every gate-driven run.
_DEFAULT_MODEL_DIRECT = "qwen3-coder:30b"
_DEFAULT_MODEL_GATEWAY = "rit-qwen3-coder-30b"
_DEFAULT_VERIFICATION_MODEL_DIRECT = "gemma4:latest"
_DEFAULT_VERIFICATION_MODEL_GATEWAY = "rit-gemma4-latest"
_DEFAULT_SEED = 42
_DEFAULT_METRIC_NAMES = ["roc_auc", "pr_auc", "f1", "accuracy"]
# 2 (run_orchestrator.py's own --max-candidates default) turned out too
# thin a retry budget here: label_permutation_test is a real statistical
# boundary call with per-candidate variance, not a hard leakage check, so
# 2-for-2 rejections happen by chance more often than "no usable model
# exists" would suggest — confirmed 2026-09-08 reproducing a sandbox
# titanic-static-copy run's no_candidate result via this same CLI: one
# candidate failed label_permutation_test at 0.5903 vs a ~0.58 boundary
# (tolerance 0.08 around chance=0.5), the other passed cleanly. Raised to
# 4 to cut the odds of a bad-luck no_candidate outcome; costs up to 2 more
# LLM round trips (time/$) per modeling_and_verification gate call when
# earlier candidates keep failing.
_DEFAULT_MAX_CANDIDATES = 4


def _make_client(model_default: str, gateway_default: str) -> tuple[ModelClient, str]:
    base_url, api_key, model = resolve_model_endpoint(False, None, model_default, gateway_default, use_local=False)
    return ModelClient(base_url=base_url, api_key=api_key, default_model=model), model


def _write_manifest(path: Path, data: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))
    return str(path)


def _read_manifest(path: str) -> dict:
    return json.loads(Path(path).read_text())


def _rebuild_dataset_spec(d: dict) -> DatasetSpec:
    """DatasetSpec has to_dict()/write_dataset_spec() but no read-side
    counterpart anywhere in agentic_ml — to_dict()'s keys line up 1:1 with
    the constructor's, so this is a safe, direct reconstruction."""
    return DatasetSpec(
        path=d["csv_path"],
        target_column=d["target_column"],
        id_columns=d.get("id_columns") or [],
        group_column=d.get("group_column"),
        time_column=d.get("time_column"),
    )


def _rebuild_split_manifest(d: dict) -> SplitManifest:
    """Same story as DatasetSpec: SplitManifest.write() has no read
    counterpart in agentic_ml either."""
    return SplitManifest(
        strategy=d["strategy"], seed=d["seed"], data_hash=d["data_hash"],
        train_idx=d["train_idx"], val_idx=d["val_idx"], test_idx=d["test_idx"],
        target_distribution=d.get("target_distribution", {}),
        group_overlap_ok=d.get("group_overlap_ok", True),
        time_range=d.get("time_range", {}),
    )


def _load_engineered(manifest: dict) -> LoadedDataset:
    """Reconstructs a LoadedDataset from a manifest's features_path +
    dataset fields. data_hash is only meaningful for split reproducibility
    (already baked into split_manifest.json by the time this is called for
    modeling/finalize), so it's left blank past the split stage."""
    df = pd.read_parquet(manifest["features_path"])
    spec = _rebuild_dataset_spec(manifest)
    return LoadedDataset(df=df, spec=spec, data_hash=manifest.get("data_hash", ""))


# -- 1. Intake ----------------------------------------------------------


def run_intake(outputs: dict[str, str]) -> tuple[str, str]:
    task = outputs["__task__"]
    run_id, run_dir = make_run_dir(None)
    manifest_path = run_dir / "intake_manifest.json"
    on_event = make_event_emitter(run_id, persist_fn=make_event_logger(run_dir))

    # task may be a local CSV path or an http(s) URL (e.g. a raw GitHub
    # dataset link) — resolve_dataset_path downloads the latter once, into
    # this run's own directory, and hands back a plain local path. Doing
    # this here, before any other gate reads csv_path off the manifest,
    # means every downstream stage (feature_engineering, profiler_and_split,
    # ...) re-reads a stable local file exactly as it already does today —
    # no repeat network calls, and no risk of a mutable remote resource
    # changing mid-run and silently breaking split reproducibility.
    csv_path = resolve_dataset_path(
        task, cache_dir=run_dir,
        on_network_fetch=lambda meta: emit_event(on_event, "intake", "network_fetch", meta),
    )

    client, model = _make_client(_DEFAULT_MODEL_DIRECT, _DEFAULT_MODEL_GATEWAY)
    raw_df = read_dataframe(csv_path)
    result = run_intake_step(raw_df, "", client, model=model)

    if not result.ok:
        return "failed", _write_manifest(
            manifest_path, {"run_id": run_id, "run_dir": str(run_dir), "errors": result.validation_errors}
        )

    proposal = result.dataset_spec_proposal
    manifest = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "csv_path": csv_path,
        "target_column": proposal["target_column"],
        "group_column": proposal.get("group_column"),
        "time_column": proposal.get("time_column"),
        "id_columns": proposal.get("id_columns") or [],
    }
    return "ok", _write_manifest(manifest_path, manifest)


# -- 2. Feature engineering ----------------------------------------------


def run_feature_engineering(outputs: dict[str, str]) -> tuple[str, str]:
    intake = _read_manifest(outputs["intake"])
    run_dir = Path(intake["run_dir"])
    manifest_path = run_dir / "feature_engineering_manifest.json"
    on_event = make_event_emitter(intake["run_id"], persist_fn=make_event_logger(run_dir))

    client, model = _make_client(_DEFAULT_MODEL_DIRECT, _DEFAULT_MODEL_GATEWAY)
    raw_df = read_dataframe(intake["csv_path"])
    result = run_feature_engineering_step(
        raw_df, intake["target_column"], client,
        group_column=intake["group_column"], time_column=intake["time_column"], model=model,
        on_event=on_event,
    )

    if not result.ok:
        return "failed", _write_manifest(manifest_path, {**intake, "errors": result.errors})

    features_path = run_dir / "features.parquet"
    result.df.to_parquet(features_path)

    manifest = {
        "run_id": intake["run_id"],
        "run_dir": str(run_dir),
        "csv_path": intake["csv_path"],
        "target_column": intake["target_column"],
        "group_column": intake["group_column"],
        "time_column": intake["time_column"],
        "id_columns": sorted(set(intake["id_columns"]) | set(result.drop_columns)),
        "features_path": str(features_path),
    }
    return "ok", _write_manifest(manifest_path, manifest)


# -- 3. Profiler + split + split-leakage checks --------------------------


def run_profiler_and_split(outputs: dict[str, str]) -> tuple[str, str]:
    fe = _read_manifest(outputs["feature_engineering"])
    run_dir = Path(fe["run_dir"])
    manifest_path = run_dir / "profiler_and_split_manifest.json"
    on_event = make_event_emitter(fe["run_id"], persist_fn=make_event_logger(run_dir))

    engineered_df = pd.read_parquet(fe["features_path"])
    spec = _rebuild_dataset_spec(fe)
    write_dataset_spec(spec, run_dir / "dataset_spec.json")
    raw_loaded = load_dataset(DatasetSpec(path=fe["csv_path"], target_column=fe["target_column"]))
    loaded = LoadedDataset(df=engineered_df, spec=spec, data_hash=raw_loaded.data_hash)

    client, model = _make_client(_DEFAULT_MODEL_DIRECT, _DEFAULT_MODEL_GATEWAY)
    profiler_result = run_profiler_step(
        loaded.df, fe["target_column"], client,
        group_column=fe["group_column"], time_column=fe["time_column"], model=model,
        on_event=on_event,
    )
    if not profiler_result.ok:
        return "failed", _write_manifest(
            manifest_path, {**fe, "errors": ["ProfilerAgent never called get_dataset_profile"]}
        )

    # run_split_step (steps/split_step.py) bundles resolve_split_columns +
    # make_split + run_all_split_leakage_checks + per-check event emission
    # in one call — used here instead of duplicating that sequence by hand
    # (which is what this function did before 2026-09-08) since
    # run_all_split_leakage_checks itself has no on_event parameter of its
    # own to wire; run_split_step is the only way to get leakage_gate_result
    # events out of the split stage at all.
    split_result = run_split_step(
        df=loaded.df, target_column=fe["target_column"], data_hash=loaded.data_hash,
        profiler_report=profiler_result.deterministic_report,
        group_column=fe["group_column"], time_column=fe["time_column"],
        seed=_DEFAULT_SEED, on_event=on_event,
    )
    split_result.manifest.write(run_dir / "split_manifest.json")

    if not split_result.ok:
        # Same shape as before the run_split_step switch (a list of failed-
        # check dicts, not run_split_step's own formatted `errors` strings)
        # so profiler_and_split_manifest.json's "errors" field doesn't
        # change shape for anything already reading it.
        return "failed", _write_manifest(
            manifest_path, {**fe, "errors": [c for c in split_result.leakage_checks if not c["passed"]]}
        )

    manifest = {
        "run_id": fe["run_id"],
        "run_dir": str(run_dir),
        "csv_path": fe["csv_path"],
        "target_column": fe["target_column"],
        "id_columns": fe["id_columns"],
        "group_column": split_result.group_column,
        "time_column": split_result.time_column,
        "features_path": fe["features_path"],
        "data_hash": loaded.data_hash,
        "split_manifest_path": str(run_dir / "split_manifest.json"),
        "strategy": split_result.strategy_used,
        "profiler_report": profiler_result.deterministic_report,
    }
    return "ok", _write_manifest(manifest_path, manifest)


# -- 4. Modeling (up to N candidates) + select/verify best-first ---------


def run_modeling_and_verification(outputs: dict[str, str]) -> tuple[str, str]:
    prof = _read_manifest(outputs["profiler_and_split"])
    run_dir = Path(prof["run_dir"])
    manifest_path = run_dir / "modeling_manifest.json"
    on_event = make_event_emitter(prof["run_id"], persist_fn=make_event_logger(run_dir))

    loaded = _load_engineered(prof)
    split_manifest = _rebuild_split_manifest(json.loads(Path(prof["split_manifest_path"]).read_text()))

    metric_names = _DEFAULT_METRIC_NAMES
    primary_metric = metric_names[0]
    client, model = _make_client(_DEFAULT_MODEL_DIRECT, _DEFAULT_MODEL_GATEWAY)

    passing_candidates = []
    tried_template_ids: list[str] = []
    for _ in range(_DEFAULT_MAX_CANDIDATES):
        step_result = run_modeling_step(
            full_df=loaded.df, X=loaded.X, y=loaded.y, target_column=prof["target_column"],
            group_column=prof["group_column"], time_column=prof["time_column"],
            train_idx=split_manifest.train_idx, val_idx=split_manifest.val_idx,
            client=client, model=model, metric_names=metric_names, seed=_DEFAULT_SEED,
            already_tried_template_ids=tried_template_ids, on_event=on_event,
        )
        if step_result.template_id:
            tried_template_ids.append(step_result.template_id)
        if step_result.ok:
            passing_candidates.append(step_result)

    if not passing_candidates:
        return "no_candidate", _write_manifest(
            manifest_path, {**prof, "errors": ["no candidate passed the harness's leakage gates"]}
        )

    verification_client, verification_model = _make_client(
        _DEFAULT_VERIFICATION_MODEL_DIRECT, _DEFAULT_VERIFICATION_MODEL_GATEWAY
    )
    ranked = sorted(passing_candidates, key=lambda r: r.metrics[primary_metric]["value"], reverse=True)

    best = None
    best_verification = None
    for candidate in ranked:
        template = get_template(candidate.template_id)
        bundle = build_review_bundle(
            candidate_id=candidate.candidate_id, template_id=candidate.template_id,
            template_description=template.description, template_when_to_use=template.when_to_use,
            config=candidate.config, explanation=candidate.explanation, metrics=candidate.metrics,
            label_permutation_check=candidate.label_permutation_check,
            feature_correlation_check=candidate.feature_correlation_check,
            profiler_report=prof["profiler_report"],
        )
        v_result = run_verification_step(bundle, verification_client, model=verification_model, on_event=on_event)
        if v_result.verdict == "rejected":
            continue
        best = candidate
        best_verification = v_result
        break

    if best is None:
        return "no_candidate", _write_manifest(
            manifest_path, {**prof, "errors": ["every gate-passing candidate was rejected by verification"]}
        )

    candidate_path = run_dir / "candidate.joblib"
    joblib.dump(best.pipeline, candidate_path)

    manifest = {
        "run_id": prof["run_id"],
        "run_dir": str(run_dir),
        "csv_path": prof["csv_path"],
        "target_column": prof["target_column"],
        "id_columns": prof["id_columns"],
        "group_column": prof["group_column"],
        "time_column": prof["time_column"],
        "features_path": prof["features_path"],
        "data_hash": prof["data_hash"],
        "split_manifest_path": prof["split_manifest_path"],
        "candidate_path": str(candidate_path),
        "candidate_id": best.candidate_id,
        "template_id": best.template_id,
        "validation_metrics": best.metrics,
        "verification_verdict": best_verification.verdict,
        "verification_concerns": best_verification.concerns,
    }
    return "selected", _write_manifest(manifest_path, manifest)


# -- 5. Finalize: refit on train+val, one-shot test eval, persist --------


def run_finalize(outputs: dict[str, str]) -> tuple[str, str]:
    modeling = _read_manifest(outputs["modeling_and_verification"])
    run_dir = Path(modeling["run_dir"])
    manifest_path = run_dir / "finalize_manifest.json"
    on_event = make_event_emitter(modeling["run_id"], persist_fn=make_event_logger(run_dir))
    # No *_step function here to wire on_event into (this stage is plain
    # sklearn/joblib, no LLM call) — a couple of manual milestone events
    # instead, so the timeline doesn't have a gap here.
    emit_event(on_event, "finalize", "finalize_started", {"candidate_id": modeling["candidate_id"]})

    loaded = _load_engineered(modeling)
    split_manifest = _rebuild_split_manifest(json.loads(Path(modeling["split_manifest_path"]).read_text()))

    candidate_pipeline = joblib.load(modeling["candidate_path"])
    train_and_val_idx = sorted(split_manifest.train_idx + split_manifest.val_idx)
    final_pipeline = clone(candidate_pipeline)
    final_pipeline.fit(loaded.X.iloc[train_and_val_idx], loaded.y.iloc[train_and_val_idx])

    y_pred = final_pipeline.predict(loaded.X.iloc[split_manifest.test_idx])
    proba = final_pipeline.predict_proba(loaded.X.iloc[split_manifest.test_idx])
    test_results = compute_metrics(
        loaded.y.iloc[split_manifest.test_idx].values, y_pred, proba, _DEFAULT_METRIC_NAMES,
        n_bootstrap=200, seed=_DEFAULT_SEED,
    )
    test_metrics = {m: test_results[m].to_dict() for m in _DEFAULT_METRIC_NAMES}

    model_path = None
    if loaded.y.iloc[train_and_val_idx].nunique() == 2:
        background = compute_background(
            loaded.X.iloc[train_and_val_idx], list(loaded.X.columns),
            normal_mask=(loaded.y.iloc[train_and_val_idx] == 0),
        )
        model_path = run_dir / "final_model.joblib"
        joblib.dump(
            {"model": final_pipeline, "feature_columns": list(loaded.X.columns), "background": background},
            model_path,
        )

    manifest = {
        "run_id": modeling["run_id"],
        "run_dir": str(run_dir),
        "candidate_id": modeling["candidate_id"],
        "template_id": modeling["template_id"],
        "validation_metrics": modeling["validation_metrics"],
        "verification_verdict": modeling["verification_verdict"],
        "verification_concerns": modeling["verification_concerns"],
        "test_metrics": test_metrics,
        "model_path": str(model_path) if model_path else None,
    }
    emit_event(on_event, "finalize", "finalize_completed", {"test_metrics": test_metrics})
    return "done", _write_manifest(manifest_path, manifest)


# -- 6. Summarize: plain-text narration, no tools, no decisions ----------


def run_summarize(outputs: dict[str, str]) -> tuple[str, str]:
    final = _read_manifest(outputs["finalize"])
    run_dir = Path(final["run_dir"])
    on_event = make_event_emitter(final["run_id"], persist_fn=make_event_logger(run_dir))
    client, model = _make_client(_DEFAULT_MODEL_DIRECT, _DEFAULT_MODEL_GATEWAY)
    # Bare client.call, no *_step wrapper of its own — manual milestone
    # events, same reasoning as run_finalize above.
    emit_event(on_event, "summarize", "summarize_started", {"candidate_id": final["candidate_id"]})

    messages = [
        {"role": "system", "content": (
            "You are the Analyst step of a deterministic ML pipeline. Respond "
            "with 4-6 sentences of plain prose only (no JSON, no markdown "
            "fences) summarizing the given facts for a non-technical "
            "stakeholder. Do not invent any numbers not present in the input."
        )},
        {"role": "user", "content": json.dumps({
            "candidate_id": final["candidate_id"],
            "template_id": final["template_id"],
            "validation_metrics": final["validation_metrics"],
            "verification_verdict": final["verification_verdict"],
            "test_metrics": final["test_metrics"],
            "note_for_summary": (
                "If verification_concerns is non-empty, mention it briefly as a caveat for human review."
                if final["verification_concerns"] else None
            ),
        }, indent=2)},
    ]
    response = client.call(messages, model=model, max_tokens=400)

    summary_path = run_dir / "summary.txt"
    summary_path.write_text(response.text)
    emit_event(on_event, "summarize", "summarize_completed", {"summary_path": str(summary_path)})
    return "done", str(summary_path)
