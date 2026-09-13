"""agent-sandbox GateStep functions for agentic_ml's pipeline stages, each
resolvable as a GateStep.gate path ("agentic_ml.gate_adapters:prepare_intake"
etc.). Every function here is purely deterministic: no ModelClient, no
ToolCallingAgent, no LLM call of any kind.

Shape: each stage that needs a judgment call is split into a pair of gates
with an agent step, owned entirely outside agentic_ml, in between:

    prepare_<stage>  ->  (external agent proposes)  ->  <stage>_decide

prepare_<stage> computes the stage's harness facts with the same calls
McpToolProvider makes (mcp_facts/provider.py), persists them via
fact_store.write_fact so mcp_facts/server.py can serve them, writes a
manifest naming the MCP tools the proposer should read, and outputs only the
bare run_id. It never waits for or produces a proposal.

<stage>_decide re-reads that manifest from disk by run_id, and takes the
proposal text (the agent step's
raw output, held to exactly the JSON contract the stage's own LLM output had)
and runs the stage's post-proposal logic — the same steps/*_step.py functions
run_*_step uses (decide_intake_proposal, apply_feature_engineering_proposal,
record_profiler_narrative, evaluate_modeling_candidate,
interpret_verification_verdict), so a proposal is judged identically however
it was produced. How a proposal gets made is none of this module's business;
only what shape it must arrive in.

Pipeline step_ids read from `outputs` are the STEP_* constants below. The
decide gates keep the step_ids the old single-call gates had (intake,
feature_engineering, profiler_and_split, modeling_and_verification), so each
later stage — including run_finalize, unchanged — reads its input under the
same key it always did.

Modeling runs one candidate per modeling_decide call instead of a Python loop.
Every attempt (accepted or not) is appended to the `modeling_attempts` fact,
served by the get_modeling_attempts MCP tool so the next proposer can see what
was tried and why it failed. modeling_decide itself enforces the
_DEFAULT_MAX_CANDIDATES budget — past it, it refuses to build anything — so the
bound doesn't depend on whatever drives the proposals. An accepted candidate's
review bundle is written as the `review_bundle` fact for the verifier.

Behavior change vs. the old run_modeling_and_verification: there is no
pre-built, pre-ranked batch any more. A verification rejection returns
"rejected" and the orchestration goes back to proposing a fresh modeling
candidate (not to the next-best already-built one), and the first accepted
candidate is the one verified rather than the best of N by roc_auc.

Data flow: every gate writes a JSON "manifest" (paths + small JSON-safe
fields) under run_dir. A gate whose output feeds an agent step (prepare_*,
and modeling_decide on "accepted") outputs only the bare run_id: an agent
step substitutes a prior output into its prompt verbatim, can't read this
filesystem, and must never see raw paths — and run_id is the one argument
every run-scoped MCP tool takes. The gate after the agent re-derives the
manifest path from run_id (verification_decide via the fixed-name
pending_verification_manifest.json pointer, since attempt manifests are
numbered). Every other gate outputs its manifest's path, as before. Never a
bare DataFrame/fitted pipeline in `outputs`, since GateStepResult.output is
a plain string. Fitted pipelines cross stage
boundaries via joblib, referenced by path in the manifest — same pattern
agentic_ml's own artifacts/models/*.joblib already uses. prepare_* manifests
are the upstream manifest plus one "prepare" key; decide gates strip it, so
their own manifests have exactly the shape the old single-call gates wrote.

`run_dir` is created once, by prepare_intake, via agentic_ml's own
cli_common.make_run_dir() — every later stage reads it from the manifest
chain rather than recomputing it, so agentic_ml keeps owning its own
filesystem conventions (CLAUDE.md invariant #9). AGENTIC_ML_DATA_ROOT must
be set (see repo-root .env) so that run_dir (and with it runs/<run_id>/facts/)
lands under agentic-ml-classification/ instead of agent-sandbox/runs/, which
RunManager.list_runs() scans for agent runs.

Every gate builds one `on_event` (via make_event_emitter/make_event_logger)
appending to that run's single events.jsonl, so leakage_gate_result,
candidate_scored, candidate_rejected, verification_verdict, ... events are
recorded exactly as the step functions emit them.

v1 scope: no --target/--skip-feature-engineering equivalents (intake and
feature_engineering always run), no natural-language --goal (the seed task
is just a bare CSV path or URL, intake infers everything from schema alone).
"""

import json
import shutil
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone

from agentic_ml.cli_common import make_run_dir
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
from agentic_ml.harness.intake import raw_schema_summary
from agentic_ml.harness.metrics import compute_metrics
from agentic_ml.harness.splits import SplitManifest
from agentic_ml.harness.verification import build_review_bundle
from agentic_ml.mcp_facts.fact_store import read_fact, read_fact_or_default, write_fact
from agentic_ml.mcp_facts.server import FACT_TOOL_NAMES
from agentic_ml.paths import run_dir as resolve_run_dir
from agentic_ml.steps.feature_engineering_step import apply_feature_engineering_proposal
from agentic_ml.steps.intake_step import decide_intake_proposal
from agentic_ml.steps.modeling_step import evaluate_modeling_candidate
from agentic_ml.steps.profiler_step import record_profiler_narrative
from agentic_ml.steps.split_step import run_split_step
from agentic_ml.steps.verification_step import interpret_verification_verdict
from agentic_ml.templates.registry import get_template
from agentic_ml.tools.profiler_tool import build_profile_fact

STEP_PREPARE_INTAKE = "prepare_intake"
STEP_PROPOSE_INTAKE = "propose_intake"
STEP_INTAKE = "intake"
STEP_PREPARE_FEATURE_ENGINEERING = "prepare_feature_engineering"
STEP_PROPOSE_FEATURE_ENGINEERING = "propose_feature_engineering"
STEP_FEATURE_ENGINEERING = "feature_engineering"
STEP_PREPARE_PROFILER_AND_SPLIT = "prepare_profiler_and_split"
STEP_PROPOSE_PROFILER = "propose_profiler"
STEP_PROFILER_AND_SPLIT = "profiler_and_split"
STEP_PREPARE_MODELING = "prepare_modeling"
STEP_PROPOSE_MODELING = "propose_modeling"
STEP_MODELING = "modeling"
STEP_PROPOSE_VERIFICATION = "propose_verification"
STEP_MODELING_AND_VERIFICATION = "modeling_and_verification"
STEP_FINALIZE = "finalize"

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
# 4 to cut the odds of a bad-luck no_candidate outcome. Counts every
# modeling_decide call in a run, accepted or not.
_DEFAULT_MAX_CANDIDATES = 4

_PREPARE_KEY = "prepare"
_ATTEMPT_KEY = "modeling_attempt"
_ATTEMPTS_FACT = "modeling_attempts"


def _write_manifest(path: Path, data: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))
    return str(path)


def _read_manifest(path: str) -> dict:
    return json.loads(Path(path).read_text())


def _without(manifest: dict, key: str) -> dict:
    return {k: v for k, v in manifest.items() if k != key}


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


def _publish_facts(
    run_id: str, on_event, stage: str, facts: dict[str, dict], extra_tools: tuple[str, ...] = (),
) -> dict:
    """Writes each fact for mcp_facts/server.py to serve and returns the
    manifest's "prepare" block: which stage now awaits a proposal, and the
    MCP tools its proposer should read (the facts just written, plus any
    static or pre-existing ones in `extra_tools`)."""
    for name, payload in facts.items():
        write_fact(run_id, name, payload)
    mcp_tools = [FACT_TOOL_NAMES[name] for name in facts] + list(extra_tools)
    emit_event(on_event, stage, "facts_ready", {"mcp_tools": mcp_tools})
    return {"stage": stage, "mcp_tools": mcp_tools}


# -- Modeling attempt bookkeeping (the modeling_attempts fact) ------------


def _record_modeling_attempt(run_id: str, attempts: dict, result, accepted: bool, reason) -> dict:
    index = len(attempts["attempts"])
    attempts["attempts"].append({
        "attempt_index": index, "candidate_id": result.candidate_id,
        "template_id": result.template_id, "accepted": accepted, "reason": reason,
    })
    if result.template_id:
        attempts["tried_template_ids"].append(result.template_id)
    if not accepted:
        attempts["rejections"].append({
            "attempt_index": index, "candidate_id": result.candidate_id,
            "template_id": result.template_id, "stage": "modeling", "reason": reason,
        })
    write_fact(run_id, _ATTEMPTS_FACT, attempts)
    return attempts


def _record_verification_rejection(run_id: str, attempts: dict, candidate: dict, reason: str) -> dict:
    entry = attempts["attempts"][candidate["attempt_index"]]
    entry["accepted"] = False
    entry["reason"] = reason
    attempts["rejections"].append({
        "attempt_index": candidate["attempt_index"], "candidate_id": candidate["candidate_id"],
        "template_id": candidate["template_id"], "stage": "verification", "reason": reason,
    })
    write_fact(run_id, _ATTEMPTS_FACT, attempts)
    return attempts


def _is_pending_acceptance(attempts: dict, candidate: dict | None) -> bool:
    """True iff `candidate` is an attempt modeling_decide accepted and
    verification hasn't since rejected — the only thing verification_decide
    may act on."""
    if not candidate:
        return False
    index = candidate.get("attempt_index")
    if not isinstance(index, int) or not 0 <= index < len(attempts["attempts"]):
        return False
    entry = attempts["attempts"][index]
    return entry["accepted"] and entry["candidate_id"] == candidate.get("candidate_id")


def _budget_exhausted(attempts: dict) -> bool:
    return len(attempts["attempts"]) >= _DEFAULT_MAX_CANDIDATES and not any(
        a["accepted"] for a in attempts["attempts"]
    )


def _no_candidate_errors(attempts: dict) -> list[str]:
    # Same two messages the old batch loop used: whether anything ever got
    # past the leakage gates decides which one applies.
    if any(r["stage"] == "verification" for r in attempts["rejections"]):
        return ["every gate-passing candidate was rejected by verification"]
    return ["no candidate passed the harness's leakage gates"]


# -- 1. Intake ----------------------------------------------------------


def prepare_intake(outputs: dict[str, str]) -> tuple[str, str]:
    task = outputs["__task__"]
    run_id, run_dir = make_run_dir(None)
    manifest_path = run_dir / "prepare_intake_manifest.json"
    on_event = make_event_emitter(run_id, persist_fn=make_event_logger(run_dir))

    # task may be a local CSV path or an http(s) URL (e.g. a raw GitHub
    # dataset link) — resolve_dataset_path downloads the latter once, into
    # this run's own directory, and hands back a plain local path. Doing
    # this here, before any other gate reads csv_path off the manifest,
    # means every downstream stage re-reads a stable local file — no repeat
    # network calls, and no risk of a mutable remote resource changing
    # mid-run and silently breaking split reproducibility.
    csv_path = resolve_dataset_path(
        task, cache_dir=run_dir,
        on_network_fetch=lambda meta: emit_event(on_event, "intake", "network_fetch", meta),
    )
    raw_df = read_dataframe(csv_path)
    prepare = _publish_facts(run_id, on_event, "intake", {"raw_schema": raw_schema_summary(raw_df)})

    _write_manifest(manifest_path, {
        "run_id": run_id, "run_dir": str(run_dir), "csv_path": csv_path, _PREPARE_KEY: prepare,
    })
    return "ready", run_id


def intake_decide(outputs: dict[str, str]) -> tuple[str, str]:
    prep = _read_manifest(resolve_run_dir(outputs[STEP_PREPARE_INTAKE]) / "prepare_intake_manifest.json")
    run_id, run_dir = prep["run_id"], Path(prep["run_dir"])
    manifest_path = run_dir / "intake_manifest.json"
    on_event = make_event_emitter(run_id, persist_fn=make_event_logger(run_dir))

    raw_df = read_dataframe(prep["csv_path"])
    result = decide_intake_proposal(raw_df, outputs[STEP_PROPOSE_INTAKE], on_event=on_event)

    if not result.ok:
        return "failed", _write_manifest(
            manifest_path, {"run_id": run_id, "run_dir": str(run_dir), "errors": result.validation_errors}
        )

    proposal = result.dataset_spec_proposal
    manifest = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "csv_path": prep["csv_path"],
        "target_column": proposal["target_column"],
        "group_column": proposal.get("group_column"),
        "time_column": proposal.get("time_column"),
        "id_columns": proposal.get("id_columns") or [],
    }
    return "ok", _write_manifest(manifest_path, manifest)


# -- 2. Feature engineering ----------------------------------------------


def prepare_feature_engineering(outputs: dict[str, str]) -> tuple[str, str]:
    intake = _read_manifest(outputs[STEP_INTAKE])
    run_dir = Path(intake["run_dir"])
    manifest_path = run_dir / "prepare_feature_engineering_manifest.json"
    on_event = make_event_emitter(intake["run_id"], persist_fn=make_event_logger(run_dir))

    raw_df = read_dataframe(intake["csv_path"])
    profile = build_profile_fact(raw_df, intake["target_column"], intake["group_column"], intake["time_column"])
    prepare = _publish_facts(
        intake["run_id"], on_event, "feature_engineering", {"dataset_profile": profile},
        extra_tools=("list_feature_ops",),
    )
    _write_manifest(manifest_path, {**intake, _PREPARE_KEY: prepare})
    return "ready", intake["run_id"]


def feature_engineering_decide(outputs: dict[str, str]) -> tuple[str, str]:
    intake = _without(_read_manifest(
        resolve_run_dir(outputs[STEP_PREPARE_FEATURE_ENGINEERING]) / "prepare_feature_engineering_manifest.json"
    ), _PREPARE_KEY)
    run_dir = Path(intake["run_dir"])
    manifest_path = run_dir / "feature_engineering_manifest.json"
    on_event = make_event_emitter(intake["run_id"], persist_fn=make_event_logger(run_dir))

    raw_df = read_dataframe(intake["csv_path"])
    # Validated against the exact profile the proposer was served.
    profile_report = read_fact(intake["run_id"], "dataset_profile")
    result = apply_feature_engineering_proposal(
        raw_df, intake["target_column"], outputs[STEP_PROPOSE_FEATURE_ENGINEERING], profile_report,
        group_column=intake["group_column"], time_column=intake["time_column"], on_event=on_event,
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


def prepare_profiler_and_split(outputs: dict[str, str]) -> tuple[str, str]:
    fe = _read_manifest(outputs[STEP_FEATURE_ENGINEERING])
    run_dir = Path(fe["run_dir"])
    manifest_path = run_dir / "prepare_profiler_and_split_manifest.json"
    on_event = make_event_emitter(fe["run_id"], persist_fn=make_event_logger(run_dir))

    # Recomputed on the ENGINEERED frame rather than reusing
    # prepare_feature_engineering's profile: that one describes the raw CSV,
    # which lacks every column feature_engineering_decide just derived, and
    # this profile is what the split and (via profiler_report) the review
    # bundle are built from — same frame the old run_profiler_step profiled.
    engineered_df = pd.read_parquet(fe["features_path"])
    profile = build_profile_fact(engineered_df, fe["target_column"], fe["group_column"], fe["time_column"])
    prepare = _publish_facts(fe["run_id"], on_event, "profiler", {"dataset_profile": profile})
    _write_manifest(manifest_path, {**fe, _PREPARE_KEY: prepare})
    return "ready", fe["run_id"]


def profiler_and_split_decide(outputs: dict[str, str]) -> tuple[str, str]:
    fe = _without(_read_manifest(
        resolve_run_dir(outputs[STEP_PREPARE_PROFILER_AND_SPLIT]) / "prepare_profiler_and_split_manifest.json"
    ), _PREPARE_KEY)
    run_dir = Path(fe["run_dir"])
    manifest_path = run_dir / "profiler_and_split_manifest.json"
    on_event = make_event_emitter(fe["run_id"], persist_fn=make_event_logger(run_dir))

    engineered_df = pd.read_parquet(fe["features_path"])
    spec = _rebuild_dataset_spec(fe)
    write_dataset_spec(spec, run_dir / "dataset_spec.json")
    raw_loaded = load_dataset(DatasetSpec(path=fe["csv_path"], target_column=fe["target_column"]))
    loaded = LoadedDataset(df=engineered_df, spec=spec, data_hash=raw_loaded.data_hash)

    # The narrative has no decision impact (the deterministic report exists
    # before any agent runs) — kept on disk for the record only.
    narrative = outputs[STEP_PROPOSE_PROFILER]
    (run_dir / "profiler_narrative.txt").write_text(narrative)
    profiler_result = record_profiler_narrative(
        read_fact(fe["run_id"], "dataset_profile"), narrative, on_event=on_event,
    )

    # run_split_step bundles resolve_split_columns + make_split +
    # run_all_split_leakage_checks + per-check event emission in one call.
    split_result = run_split_step(
        df=loaded.df, target_column=fe["target_column"], data_hash=loaded.data_hash,
        profiler_report=profiler_result.deterministic_report,
        group_column=fe["group_column"], time_column=fe["time_column"],
        seed=_DEFAULT_SEED, on_event=on_event,
    )
    split_result.manifest.write(run_dir / "split_manifest.json")

    if not split_result.ok:
        # A list of failed-check dicts, not run_split_step's formatted
        # `errors` strings, so profiler_and_split_manifest.json's "errors"
        # field keeps the shape anything already reading it expects.
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


# -- 4. Modeling (one candidate per call) + verification -----------------


def prepare_modeling(outputs: dict[str, str]) -> tuple[str, str]:
    """Idempotent: safe to route back to before every attempt. Never
    touches the modeling_attempts fact."""
    prof = _read_manifest(outputs[STEP_PROFILER_AND_SPLIT])
    run_dir = Path(prof["run_dir"])
    manifest_path = run_dir / "prepare_modeling_manifest.json"
    on_event = make_event_emitter(prof["run_id"], persist_fn=make_event_logger(run_dir))

    # Profiled with the split-resolved group/time columns (not intake's
    # declared ones), since those are what modeling's column validation
    # excludes.
    df = pd.read_parquet(prof["features_path"])
    profile = build_profile_fact(df, prof["target_column"], prof["group_column"], prof["time_column"])
    prepare = _publish_facts(
        prof["run_id"], on_event, "modeling", {"dataset_profile": profile},
        extra_tools=("list_templates", FACT_TOOL_NAMES[_ATTEMPTS_FACT]),
    )
    attempts = read_fact_or_default(prof["run_id"], _ATTEMPTS_FACT)
    prepare["attempts_used"] = len(attempts["attempts"])
    prepare["max_candidates"] = _DEFAULT_MAX_CANDIDATES
    _write_manifest(manifest_path, {**prof, _PREPARE_KEY: prepare})
    return "ready", prof["run_id"]


def modeling_decide(outputs: dict[str, str]) -> tuple[str, str]:
    """Judges exactly one proposed candidate. Decisions: "accepted" (every
    check passed; review_bundle fact written for verification), "rejected"
    (try another), or "no_candidate" (budget spent with nothing accepted)."""
    prof = _without(_read_manifest(
        resolve_run_dir(outputs[STEP_PREPARE_MODELING]) / "prepare_modeling_manifest.json"
    ), _PREPARE_KEY)
    run_id, run_dir = prof["run_id"], Path(prof["run_dir"])
    on_event = make_event_emitter(run_id, persist_fn=make_event_logger(run_dir))

    attempts = read_fact_or_default(run_id, _ATTEMPTS_FACT)
    attempt_index = len(attempts["attempts"])
    if attempt_index >= _DEFAULT_MAX_CANDIDATES:
        # The budget is enforced here, not trusted to the caller: a
        # proposal arriving past it is refused without being built.
        emit_event(on_event, "modeling", "candidate_rejected", {
            "candidate_id": None, "template_id": None,
            "errors": [f"modeling candidate budget of {_DEFAULT_MAX_CANDIDATES} attempts already spent"],
        })
        return "no_candidate", _write_manifest(
            run_dir / "modeling_manifest.json", {**prof, "errors": _no_candidate_errors(attempts)}
        )

    loaded = _load_engineered(prof)
    split_manifest = _rebuild_split_manifest(json.loads(Path(prof["split_manifest_path"]).read_text()))
    result = evaluate_modeling_candidate(
        outputs[STEP_PROPOSE_MODELING], read_fact(run_id, "dataset_profile"),
        X=loaded.X, y=loaded.y, target_column=prof["target_column"],
        group_column=prof["group_column"], time_column=prof["time_column"],
        train_idx=split_manifest.train_idx, val_idx=split_manifest.val_idx,
        metric_names=_DEFAULT_METRIC_NAMES, seed=_DEFAULT_SEED, on_event=on_event,
    )
    manifest_path = run_dir / f"modeling_attempt_{attempt_index}_manifest.json"

    if not result.ok:
        attempts = _record_modeling_attempt(run_id, attempts, result, accepted=False, reason="; ".join(result.errors))
        if _budget_exhausted(attempts):
            return "no_candidate", _write_manifest(
                run_dir / "modeling_manifest.json", {**prof, "errors": _no_candidate_errors(attempts)}
            )
        return "rejected", _write_manifest(
            manifest_path, {**prof, "attempt_index": attempt_index, "errors": result.errors}
        )

    candidate_attempt_path = run_dir / f"candidate_attempt_{attempt_index}.joblib"
    joblib.dump(result.pipeline, candidate_attempt_path)
    # Written only here, after every deterministic gate passed — the
    # verifier is never served a candidate that failed one (invariant #4).
    template = get_template(result.template_id)
    write_fact(run_id, "review_bundle", build_review_bundle(
        candidate_id=result.candidate_id, template_id=result.template_id,
        template_description=template.description, template_when_to_use=template.when_to_use,
        config=result.config, explanation=result.explanation, metrics=result.metrics,
        label_permutation_check=result.label_permutation_check,
        feature_correlation_check=result.feature_correlation_check,
        profiler_report=prof["profiler_report"],
    ))
    _record_modeling_attempt(run_id, attempts, result, accepted=True, reason=None)

    _write_manifest(manifest_path, {**prof, _ATTEMPT_KEY: {
        "attempt_index": attempt_index,
        "candidate_attempt_path": str(candidate_attempt_path),
        "candidate_id": result.candidate_id,
        "template_id": result.template_id,
        "config": result.config,
        "explanation": result.explanation,
        "validation_metrics": result.metrics,
        "label_permutation_check": result.label_permutation_check,
        "feature_correlation_check": result.feature_correlation_check,
        "train_cv_consistency_check": result.train_cv_consistency_check,
    }})
    # Attempt manifests are numbered, so verification_decide (given only
    # run_id) finds this one through a fixed-name pointer.
    _write_manifest(run_dir / "pending_verification_manifest.json", {"attempt_index": attempt_index})
    return "accepted", run_id


def verification_decide(outputs: dict[str, str]) -> tuple[str, str]:
    """Decisions: "selected" (approved or flagged — the candidate proceeds
    to finalize), "rejected" (propose a fresh modeling candidate), or
    "no_candidate" (rejected with the modeling budget spent)."""
    run_id = outputs[STEP_MODELING]
    pointer_path = resolve_run_dir(run_id) / "pending_verification_manifest.json"
    decided = {}
    if pointer_path.is_file():  # no pointer = nothing ever accepted; refused by the ratchet below
        attempt_index = _read_manifest(pointer_path)["attempt_index"]
        decided = _read_manifest(resolve_run_dir(run_id) / f"modeling_attempt_{attempt_index}_manifest.json")
    candidate = decided.get(_ATTEMPT_KEY)
    attempts = read_fact_or_default(run_id, _ATTEMPTS_FACT)

    # One-way ratchet (CLAUDE.md invariant #4), held across separate gate
    # calls: only a candidate modeling_decide accepted, and verification
    # hasn't already rejected, can be verified. Otherwise re-running this
    # gate with a different verdict would be exactly the override path the
    # invariant forbids.
    if not _is_pending_acceptance(attempts, candidate):
        raise ValueError(
            "verification_decide refuses: the modeling manifest is not a candidate awaiting "
            "verification (never accepted by modeling_decide, or already rejected by verification)"
        )

    prof = _without(decided, _ATTEMPT_KEY)
    run_dir = Path(decided["run_dir"])
    on_event = make_event_emitter(run_id, persist_fn=make_event_logger(run_dir))
    verification = interpret_verification_verdict(
        outputs[STEP_PROPOSE_VERIFICATION], candidate["candidate_id"], on_event=on_event,
    )

    if verification.verdict == "rejected":
        attempts = _record_verification_rejection(
            run_id, attempts, candidate, "; ".join(verification.concerns) or "rejected by verification",
        )
        if _budget_exhausted(attempts):
            return "no_candidate", _write_manifest(
                run_dir / "modeling_manifest.json", {**prof, "errors": _no_candidate_errors(attempts)}
            )
        return "rejected", _write_manifest(
            run_dir / f"verification_attempt_{candidate['attempt_index']}_manifest.json",
            {**prof, "candidate_id": candidate["candidate_id"], "template_id": candidate["template_id"],
             "verification_verdict": verification.verdict, "verification_concerns": verification.concerns},
        )

    candidate_path = run_dir / "candidate.joblib"
    shutil.copyfile(candidate["candidate_attempt_path"], candidate_path)

    manifest = {
        "run_id": run_id,
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
        "candidate_id": candidate["candidate_id"],
        "template_id": candidate["template_id"],
        "validation_metrics": candidate["validation_metrics"],
        "verification_verdict": verification.verdict,
        "verification_concerns": verification.concerns,
    }
    return "selected", _write_manifest(run_dir / "modeling_manifest.json", manifest)


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


# -- 6. Summarize: hand over the facts to narrate, nothing more ----------


def run_summarize(outputs: dict[str, str]) -> tuple[str, str]:
    """Returns the facts JSON itself (not a path) as the output, so an agent
    step's task_template can substitute it straight into its prompt. The
    prose summary is that agent's output; there's nothing to decide."""
    final = _read_manifest(outputs[STEP_FINALIZE])
    run_dir = Path(final["run_dir"])
    on_event = make_event_emitter(final["run_id"], persist_fn=make_event_logger(run_dir))

    facts_json = json.dumps({
        "candidate_id": final["candidate_id"],
        "template_id": final["template_id"],
        "validation_metrics": final["validation_metrics"],
        "verification_verdict": final["verification_verdict"],
        "test_metrics": final["test_metrics"],
        "note_for_summary": (
            "If verification_concerns is non-empty, mention it briefly as a caveat for human review."
            if final["verification_concerns"] else None
        ),
    }, indent=2)

    facts_path = run_dir / "summary_facts.json"
    facts_path.write_text(facts_json)
    emit_event(on_event, "summarize", "summary_facts_ready", {"summary_facts_path": str(facts_path)})
    return "done", facts_json
