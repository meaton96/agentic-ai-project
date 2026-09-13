"""
gate_adapters.py's prepare_*/*_decide gates: agentic_ml's pipeline stages
with every internal LLM call removed, driven here by hand-written proposals
standing in for an external agent. Things worth proving, not just
exercising:

1. Nothing here can call a model. Every test runs with ModelClient.call and
   ToolCallingAgent.run patched to raise, and the full pipeline still
   completes end to end through finalize and summarize.
2. Each prepare_* gate publishes the fact its MCP tool serves (read back
   over a real in-memory MCP session), and each *_decide gate judges a
   proposal by the step functions' rules and writes the manifest shape the
   old single-call gates wrote (LEGACY_*_KEYS below).
3. What an agent step can receive from a gate — every prepare_* output, and
   modeling_decide's on "accepted" — is only the bare run_id, never a path
   or manifest content. The gate after the agent finds the manifest from
   run_id alone.
4. The modeling budget is enforced by modeling_decide itself: once it's
   spent, a valid candidate is refused without being built, however many
   proposals the caller keeps sending.
5. Verification stays a one-way ratchet across separate gate calls: it
   refuses when nothing was accepted, and a candidate it rejected can't be
   re-verified into "approved".
6. The profiler stage profiles the engineered frame, so derived feature
   columns are visible downstream — reusing feature engineering's raw-CSV
   profile would hide them.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agentic_ml import gate_adapters as ga
from agentic_ml.agent_runtime import ToolCallingAgent
from agentic_ml.harness.column_grouping import expand_grouped_columns
from agentic_ml.harness.dataset import read_dataframe
from agentic_ml.harness.intake import raw_schema_summary
from agentic_ml.mcp_facts.fact_store import FACT_DEFAULTS, FactNotFoundError, read_fact
from agentic_ml.mcp_facts.server import build_server
from agentic_ml.mcp_facts.transport import InMemoryMcpTransport
from agentic_ml.model_client import ModelClient
from agentic_ml.paths import run_dir as resolve_run_dir

LEGACY_INTAKE_KEYS = {"run_id", "run_dir", "csv_path", "target_column", "group_column", "time_column", "id_columns"}
LEGACY_FEATURE_ENGINEERING_KEYS = LEGACY_INTAKE_KEYS | {"features_path"}
LEGACY_PROFILER_AND_SPLIT_KEYS = LEGACY_FEATURE_ENGINEERING_KEYS | {
    "data_hash", "split_manifest_path", "strategy", "profiler_report",
}
LEGACY_SELECTED_KEYS = {
    "run_id", "run_dir", "csv_path", "target_column", "id_columns", "group_column", "time_column",
    "features_path", "data_hash", "split_manifest_path", "candidate_path", "candidate_id", "template_id",
    "validation_metrics", "verification_verdict", "verification_concerns",
}

INTAKE_PROPOSAL = json.dumps({
    "target_column": "churned", "task": "binary_classification", "id_columns": ["customer_id"],
    "group_column": None, "time_column": None, "positive_label": "1",
    "reasoning": "churned is the binary outcome.",
})
NO_CHANGES_PROPOSAL = json.dumps({"drop_columns": [], "derived_features": [], "explanation": "none needed"})
PROFILER_NARRATIVE = json.dumps({"summary": "Synthetic churn data.", "recommended_split_strategy": "stratified"})
CANDIDATE_A = json.dumps({
    "candidate_id": "candidate_a", "template_id": "sklearn_mixed_pipeline",
    "config": {"numeric_cols": ["age", "income"], "categorical_cols": ["plan_type", "region"],
               "classifier": "logistic_regression"},
    "explanation": "Mixed baseline.",
})
BAD_COLUMN_CANDIDATE = json.dumps({
    "candidate_id": "candidate_bad", "template_id": "sklearn_mixed_pipeline",
    "config": {"numeric_cols": ["not_a_column"], "categorical_cols": [], "classifier": "logistic_regression"},
    "explanation": "References a column that doesn't exist.",
})
APPROVED = json.dumps({"verdict": "approved", "concerns": [], "reasoning": "looks fine"})
REJECTED = json.dumps({"verdict": "rejected", "concerns": ["explanation doesn't match config"], "reasoning": "no"})


@pytest.fixture(autouse=True)
def isolated_run_root(tmp_path, monkeypatch):
    for var in ("AGENTIC_ML_DATA_ROOT", "AGENTIC_ML_RUNS_DIR", "AGENTIC_ML_ARTIFACTS_DIR", "AGENTIC_ML_DATASETS_DIR"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)


@pytest.fixture(autouse=True)
def no_llm_calls(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("gate_adapters made an LLM call")
    monkeypatch.setattr(ModelClient, "call", refuse)
    monkeypatch.setattr(ToolCallingAgent, "run", refuse)


@pytest.fixture
def dataset_csv(tmp_path):
    rng = np.random.RandomState(0)
    n = 400
    df = pd.DataFrame({
        "customer_id": [f"C{i}" for i in range(n)],
        "age": rng.randint(18, 80, size=n),
        "income": rng.exponential(50000, size=n),
        "plan_type": rng.choice(["basic", "premium", "pro"], size=n),
        "region": rng.choice(["north", "south", "east", "west"], size=n),
        "churned": rng.binomial(1, 0.35, size=n),
    })
    path = tmp_path / "churn.csv"
    df.to_csv(path, index=False)
    return path


def _manifest(path) -> dict:
    return json.loads(Path(path).read_text())


def _prepare_manifest(run_id: str, stage: str) -> dict:
    return _manifest(resolve_run_dir(run_id) / f"prepare_{stage}_manifest.json")


def _mcp(tool: str, run_id: str) -> dict:
    return InMemoryMcpTransport(build_server()).call_tool(tool, {"run_id": run_id})


def _through_intake(dataset_csv, proposal=INTAKE_PROPOSAL) -> dict:
    outputs = {"__task__": str(dataset_csv)}
    _, outputs[ga.STEP_PREPARE_INTAKE] = ga.prepare_intake(outputs)
    outputs[ga.STEP_PROPOSE_INTAKE] = proposal
    decision, outputs[ga.STEP_INTAKE] = ga.intake_decide(outputs)
    assert decision == "ok"
    return outputs


def _through_feature_engineering(outputs: dict, proposal=NO_CHANGES_PROPOSAL) -> dict:
    _, outputs[ga.STEP_PREPARE_FEATURE_ENGINEERING] = ga.prepare_feature_engineering(outputs)
    outputs[ga.STEP_PROPOSE_FEATURE_ENGINEERING] = proposal
    decision, outputs[ga.STEP_FEATURE_ENGINEERING] = ga.feature_engineering_decide(outputs)
    assert decision == "ok"
    return outputs


def _through_prepare_modeling(dataset_csv) -> dict:
    outputs = _through_feature_engineering(_through_intake(dataset_csv))
    _, outputs[ga.STEP_PREPARE_PROFILER_AND_SPLIT] = ga.prepare_profiler_and_split(outputs)
    outputs[ga.STEP_PROPOSE_PROFILER] = PROFILER_NARRATIVE
    decision, outputs[ga.STEP_PROFILER_AND_SPLIT] = ga.profiler_and_split_decide(outputs)
    assert decision == "ok"
    decision, outputs[ga.STEP_PREPARE_MODELING] = ga.prepare_modeling(outputs)
    assert decision == "ready"
    return outputs


def _propose_candidate(outputs: dict, candidate: str) -> str:
    outputs[ga.STEP_PROPOSE_MODELING] = candidate
    decision, outputs[ga.STEP_MODELING] = ga.modeling_decide(outputs)
    return decision


def _verify(outputs: dict, verdict: str) -> str:
    outputs[ga.STEP_PROPOSE_VERIFICATION] = verdict
    decision, outputs[ga.STEP_MODELING_AND_VERIFICATION] = ga.verification_decide(outputs)
    return decision


# --- 2: intake ---

def test_prepare_intake_publishes_raw_schema_over_mcp_and_awaits_a_proposal(dataset_csv):
    decision, run_id = ga.prepare_intake({"__task__": str(dataset_csv)})
    prep = _prepare_manifest(run_id, "intake")
    assert decision == "ready"
    assert prep["run_id"] == run_id
    assert prep["prepare"] == {"stage": "intake", "mcp_tools": ["get_raw_schema"]}
    expected = json.loads(json.dumps(raw_schema_summary(read_dataframe(str(dataset_csv))), default=str))
    assert _mcp("get_raw_schema", run_id) == expected


def test_intake_decide_ok_writes_legacy_manifest_shape(dataset_csv):
    manifest = _manifest(_through_intake(dataset_csv)[ga.STEP_INTAKE])
    assert set(manifest) == LEGACY_INTAKE_KEYS
    assert manifest["target_column"] == "churned"
    assert manifest["id_columns"] == ["customer_id"]


@pytest.mark.parametrize("proposal, expected_error", [
    ("not json", "did not parse as JSON"),
    (json.dumps({"target_column": "no_such_column"}), "not found in dataset columns"),
])
def test_intake_decide_rejects_bad_proposals(dataset_csv, proposal, expected_error):
    outputs = {"__task__": str(dataset_csv)}
    _, outputs[ga.STEP_PREPARE_INTAKE] = ga.prepare_intake(outputs)
    outputs[ga.STEP_PROPOSE_INTAKE] = proposal
    decision, path = ga.intake_decide(outputs)
    manifest = _manifest(path)
    assert decision == "failed"
    assert set(manifest) == {"run_id", "run_dir", "errors"}
    assert expected_error in manifest["errors"][0]


# --- 3: what an agent step can see ---

def test_every_agent_facing_output_is_the_bare_run_id(dataset_csv):
    """An agent step gets a prior output substituted verbatim into its
    prompt, can't read the gates' filesystem, and must never see raw paths.
    run_id is the one argument every run-scoped MCP tool takes."""
    outputs = _through_prepare_modeling(dataset_csv)
    run_id = _manifest(outputs[ga.STEP_INTAKE])["run_id"]
    assert "/" not in run_id
    for step in (ga.STEP_PREPARE_INTAKE, ga.STEP_PREPARE_FEATURE_ENGINEERING,
                 ga.STEP_PREPARE_PROFILER_AND_SPLIT, ga.STEP_PREPARE_MODELING):
        assert outputs[step] == run_id
    assert "columns" in _mcp("get_dataset_profile", outputs[ga.STEP_PREPARE_MODELING])


def test_accepted_candidate_is_found_for_verification_from_run_id_alone(dataset_csv):
    outputs = _through_prepare_modeling(dataset_csv)
    run_id = outputs[ga.STEP_PREPARE_MODELING]
    run_dir = resolve_run_dir(run_id)

    # accept attempt 1, not 0, so the pointer can't be right by coincidence
    assert _propose_candidate(outputs, BAD_COLUMN_CANDIDATE) == "rejected"
    assert _propose_candidate(outputs, CANDIDATE_A) == "accepted"
    assert outputs[ga.STEP_MODELING] == run_id
    assert _manifest(run_dir / "pending_verification_manifest.json") == {"attempt_index": 1}
    assert _manifest(run_dir / "modeling_attempt_1_manifest.json")["modeling_attempt"]["candidate_id"] == "candidate_a"

    decision, path = ga.verification_decide({ga.STEP_MODELING: run_id, ga.STEP_PROPOSE_VERIFICATION: APPROVED})
    assert decision == "selected"
    assert _manifest(path)["candidate_id"] == "candidate_a"


# --- 2: feature engineering ---

def test_feature_engineering_decide_applies_ops_and_merges_drop_columns(dataset_csv):
    proposal = json.dumps({
        "drop_columns": ["region"],
        "derived_features": [{"op_id": "missing_indicator", "params": {"col": "age"}}],
        "explanation": "drop region, flag missing age",
    })
    outputs = _through_intake(dataset_csv)
    _, run_id = ga.prepare_feature_engineering(outputs)
    assert _prepare_manifest(run_id, "feature_engineering")["prepare"]["mcp_tools"] == [
        "get_dataset_profile", "list_feature_ops",
    ]

    manifest = _manifest(_through_feature_engineering(outputs, proposal)[ga.STEP_FEATURE_ENGINEERING])
    assert set(manifest) == LEGACY_FEATURE_ENGINEERING_KEYS
    assert manifest["id_columns"] == ["customer_id", "region"]
    assert "age_was_missing" in pd.read_parquet(manifest["features_path"]).columns


def test_feature_engineering_decide_invalid_proposal_fails_with_intake_manifest(dataset_csv):
    outputs = _through_intake(dataset_csv)
    _, outputs[ga.STEP_PREPARE_FEATURE_ENGINEERING] = ga.prepare_feature_engineering(outputs)
    outputs[ga.STEP_PROPOSE_FEATURE_ENGINEERING] = json.dumps({"drop_columns": ["churned"], "derived_features": []})
    decision, path = ga.feature_engineering_decide(outputs)
    manifest = _manifest(path)
    assert decision == "failed"
    assert manifest == {**_manifest(outputs[ga.STEP_INTAKE]), "errors": manifest["errors"]}
    assert any("target/group/time" in e for e in manifest["errors"])


# --- 2 + 6: profiler + split ---

def test_profiler_stage_profiles_the_engineered_frame_not_the_raw_csv(dataset_csv):
    proposal = json.dumps({
        "drop_columns": [], "derived_features": [{"op_id": "missing_indicator", "params": {"col": "age"}}],
    })
    outputs = _through_intake(dataset_csv)
    _, outputs[ga.STEP_PREPARE_FEATURE_ENGINEERING] = ga.prepare_feature_engineering(outputs)
    run_id = _manifest(outputs[ga.STEP_INTAKE])["run_id"]
    assert "age_was_missing" not in expand_grouped_columns(_mcp("get_dataset_profile", run_id)["columns"])

    _through_feature_engineering(outputs, proposal)
    ga.prepare_profiler_and_split(outputs)
    assert "age_was_missing" in expand_grouped_columns(_mcp("get_dataset_profile", run_id)["columns"])


def test_profiler_and_split_decide_writes_legacy_shape_and_keeps_the_narrative(dataset_csv):
    outputs = _through_prepare_modeling(dataset_csv)
    manifest = _manifest(outputs[ga.STEP_PROFILER_AND_SPLIT])
    run_dir = Path(manifest["run_dir"])
    assert set(manifest) == LEGACY_PROFILER_AND_SPLIT_KEYS
    assert Path(manifest["split_manifest_path"]).exists()
    assert (run_dir / "profiler_narrative.txt").read_text() == PROFILER_NARRATIVE
    assert manifest["profiler_report"]["recommended_split_strategy"] == manifest["strategy"]


# --- 4: modeling attempts + budget ---

def test_prepare_modeling_advertises_attempts_tool_which_starts_empty(dataset_csv):
    outputs = _through_prepare_modeling(dataset_csv)
    run_id = outputs[ga.STEP_PREPARE_MODELING]
    prep = _prepare_manifest(run_id, "modeling")
    assert prep["prepare"]["mcp_tools"] == ["get_dataset_profile", "list_templates", "get_modeling_attempts"]
    assert prep["prepare"]["attempts_used"] == 0
    assert prep["prepare"]["max_candidates"] == ga._DEFAULT_MAX_CANDIDATES
    assert _mcp("get_modeling_attempts", run_id) == FACT_DEFAULTS["modeling_attempts"]


def test_modeling_rejection_is_recorded_for_the_next_proposer(dataset_csv):
    outputs = _through_prepare_modeling(dataset_csv)
    run_id = outputs[ga.STEP_PREPARE_MODELING]

    assert _propose_candidate(outputs, BAD_COLUMN_CANDIDATE) == "rejected"
    attempts = _mcp("get_modeling_attempts", run_id)
    assert attempts["tried_template_ids"] == ["sklearn_mixed_pipeline"]
    assert attempts["rejections"][0]["stage"] == "modeling"
    assert "unknown column 'not_a_column'" in attempts["rejections"][0]["reason"]
    assert attempts["attempts"][0]["accepted"] is False
    # nothing for a verifier to see: the bundle only exists for accepted candidates
    with pytest.raises(FactNotFoundError):
        read_fact(run_id, "review_bundle")
    ga.prepare_modeling(outputs)
    assert _prepare_manifest(run_id, "modeling")["prepare"]["attempts_used"] == 1


def test_modeling_budget_is_enforced_by_the_gate_not_the_caller(dataset_csv):
    outputs = _through_prepare_modeling(dataset_csv)
    run_id = outputs[ga.STEP_PREPARE_MODELING]
    run_dir = resolve_run_dir(run_id)

    decisions = [_propose_candidate(outputs, BAD_COLUMN_CANDIDATE) for _ in range(ga._DEFAULT_MAX_CANDIDATES)]
    assert decisions == ["rejected"] * (ga._DEFAULT_MAX_CANDIDATES - 1) + ["no_candidate"]
    assert _manifest(outputs[ga.STEP_MODELING])["errors"] == ["no candidate passed the harness's leakage gates"]

    # a perfectly valid candidate past the budget is refused, never built
    assert _propose_candidate(outputs, CANDIDATE_A) == "no_candidate"
    assert len(read_fact(run_id, "modeling_attempts")["attempts"]) == ga._DEFAULT_MAX_CANDIDATES
    assert not list(run_dir.glob("candidate_attempt_*.joblib"))
    assert not (run_dir / "pending_verification_manifest.json").exists()


# --- 1 + 2: end to end ---

def test_full_pipeline_runs_without_any_llm_call_and_writes_legacy_shapes(dataset_csv):
    outputs = _through_prepare_modeling(dataset_csv)
    run_id = outputs[ga.STEP_PREPARE_MODELING]

    assert _propose_candidate(outputs, CANDIDATE_A) == "accepted"
    assert _mcp("get_candidate_review_bundle", run_id)["candidate_id"] == "candidate_a"

    assert _verify(outputs, APPROVED) == "selected"
    selected = _manifest(outputs[ga.STEP_MODELING_AND_VERIFICATION])
    assert set(selected) == LEGACY_SELECTED_KEYS
    assert selected["verification_verdict"] == "approved"
    assert Path(selected["candidate_path"]).exists()

    decision, outputs[ga.STEP_FINALIZE] = ga.run_finalize(outputs)
    assert decision == "done"
    decision, facts_json = ga.run_summarize(outputs)
    assert decision == "done"
    facts = json.loads(facts_json)
    assert facts["candidate_id"] == "candidate_a"
    assert facts["test_metrics"] == _manifest(outputs[ga.STEP_FINALIZE])["test_metrics"]

    event_types = {
        json.loads(line)["type"]
        for line in (Path(selected["run_dir"]) / "events.jsonl").read_text().splitlines()
    }
    assert {"facts_ready", "leakage_gate_result", "candidate_scored", "verification_verdict",
            "split_completed", "finalize_completed", "summary_facts_ready"} <= event_types


def test_unparseable_verdict_degrades_to_flagged_never_approved(dataset_csv):
    outputs = _through_prepare_modeling(dataset_csv)
    assert _propose_candidate(outputs, CANDIDATE_A) == "accepted"
    assert _verify(outputs, "LGTM!") == "selected"
    selected = _manifest(outputs[ga.STEP_MODELING_AND_VERIFICATION])
    assert selected["verification_verdict"] == "flagged"
    assert "did not parse" in selected["verification_concerns"][0]


# --- 5: verification ratchet ---

def test_verification_refuses_when_modeling_never_accepted_anything(dataset_csv):
    outputs = _through_prepare_modeling(dataset_csv)
    assert _propose_candidate(outputs, BAD_COLUMN_CANDIDATE) == "rejected"
    outputs[ga.STEP_MODELING] = outputs[ga.STEP_PREPARE_MODELING]  # the run_id an accepted output would carry
    with pytest.raises(ValueError, match="refuses"):
        _verify(outputs, APPROVED)


def test_verification_rejection_cannot_be_overridden_and_counts_toward_the_budget(dataset_csv):
    outputs = _through_prepare_modeling(dataset_csv)
    run_id = outputs[ga.STEP_PREPARE_MODELING]

    assert _propose_candidate(outputs, CANDIDATE_A) == "accepted"
    assert _verify(outputs, REJECTED) == "rejected"
    rejection = _mcp("get_modeling_attempts", run_id)["rejections"][-1]
    assert (rejection["stage"], rejection["candidate_id"]) == ("verification", "candidate_a")

    # re-running verification for the same run with a friendlier verdict
    outputs[ga.STEP_MODELING] = run_id
    with pytest.raises(ValueError, match="refuses"):
        _verify(outputs, APPROVED)

    decisions = [_propose_candidate(outputs, BAD_COLUMN_CANDIDATE) for _ in range(ga._DEFAULT_MAX_CANDIDATES - 1)]
    assert decisions[-1] == "no_candidate"
    assert _manifest(outputs[ga.STEP_MODELING])["errors"] == [
        "every gate-passing candidate was rejected by verification"
    ]
