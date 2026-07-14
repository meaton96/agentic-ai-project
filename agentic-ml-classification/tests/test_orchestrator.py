"""
Phase 5 integration test: the full orchestrator loop (intake ->
profiler -> N modeling candidates -> selection -> one locked test-set
evaluation -> narrated summary) driven end to end with a stubbed
ModelClient — no real network/LLM call. This is the automated version
of "does prompt (or just a dataset) + dataset -> finished
classification actually work", run against a synthetic in-memory
dataset instead of a real file so it stays fast and hermetic.

The fake model dispatches purely on (a) which system prompt is active
(identifies which agent is calling) and (b) how many messages are in
the conversation so far (identifies which turn within that agent's own
run() this is) — both are stable, real signals already present in
every call, so no external mutable call-counter is needed and multiple
independent agent.run() invocations (e.g. two modeling candidates in
one orchestrator run) are handled correctly without cross-talk.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from agentic_ml.model_client import ModelClient, ModelResponse

import run_orchestrator  # noqa: E402  (needs the sys.path insert above first)

INTAKE_PROPOSAL = json.dumps({
    "target_column": "churned",
    "task": "binary_classification",
    "id_columns": ["customer_id"],
    "group_column": None,
    "time_column": None,
    "positive_label": "1",
    "reasoning": "churned looks like the binary outcome column given the goal.",
})

PROFILER_NARRATIVE = json.dumps({
    "summary": "Synthetic churn dataset with mixed numeric/categorical features.",
    "recommended_split_strategy": "stratified",
    "key_risks": [],
    "recommended_next_steps": ["try a mixed baseline"],
})

CANDIDATE_A = json.dumps({
    "candidate_id": "candidate_a",
    "template_id": "sklearn_mixed_pipeline",
    "config": {
        "numeric_cols": ["age", "income"],
        "categorical_cols": ["plan_type", "region"],
        "classifier": "logistic_regression",
    },
    "explanation": "Mixed baseline; logistic regression as a cheap first pass.",
})

CANDIDATE_B = json.dumps({
    "candidate_id": "candidate_b",
    "template_id": "imbalanced_binary_boosted",
    "config": {
        "numeric_cols": ["age", "income"],
        "categorical_cols": ["plan_type", "region"],
    },
    "explanation": "Try class-weighted boosting given moderate imbalance.",
})


def _resp(text=None, tool_calls=None):
    return ModelResponse(
        text=text, tool_calls=tool_calls or [], raw=None, latency_seconds=0.01,
        model="fake-model", input_tokens=1, output_tokens=1,
    )


def fake_call(self, messages, model=None, tools=None, temperature=0.0, max_tokens=1024):
    system_content = messages[0]["content"]
    n = len(messages)

    if "Intake agent" in system_content:
        if n == 2:
            return _resp(tool_calls=[{"id": "t1", "name": "get_raw_schema", "arguments": "{}"}])
        return _resp(text=INTAKE_PROPOSAL)

    if "Profiler agent" in system_content:
        if n == 2:
            return _resp(tool_calls=[{"id": "t2", "name": "get_dataset_profile", "arguments": "{}"}])
        return _resp(text=PROFILER_NARRATIVE)

    if "Modeling agent" in system_content:
        if n == 2:
            return _resp(tool_calls=[{"id": "t3", "name": "get_dataset_profile", "arguments": "{}"}])
        if n == 4:
            return _resp(tool_calls=[{"id": "t4", "name": "list_templates", "arguments": "{}"}])
        candidate = CANDIDATE_B if "Templates already tried" in system_content else CANDIDATE_A
        return _resp(text=candidate)

    # Analyst-style final summary: a plain text call with no tools.
    return _resp(text="This is a plain-language summary of the modeling run.")


@pytest.fixture(autouse=True)
def patch_model_client(monkeypatch):
    monkeypatch.setattr(ModelClient, "call", fake_call)
    monkeypatch.setenv("RIT_BASE_URL", "http://example.invalid/v1")
    monkeypatch.setenv("RIT_API_KEY", "dummy")


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


def _run_orchestrator_in(tmp_path, argv):
    """Runs run_orchestrator.main() with cwd pointed at tmp_path, so
    runs/ and artifacts/reports/leaderboard.jsonl land there instead of
    the real repo — no manual cleanup needed."""
    old_cwd = Path.cwd()
    old_argv = sys.argv
    os.chdir(tmp_path)
    sys.argv = argv
    try:
        run_orchestrator.main()
    finally:
        os.chdir(old_cwd)
        sys.argv = old_argv


def test_orchestrator_explicit_target_skips_intake(dataset_csv, tmp_path):
    _run_orchestrator_in(tmp_path, [
        "run_orchestrator.py",
        "--data", str(dataset_csv),
        "--target", "churned",
        "--id-columns", "customer_id",
        "--max-candidates", "2",
        "--run-id", "test_run_explicit",
    ])

    report = json.loads((tmp_path / "runs" / "test_run_explicit" / "orchestrator_report.json").read_text())

    assert report["status"] == "success"
    assert report["intake"]["skipped"] is True
    assert len(report["candidates"]) == 2
    assert {c["template_id"] for c in report["candidates"]} == {"sklearn_mixed_pipeline", "imbalanced_binary_boosted"}
    assert all(c["passed_gate"] for c in report["candidates"])
    assert report["selected_candidate"]["candidate_id"] in {"candidate_a", "candidate_b"}
    assert set(report["final_test_metrics"]) == {"roc_auc", "pr_auc", "f1", "accuracy"}
    assert report["final_summary"]

    leaderboard_path = tmp_path / "artifacts" / "reports" / "leaderboard.jsonl"
    entries = [json.loads(line) for line in leaderboard_path.read_text().splitlines()]
    assert len(entries) == 2
    assert all(e["source"] == "orchestrator" for e in entries)


def test_orchestrator_full_intake_no_target_given(dataset_csv, tmp_path):
    _run_orchestrator_in(tmp_path, [
        "run_orchestrator.py",
        "--data", str(dataset_csv),
        "--goal", "predict which customers will churn",
        "--max-candidates", "1",
        "--run-id", "test_run_intake",
    ])

    report = json.loads((tmp_path / "runs" / "test_run_intake" / "orchestrator_report.json").read_text())

    assert report["status"] == "success"
    assert report["intake"]["target_column"] == "churned"
    assert report["intake"]["id_columns"] == ["customer_id"]
    assert len(report["candidates"]) == 1
    assert report["selected_candidate"]["candidate_id"] == "candidate_a"
    assert report["final_test_metrics"]["roc_auc"]["value"] is not None

    intake_report = json.loads((tmp_path / "runs" / "test_run_intake" / "intake_report.json").read_text())
    assert intake_report["dataset_spec_proposal"]["target_column"] == "churned"


def test_orchestrator_dataset_only_no_goal_no_target(dataset_csv, tmp_path):
    """The 'just a dataset' case: no --goal, no --target at all."""
    _run_orchestrator_in(tmp_path, [
        "run_orchestrator.py",
        "--data", str(dataset_csv),
        "--max-candidates", "1",
        "--run-id", "test_run_dataset_only",
    ])

    report = json.loads((tmp_path / "runs" / "test_run_dataset_only" / "orchestrator_report.json").read_text())
    assert report["status"] == "success"
    assert report["intake"]["target_column"] == "churned"
