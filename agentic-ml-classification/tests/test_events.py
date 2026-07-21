"""
M0 instrumentation: the optional on_event callback (agentic_ml.events)
and the explicit data-root config (agentic_ml.paths). Three things worth
proving, not just exercising:

1. A static orchestrator run produces the expected event sequence in
   runs/<run_id>/events.jsonl — phases in order, both leakage gates
   present, a candidate-scored event, a verification verdict, and final
   test-set metrics — using the same stubbed-ModelClient pattern as
   tests/test_orchestrator.py.
2. A dynamic orchestrator run's events.jsonl includes a rejected planner
   proposal (reusing test_dynamic_orchestrator.py's hallucinated-agent_id
   fixture) — proving on_event captures control-flow rejections, not
   just successful steps.
3. AGENTIC_ML_DATA_ROOT actually redirects where events.jsonl, the
   leaderboard, and run artifacts get written, while leaving them alone
   (today's exact cwd-relative paths) when unset.
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

import run_dynamic_orchestrator  # noqa: E402
import run_orchestrator  # noqa: E402

# --- shared response constants (mirrors tests/test_orchestrator.py's) ---

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

VERIFICATION_APPROVED = json.dumps({"verdict": "approved", "concerns": [], "reasoning": "looks fine"})

INTAKE_PROPOSAL = json.dumps({
    "target_column": "churned", "task": "binary_classification",
    "id_columns": ["customer_id"], "group_column": None, "time_column": None,
    "positive_label": "1", "reasoning": "churned looks like the binary outcome column.",
})


def _resp(text=None, tool_calls=None):
    return ModelResponse(
        text=text, tool_calls=tool_calls or [], raw=None, latency_seconds=0.01,
        model="fake-model", input_tokens=1, output_tokens=1,
    )


def static_fake_call(self, messages, model=None, tools=None, temperature=0.0, max_tokens=1024):
    system_content = messages[0]["content"]
    n = len(messages)

    if "Profiler agent" in system_content:
        if n == 2:
            return _resp(tool_calls=[{"id": "t2", "name": "get_dataset_profile", "arguments": "{}"}])
        return _resp(text=PROFILER_NARRATIVE)

    if "Modeling agent" in system_content:
        if n == 2:
            return _resp(tool_calls=[{"id": "t3", "name": "get_dataset_profile", "arguments": "{}"}])
        if n == 4:
            return _resp(tool_calls=[{"id": "t4", "name": "list_templates", "arguments": "{}"}])
        return _resp(text=CANDIDATE_A)

    if "Verification agent" in system_content:
        if n == 2:
            return _resp(tool_calls=[{"id": "v1", "name": "get_candidate_review_bundle", "arguments": "{}"}])
        return _resp(text=VERIFICATION_APPROVED)

    # Analyst-style final summary: a plain text call with no tools.
    return _resp(text="This is a plain-language summary of the modeling run.")


_planner_state = {"call_index": 0}


def rejecting_planner_fake_call(self, messages, model=None, tools=None, temperature=0.0, max_tokens=1024):
    n = len(messages)
    if n == 2:
        return _resp(tool_calls=[{"id": "p1", "name": "get_planning_context", "arguments": "{}"}])
    idx = _planner_state["call_index"]
    _planner_state["call_index"] += 1
    if idx == 0:
        return _resp(text=json.dumps({
            "action": "run_agent", "agent_id": "delete_all_data", "args": {}, "reasoning": "???",
        }))
    return _resp(text=json.dumps({"action": "finish", "agent_id": None, "args": {}, "reasoning": "done"}))


@pytest.fixture(autouse=True)
def patch_env(monkeypatch):
    monkeypatch.setenv("RIT_BASE_URL", "http://example.invalid/v1")
    monkeypatch.setenv("RIT_API_KEY", "dummy")
    _planner_state["call_index"] = 0
    yield
    _planner_state["call_index"] = 0


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


def _run_in(tmp_path, module, argv):
    old_cwd = Path.cwd()
    old_argv = sys.argv
    os.chdir(tmp_path)
    sys.argv = argv
    try:
        module.main()
    finally:
        os.chdir(old_cwd)
        sys.argv = old_argv


def _read_events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


# --- 1. Static orchestrator: full expected event sequence ---

def test_static_orchestrator_writes_expected_event_sequence(dataset_csv, tmp_path, monkeypatch):
    monkeypatch.setattr(ModelClient, "call", static_fake_call)
    _run_in(tmp_path, run_orchestrator, [
        "run_orchestrator.py",
        "--data", str(dataset_csv),
        "--target", "churned",
        "--id-columns", "customer_id",
        "--skip-feature-engineering",
        "--max-candidates", "1",
        "--run-id", "test_events_static",
    ])

    events_path = tmp_path / "runs" / "test_events_static" / "events.jsonl"
    assert events_path.exists()
    events = _read_events(events_path)

    # every event is a small, JSON-safe dict with the required keys
    for e in events:
        assert set(e) == {"ts", "run_id", "phase", "type", "payload"}
        assert e["run_id"] == "test_events_static"
        assert isinstance(e["payload"], dict)

    assert events[0]["type"] == "run_started"
    assert events[-1] == events[-1]  # sanity: list is non-empty and ordered below
    assert events[-1]["type"] == "run_completed"
    assert events[-1]["payload"]["status"] == "success"

    # phases visited, in order (feature_engineering skipped via the flag above)
    phases_in_order = [e["phase"] for e in events]
    for expected_phase in ["profiler", "split_and_check_leakage", "modeling", "verification", "finalize"]:
        assert expected_phase in phases_in_order

    def first_index(predicate):
        return next(i for i, e in enumerate(events) if predicate(e))

    assert first_index(lambda e: e["phase"] == "profiler") < first_index(lambda e: e["phase"] == "split_and_check_leakage")
    assert first_index(lambda e: e["phase"] == "split_and_check_leakage") < first_index(lambda e: e["phase"] == "modeling")
    assert first_index(lambda e: e["phase"] == "modeling") < first_index(lambda e: e["phase"] == "verification")
    assert first_index(lambda e: e["phase"] == "verification") < first_index(lambda e: e["phase"] == "finalize")

    # both deterministic leakage gates present for the split, at least one
    # candidate-scored event, a verification verdict, and final metrics
    split_gate_events = [e for e in events if e["phase"] == "split_and_check_leakage" and e["type"] == "leakage_gate_result"]
    assert len(split_gate_events) >= 1

    candidate_scored = [e for e in events if e["type"] == "candidate_scored"]
    assert len(candidate_scored) == 1
    assert candidate_scored[0]["payload"]["candidate_id"] == "candidate_a"
    assert candidate_scored[0]["payload"]["passed_gate"] is True

    verdicts = [e for e in events if e["type"] == "verification_verdict"]
    assert len(verdicts) == 1
    assert verdicts[0]["payload"]["verdict"] == "approved"

    final_metrics = [e for e in events if e["type"] == "final_test_metrics"]
    assert len(final_metrics) == 1
    assert set(final_metrics[0]["payload"]["test_metrics"]) == {"roc_auc", "pr_auc", "f1", "accuracy"}


# --- 2. Dynamic orchestrator: rejected planner proposal shows up in events.jsonl ---

def test_dynamic_orchestrator_events_include_rejected_planner_proposal(dataset_csv, tmp_path, monkeypatch):
    monkeypatch.setattr(ModelClient, "call", rejecting_planner_fake_call)
    _run_in(tmp_path, run_dynamic_orchestrator, [
        "run_dynamic_orchestrator.py", "--data", str(dataset_csv), "--target", "churned",
        "--run-id", "test_events_dynamic", "--max-iterations", "3",
    ])

    events_path = tmp_path / "runs" / "test_events_dynamic" / "events.jsonl"
    assert events_path.exists()
    events = _read_events(events_path)

    rejected = [e for e in events if e["type"] == "planner_proposal_rejected"]
    assert len(rejected) == 1
    assert "delete_all_data" in rejected[0]["payload"]["errors"][0]
    assert "not a known/available agent" in rejected[0]["payload"]["errors"][0]

    accepted = [e for e in events if e["type"] == "planner_proposal_accepted"]
    assert len(accepted) == 1
    assert accepted[0]["payload"]["proposal"]["action"] == "finish"

    assert events[0]["type"] == "run_started"
    assert events[-1]["type"] == "run_completed"
    assert events[-1]["payload"]["status"] == "success"


# --- 3. Explicit data-root config ---

def test_data_root_env_var_redirects_run_artifacts(dataset_csv, tmp_path, monkeypatch):
    custom_root = tmp_path / "elsewhere"
    monkeypatch.setenv("AGENTIC_ML_DATA_ROOT", str(custom_root))
    monkeypatch.setattr(ModelClient, "call", static_fake_call)

    _run_in(tmp_path, run_orchestrator, [
        "run_orchestrator.py",
        "--data", str(dataset_csv),
        "--target", "churned",
        "--id-columns", "customer_id",
        "--skip-feature-engineering",
        "--max-candidates", "1",
        "--run-id", "test_events_custom_root",
    ])

    # redirected: nothing lands under cwd's own runs/artifacts...
    assert not (tmp_path / "runs" / "test_events_custom_root").exists()
    assert not (tmp_path / "artifacts").exists()

    # ...it all lands under AGENTIC_ML_DATA_ROOT instead
    run_dir = custom_root / "runs" / "test_events_custom_root"
    assert (run_dir / "events.jsonl").exists()
    assert (run_dir / "orchestrator_report.json").exists()

    leaderboard_path = custom_root / "artifacts" / "reports" / "leaderboard.jsonl"
    assert leaderboard_path.exists()

    model_files = list((custom_root / "artifacts" / "models").glob("*.joblib"))
    assert len(model_files) == 1


def test_data_root_unset_reproduces_default_cwd_relative_paths(dataset_csv, tmp_path, monkeypatch):
    monkeypatch.delenv("AGENTIC_ML_DATA_ROOT", raising=False)
    monkeypatch.delenv("AGENTIC_ML_RUNS_DIR", raising=False)
    monkeypatch.delenv("AGENTIC_ML_ARTIFACTS_DIR", raising=False)
    monkeypatch.setattr(ModelClient, "call", static_fake_call)

    _run_in(tmp_path, run_orchestrator, [
        "run_orchestrator.py",
        "--data", str(dataset_csv),
        "--target", "churned",
        "--id-columns", "customer_id",
        "--skip-feature-engineering",
        "--max-candidates", "1",
        "--run-id", "test_events_default_root",
    ])

    # today's exact behavior: runs/ and artifacts/ directly under cwd (tmp_path)
    run_dir = tmp_path / "runs" / "test_events_default_root"
    assert (run_dir / "events.jsonl").exists()
    assert (run_dir / "orchestrator_report.json").exists()
    assert (tmp_path / "artifacts" / "reports" / "leaderboard.jsonl").exists()
    assert list((tmp_path / "artifacts" / "models").glob("*.joblib"))
