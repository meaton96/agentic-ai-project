"""
The dynamic orchestrator: an agent catalog (orchestrator/agent_registry.py)
plus a planning agent (steps/planner_step.py) that decides which catalog
agent runs next, validated deterministically
(orchestrator/dynamic_loop.py::validate_plan) before anything executes.

Six things worth proving, not just exercising:

1. Parity: given the same synthetic dataset and a scripted-but-realistic
   sequence of agent proposals, the dynamic path visits the same
   effective stages as the static orchestrator (intake -> feature
   engineering -> profiler -> split -> modeling -> verification ->
   finalize -> summarize) and produces well-formed final metrics — the
   concrete "this is not a regression" proof.
2. Task-routing: a goal that only needs deep-dive, with a model already
   available, makes the planner skip classification entirely — the
   concrete "different task, different agent sequence" proof a static
   script cannot express.
3. A hallucinated agent_id is rejected without executing anything.
4. A precondition violation (proposing an agent before its dependencies
   are satisfied) is rejected without executing anything.
5. finalize's one-shot guard: it cannot run a second time once the test
   set has already been touched.
6. The planner loop actually stops at max_iterations rather than
   looping forever if a (real or malfunctioning) planner never proposes
   "finish".
7. normalize_proposal() repairs the specific schema confusion a real
   local model made during evaluation (putting the agent id directly
   in 'action', e.g. {"action": "finalize"}) WITHOUT relaxing what's
   actually allowed to execute — a proposal naming a non-agent action
   string is left alone and still rejected downstream.
8. Long-format time-series auto-routing: a dataset detected as raw
   long-format (harness/dataset.py::detect_dataset_shape) blocks
   intake/feature_engineering until featurize_timeseries has run, and
   a full scripted run — featurize_timeseries -> intake -> ... ->
   finalize — completes successfully end to end, proving the routing
   mechanism itself, not just its precondition gate in isolation.
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

from agentic_ml.harness.attribution import compute_background
from agentic_ml.model_client import ModelClient, ModelResponse
from agentic_ml.orchestrator import dynamic_loop as dynamic_loop_module
from agentic_ml.orchestrator.dynamic_loop import normalize_proposal, run_dynamic_loop, validate_plan
from agentic_ml.orchestrator.run_state import CandidateSummary, DynamicRunContext, RunStateSummary
from agentic_ml.templates.sources.xgboost_mixed import build_pipeline

import run_dynamic_orchestrator  # noqa: E402  (needs the sys.path insert above first)

# --- shared response constants (mirrors tests/test_orchestrator.py's) ---

INTAKE_PROPOSAL = json.dumps({
    "target_column": "churned", "task": "binary_classification",
    "id_columns": ["customer_id"], "group_column": None, "time_column": None,
    "positive_label": "1", "reasoning": "churned looks like the binary outcome column.",
})
FEATURE_ENGINEERING_PROPOSAL = json.dumps({
    "drop_columns": [], "derived_features": [], "explanation": "No changes needed.",
})
PROFILER_NARRATIVE = json.dumps({
    "summary": "Synthetic churn dataset.", "recommended_split_strategy": "stratified",
    "key_risks": [], "recommended_next_steps": [],
})
CANDIDATE_A = json.dumps({
    "candidate_id": "candidate_a", "template_id": "sklearn_mixed_pipeline",
    "config": {"numeric_cols": ["age", "income"], "categorical_cols": ["plan_type", "region"],
              "classifier": "logistic_regression"},
    "explanation": "Mixed baseline.",
})
VERIFICATION_APPROVED = json.dumps({"verdict": "approved", "concerns": [], "reasoning": "looks fine"})

PLANNER_ACTIONS = [
    {"action": "run_agent", "agent_id": "intake", "args": {}, "reasoning": "need a target"},
    {"action": "run_agent", "agent_id": "feature_engineering", "args": {}, "reasoning": "check for useful features"},
    {"action": "run_agent", "agent_id": "profiler", "args": {}, "reasoning": "characterize the dataset"},
    {"action": "run_agent", "agent_id": "split_and_check_leakage", "args": {}, "reasoning": "split the data"},
    {"action": "run_agent", "agent_id": "modeling", "args": {}, "reasoning": "try a candidate"},
    {"action": "run_agent", "agent_id": "verification", "args": {}, "reasoning": "get a second opinion"},
    {"action": "run_agent", "agent_id": "finalize", "args": {}, "reasoning": "lock in the result"},
    {"action": "run_agent", "agent_id": "summarize", "args": {}, "reasoning": "narrate the outcome"},
    {"action": "finish", "agent_id": None, "args": {}, "reasoning": "goal satisfied"},
]


def _resp(text=None, tool_calls=None):
    return ModelResponse(
        text=text, tool_calls=tool_calls or [], raw=None, latency_seconds=0.01,
        model="fake-model", input_tokens=1, output_tokens=1,
    )


_state = {"planner_call_index": 0}


def _reset_state():
    _state["planner_call_index"] = 0


def parity_fake_call(self, messages, model=None, tools=None, temperature=0.0, max_tokens=1024):
    system_content = messages[0]["content"]
    n = len(messages)

    if "planning agent" in system_content:
        if n == 2:
            return _resp(tool_calls=[{"id": "p1", "name": "get_planning_context", "arguments": "{}"}])
        idx = min(_state["planner_call_index"], len(PLANNER_ACTIONS) - 1)
        _state["planner_call_index"] += 1
        return _resp(text=json.dumps(PLANNER_ACTIONS[idx]))

    if "Intake agent" in system_content:
        if n == 2:
            return _resp(tool_calls=[{"id": "t1", "name": "get_raw_schema", "arguments": "{}"}])
        return _resp(text=INTAKE_PROPOSAL)

    if "Feature Engineering agent" in system_content:
        if n == 2:
            return _resp(tool_calls=[{"id": "f1", "name": "get_dataset_profile", "arguments": "{}"}])
        if n == 4:
            return _resp(tool_calls=[{"id": "f2", "name": "list_feature_ops", "arguments": "{}"}])
        return _resp(text=FEATURE_ENGINEERING_PROPOSAL)

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
    return _resp(text="This is a plain-language summary of the dynamic run.")


@pytest.fixture(autouse=True)
def patch_env(monkeypatch):
    monkeypatch.setenv("RIT_BASE_URL", "http://example.invalid/v1")
    monkeypatch.setenv("RIT_API_KEY", "dummy")
    _reset_state()
    yield
    _reset_state()


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


def _run_dynamic_in(tmp_path, argv):
    old_cwd = Path.cwd()
    old_argv = sys.argv
    os.chdir(tmp_path)
    sys.argv = argv
    try:
        run_dynamic_orchestrator.main()
    finally:
        os.chdir(old_cwd)
        sys.argv = old_argv


# --- 1. Parity ---

def test_dynamic_orchestrator_visits_expected_sequence_and_produces_valid_metrics(
    dataset_csv, tmp_path, monkeypatch,
):
    monkeypatch.setattr(ModelClient, "call", parity_fake_call)
    _run_dynamic_in(tmp_path, [
        "run_dynamic_orchestrator.py",
        "--data", str(dataset_csv),
        "--goal", "predict which customers will churn",
        "--run-id", "test_dynamic_parity",
    ])

    report = json.loads(
        (tmp_path / "runs" / "test_dynamic_parity" / "dynamic_orchestrator_report.json").read_text()
    )
    assert report["status"] == "success"

    executed_agent_ids = [h["agent_id"] for h in report["history"] if h.get("executed")]
    assert executed_agent_ids == [
        "intake", "feature_engineering", "profiler", "split_and_check_leakage",
        "modeling", "verification", "finalize", "summarize",
    ]
    assert all(h["ok"] for h in report["history"] if h.get("executed"))

    assert report["final_state"]["target_column"] == "churned"
    assert report["final_state"]["has_verified_candidate"] is True
    assert report["final_state"]["final_test_metrics_present"] is True
    assert set(report["final_test_metrics"]) == {"roc_auc", "pr_auc", "f1", "accuracy"}
    assert 0.0 <= report["final_test_metrics"]["roc_auc"]["value"] <= 1.0
    assert report["summary"]

    leaderboard_path = tmp_path / "artifacts" / "reports" / "leaderboard.jsonl"
    entries = [json.loads(line) for line in leaderboard_path.read_text().splitlines()]
    assert entries[0]["source"] == "dynamic_orchestrator"


# --- 2. Task-routing: explain-only goal skips classification entirely ---

NGAFID_TEST_SENSORS = [
    "volt1", "volt2", "amp1", "amp2", "FQtyL", "FQtyR", "E1 FFlow",
    "E1 OilT", "E1 OilP", "E1 RPM", "E1 CHT1", "E1 CHT2", "E1 CHT3",
    "E1 CHT4", "E1 EGT1", "E1 EGT2", "E1 EGT3", "E1 EGT4", "OAT",
    "IAS", "VSpd", "NormAc", "AltMSL",
]


@pytest.fixture
def aviation_fixtures(tmp_path):
    rng = np.random.RandomState(0)
    n = 30
    raw_rows = {c: rng.normal(100.0, 5.0, n) for c in NGAFID_TEST_SENSORS}
    raw_rows["AltMSL"] = np.linspace(500, 600, n)
    raw_rows["id"] = 5
    raw_df = pd.DataFrame(raw_rows)
    raw_csv = tmp_path / "raw_flights.csv"
    raw_df.to_csv(raw_csv, index=False)

    feature_cols = [f"E1 EGT{i}__mean" for i in range(1, 5)] + [f"E1 CHT{i}__mean" for i in range(1, 5)]
    features_df = pd.DataFrame([{**{c: 1200.0 for c in feature_cols}, "id": 5}])
    features_csv = tmp_path / "features.csv"
    features_df.to_csv(features_csv, index=False)

    train_rows = []
    for i in range(40):
        label = i % 2
        row = {c: rng.normal(1200.0, 5.0) for c in feature_cols}
        if label == 1:
            row["E1 EGT3__mean"] += 80.0
        row["label"] = label
        train_rows.append(row)
    train_tbl = pd.DataFrame(train_rows)
    pipeline = build_pipeline({"numeric_cols": feature_cols, "categorical_cols": [], "seed": 0})
    pipeline.fit(train_tbl[feature_cols], train_tbl["label"])
    background = compute_background(train_tbl, feature_cols, normal_mask=(train_tbl["label"] == 0))

    import joblib
    model_path = tmp_path / "existing_model.joblib"
    joblib.dump({"model": pipeline, "feature_columns": feature_cols, "background": background}, model_path)

    return raw_csv, features_csv, model_path


def routing_fake_call(self, messages, model=None, tools=None, temperature=0.0, max_tokens=1024):
    system_content = messages[0]["content"]
    n = len(messages)

    if "planning agent" in system_content:
        if n == 2:
            return _resp(tool_calls=[{"id": "p1", "name": "get_planning_context", "arguments": "{}"}])
        idx = _state["planner_call_index"]
        _state["planner_call_index"] += 1
        actions = [
            {"action": "run_agent", "agent_id": "deep_dive", "args": {"flight_id": "5"},
             "reasoning": "goal asks to explain flight 5 and a model is already available"},
            {"action": "finish", "agent_id": None, "args": {}, "reasoning": "explanation produced"},
        ]
        return _resp(text=json.dumps(actions[min(idx, len(actions) - 1)]))

    if "maintenance analyst" in system_content:
        if n == 2:
            return _resp(tool_calls=[{"id": "d1", "name": "get_flight_deep_dive_evidence", "arguments": "{}"}])
        return _resp(text=json.dumps({
            "hypothesis": "E1 EGT3 drove the flag.", "agrees_with_localization": None, "confidence": "medium",
        }))

    raise AssertionError(f"unexpected system prompt in routing test: {system_content[:120]!r}")


def test_dynamic_orchestrator_routes_explain_goal_straight_to_deep_dive(
    aviation_fixtures, tmp_path, monkeypatch,
):
    raw_csv, features_csv, model_path = aviation_fixtures
    monkeypatch.setattr(ModelClient, "call", routing_fake_call)

    _run_dynamic_in(tmp_path, [
        "run_dynamic_orchestrator.py",
        "--data", str(features_csv),
        "--goal", "explain why flight 5 was flagged for maintenance",
        "--existing-model", str(model_path),
        "--raw-csv", str(raw_csv),
        "--features-csv", str(features_csv),
        "--run-id", "test_dynamic_routing",
    ])

    report = json.loads(
        (tmp_path / "runs" / "test_dynamic_routing" / "dynamic_orchestrator_report.json").read_text()
    )
    assert report["status"] == "success"

    executed_agent_ids = [h["agent_id"] for h in report["history"] if h.get("executed")]
    # the whole point: classification agents never ran at all
    assert executed_agent_ids == ["deep_dive"]
    assert "5" in report["deep_dive_results"]
    assert report["deep_dive_results"]["5"]["hypothesis"] == "E1 EGT3 drove the flag."
    # --existing-model must mark target_known=True even without --target,
    # or intake looks like a legitimate next step to the planner and (for
    # a feature table with no plausible target column, like this one)
    # burns iterations failing repeatedly instead of going to deep_dive —
    # a real failure mode this exact scenario hit before the fix.
    assert report["final_state"]["target_known"] is True
    # so a real planner can see this flight was already explained and
    # propose "finish" instead of repeating deep_dive on it forever — a
    # real failure mode this exact scenario hit before deep_dive_completed_
    # flight_ids was added (the planner kept re-running deep_dive on flight
    # 5 until max_iterations, since nothing in state signaled it was done).
    assert report["final_state"]["deep_dive_completed_flight_ids"] == ["5"]


def realistic_routing_fake_call(self, messages, model=None, tools=None, temperature=0.0, max_tokens=1024):
    """Unlike routing_fake_call (which hardcodes the 'correct' two-step
    answer regardless of state), this planner stub actually reads
    current_state from the tool result and would propose 'intake' if it
    ever saw target_known=False — reproducing the real behavior a real
    model showed in practice. This is what actually proves the
    --existing-model target_known=True seeding fix works, rather than
    just checking a flag in isolation."""
    system_content = messages[0]["content"]
    n = len(messages)

    if "planning agent" in system_content:
        if n == 2:
            return _resp(tool_calls=[{"id": "p1", "name": "get_planning_context", "arguments": "{}"}])
        return _resp(text=json.dumps({
            "action": "run_agent", "agent_id": "intake", "args": {},
            "reasoning": "target isn't known yet",
        }))
        # deliberately never proposes deep_dive or finish — if target_known
        # seeding is broken, this reveals it immediately via "intake" being
        # (wrongly) proposed instead of aborting/looping harmlessly, rather
        # than a hardcoded stub masking the problem.

    raise AssertionError(f"unexpected system prompt: {system_content[:120]!r}")


def test_existing_model_prevents_planner_from_considering_intake(
    aviation_fixtures, tmp_path, monkeypatch,
):
    raw_csv, features_csv, model_path = aviation_fixtures
    monkeypatch.setattr(ModelClient, "call", realistic_routing_fake_call)

    # this stub planner never proposes "finish", so the run is expected to
    # end via planner_failed (sys.exit(1)) once retries are exhausted —
    # what's under test is that it fails WITHOUT ever running intake, not
    # that the run completes successfully.
    with pytest.raises(SystemExit):
        _run_dynamic_in(tmp_path, [
            "run_dynamic_orchestrator.py",
            "--data", str(features_csv),
            "--goal", "explain why flight 5 was flagged for maintenance",
            "--existing-model", str(model_path),
            "--raw-csv", str(raw_csv),
            "--features-csv", str(features_csv),
            "--run-id", "test_intake_excluded",
            "--max-iterations", "2",
        ])
    report = json.loads(
        (tmp_path / "runs" / "test_intake_excluded" / "dynamic_orchestrator_report.json").read_text()
    )
    # intake requires target_known=False; with the seeding fix, target_known
    # is True, so every "intake" proposal is rejected by validate_plan
    # before it can execute — the planner never gets to actually run it.
    assert all(h.get("agent_id") != "intake" for h in report["history"] if h.get("executed"))
    rejected_reasons = [
        a["errors"] for h in report["history"] for a in h.get("attempts", [])
    ]
    assert any("target_known" in err for errs in rejected_reasons for err in errs)


# --- 3 & 4. Rejections: hallucinated agent_id / precondition violation ---

def _rejecting_planner_fake_call(bad_proposal):
    def fake_call(self, messages, model=None, tools=None, temperature=0.0, max_tokens=1024):
        n = len(messages)
        if n == 2:
            return _resp(tool_calls=[{"id": "p1", "name": "get_planning_context", "arguments": "{}"}])
        idx = _state["planner_call_index"]
        _state["planner_call_index"] += 1
        if idx == 0:
            return _resp(text=json.dumps(bad_proposal))
        return _resp(text=json.dumps({"action": "finish", "agent_id": None, "args": {}, "reasoning": "done"}))
    return fake_call


def test_hallucinated_agent_id_is_rejected_without_executing(dataset_csv, tmp_path, monkeypatch):
    monkeypatch.setattr(ModelClient, "call", _rejecting_planner_fake_call({
        "action": "run_agent", "agent_id": "delete_all_data", "args": {}, "reasoning": "???",
    }))
    _run_dynamic_in(tmp_path, [
        "run_dynamic_orchestrator.py", "--data", str(dataset_csv), "--target", "churned",
        "--run-id", "test_hallucination", "--max-iterations", "3",
    ])
    report = json.loads(
        (tmp_path / "runs" / "test_hallucination" / "dynamic_orchestrator_report.json").read_text()
    )
    assert report["status"] == "success"  # recovered via retry -> finish
    # nothing ever executed: the bad proposal was caught before execution
    assert all(not h.get("executed") for h in report["history"])
    rejected_attempts = report["history"][0]["attempts"]
    assert len(rejected_attempts) == 1
    assert "delete_all_data" in rejected_attempts[0]["errors"][0]
    assert "not a known/available agent" in rejected_attempts[0]["errors"][0]


def test_precondition_violation_is_rejected_without_executing(dataset_csv, tmp_path, monkeypatch):
    monkeypatch.setattr(ModelClient, "call", _rejecting_planner_fake_call({
        "action": "run_agent", "agent_id": "modeling", "args": {},
        "reasoning": "let's just try modeling immediately",
    }))
    _run_dynamic_in(tmp_path, [
        "run_dynamic_orchestrator.py", "--data", str(dataset_csv), "--target", "churned",
        "--run-id", "test_precondition", "--max-iterations", "3",
    ])
    report = json.loads(
        (tmp_path / "runs" / "test_precondition" / "dynamic_orchestrator_report.json").read_text()
    )
    assert report["status"] == "success"
    assert all(not h.get("executed") for h in report["history"])
    rejected_attempts = report["history"][0]["attempts"]
    assert "precondition failed for 'modeling'" in rejected_attempts[0]["errors"][0]
    assert "split_leakage_passed" in rejected_attempts[0]["errors"][0]


# --- failed split must not dead-end the run ---

def _degenerate_group_time_dataset(tmp_path):
    """A dataset whose profile recommends group_time and whose group_time
    split puts only one target class in the test fold — the exact shape
    of the real NGAFID run run_260805_142454_8721: customers 15-19 have
    only churned=0 rows AND the latest earliest-timestamps, so group_time
    assigns them wholesale to the test fold and fold_class_presence fails."""
    rng = np.random.RandomState(0)
    rows = []
    for cust in range(20):
        if cust < 15:
            rows.append({"customer_id": f"C{cust}", "day": 0, "x": rng.normal(), "churned": 1})
            rows.append({"customer_id": f"C{cust}", "day": 1, "x": rng.normal(), "churned": 0})
        else:
            rows.append({"customer_id": f"C{cust}", "day": 5, "x": rng.normal(), "churned": 0})
            rows.append({"customer_id": f"C{cust}", "day": 6, "x": rng.normal(), "churned": 0})
    path = tmp_path / "degenerate.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _failed_split_fake_call(self, messages, model=None, tools=None, temperature=0.0, max_tokens=1024):
    system_content = messages[0]["content"]
    n = len(messages)

    if "planning agent" in system_content:
        if n == 2:
            return _resp(tool_calls=[{"id": "p1", "name": "get_planning_context", "arguments": "{}"}])
        actions = [
            {"action": "run_agent", "agent_id": "feature_engineering", "args": {}, "reasoning": "check features"},
            {"action": "run_agent", "agent_id": "profiler", "args": {}, "reasoning": "characterize"},
            {"action": "run_agent", "agent_id": "split_and_check_leakage", "args": {}, "reasoning": "split"},
            {"action": "run_agent", "agent_id": "split_and_check_leakage", "args": {},
             "reasoning": "split failed its gates; retry"},
            {"action": "finish", "agent_id": None, "args": {},
             "reasoning": "split cannot pass its gates on this data; stopping honestly"},
        ]
        idx = min(_state["planner_call_index"], len(actions) - 1)
        _state["planner_call_index"] += 1
        return _resp(text=json.dumps(actions[idx]))

    if "Feature Engineering agent" in system_content:
        if n == 2:
            return _resp(tool_calls=[{"id": "f1", "name": "get_dataset_profile", "arguments": "{}"}])
        if n == 4:
            return _resp(tool_calls=[{"id": "f2", "name": "list_feature_ops", "arguments": "{}"}])
        return _resp(text=FEATURE_ENGINEERING_PROPOSAL)

    if "Profiler agent" in system_content:
        if n == 2:
            return _resp(tool_calls=[{"id": "t2", "name": "get_dataset_profile", "arguments": "{}"}])
        return _resp(text=PROFILER_NARRATIVE)

    raise AssertionError(f"unexpected system prompt in failed-split test: {system_content[:120]!r}")


def test_failed_split_stays_retryable_and_run_can_finish_instead_of_dead_ending(tmp_path, monkeypatch):
    """Regression for the real NGAFID run that ended planner_failed: the
    split failed a leakage gate but was still marked split_done=True, so
    split_and_check_leakage (requires split_done=False) could never
    re-run AND modeling (requires split_leakage_passed=True) could never
    start — no catalog agent was proposable and the planner burned its
    retries. A failed split must leave split_done=False so the planner
    can retry it or finish gracefully. The gates themselves are
    untouched: both split attempts here still FAIL."""
    monkeypatch.setattr(ModelClient, "call", _failed_split_fake_call)
    data_csv = _degenerate_group_time_dataset(tmp_path)
    _run_dynamic_in(tmp_path, [
        "run_dynamic_orchestrator.py", "--data", str(data_csv), "--target", "churned",
        "--run-id", "test_failed_split", "--max-iterations", "8",
    ])
    report = json.loads(
        (tmp_path / "runs" / "test_failed_split" / "dynamic_orchestrator_report.json").read_text()
    )

    # the run ends by the planner's own choice, not planner_failed
    assert report["status"] == "success"

    executed = [h["agent_id"] for h in report["history"] if h.get("executed")]
    assert executed == [
        "feature_engineering", "profiler",
        "split_and_check_leakage", "split_and_check_leakage",
    ]
    # the second split proposal EXECUTED (was not rejected as a
    # precondition violation) — the dead-end regression itself
    split_entries = [h for h in report["history"] if h.get("agent_id") == "split_and_check_leakage"]
    assert all(not h.get("attempts") for h in split_entries)
    # ...and both attempts genuinely failed their gate: no gate weakened
    assert all(h["ok"] is False for h in split_entries)
    assert any("fold_class_presence" in e for h in split_entries for e in h["errors"])

    assert report["final_state"]["split_done"] is False
    assert report["final_state"]["split_leakage_passed"] is False


# --- 5. Finalize one-shot guard (direct validate_plan unit test) ---

def test_finalize_cannot_run_twice():
    state = RunStateSummary(goal="predict x")
    state.candidates = [CandidateSummary(
        candidate_id="c1", template_id="sklearn_mixed_pipeline", metric_name="roc_auc",
        metric_value=0.9, passed_gate=True, verification_verdict="approved",
    )]
    proposal = {"action": "run_agent", "agent_id": "finalize", "args": {}, "reasoning": "lock it in"}

    # first time: allowed (no prior final_test_metrics)
    assert validate_plan(proposal, state, capabilities=set()) == []

    # simulate finalize having already run
    state.final_test_metrics_present = True
    errors = validate_plan(proposal, state, capabilities=set())
    assert len(errors) == 1
    assert "final_test_metrics_present" in errors[0]


# --- ablation: Dynamic Planner + Verification + Finalize (Phase 1g) ---
# See docs/ablation_study_report.md and scripts/run_ablation_study.py.

def test_ablation_planner_registry_check_disabled_crashes_inside_validate_plan():
    """A hallucinated agent_id is rejected cleanly by default. Disabling
    the registry check doesn't let it through -- get_agent() a few lines
    later raises KeyError inside validate_plan itself."""
    from agentic_ml.ablation import AblationConfig

    state = RunStateSummary(goal="predict x")
    proposal = {"action": "run_agent", "agent_id": "not_a_real_agent", "args": {}, "reasoning": "x"}

    assert validate_plan(proposal, state, capabilities=set()) != []

    with pytest.raises(KeyError):
        validate_plan(proposal, state, capabilities=set(), ablation=AblationConfig(skip_planner_registry_check=True))


def test_ablation_disabling_precondition_check_reproduces_finalize_running_twice():
    """This IS the Finalize one-shot guard -- steps/finalize_step.py has
    no check of its own; the entire guarantee lives in validate_plan's
    required_state precondition. Disabling the precondition check
    silently permits a second 'finalize' proposal after the test set has
    already been touched once."""
    from agentic_ml.ablation import AblationConfig

    state = RunStateSummary(goal="predict x")
    state.candidates = [CandidateSummary(
        candidate_id="c1", template_id="sklearn_mixed_pipeline", metric_name="roc_auc",
        metric_value=0.9, passed_gate=True, verification_verdict="approved",
    )]
    state.final_test_metrics_present = True
    proposal = {"action": "run_agent", "agent_id": "finalize", "args": {}, "reasoning": "lock it in again"}

    assert validate_plan(proposal, state, capabilities=set()) != []
    assert validate_plan(proposal, state, capabilities=set(),
                          ablation=AblationConfig(skip_planner_precondition_check=True)) == []


def test_ablation_verification_gate_status_check_found_a_real_shipped_gap():
    """Not a hypothetical ablation like the rest of this study: this
    check was ADDED by the ablation study after finding it was missing
    entirely. An explicit args={"candidate_id": ...} targeting a
    gate-FAILED candidate bypasses best_unverified_candidate_id()'s
    filter, and nothing downstream re-checked it -- confirmed to let a
    leaky candidate reach the verification LLM and be approved. This
    test locks in the fix; skip_verification_gate_status_check
    reproduces the original bug for comparison."""
    from types import SimpleNamespace

    from agentic_ml.ablation import AblationConfig
    from agentic_ml.orchestrator.dynamic_loop import execute_agent_step

    def make_fixtures():
        passed = SimpleNamespace(
            candidate_id="c_passed", template_id="logistic_numeric", config={"numeric_cols": ["age"]},
            explanation="ok", metrics={"roc_auc": {"value": 0.7}},
            label_permutation_check={"passed": True, "detail": "ok", "check": "x"},
            feature_correlation_check={"passed": True, "detail": "ok", "check": "y"},
            pipeline=None, ok=True,
        )
        failed = SimpleNamespace(
            candidate_id="c_failed", template_id="logistic_numeric", config={"numeric_cols": ["age", "leak"]},
            explanation="uses a leaky column", metrics={"roc_auc": {"value": 0.99}},
            label_permutation_check={"passed": True, "detail": "ok", "check": "x"},
            feature_correlation_check={"passed": False, "detail": "leak corr=1.0", "check": "y"},
            pipeline=None, ok=False,
        )
        state = RunStateSummary(goal="predict x")
        state.candidates = [
            CandidateSummary(candidate_id="c_passed", template_id="logistic_numeric", metric_name="roc_auc",
                              metric_value=0.7, passed_gate=True),
            CandidateSummary(candidate_id="c_failed", template_id="logistic_numeric", metric_name="roc_auc",
                              metric_value=0.99, passed_gate=False),
        ]
        ctx = DynamicRunContext(data_path="x", goal="predict x")
        ctx.modeling_results = {"c_passed": passed, "c_failed": failed}
        ctx.profiler_report = {"is_imbalanced": False, "class_imbalance_ratio": 0.9, "leakage_risk_flags": []}
        ctx.run_id = "test"
        ctx.loaded = SimpleNamespace(data_hash="fakehash")
        return state, ctx

    class ApprovesEverythingClient:
        def call(self, messages, model=None, tools=None, temperature=0.0, max_tokens=1024):
            return ModelResponse(
                text=json.dumps({"verdict": "approved", "concerns": [], "reasoning": "fine"}),
                tool_calls=[], raw=None, latency_seconds=0.0, model="fake", input_tokens=0, output_tokens=0,
            )

    # fixed behavior (default): blocked before ever reaching the LLM
    state, ctx = make_fixtures()
    ok, errors = execute_agent_step(
        "verification", {"candidate_id": "c_failed"}, ctx, state, ApprovesEverythingClient(),
        None, None, None, lambda name, msgs: None, on_event=None,
    )
    assert ok is False
    assert "failed its harness gates" in errors[0]
    failed_summary = next(c for c in state.candidates if c.candidate_id == "c_failed")
    assert failed_summary.verification_verdict is None

    # ablated: reproduces the original bug
    state2, ctx2 = make_fixtures()
    ok2, errors2 = execute_agent_step(
        "verification", {"candidate_id": "c_failed"}, ctx2, state2, ApprovesEverythingClient(),
        None, None, None, lambda name, msgs: None, on_event=None,
        ablation=AblationConfig(skip_verification_gate_status_check=True),
    )
    failed_summary2 = next(c for c in state2.candidates if c.candidate_id == "c_failed")
    assert failed_summary2.verification_verdict == "approved", "reproduces the exact pre-fix bug"


# --- normalize_proposal: repairs a real schema confusion without relaxing validation ---

def test_normalize_proposal_repairs_agent_name_in_action_field():
    proposal = {"action": "finalize", "args": {}, "reasoning": "done modeling, wrap up"}
    normalized = normalize_proposal(proposal)
    assert normalized == {
        "action": "run_agent", "agent_id": "finalize", "args": {}, "reasoning": "done modeling, wrap up",
    }


def test_normalize_proposal_repairs_even_when_agent_id_redundantly_set():
    # the real failure mode observed in evaluation: the model set BOTH
    # action='finalize' AND agent_id='finalize', not just one or the other.
    proposal = {"action": "finalize", "agent_id": "finalize", "args": {}, "reasoning": "wrap up"}
    normalized = normalize_proposal(proposal)
    assert normalized == {
        "action": "run_agent", "agent_id": "finalize", "args": {}, "reasoning": "wrap up",
    }


def test_normalize_proposal_leaves_well_formed_proposals_alone():
    proposal = {"action": "run_agent", "agent_id": "modeling", "args": {}, "reasoning": "try a candidate"}
    assert normalize_proposal(proposal) == proposal

    finish = {"action": "finish", "agent_id": None, "args": {}, "reasoning": "done"}
    assert normalize_proposal(finish) == finish


def test_normalize_proposal_does_not_repair_a_genuinely_unknown_action():
    # "delete_everything" isn't a real agent_id, so this must NOT be
    # silently rewritten into something that could pass validation —
    # normalization only fixes an unambiguous, known-safe confusion.
    proposal = {"action": "delete_everything", "args": {}, "reasoning": "???"}
    assert normalize_proposal(proposal) == proposal
    state = RunStateSummary(goal="predict x")
    assert validate_plan(normalize_proposal(proposal), state, capabilities=set()) != []


# --- 6. max_iterations guard ---

def test_loop_stops_at_max_iterations_when_planner_never_finishes(monkeypatch):
    def never_finishes(self, messages, model=None, tools=None, temperature=0.0, max_tokens=1024):
        n = len(messages)
        if n == 2:
            return _resp(tool_calls=[{"id": "p1", "name": "get_planning_context", "arguments": "{}"}])
        return _resp(text=json.dumps({
            "action": "run_agent", "agent_id": "modeling", "args": {}, "reasoning": "always try again",
        }))

    monkeypatch.setattr(ModelClient, "call", never_finishes)
    # bypass real modeling execution — this test is about the LOOP's stopping
    # condition, not modeling itself, which stays fully covered by test 1.
    monkeypatch.setattr(dynamic_loop_module, "execute_agent_step", lambda *a, **k: (True, []))

    client = ModelClient(base_url="http://example.invalid/v1", api_key="dummy", default_model="fake-model")
    ctx = DynamicRunContext(data_path="unused.csv", goal="predict x", run_id="test_max_iter")
    state = RunStateSummary(goal="predict x")
    state.split_leakage_passed = True  # so "modeling" is always a valid proposal

    result = run_dynamic_loop(ctx, state, client, model="fake-model", max_iterations=4)

    assert result.status == "max_iterations_reached"
    assert len([h for h in result.history if h.get("executed")]) == 4


# --- 8. Long-format time-series auto-routing ---

def test_long_format_precondition_blocks_intake_and_feature_engineering():
    """Direct validate_plan unit test — the state-gating mechanism in
    isolation. A real end-to-end proof follows below."""
    state = RunStateSummary(goal="classify which flights need maintenance")
    state.looks_long_format = True  # featurization_done defaults False -> data_ready False

    intake_errors = validate_plan(
        {"action": "run_agent", "agent_id": "intake", "args": {}}, state, capabilities=set(),
    )
    assert any("data_ready" in e for e in intake_errors)

    # target_known=True simulates the --target-given entry point, which
    # bypasses intake entirely — feature_engineering needs its own gate.
    state.target_known = True
    fe_errors = validate_plan(
        {"action": "run_agent", "agent_id": "feature_engineering", "args": {}}, state, capabilities=set(),
    )
    assert any("data_ready" in e for e in fe_errors)

    # featurize_timeseries itself is unaffected — that's the one agent this
    # state is supposed to allow.
    featurize_errors = validate_plan(
        {"action": "run_agent", "agent_id": "featurize_timeseries", "args": {}}, state, capabilities=set(),
    )
    assert featurize_errors == []

    # once featurization has run, both gates lift with no other change.
    state.featurization_done = True
    assert validate_plan(
        {"action": "run_agent", "agent_id": "feature_engineering", "args": {}}, state, capabilities=set(),
    ) == []


NGAFID_SENSORS_TEST = [
    "volt1", "volt2", "amp1", "amp2", "FQtyL", "FQtyR", "E1 FFlow",
    "E1 OilT", "E1 OilP", "E1 RPM", "E1 CHT1", "E1 CHT2", "E1 CHT3",
    "E1 CHT4", "E1 EGT1", "E1 EGT2", "E1 EGT3", "E1 EGT4", "OAT",
    "IAS", "VSpd", "NormAc", "AltMSL",
]


@pytest.fixture
def long_format_dataset_csv(tmp_path):
    """A small raw long-format CSV shaped exactly like NGAFID's real
    columns (agentic_ml.domain.aviation.ngafid_config) — small enough to
    be fast, but structurally identical to what build_flight_feature_table_streaming
    expects, so featurize_timeseries's real (non-stubbed) execution runs
    against it unmodified. before_after is independent random noise
    (matching dataset_csv's churned column above) — the leakage gates
    check for suspiciously-too-good validation behavior, not for the
    presence of a real learnable signal, so this is deliberately the
    same proven-safe pattern already used for the Titanic parity test
    above rather than a hand-tuned planted effect size."""
    rng = np.random.RandomState(0)
    n_flights, steps_per_flight = 24, 15
    rows = []
    for fid in range(n_flights):
        plane = f"plane_{fid % 8}"
        label = "before" if rng.random() < 0.5 else "post"
        for _ in range(steps_per_flight):
            row = {c: rng.normal(100.0, 5.0) for c in NGAFID_SENSORS_TEST}
            row.update({"id": fid, "plane_id": plane, "before_after": label,
                       "date_diff": fid, "split": "train"})
            rows.append(row)
    path = tmp_path / "long_format.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


LONG_FORMAT_INTAKE_PROPOSAL = json.dumps({
    "target_column": "before_after", "task": "binary_classification",
    "id_columns": ["id", "date_diff", "split"], "group_column": "plane_id", "time_column": None,
    "positive_label": "1", "reasoning": "before_after is the binary outcome; plane_id repeats per flight.",
})
LONG_FORMAT_CANDIDATE = json.dumps({
    "candidate_id": "candidate_a", "template_id": "logistic_numeric",
    "config": {"numeric_cols": ["volt1__mean", "OAT__mean"]},
    "explanation": "Cheap numeric-only baseline.",
})

LONG_FORMAT_PLANNER_ACTIONS = [
    {"action": "run_agent", "agent_id": "featurize_timeseries", "args": {},
     "reasoning": "raw data is long-format — must roll up before anything else can see it"},
    {"action": "run_agent", "agent_id": "intake", "args": {}, "reasoning": "need a target"},
    {"action": "run_agent", "agent_id": "feature_engineering", "args": {}, "reasoning": "check for useful features"},
    {"action": "run_agent", "agent_id": "profiler", "args": {}, "reasoning": "characterize the dataset"},
    {"action": "run_agent", "agent_id": "split_and_check_leakage", "args": {}, "reasoning": "split the data"},
    {"action": "run_agent", "agent_id": "modeling", "args": {}, "reasoning": "try a candidate"},
    {"action": "run_agent", "agent_id": "verification", "args": {}, "reasoning": "get a second opinion"},
    {"action": "run_agent", "agent_id": "finalize", "args": {}, "reasoning": "lock in the result"},
    {"action": "run_agent", "agent_id": "summarize", "args": {}, "reasoning": "narrate the outcome"},
    {"action": "finish", "agent_id": None, "args": {}, "reasoning": "goal satisfied"},
]


def long_format_fake_call(self, messages, model=None, tools=None, temperature=0.0, max_tokens=1024):
    system_content = messages[0]["content"]
    n = len(messages)

    if "planning agent" in system_content:
        if n == 2:
            return _resp(tool_calls=[{"id": "p1", "name": "get_planning_context", "arguments": "{}"}])
        idx = min(_state["planner_call_index"], len(LONG_FORMAT_PLANNER_ACTIONS) - 1)
        _state["planner_call_index"] += 1
        return _resp(text=json.dumps(LONG_FORMAT_PLANNER_ACTIONS[idx]))

    if "Intake agent" in system_content:
        if n == 2:
            return _resp(tool_calls=[{"id": "t1", "name": "get_raw_schema", "arguments": "{}"}])
        return _resp(text=LONG_FORMAT_INTAKE_PROPOSAL)

    if "Feature Engineering agent" in system_content:
        if n == 2:
            return _resp(tool_calls=[{"id": "f1", "name": "get_dataset_profile", "arguments": "{}"}])
        if n == 4:
            return _resp(tool_calls=[{"id": "f2", "name": "list_feature_ops", "arguments": "{}"}])
        return _resp(text=FEATURE_ENGINEERING_PROPOSAL)

    if "Profiler agent" in system_content:
        if n == 2:
            return _resp(tool_calls=[{"id": "t2", "name": "get_dataset_profile", "arguments": "{}"}])
        return _resp(text=PROFILER_NARRATIVE)

    if "Modeling agent" in system_content:
        if n == 2:
            return _resp(tool_calls=[{"id": "t3", "name": "get_dataset_profile", "arguments": "{}"}])
        if n == 4:
            return _resp(tool_calls=[{"id": "t4", "name": "list_templates", "arguments": "{}"}])
        return _resp(text=LONG_FORMAT_CANDIDATE)

    if "Verification agent" in system_content:
        if n == 2:
            return _resp(tool_calls=[{"id": "v1", "name": "get_candidate_review_bundle", "arguments": "{}"}])
        return _resp(text=VERIFICATION_APPROVED)

    return _resp(text="This is a plain-language summary of the long-format run.")


def test_dynamic_loop_routes_long_format_dataset_through_featurize_timeseries(
    long_format_dataset_csv, tmp_path, monkeypatch,
):
    # featurize_timeseries writes its rolled-up output under
    # datasets_root()/processed/ (cwd-relative by default, paths.py) —
    # isolate that from the real repo the same way every other test
    # touching runs/datasets does (see test_events.py, test_mcp_facts.py).
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ModelClient, "call", long_format_fake_call)
    client = ModelClient(base_url="http://example.invalid/v1", api_key="dummy", default_model="fake-model")

    ctx = DynamicRunContext(
        data_path=str(long_format_dataset_csv), goal="classify which flights need maintenance",
        run_id="test_long_format_routing",
    )
    state = RunStateSummary(goal=ctx.goal)
    # Set directly rather than via detect_dataset_shape — that function has
    # its own dedicated tests (tests/test_harness.py); this test is about
    # what happens once long-format is known, not about detecting it.
    state.looks_long_format = True
    assert ctx.raw_df is None  # never loaded — the whole point

    result = run_dynamic_loop(ctx, state, client, model="fake-model", verification_model="fake-model", max_iterations=15)

    assert result.status == "success"
    executed_agent_ids = [h["agent_id"] for h in result.history if h.get("executed")]
    assert executed_agent_ids == [
        "featurize_timeseries", "intake", "feature_engineering", "profiler",
        "split_and_check_leakage", "modeling", "verification", "finalize", "summarize",
    ]
    assert all(h["ok"] for h in result.history if h.get("executed"))

    # featurize_timeseries actually populated ctx from the raw file — not
    # a no-op, not stubbed.
    assert state.featurization_done is True
    assert state.featurization_summary["n_examples"] == 24
    assert ctx.raw_csv == str(long_format_dataset_csv)
    assert ctx.features_csv is not None and Path(ctx.features_csv).exists()
    assert "volt1__mean" in ctx.raw_df.columns

    assert state.target_column == "before_after"
    assert state.final_test_metrics_present is True
    assert 0.0 <= ctx.final_test_metrics["roc_auc"]["value"] <= 1.0


# --- 9. Retry-with-feedback: a rejected proposal doesn't repeat forever ---
#
# Real incident: feature_engineering proposed dropping a column intake
# had misclassified as the (protected) time_column, got rejected, and —
# because every call was a fresh conversation with no memory of the
# rejection, at ModelClient's default temperature=0.0 — proposed the
# EXACT SAME thing again on every subsequent iteration, 14 times, until
# max_iterations. This reproduces that shape hermetically: a stubbed
# feature_engineering agent that only stops repeating the bad proposal
# once it actually sees the rejection reason in its own prompt — proving
# dynamic_loop.py's previous_error wiring works end to end, not just
# that the parameter exists.

def _fe_retry_dataset_csv(tmp_path):
    rng = np.random.RandomState(0)
    n = 100
    df = pd.DataFrame({
        "customer_id": [f"C{i}" for i in range(n)],
        "signup_day": [f"day_{i}" for i in range(n)],  # stands in for the misclassified time_column
        "age": rng.randint(18, 80, size=n),
        "churned": rng.binomial(1, 0.35, size=n),
    })
    path = tmp_path / "fe_retry.csv"
    df.to_csv(path, index=False)
    return path


def test_feature_engineering_retry_sees_previous_rejection_reason_and_recovers(tmp_path, monkeypatch):
    dataset_csv = _fe_retry_dataset_csv(tmp_path)
    call_count = {"feature_engineering_final": 0}

    def fake_call(self, messages, model=None, tools=None, temperature=0.0, max_tokens=1024):
        system_content = messages[0]["content"]
        n = len(messages)

        if "planning agent" in system_content:
            if n == 2:
                return _resp(tool_calls=[{"id": "p1", "name": "get_planning_context", "arguments": "{}"}])
            idx = min(_state["planner_call_index"], 2)
            _state["planner_call_index"] += 1
            actions = [
                {"action": "run_agent", "agent_id": "feature_engineering", "args": {}, "reasoning": "check features"},
                {"action": "run_agent", "agent_id": "feature_engineering", "args": {}, "reasoning": "retry"},
                {"action": "finish", "agent_id": None, "args": {}, "reasoning": "done"},
            ]
            return _resp(text=json.dumps(actions[idx]))

        if "Feature Engineering agent" in system_content:
            if n == 2:
                return _resp(tool_calls=[{"id": "f1", "name": "get_dataset_profile", "arguments": "{}"}])
            if n == 4:
                return _resp(tool_calls=[{"id": "f2", "name": "list_feature_ops", "arguments": "{}"}])
            # Final text-response turn: only stop proposing the rejected
            # drop once the retry feedback is actually present in the
            # conversation — a stand-in for a real model reacting to it.
            call_count["feature_engineering_final"] += 1
            user_message = messages[1]["content"]
            if "Your previous proposal was rejected" in user_message:
                return _resp(text=json.dumps(
                    {"drop_columns": [], "derived_features": [], "explanation": "leaving signup_day alone now"},
                ))
            return _resp(text=json.dumps({
                "drop_columns": ["signup_day"], "derived_features": [],
                "explanation": "looks uninformative",
            }))

        raise AssertionError(f"unexpected system prompt: {system_content[:120]!r}")

    monkeypatch.setattr(ModelClient, "call", fake_call)
    client = ModelClient(base_url="http://example.invalid/v1", api_key="dummy", default_model="fake-model")

    ctx = DynamicRunContext(
        data_path=str(dataset_csv), goal="predict churn", seed=42,
        target_column="churned", time_column="signup_day",  # mirrors intake's real misclassification
        run_id="test_fe_retry",
    )
    ctx.raw_df = pd.read_csv(dataset_csv)
    ctx.engineered_df = ctx.raw_df  # what a real run has by the time feature_engineering runs
    state = RunStateSummary(goal=ctx.goal)
    state.target_known = True
    state.target_column = "churned"

    result = run_dynamic_loop(ctx, state, client, model="fake-model", max_iterations=6)

    assert result.status == "success"
    executed = [h["agent_id"] for h in result.history if h.get("executed")]
    assert executed.count("feature_engineering") == 2  # one rejection, one recovery — not stuck
    assert state.feature_engineering_done is True
    # the rejection really was surfaced back to the agent, not just
    # coincidentally succeeding a second time
    assert call_count["feature_engineering_final"] == 2


def test_feature_engineering_degrades_to_zero_changes_if_model_never_self_corrects(tmp_path, monkeypatch):
    """The stronger guarantee retry-with-feedback alone can't provide: even
    a model that NEVER reacts to previous_error — proposing the exact same
    rejected drop every single time, exactly what the real incident showed
    — cannot leave the run stuck. dynamic_loop.py's
    MAX_FEATURE_ENGINEERING_ATTEMPTS bound takes over and deterministically
    finishes feature_engineering with no changes, the same "legitimate
    outcome" this agent is already allowed to propose itself."""
    dataset_csv = _fe_retry_dataset_csv(tmp_path)
    call_count = {"feature_engineering_final": 0}

    def fake_call(self, messages, model=None, tools=None, temperature=0.0, max_tokens=1024):
        system_content = messages[0]["content"]
        n = len(messages)

        if "planning agent" in system_content:
            if n == 2:
                return _resp(tool_calls=[{"id": "p1", "name": "get_planning_context", "arguments": "{}"}])
            # always propose feature_engineering again — a real planner
            # would too, since nothing in state changes until it succeeds
            return _resp(text=json.dumps(
                {"action": "run_agent", "agent_id": "feature_engineering", "args": {}, "reasoning": "retry"},
            ))

        if "Feature Engineering agent" in system_content:
            if n == 2:
                return _resp(tool_calls=[{"id": "f1", "name": "get_dataset_profile", "arguments": "{}"}])
            if n == 4:
                return _resp(tool_calls=[{"id": "f2", "name": "list_feature_ops", "arguments": "{}"}])
            call_count["feature_engineering_final"] += 1
            # NEVER reacts to previous_error — always the same rejected proposal.
            return _resp(text=json.dumps({
                "drop_columns": ["signup_day"], "derived_features": [],
                "explanation": "looks uninformative",
            }))

        raise AssertionError(f"unexpected system prompt: {system_content[:120]!r}")

    monkeypatch.setattr(ModelClient, "call", fake_call)
    client = ModelClient(base_url="http://example.invalid/v1", api_key="dummy", default_model="fake-model")

    ctx = DynamicRunContext(
        data_path=str(dataset_csv), goal="predict churn", seed=42,
        target_column="churned", time_column="signup_day",
        run_id="test_fe_degrade",
    )
    ctx.raw_df = pd.read_csv(dataset_csv)
    ctx.engineered_df = ctx.raw_df
    state = RunStateSummary(goal=ctx.goal)
    state.target_known = True
    state.target_column = "churned"

    # max_iterations well above what degradation needs, so a status of
    # "max_iterations_reached" would mean degradation DIDN'T kick in, not
    # that the test under-budgeted iterations.
    result = run_dynamic_loop(ctx, state, client, model="fake-model", max_iterations=10)

    executed = [h["agent_id"] for h in result.history if h.get("executed")]
    # MAX_FEATURE_ENGINEERING_ATTEMPTS real (rejected) LLM attempts, plus
    # one more execution that's the deterministic degraded success — no
    # LLM call for that last one, which is exactly what the separate
    # call_count assertion below confirms.
    assert executed.count("feature_engineering") == dynamic_loop_module.MAX_FEATURE_ENGINEERING_ATTEMPTS + 1
    assert state.feature_engineering_done is True
    assert result.status != "max_iterations_reached"
    # the model really did keep proposing the same rejected thing for
    # every REAL attempt — this proves degradation, not a lucky model
    # recovery, and that the degraded execution skipped the LLM entirely.
    assert call_count["feature_engineering_final"] == dynamic_loop_module.MAX_FEATURE_ENGINEERING_ATTEMPTS
    # no drops were actually applied — a real "zero changes" outcome
    assert list(ctx.engineered_df.columns) == list(ctx.raw_df.columns)
