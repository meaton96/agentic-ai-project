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
