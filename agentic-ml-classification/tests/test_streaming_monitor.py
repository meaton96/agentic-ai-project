"""
Orchestrator-level (stubbed-client) scenarios for Phase 9. Complements
tests/test_streaming.py's hermetic harness-module tests by exercising
the whole loop: scripts/run_streaming_monitor.py driving repeated
run_dynamic_loop calls over a cold start plus several simulated
batches.

What's proven here that a harness-only test can't show:
1. A low-drift batch -> retrain_decision picks "infer_only", no
   retrain happens, model_version stays unchanged.
2. A high-drift batch -> "retrain": the FULL classification
   sub-sequence (feature_engineering -> profiler ->
   split_and_check_leakage -> modeling -> verification -> finalize)
   re-executes end to end and model_version increments.
3. TWO separate retrain cycles in the same session both complete
   (not just one) — proving the reset logic in
   orchestrator/dynamic_loop.py's retrain_decision branch, not merely
   that a single retrain works. finalize's pre-existing one-shot guard
   (final_test_metrics_present=False) would block a second finalize
   forever if the reset didn't correctly flip it back to False.
4. An unparseable retrain_decision response degrades to "infer_only"
   (unit-level, on steps/retrain_decision_step.py directly).
5. The streaming-only catalog entries are invisible to an ordinary
   (non-streaming) dynamic-orchestrator run's planner.
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
from agentic_ml.orchestrator.agent_registry import list_agent_summaries
from agentic_ml.harness.streaming import simulate_batches
from agentic_ml.steps.retrain_decision_step import run_retrain_decision_step

import run_streaming_monitor  # noqa: E402  (needs the sys.path insert above first)

FEATURE_ENGINEERING_PROPOSAL = json.dumps({
    "drop_columns": [], "derived_features": [], "explanation": "No changes needed.",
})
PROFILER_NARRATIVE = json.dumps({
    "summary": "Synthetic streaming dataset.", "recommended_split_strategy": "group",
    "key_risks": [], "recommended_next_steps": [],
})
CANDIDATE_A = json.dumps({
    "candidate_id": "candidate_a", "template_id": "sklearn_mixed_pipeline",
    "config": {"numeric_cols": ["feat_a", "feat_b"], "categorical_cols": [], "classifier": "logistic_regression"},
    "explanation": "Mixed baseline.",
})
VERIFICATION_APPROVED = json.dumps({"verdict": "approved", "concerns": [], "reasoning": "looks fine"})

# --- Phase 9 dataset geometry: shared between the fixture and the test ---
N_INITIAL_GROUPS = 20
BATCH_SIZE_GROUPS = 5
N_EXTRA_GROUPS = 15  # -> exactly 3 batches of 5
SEED = 42
ROWS_PER_PLANE = 3


def _resp(text=None, tool_calls=None):
    return ModelResponse(
        text=text, tool_calls=tool_calls or [], raw=None, latency_seconds=0.01,
        model="fake-model", input_tokens=1, output_tokens=1,
    )


def _build_neutral_df(n_planes: int, seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    rows = []
    row_id = 0
    for plane in range(n_planes):
        for _ in range(ROWS_PER_PLANE):
            rows.append({
                "id": row_id, "plane_id": plane,
                "feat_a": rng.normal(0, 1), "feat_b": rng.normal(0, 1),
                "label": rng.binomial(1, 0.5),
            })
            row_id += 1
    return pd.DataFrame(rows)


@pytest.fixture
def streaming_dataset_csv(tmp_path):
    """Builds a full flight-level table where the SAME group-shuffle the
    script will perform (identical group_column/id_column/seed/sizes)
    determines which plane_ids fall in each batch — then injects a huge
    shift into batch 1's planes on feat_a, and batch 2's planes on
    feat_b (a DIFFERENT feature), so each batch is independently
    drift-triggering. This has to be done in two passes: group
    membership doesn't depend on feature values, so a neutral first
    pass discovers membership, and a second pass injects the shift
    before writing the real CSV.

    Shifting different features per batch (rather than the same one
    twice) matters: once batch 1 triggers a retrain, its shifted feat_a
    values become part of the new baseline — a second feat_a shift of
    the same size would no longer look like drift relative to the
    ALREADY-ADAPTED model. Shifting feat_b instead proves the second
    retrain is a genuinely independent decision, not a lucky repeat."""
    n_total = N_INITIAL_GROUPS + N_EXTRA_GROUPS
    neutral_df = _build_neutral_df(n_total, seed=0)

    _, neutral_batches = simulate_batches(
        neutral_df, group_column="plane_id", id_column="id",
        n_initial_groups=N_INITIAL_GROUPS, batch_size_groups=BATCH_SIZE_GROUPS, seed=SEED,
    )
    assert len(neutral_batches) == 3
    low_drift_planes = set(neutral_batches[0]["plane_id"])
    high_drift_planes_1 = set(neutral_batches[1]["plane_id"])
    high_drift_planes_2 = set(neutral_batches[2]["plane_id"])

    full_df = neutral_df.copy()
    rng = np.random.RandomState(1)
    mask1 = full_df["plane_id"].isin(high_drift_planes_1)
    full_df.loc[mask1, "feat_a"] = rng.normal(12.0, 1.0, size=mask1.sum())
    mask2 = full_df["plane_id"].isin(high_drift_planes_2)
    full_df.loc[mask2, "feat_b"] = rng.normal(12.0, 1.0, size=mask2.sum())

    path = tmp_path / "streaming_full.csv"
    full_df.to_csv(path, index=False)
    return path, low_drift_planes, high_drift_planes_1 | high_drift_planes_2


def make_smart_stream_call():
    """A planner stub driven entirely by the CURRENT run state (read back
    from the tool result each turn) rather than a hardcoded action index —
    the same discipline test_dynamic_orchestrator.py's
    realistic_routing_fake_call uses, generalized to walk BOTH the
    classification sequence and the streaming monitoring cycle, however
    many times either needs to repeat across a whole session."""

    def fake_call(self, messages, model=None, tools=None, temperature=0.0, max_tokens=1024):
        system_content = messages[0]["content"]
        n = len(messages)

        if "planning agent" in system_content:
            if n == 2:
                return _resp(tool_calls=[{"id": "p1", "name": "get_planning_context", "arguments": "{}"}])
            tool_msg = next(m for m in messages if m.get("role") == "tool")
            state = json.loads(tool_msg["content"])["current_state"]

            if not state["feature_engineering_done"]:
                action = {"action": "run_agent", "agent_id": "feature_engineering", "args": {}}
            elif not state["profiler_done"]:
                action = {"action": "run_agent", "agent_id": "profiler", "args": {}}
            elif not state["split_done"]:
                action = {"action": "run_agent", "agent_id": "split_and_check_leakage", "args": {}}
            elif not state["has_verified_candidate"] and not state["has_unverified_passing_candidate"]:
                action = {"action": "run_agent", "agent_id": "modeling", "args": {}}
            elif state["has_unverified_passing_candidate"]:
                action = {"action": "run_agent", "agent_id": "verification", "args": {}}
            elif not state["final_test_metrics_present"]:
                action = {"action": "run_agent", "agent_id": "finalize", "args": {}}
            elif state["new_batch_pending"] and not state["drift_checked"]:
                action = {"action": "run_agent", "agent_id": "monitor_drift", "args": {}}
            elif state["drift_checked"] and state["pending_retrain_action"] is None:
                action = {"action": "run_agent", "agent_id": "retrain_decision", "args": {}}
            elif state["pending_retrain_action"] == "infer_only" and not state["batch_action_completed"]:
                action = {"action": "run_agent", "agent_id": "infer_batch", "args": {}}
            else:
                action = {"action": "finish", "agent_id": None, "args": {}}
            action["reasoning"] = "deterministic test planner"
            return _resp(text=json.dumps(action))

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

        if "Retrain-Decision agent" in system_content:
            if n == 2:
                return _resp(tool_calls=[{"id": "r1", "name": "get_monitoring_context", "arguments": "{}"}])
            tool_msg = next(m for m in messages if m.get("role") == "tool")
            monitoring_context = json.loads(tool_msg["content"])
            drift = monitoring_context.get("drift_summary") or {}
            mean_shift = drift.get("mean_abs_shift", 0.0)
            action = "retrain" if mean_shift > 3.0 else "infer_only"
            return _resp(text=json.dumps({"action": action, "reasoning": f"mean_abs_shift={mean_shift}"}))

        raise AssertionError(f"unexpected system prompt in streaming test: {system_content[:150]!r}")

    return fake_call


def _run_streaming_in(tmp_path, argv):
    old_cwd = Path.cwd()
    old_argv = sys.argv
    os.chdir(tmp_path)
    sys.argv = argv
    try:
        run_streaming_monitor.main()
    finally:
        os.chdir(old_cwd)
        sys.argv = old_argv


def test_streaming_monitor_low_drift_infers_only_and_high_drift_retrains_twice(
    streaming_dataset_csv, tmp_path, monkeypatch,
):
    data_path, low_drift_planes, high_drift_planes = streaming_dataset_csv
    monkeypatch.setenv("RIT_BASE_URL", "http://example.invalid/v1")
    monkeypatch.setenv("RIT_API_KEY", "dummy")
    monkeypatch.setattr(ModelClient, "call", make_smart_stream_call())

    _run_streaming_in(tmp_path, [
        "run_streaming_monitor.py",
        "--data", str(data_path), "--target", "label", "--group-column", "plane_id",
        "--id-columns", "id,plane_id",
        "--n-initial-groups", str(N_INITIAL_GROUPS), "--batch-size-groups", str(BATCH_SIZE_GROUPS),
        "--seed", str(SEED), "--strategy", "group",
        "--max-iterations-per-batch", "12",
        "--run-id", "test_streaming_basic",
    ])

    report = json.loads(
        (tmp_path / "runs" / "test_streaming_basic" / "streaming_monitor_report.json").read_text()
    )
    log_lines = [
        json.loads(line)
        for line in (tmp_path / "runs" / "test_streaming_basic" / "streaming_log.jsonl").read_text().splitlines()
    ]
    assert len(log_lines) == 3

    # batch 0: low drift -> infer_only, no retrain, model_version unchanged
    b0 = log_lines[0]
    assert b0["decision"] == "infer_only"
    assert b0["retrained"] is False
    assert b0["model_version_after"] == b0["model_version_before"] == 1

    # batch 1: high drift -> retrain, full classification cycle re-runs, version bumps
    b1 = log_lines[1]
    assert b1["decision"] == "retrain"
    assert b1["retrained"] is True
    assert b1["model_version_before"] == 1
    assert b1["model_version_after"] == 2
    assert b1["test_metrics"] is not None

    # batch 2: ALSO high drift -> a SECOND retrain cycle in the same session.
    # This is the proof that matters: finalize's one-shot guard
    # (final_test_metrics_present=False) must have been correctly reset by
    # the first retrain, or this second finalize could never execute.
    b2 = log_lines[2]
    assert b2["decision"] == "retrain"
    assert b2["retrained"] is True
    assert b2["model_version_before"] == 2
    assert b2["model_version_after"] == 3
    assert b2["test_metrics"] is not None

    assert report["final_model_version"] == 3
    assert len(report["model_history"]) == 3  # cold start + 2 retrains
    assert [m["version"] for m in report["model_history"]] == [1, 2, 3]


# --- Unit: unparseable retrain_decision response degrades to infer_only ---

def test_retrain_decision_step_degrades_to_infer_only_on_unparseable_response(monkeypatch):
    monkeypatch.setenv("RIT_BASE_URL", "http://example.invalid/v1")
    monkeypatch.setenv("RIT_API_KEY", "dummy")

    def bad_call(self, messages, model=None, tools=None, temperature=0.0, max_tokens=1024):
        n = len(messages)
        if n == 2:
            return _resp(tool_calls=[{"id": "r1", "name": "get_monitoring_context", "arguments": "{}"}])
        return _resp(text="not valid json at all")

    monkeypatch.setattr(ModelClient, "call", bad_call)
    client = ModelClient(base_url="http://example.invalid/v1", api_key="dummy", default_model="fake-model")
    result = run_retrain_decision_step({"drift_summary": {"mean_abs_shift": 5.0}}, client, model="fake-model")

    assert result.ok is False
    assert result.action == "infer_only"


def test_retrain_decision_step_degrades_to_infer_only_on_invalid_action(monkeypatch):
    monkeypatch.setenv("RIT_BASE_URL", "http://example.invalid/v1")
    monkeypatch.setenv("RIT_API_KEY", "dummy")

    def bad_action_call(self, messages, model=None, tools=None, temperature=0.0, max_tokens=1024):
        n = len(messages)
        if n == 2:
            return _resp(tool_calls=[{"id": "r1", "name": "get_monitoring_context", "arguments": "{}"}])
        return _resp(text=json.dumps({"action": "delete_everything", "reasoning": "???"}))

    monkeypatch.setattr(ModelClient, "call", bad_action_call)
    client = ModelClient(base_url="http://example.invalid/v1", api_key="dummy", default_model="fake-model")
    result = run_retrain_decision_step({"drift_summary": {}}, client, model="fake-model")

    assert result.ok is False
    assert result.action == "infer_only"


# --- Capability gating: streaming-only agents invisible to ordinary runs ---

def test_streaming_agents_hidden_without_streaming_capability():
    ordinary_ids = {a["agent_id"] for a in list_agent_summaries(capabilities=set())}
    assert "monitor_drift" not in ordinary_ids
    assert "retrain_decision" not in ordinary_ids
    assert "infer_batch" not in ordinary_ids

    streaming_ids = {a["agent_id"] for a in list_agent_summaries(capabilities={"streaming"})}
    assert {"monitor_drift", "retrain_decision", "infer_batch"} <= streaming_ids
