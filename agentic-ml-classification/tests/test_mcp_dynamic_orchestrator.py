"""
End-to-end proof that routing agent tool facts through the MCP fact
server (agentic_ml.mcp_facts) produces the SAME outcome as the default
in-process path (tests/test_dynamic_orchestrator.py's parity test) —
same scripted planner sequence, same stubbed ModelClient, same
synthetic dataset, run through run_dynamic_loop directly with an
McpToolProvider (in-memory transport — no network, no real model)
instead of the default LocalToolProvider.

This is the two-birds test the milestone needs: it proves adoption is
truly opt-in (nothing here uses --use-mcp or touches the CLI's default
path) AND that the MCP path is a real substitute, not just a parallel
code path that happens to also run — intake through summarize all
route through get_raw_schema/get_dataset_profile/list_templates/etc.
served over an actual (in-memory) MCP session, and the harness-side
facts persisted along the way are readable back off disk afterward.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from agentic_ml.harness.dataset import read_dataframe
from agentic_ml.mcp_facts.fact_store import read_fact
from agentic_ml.mcp_facts.provider import McpToolProvider
from agentic_ml.mcp_facts.server import build_server
from agentic_ml.mcp_facts.transport import InMemoryMcpTransport
from agentic_ml.model_client import ModelClient, ModelResponse
from agentic_ml.orchestrator.dynamic_loop import run_dynamic_loop
from agentic_ml.orchestrator.run_state import DynamicRunContext, RunStateSummary

# --- identical scripted responses to test_dynamic_orchestrator.py's parity test ---

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


@pytest.fixture(autouse=True)
def reset_state():
    _state["planner_call_index"] = 0
    yield
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


def test_dynamic_loop_via_mcp_provider_matches_local_provider_outcome(
    dataset_csv, tmp_path, monkeypatch,
):
    monkeypatch.chdir(tmp_path)  # so runs/<run_id>/facts/ lands under tmp_path
    monkeypatch.setattr(ModelClient, "call", parity_fake_call)

    client = ModelClient(base_url="http://example.invalid/v1", api_key="dummy", default_model="fake-model")
    run_id = "test_mcp_parity"
    tool_provider = McpToolProvider(run_id, InMemoryMcpTransport(build_server()))

    ctx = DynamicRunContext(data_path=str(dataset_csv), goal="predict which customers will churn", run_id=run_id)
    ctx.raw_df = read_dataframe(str(dataset_csv))
    state = RunStateSummary(goal=ctx.goal)

    result = run_dynamic_loop(
        ctx, state, client, model="fake-model", verification_model="fake-model",
        max_iterations=15, tool_provider=tool_provider,
    )

    assert result.status == "success"
    executed_agent_ids = [h["agent_id"] for h in result.history if h.get("executed")]
    assert executed_agent_ids == [
        "intake", "feature_engineering", "profiler", "split_and_check_leakage",
        "modeling", "verification", "finalize", "summarize",
    ]
    assert all(h["ok"] for h in result.history if h.get("executed"))

    assert state.target_column == "churned"
    assert state.has_verified_candidate is True
    assert state.final_test_metrics_present is True
    assert 0.0 <= ctx.final_test_metrics["roc_auc"]["value"] <= 1.0
    assert ctx.summary_text

    # the facts an MCP-served agent actually saw are a durable, audit-visible
    # artifact under runs/<run_id>/facts/ — not just something that happened
    # to flow through a socket and vanish.
    assert read_fact(run_id, "raw_schema")["n_columns"] == 6
    assert read_fact(run_id, "dataset_profile")["recommended_split_strategy"]
    assert read_fact(run_id, "review_bundle")["candidate_id"] == "candidate_a"
    assert read_fact(run_id, "planning_context")["goal"] == ctx.goal
