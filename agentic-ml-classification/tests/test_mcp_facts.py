"""
The MCP fact server (agentic_ml.mcp_facts): a standardized, network-
reachable replacement for the in-process Tool closures in tools/*.py,
built as a strict "fact server" — the harness computes every fact
exactly as it always has, persists it as JSON under
runs/<run_id>/facts/, and the server only ever reads that directory
back (plus two static, stateless registries). No dataframe, fitted
pipeline, or raw file path crosses the process boundary.

Four things worth proving, not just exercising:

1. fact_store round-trips a payload and raises a clean, typed error
   for a fact that was never written — never a bare FileNotFoundError
   or a crash.
2. The server serves what was persisted, over a real MCP session (no
   network — mcp.shared.memory's in-process transport), rejects an
   unknown run_id with a structured MCP error instead of crashing, and
   `enabled_tools` in config actually removes a tool from what's
   registered.
3. list_templates/list_feature_ops over MCP can't drift from the
   local registries, because the server computes them from the exact
   same functions tools/*.py's local factories call.
4. Provider parity: for every one of the 8 tool factories,
   LocalToolProvider and McpToolProvider (in-memory transport) return
   a Tool with the identical name/description/schema, and calling the
   handler returns the identical payload for the same inputs. This is
   the test that would fail if the MCP path ever served a different
   fact than agents see today — see mcp_facts/provider.py's docstring.
5. transport.call_tool() works when called from a thread that already
   has an asyncio event loop running — e.g. Jupyter/ipykernel, which
   runs its own loop in the main thread. A real bug (anyio.run() raises
   "Already running asyncio in this thread" in exactly that case) was
   found running notebooks/dynamic_orchestrator.ipynb with --use-mcp;
   this reproduces it hermetically via asyncio.run() instead of an
   actual notebook.
"""
from __future__ import annotations

import asyncio

import pandas as pd
import pytest

from agentic_ml.harness.feature_engineering import list_feature_ops
from agentic_ml.harness.profiler import profile_dataset
from agentic_ml.mcp_facts.fact_store import FactNotFoundError, read_fact, write_fact
from agentic_ml.mcp_facts.provider import LocalToolProvider, McpToolProvider
from agentic_ml.mcp_facts.server import ALL_TOOL_NAMES, build_server
from agentic_ml.mcp_facts.transport import InMemoryMcpTransport, McpToolError
from agentic_ml.templates.registry import list_template_summaries


@pytest.fixture(autouse=True)
def isolated_runs_dir(tmp_path, monkeypatch):
    # fact_store resolves runs/<run_id>/facts/ via paths.run_dir(), which
    # is cwd-relative by default — same isolation pattern every other test
    # touching runs/ already uses (see test_events.py, test_dynamic_orchestrator.py).
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "age": [25, 40, 33, 51, 29],
        "plan": ["basic", "pro", "basic", "premium", "pro"],
        "churned": [0, 1, 0, 1, 0],
    })


# --- 1. fact_store ---

def test_fact_store_round_trips_a_payload():
    payload = {"n_rows": 10, "columns": [{"name": "a", "dtype": "int64"}]}
    write_fact("run_a", "dataset_profile", payload)
    assert read_fact("run_a", "dataset_profile") == payload


def test_fact_store_raises_typed_error_for_missing_fact():
    with pytest.raises(FactNotFoundError):
        read_fact("run_a", "dataset_profile")


def test_fact_store_is_scoped_per_run_id():
    write_fact("run_a", "dataset_profile", {"n_rows": 1})
    write_fact("run_b", "dataset_profile", {"n_rows": 2})
    assert read_fact("run_a", "dataset_profile") == {"n_rows": 1}
    assert read_fact("run_b", "dataset_profile") == {"n_rows": 2}


# --- 2. server behavior ---

def test_server_serves_a_fact_that_was_written():
    write_fact("run_c", "dataset_profile", {"n_rows": 5})
    server = build_server()
    transport = InMemoryMcpTransport(server)
    assert transport.call_tool("get_dataset_profile", {"run_id": "run_c"}) == {"n_rows": 5}


def test_server_returns_structured_error_for_unknown_run_id_not_a_crash():
    server = build_server()
    transport = InMemoryMcpTransport(server)
    with pytest.raises(McpToolError, match="does-not-exist"):
        transport.call_tool("get_dataset_profile", {"run_id": "does-not-exist"})


def test_disabled_tool_is_not_registered():
    server = build_server({"enabled_tools": ["list_templates"]})
    transport = InMemoryMcpTransport(server)
    with pytest.raises(McpToolError):
        transport.call_tool("get_dataset_profile", {"run_id": "irrelevant"})
    # the enabled one still works
    assert "templates" in transport.call_tool("list_templates", {})


def test_build_server_rejects_unknown_tool_name_in_config():
    with pytest.raises(ValueError, match="not_a_real_tool"):
        build_server({"enabled_tools": ["not_a_real_tool"]})


def test_call_tool_works_from_inside_an_already_running_event_loop():
    # anyio.run() (used naively) refuses to start a second event loop in a
    # thread that already has one running — which is exactly Jupyter's
    # setup, since ipykernel runs its own asyncio loop in the main thread.
    # Wrapping the whole test body in asyncio.run() reproduces that
    # "there's already a running loop here" condition without needing an
    # actual notebook.
    write_fact("run_loop", "dataset_profile", {"n_rows": 3})
    server = build_server()
    transport = InMemoryMcpTransport(server)

    async def call_from_within_a_running_loop():
        return transport.call_tool("get_dataset_profile", {"run_id": "run_loop"})

    assert asyncio.run(call_from_within_a_running_loop()) == {"n_rows": 3}


# --- 3. registry parity ---

def test_list_templates_over_mcp_matches_local_registry():
    server = build_server()
    transport = InMemoryMcpTransport(server)
    over_mcp = transport.call_tool("list_templates", {})
    assert over_mcp == {"templates": list_template_summaries()}


def test_list_feature_ops_over_mcp_matches_local_registry():
    server = build_server()
    transport = InMemoryMcpTransport(server)
    over_mcp = transport.call_tool("list_feature_ops", {})
    assert over_mcp == {"feature_ops": list_feature_ops()}


# --- 4. provider parity: the design-claim test ---

def _server_transport():
    return InMemoryMcpTransport(build_server())


def test_provider_parity_raw_schema(sample_df):
    local = LocalToolProvider().make_raw_schema_tool(sample_df)
    mcp = McpToolProvider("run_parity", _server_transport()).make_raw_schema_tool(sample_df)
    assert (local.name, local.description, local.parameters) == (mcp.name, mcp.description, mcp.parameters)
    assert local.handler() == mcp.handler()


def test_provider_parity_dataset_profile(sample_df):
    local = LocalToolProvider().make_profiler_tool(sample_df, "churned")
    mcp = McpToolProvider("run_parity", _server_transport()).make_profiler_tool(sample_df, "churned")
    assert (local.name, local.description, local.parameters) == (mcp.name, mcp.description, mcp.parameters)
    assert local.handler() == mcp.handler()
    # the served fact is build_profile_fact — profile_dataset's report plus
    # the declared-role annotations (declared_group_column /
    # declared_time_column / excluded_columns); everything the profiler
    # computed must still be present verbatim underneath.
    payload = local.handler()
    expected_report = profile_dataset(sample_df, target_column="churned").to_dict()
    assert {k: v for k, v in payload.items() if k in expected_report} == expected_report
    assert payload["declared_group_column"] is None
    assert payload["declared_time_column"] is None
    assert "churned" in payload["excluded_columns"]


def test_provider_parity_dataset_profile_with_declared_columns(sample_df):
    """Both providers must serve identical declared-role annotations —
    parity breaking exactly on the do-not-use column list is the failure
    mode that burned 8 modeling iterations on a real NGAFID run (the
    agent could not see 'date_diff' was the declared time column and
    proposed it over and over, each time correctly rejected)."""
    df = sample_df.assign(customer_id=["C1", "C1", "C2", "C2", "C3"], day=[1, 2, 1, 2, 1])
    args = (df, "churned", "customer_id", "day")
    local = LocalToolProvider().make_profiler_tool(*args)
    mcp = McpToolProvider("run_parity_declared", _server_transport()).make_profiler_tool(*args)
    payload = local.handler()
    assert payload == mcp.handler()
    assert payload["declared_group_column"] == "customer_id"
    assert payload["declared_time_column"] == "day"
    assert {"churned", "customer_id", "day"} <= set(payload["excluded_columns"])


def test_provider_parity_list_templates():
    local = LocalToolProvider().make_list_templates_tool()
    mcp = McpToolProvider("run_parity", _server_transport()).make_list_templates_tool()
    assert (local.name, local.description, local.parameters) == (mcp.name, mcp.description, mcp.parameters)
    assert local.handler() == mcp.handler()


def test_provider_parity_list_feature_ops():
    local = LocalToolProvider().make_list_feature_ops_tool()
    mcp = McpToolProvider("run_parity", _server_transport()).make_list_feature_ops_tool()
    assert (local.name, local.description, local.parameters) == (mcp.name, mcp.description, mcp.parameters)
    assert local.handler() == mcp.handler()


def test_provider_parity_review_bundle():
    bundle = {"candidate_id": "c1", "metrics": {"roc_auc": {"value": 0.9}}}
    local = LocalToolProvider().make_review_bundle_tool(bundle)
    mcp = McpToolProvider("run_parity", _server_transport()).make_review_bundle_tool(bundle)
    assert (local.name, local.description, local.parameters) == (mcp.name, mcp.description, mcp.parameters)
    assert local.handler() == mcp.handler() == bundle


def test_provider_parity_monitoring_context():
    context = {"drift_summary": {"mean_abs_shift": 0.1}, "model_version": 1}
    local = LocalToolProvider().make_monitoring_context_tool(context)
    mcp = McpToolProvider("run_parity", _server_transport()).make_monitoring_context_tool(context)
    assert (local.name, local.description, local.parameters) == (mcp.name, mcp.description, mcp.parameters)
    assert local.handler() == mcp.handler() == context


def test_provider_parity_planning_context():
    args = ("predict churn", {"target_known": True}, [{"agent_id": "profiler"}], 2, 15)
    local = LocalToolProvider().make_planning_context_tool(*args)
    mcp = McpToolProvider("run_parity", _server_transport()).make_planning_context_tool(*args)
    assert (local.name, local.description, local.parameters) == (mcp.name, mcp.description, mcp.parameters)
    assert local.handler() == mcp.handler()


def test_provider_parity_deep_dive_evidence():
    import numpy as np
    from agentic_ml.harness.attribution import compute_background
    from agentic_ml.templates.sources.xgboost_mixed import build_pipeline

    rng = np.random.RandomState(0)
    feature_cols = ["a", "b", "c", "d"]
    rows = []
    for i in range(30):
        label = i % 2
        row = {c: rng.normal(0.0, 1.0) for c in feature_cols}
        if label == 1:
            row["a"] += 5.0
        row["label"] = label
        rows.append(row)
    table = pd.DataFrame(rows)
    pipeline = build_pipeline({"numeric_cols": feature_cols, "categorical_cols": [], "seed": 0})
    pipeline.fit(table[feature_cols], table["label"])
    background = compute_background(table, feature_cols, normal_mask=(table["label"] == 0))
    feature_row = table[table["label"] == 1].iloc[0][feature_cols]

    # minimal flight_df: enough for segment_flight to run; no EGT/CHT columns,
    # so localize_anomaly finds nothing — same proven shape as
    # tests/test_deep_dive.py's deep_dive_fixture.
    flight_df = pd.DataFrame({"AltMSL": np.full(30, 500.0), "IAS": np.zeros(30)})

    local = LocalToolProvider().make_deep_dive_evidence_tool(
        flight_df, feature_row, pipeline, feature_cols, background,
    )
    mcp = McpToolProvider("run_parity", _server_transport()).make_deep_dive_evidence_tool(
        flight_df, feature_row, pipeline, feature_cols, background, flight_id="5",
    )
    assert (local.name, local.description, local.parameters) == (mcp.name, mcp.description, mcp.parameters)
    assert local.handler() == mcp.handler()
