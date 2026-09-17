"""
The MCP fact server (resource_scheduler.mcp_facts): a standardized,
stdio-reachable replacement for the in-process Tool closures in
tools/*.py, once a stage's LLM call moves out of an in-process
ToolCallingAgent and into a real sandbox AgentSpec agent step. Unlike
the sibling agentic-ml-classification project's equivalent test file,
there is no LocalToolProvider/McpToolProvider split to prove parity
between here -- resource_scheduler's sandbox port only ever has one
tool-serving path (see gate_adapters.py's module docstring), so these
tests only need to prove:

1. fact_store round-trips a payload and raises a clean, typed error for
   a fact that was never written.
2. The server serves what was persisted, over a real MCP session (no
   network -- mcp.shared.memory's in-process transport), and
   `enabled_tools` in config actually removes a tool from what's
   registered.
"""
from __future__ import annotations

import asyncio

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from resource_scheduler.mcp_facts.fact_store import FactNotFoundError, read_fact, write_fact
from resource_scheduler.mcp_facts.server import ALL_TOOL_NAMES, build_server


@pytest.fixture(autouse=True)
def isolated_runs_dir(tmp_path, monkeypatch):
    # fact_store resolves runs/<run_id>/facts/ via paths.run_dir(), which
    # is cwd-relative by default -- same isolation pattern every other
    # test touching runs/ already uses.
    monkeypatch.chdir(tmp_path)


def _call_tool(server, name: str, arguments: dict) -> dict:
    async def _call():
        async with create_connected_server_and_client_session(server) as session:
            result = await session.call_tool(name, arguments)
        if result.isError:
            text = "; ".join(getattr(block, "text", str(block)) for block in result.content)
            raise RuntimeError(text or "MCP tool call failed with no error detail")
        if result.structuredContent is not None:
            return result.structuredContent
        import json
        for block in result.content:
            if getattr(block, "type", None) == "text":
                return json.loads(block.text)
        raise RuntimeError("MCP tool call returned no content")

    return asyncio.run(_call())


# --- fact_store ---


def test_fact_store_round_trips_a_payload():
    payload = {"pending_tasks": [{"task_id": "T1"}]}
    write_fact("run_a", "task_queue_profile", payload)
    assert read_fact("run_a", "task_queue_profile") == payload


def test_fact_store_raises_typed_error_for_missing_fact():
    with pytest.raises(FactNotFoundError):
        read_fact("run_a", "task_queue_profile")


def test_fact_store_is_scoped_per_run_id():
    write_fact("run_a", "task_queue_profile", {"pending_tasks": [1]})
    write_fact("run_b", "task_queue_profile", {"pending_tasks": [2]})
    assert read_fact("run_a", "task_queue_profile") == {"pending_tasks": [1]}
    assert read_fact("run_b", "task_queue_profile") == {"pending_tasks": [2]}


# --- server behavior ---


def test_server_serves_a_fact_that_was_written():
    write_fact("run_c", "resource_snapshot", {"flags": []})
    server = build_server()
    assert _call_tool(server, "get_resource_snapshot", {"run_id": "run_c"}) == {"flags": []}


def test_server_returns_structured_error_for_unknown_run_id_not_a_crash():
    server = build_server()
    with pytest.raises(RuntimeError, match="does-not-exist"):
        _call_tool(server, "get_resource_snapshot", {"run_id": "does-not-exist"})


def test_disabled_tool_is_not_registered():
    server = build_server({"enabled_tools": ["get_resource_snapshot"]})
    with pytest.raises(RuntimeError):
        _call_tool(server, "get_task_queue_profile", {"run_id": "irrelevant"})
    # the enabled one still works (once a fact exists for it)
    write_fact("run_d", "resource_snapshot", {"flags": []})
    assert _call_tool(server, "get_resource_snapshot", {"run_id": "run_d"}) == {"flags": []}


def test_build_server_rejects_unknown_tool_name_in_config():
    with pytest.raises(ValueError, match="not_a_real_tool"):
        build_server({"enabled_tools": ["not_a_real_tool"]})


def test_all_seven_tools_are_registered_by_default():
    assert set(ALL_TOOL_NAMES) == {
        "get_resource_snapshot", "get_task_queue_profile", "get_allocation_context",
        "get_incident_report", "get_policy_evidence", "get_policy_review_bundle",
        "get_decision_review_bundle",
    }
