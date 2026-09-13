"""
Additions to the MCP fact server (mcp_facts/server.py) beyond what
tests/test_mcp_facts.py covers:

1. get_modeling_attempts has a defined empty state: before the first
   modeling attempt it answers "nothing tried yet" rather than erroring,
   while every other missing fact still comes back as a structured error.
2. build_http_app's bearer-token check actually gates the streamable-HTTP
   endpoint (a missing or wrong token is a 401 before any MCP handling;
   the right one completes a real MCP initialize), and the app refuses to
   build for a non-loopback host with no token at all.
"""
from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from agentic_ml.mcp_facts.fact_store import write_fact
from agentic_ml.mcp_facts.server import AUTH_TOKEN_ENV, build_http_app, build_server
from agentic_ml.mcp_facts.transport import InMemoryMcpTransport, McpToolError

_BASE_URL = "http://127.0.0.1:8765"  # FastMCP's DNS-rebinding guard only admits loopback Host headers
_INITIALIZE = {
    "jsonrpc": "2.0", "id": 1, "method": "initialize",
    "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "test", "version": "0"}},
}
_MCP_HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}


@pytest.fixture(autouse=True)
def isolated_runs_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(AUTH_TOKEN_ENV, raising=False)
    for var in ("AGENTIC_ML_DATA_ROOT", "AGENTIC_ML_RUNS_DIR"):
        monkeypatch.delenv(var, raising=False)


# --- 1. get_modeling_attempts ---

def test_get_modeling_attempts_is_empty_before_any_attempt():
    transport = InMemoryMcpTransport(build_server())
    assert transport.call_tool("get_modeling_attempts", {"run_id": "fresh_run"}) == {
        "tried_template_ids": [], "rejections": [], "attempts": [],
    }


def test_get_modeling_attempts_serves_the_recorded_fact():
    recorded = {
        "tried_template_ids": ["logistic_numeric"],
        "rejections": [{"attempt_index": 0, "candidate_id": "c0", "template_id": "logistic_numeric",
                        "stage": "modeling", "reason": "failed label_permutation_test leakage gate"}],
        "attempts": [{"attempt_index": 0, "candidate_id": "c0", "template_id": "logistic_numeric",
                      "accepted": False, "reason": "failed label_permutation_test leakage gate"}],
    }
    write_fact("run_x", "modeling_attempts", recorded)
    assert InMemoryMcpTransport(build_server()).call_tool("get_modeling_attempts", {"run_id": "run_x"}) == recorded


def test_default_does_not_leak_to_facts_that_have_no_empty_state():
    with pytest.raises(McpToolError):
        InMemoryMcpTransport(build_server()).call_tool("get_candidate_review_bundle", {"run_id": "fresh_run"})


# --- 2. bearer auth ---

@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"}, {"Authorization": "s3cret"}])
def test_http_app_rejects_missing_or_wrong_token(headers):
    client = TestClient(build_http_app(auth_token="s3cret"), base_url=_BASE_URL)
    response = client.post("/mcp", json=_INITIALIZE, headers={**_MCP_HEADERS, **headers})
    assert response.status_code == 401


def test_http_app_serves_mcp_with_the_right_token():
    with TestClient(build_http_app(auth_token="s3cret"), base_url=_BASE_URL) as client:
        response = client.post("/mcp", json=_INITIALIZE, headers={**_MCP_HEADERS, "Authorization": "Bearer s3cret"})
    assert response.status_code == 200


def test_http_app_reads_token_from_env(monkeypatch):
    monkeypatch.setenv(AUTH_TOKEN_ENV, "from-env")
    client = TestClient(build_http_app(), base_url=_BASE_URL)
    assert client.post("/mcp", json=_INITIALIZE, headers=_MCP_HEADERS).status_code == 401


def test_non_loopback_host_requires_a_token():
    with pytest.raises(ValueError, match=AUTH_TOKEN_ENV):
        build_http_app({"host": "0.0.0.0"})
    build_http_app({"host": "0.0.0.0"}, auth_token="s3cret")  # builds fine once a token is set


def test_loopback_host_without_token_still_serves_for_local_use():
    with TestClient(build_http_app(), base_url=_BASE_URL) as client:
        assert client.post("/mcp", json=_INITIALIZE, headers=_MCP_HEADERS).status_code == 200
