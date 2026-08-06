from __future__ import annotations

from agentic_ml.mcp_facts.server import ALL_TOOL_NAMES


def test_mcp_config_reflects_the_real_pipeline_config_file(client):
    """Regression guard against configs/mcp_server.json (pipeline repo) or
    agentic_ml.mcp_facts.server drifting out from under this route: every
    tool it registers should show up here, and enabled/disabled should
    match that file's enabled_tools list — read live, not hand-copied."""
    resp = client.get("/api/mcp/config")
    assert resp.status_code == 200
    body = resp.json()

    assert body["config_exists"] is True
    assert body["name"]
    assert body["host"]
    assert isinstance(body["port"], int)
    assert body["url"] == f"http://{body['host']}:{body['port']}/mcp"

    names = {t["name"] for t in body["tools"]}
    assert names == set(ALL_TOOL_NAMES)

    enabled_names = {t["name"] for t in body["tools"] if t["enabled"]}
    assert enabled_names == set(body["enabled_tools"])

    for tool in body["tools"]:
        assert tool["description"]
        assert "properties" in tool["input_schema"]


def test_mcp_config_tool_has_expected_schema_shape(client):
    resp = client.get("/api/mcp/config")
    tools = {t["name"]: t for t in resp.json()["tools"]}
    raw_schema_tool = tools["get_raw_schema"]
    assert raw_schema_tool["input_schema"]["required"] == ["run_id"]


def test_mcp_config_reports_unreachable_when_nothing_listens(client):
    # Nothing in this test suite ever starts scripts/run_mcp_server.py, so
    # the configured port should consistently read as not listening.
    resp = client.get("/api/mcp/config")
    assert resp.json()["reachable"] is False
