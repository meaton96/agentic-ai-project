import pytest
from mcp.types import CallToolResult, TextContent
from mcp.types import Tool as McpTool

from sandbox_core.runtime.mcp_client import ConnectedMcpServer, _filter_tools, _openai_tool_schema
from sandbox_core.schemas.agent_spec import McpServerBinding

RUN_ID = "run-1"
AGENT_ID = "agent-1"


def make_binding(**overrides) -> McpServerBinding:
    fields = dict(name="fs", transport="stdio", connection={"command": "mcp-server-fs"})
    fields.update(overrides)
    return McpServerBinding(**fields)


def make_tool(name: str) -> McpTool:
    return McpTool(name=name, description=f"does {name}", input_schema={"type": "object", "properties": {}})


class FakeSession:
    def __init__(self, result: CallToolResult | None = None, error: Exception | None = None):
        self.result = result
        self.error = error
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name, args):
        self.calls.append((name, args))
        if self.error is not None:
            raise self.error
        return self.result


# --- _filter_tools ---


def test_filter_tools_returns_all_when_allowlist_is_none():
    tools = [make_tool("write_file"), make_tool("read_file"), make_tool("delete_file")]
    assert _filter_tools(tools, None) == tools


def test_filter_tools_keeps_only_allowed_names():
    tools = [make_tool("write_file"), make_tool("read_file"), make_tool("delete_file")]
    filtered = _filter_tools(tools, ["write_file", "read_file"])
    assert {t.name for t in filtered} == {"write_file", "read_file"}


# --- tool schema conversion ---


def test_openai_tool_schema_shape():
    schema = _openai_tool_schema(make_tool("write_file"))
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "write_file"
    assert schema["function"]["parameters"] == {"type": "object", "properties": {}}


# --- ConnectedMcpServer.call_tool ---


@pytest.mark.asyncio
async def test_call_tool_happy_path_returns_raw_text_and_emits_events():
    session = FakeSession(result=CallToolResult(content=[TextContent(type="text", text="wrote 12 bytes")], is_error=False))
    server = ConnectedMcpServer("fs", make_binding(allowed_tools=["write_file"]), session, [make_tool("write_file")])

    call_event, result_event, raw_value = await server.call_tool(
        tool="write_file", args={"path": "out.txt"}, call_id="call-1", run_id=RUN_ID, agent_id=AGENT_ID
    )

    assert call_event.tool == "write_file"
    assert call_event.server == "fs"
    assert result_event.error is None
    assert result_event.result == "wrote 12 bytes"
    assert raw_value == "wrote 12 bytes"
    assert session.calls == [("write_file", {"path": "out.txt"})]


@pytest.mark.asyncio
async def test_call_tool_refuses_disallowed_tool_without_calling_server():
    session = FakeSession(result=CallToolResult(content=[TextContent(type="text", text="should not run")], is_error=False))
    # allowed_tools=["read_file"] means write_file was already filtered out of the tools this server was built with
    server = ConnectedMcpServer("fs", make_binding(allowed_tools=["read_file"]), session, [make_tool("read_file")])

    call_event, result_event, raw_value = await server.call_tool(
        tool="write_file", args={"path": "out.txt"}, call_id="call-1", run_id=RUN_ID, agent_id=AGENT_ID
    )

    assert result_event.error is not None
    assert "write_file" in result_event.error
    assert session.calls == []  # never forwarded to the underlying MCP server


@pytest.mark.asyncio
async def test_call_tool_hashed_policy_redacts_log_but_not_returned_value():
    session = FakeSession(result=CallToolResult(content=[TextContent(type="text", text="super-secret-contents")], is_error=False))
    server = ConnectedMcpServer(
        "fs", make_binding(allowed_tools=["read_file"], logging_policy="hashed"), session, [make_tool("read_file")]
    )

    call_event, result_event, raw_value = await server.call_tool(
        tool="read_file", args={"path": "secret.txt"}, call_id="call-1", run_id=RUN_ID, agent_id=AGENT_ID
    )

    # raw value returned to the agent/model is unredacted
    assert raw_value == "super-secret-contents"
    # what's persisted to the event log is not
    assert "super-secret-contents" not in str(result_event.result)
    assert set(result_event.result.keys()) == {"sha256", "byte_size", "shape"}


@pytest.mark.asyncio
async def test_call_tool_session_exception_becomes_error_event_not_raised():
    session = FakeSession(error=RuntimeError("transport dropped"))
    server = ConnectedMcpServer("fs", make_binding(allowed_tools=["read_file"]), session, [make_tool("read_file")])

    call_event, result_event, raw_value = await server.call_tool(
        tool="read_file", args={"path": "x.txt"}, call_id="call-1", run_id=RUN_ID, agent_id=AGENT_ID
    )

    assert "transport dropped" in result_event.error
