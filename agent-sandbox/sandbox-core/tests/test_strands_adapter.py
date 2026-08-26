from strands.tools.mcp import MCPClient

from sandbox_core.runtime.strands_adapter import (
    _auth_headers,
    _flatten_tool_result_content,
    _mcp_client_for,
    _stdio_env,
)
from sandbox_core.schemas.agent_spec import McpServerBinding


def make_binding(**overrides) -> McpServerBinding:
    fields = dict(name="fs", transport="stdio", connection={"command": "mcp-server-fs"})
    fields.update(overrides)
    return McpServerBinding(**fields)


# --- _auth_headers / _stdio_env: credential injection for http/sse vs stdio ---


def test_auth_headers_adds_bearer_token_when_credential_present():
    binding = make_binding(transport="http", connection={"url": "http://x"})
    assert _auth_headers(binding, "secret") == {"Authorization": "Bearer secret"}


def test_auth_headers_no_credential_no_authorization_header():
    binding = make_binding(transport="http", connection={"url": "http://x"})
    assert _auth_headers(binding, None) == {}


def test_auth_headers_does_not_override_an_explicit_header():
    binding = make_binding(transport="http", connection={"url": "http://x", "headers": {"Authorization": "Basic abc"}})
    assert _auth_headers(binding, "secret") == {"Authorization": "Basic abc"}


def test_stdio_env_uses_default_var_name():
    binding = make_binding(connection={"command": "mcp-server-fs"})
    assert _stdio_env(binding, "secret") == {"MCP_TOKEN": "secret"}


def test_stdio_env_respects_custom_credential_env_var():
    binding = make_binding(connection={"command": "mcp-server-fs", "credential_env_var": "FS_TOKEN"})
    assert _stdio_env(binding, "secret") == {"FS_TOKEN": "secret"}


def test_stdio_env_none_when_no_env_and_no_credential():
    binding = make_binding(connection={"command": "mcp-server-fs"})
    assert _stdio_env(binding, None) is None


# --- _mcp_client_for: one MCPClient per transport, credential-aware ---


class FakeResolver:
    def resolve(self, ref: str) -> str:
        return f"resolved-{ref}"


def test_mcp_client_for_http_transport_builds_a_client():
    binding = make_binding(transport="http", connection={"url": "http://x"}, credential_ref="tok")
    client = _mcp_client_for(binding, FakeResolver())
    assert isinstance(client, MCPClient)


def test_mcp_client_for_stdio_transport_builds_a_client():
    binding = make_binding(transport="stdio", connection={"command": "mcp-server-fs"})
    client = _mcp_client_for(binding, None)
    assert isinstance(client, MCPClient)


def test_mcp_client_for_unknown_transport_raises():
    binding = make_binding().model_copy(update={"transport": "carrier-pigeon"})
    try:
        _mcp_client_for(binding, None)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "carrier-pigeon" in str(exc)


def test_mcp_client_for_stdio_missing_command_raises_a_clear_error_not_a_bare_keyerror():
    # empty connection JSON — e.g. AgentForm's "Connection (JSON)" field left
    # at its default `{}` — used to raise a bare KeyError('command') with no
    # indication of what to fix.
    binding = make_binding(transport="stdio", connection={})
    try:
        _mcp_client_for(binding, None)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "command" in str(exc)
        assert "fs" in str(exc)


def test_mcp_client_for_http_missing_url_raises_a_clear_error_not_a_bare_keyerror():
    binding = make_binding(transport="http", connection={})
    try:
        _mcp_client_for(binding, None)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "url" in str(exc)
        assert "fs" in str(exc)


# --- _flatten_tool_result_content: mirrors the old mcp_client._flatten_content ---


def test_flatten_single_text_block_returns_bare_string():
    assert _flatten_tool_result_content([{"text": "wrote 12 bytes"}]) == "wrote 12 bytes"


def test_flatten_single_json_block_returns_bare_value():
    assert _flatten_tool_result_content([{"json": {"bytes": 12}}]) == {"bytes": 12}


def test_flatten_multiple_blocks_returns_list():
    result = _flatten_tool_result_content([{"text": "a"}, {"text": "b"}])
    assert result == ["a", "b"]


def test_flatten_empty_content_returns_empty_list():
    assert _flatten_tool_result_content([]) == []
