import time
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncIterator

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.types import Tool as McpTool

from sandbox_core.schemas.agent_spec import McpServerBinding
from sandbox_core.schemas.credentials import CredentialResolver
from sandbox_core.schemas.events import ToolCallEvent, ToolResultEvent, redact_result


def _openai_tool_schema(tool: McpTool) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.input_schema or {"type": "object", "properties": {}},
        },
    }


def _flatten_content(content: list) -> Any:
    """Collapses an MCP CallToolResult.content list into a plain value for the
    agent/event log — a bare string for the common single-text-block case,
    a list of parsed blocks otherwise."""
    parts = []
    for block in content:
        text = getattr(block, "text", None)
        parts.append(text if text is not None else block.model_dump(mode="json"))
    if len(parts) == 1:
        return parts[0]
    return parts


def _auth_headers(binding: McpServerBinding, credential: str | None) -> dict:
    headers = dict(binding.connection.get("headers", {}))
    if credential is not None:
        headers.setdefault("Authorization", f"Bearer {credential}")
    return headers


def _stdio_env(binding: McpServerBinding, credential: str | None) -> dict | None:
    env = dict(binding.connection.get("env", {})) or None
    if credential is not None:
        var_name = binding.connection.get("credential_env_var", "MCP_TOKEN")
        env = {**(env or {}), var_name: credential}
    return env


class ConnectedMcpServer:
    """One live MCP session plus the (allowlist-filtered) tools it exposes."""

    def __init__(self, name: str, binding: McpServerBinding, session: ClientSession, tools: list[McpTool]):
        self.name = name
        self.binding = binding
        self.session = session
        self._tools_by_name = {tool.name: tool for tool in tools}

    @property
    def tool_specs(self) -> list[dict]:
        return [_openai_tool_schema(tool) for tool in self._tools_by_name.values()]

    def has_tool(self, name: str) -> bool:
        return name in self._tools_by_name

    async def call_tool(
        self,
        *,
        tool: str,
        args: dict,
        call_id: str,
        run_id: str,
        agent_id: str,
        parent_call_id: str | None = None,
    ) -> tuple[ToolCallEvent, ToolResultEvent, Any]:
        call_event = ToolCallEvent(
            run_id=run_id,
            seq=0,
            ts=datetime.now(timezone.utc),
            agent_id=agent_id,
            parent_call_id=parent_call_id,
            call_id=call_id,
            server=self.name,
            tool=tool,
            args=args,
        )

        # Enforcement point for McpServerBinding.allowed_tools: a name that was
        # filtered out of self._tools_by_name never reaches the server.
        if not self.has_tool(tool):
            message = f"tool {tool!r} is not allowed on server {self.name!r} (not in allowed_tools)"
            result_event = ToolResultEvent(
                run_id=run_id,
                seq=0,
                ts=datetime.now(timezone.utc),
                agent_id=agent_id,
                parent_call_id=parent_call_id,
                call_id=call_id,
                result=None,
                error=message,
                duration_ms=0.0,
            )
            return call_event, result_event, {"error": message}

        started = time.monotonic()
        error: str | None = None
        try:
            raw = await self.session.call_tool(tool, args)
            value = _flatten_content(raw.content)
            if getattr(raw, "is_error", False):
                error = str(value)
        except Exception as exc:  # session/transport-level failure, not a tool-reported error
            value = None
            error = str(exc)
        duration_ms = (time.monotonic() - started) * 1000

        persisted_result = redact_result(
            value, self.binding.logging_policy, tool=tool, arg_keys=sorted(args.keys())
        )
        result_event = ToolResultEvent(
            run_id=run_id,
            seq=0,
            ts=datetime.now(timezone.utc),
            agent_id=agent_id,
            parent_call_id=parent_call_id,
            call_id=call_id,
            result=persisted_result,
            error=error,
            duration_ms=duration_ms,
        )
        # NOTE: `value` (raw, unredacted) goes back to the caller/model; only
        # `result_event.result` (persisted_result) is what lands in the event log.
        return call_event, result_event, value


async def _open_session(
    binding: McpServerBinding, resolver: CredentialResolver | None, stack: AsyncExitStack
) -> ClientSession:
    credential = resolver.resolve(binding.credential_ref) if (binding.credential_ref and resolver) else None

    if binding.transport == "stdio":
        conn = binding.connection
        params = StdioServerParameters(
            command=conn["command"],
            args=conn.get("args", []),
            env=_stdio_env(binding, credential),
            cwd=conn.get("cwd"),
        )
        read, write = await stack.enter_async_context(stdio_client(params))
    elif binding.transport == "http":
        url = binding.connection["url"]
        read, write = await stack.enter_async_context(streamable_http_client(url))
    elif binding.transport == "sse":
        url = binding.connection["url"]
        read, write = await stack.enter_async_context(
            sse_client(url, headers=_auth_headers(binding, credential))
        )
    else:
        raise ValueError(f"unknown mcp transport {binding.transport!r} on server {binding.name!r}")

    session = await stack.enter_async_context(ClientSession(read, write))
    await session.initialize()
    return session


def _filter_tools(tools: list[McpTool], allowed: list[str] | None) -> list[McpTool]:
    if allowed is None:
        return list(tools)
    allowed_set = set(allowed)
    return [tool for tool in tools if tool.name in allowed_set]


@asynccontextmanager
async def connect_mcp_servers(
    bindings: list[McpServerBinding], resolver: CredentialResolver | None = None
) -> AsyncIterator[dict[str, ConnectedMcpServer]]:
    """Connects every binding, yields {binding.name: ConnectedMcpServer}, and
    guarantees every connection is closed on exit (including on exception)."""
    async with AsyncExitStack() as stack:
        servers: dict[str, ConnectedMcpServer] = {}
        for binding in bindings:
            session = await _open_session(binding, resolver, stack)
            listed = await session.list_tools()
            tools = _filter_tools(listed.tools, binding.allowed_tools)
            servers[binding.name] = ConnectedMcpServer(binding.name, binding, session, tools)
        yield servers
