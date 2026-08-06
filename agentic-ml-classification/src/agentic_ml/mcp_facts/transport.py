"""
Sync facade over the MCP client session. Everything upstream of this
module (Tool handlers built by McpToolProvider, and through them the
synchronous ToolCallingAgent loop in agent_runtime.py) is plain
synchronous Python, so each call opens a session, awaits exactly one
tool call, and tears the session down — these tools are called at most
a handful of times per agent turn, so session-per-call simplicity
beats connection pooling here.

Two implementations sharing one interface: HttpMcpTransport talks to a
real server over streamable HTTP (scripts/run_mcp_server.py);
InMemoryMcpTransport connects directly to a FastMCP server object with
no network, for tests and the provider-parity check in
tests/test_mcp_facts.py.
"""
from __future__ import annotations

import asyncio
import json
import threading
from typing import Any, Protocol

import anyio
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import CallToolResult


class McpToolError(RuntimeError):
    """Raised when an MCP tool call comes back with isError=True, or the
    transport itself fails (connection refused, timeout, ...)."""


class McpTransport(Protocol):
    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict: ...


def _run_sync(coro_func, *args):
    """Runs an async function to completion and returns its result,
    whether or not the calling thread already has a running event
    loop. anyio.run() only handles the no-loop case (plain scripts,
    pytest) — called from a thread that already has one running (e.g.
    Jupyter/ipykernel, which runs its own asyncio loop in the main
    thread) it raises 'Already running asyncio in this thread'. When a
    loop is already running, fall back to a fresh thread with its own
    loop instead — every ToolCallingAgent call site here is already
    synchronous, so blocking that thread until the coroutine finishes
    is exactly what callers expect."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return anyio.run(coro_func, *args)  # no loop running here — the common case

    outcome: dict[str, Any] = {}

    def runner() -> None:
        try:
            outcome["value"] = anyio.run(coro_func, *args)
        except BaseException as e:  # noqa: BLE001 — re-raised on the caller's thread below
            outcome["error"] = e

    thread = threading.Thread(target=runner)
    thread.start()
    thread.join()
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


def _extract_payload(result: CallToolResult) -> dict:
    if result.isError:
        text = "; ".join(getattr(block, "text", str(block)) for block in result.content)
        raise McpToolError(text or "MCP tool call failed with no error detail")
    if result.structuredContent is not None:
        return result.structuredContent
    for block in result.content:
        if getattr(block, "type", None) == "text":
            return json.loads(block.text)
    raise McpToolError("MCP tool call returned no content")


class HttpMcpTransport:
    def __init__(self, url: str, timeout_seconds: float = 30.0):
        self.url = url
        self.timeout_seconds = timeout_seconds

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict:
        return _run_sync(self._call_tool_async, name, arguments)

    async def _call_tool_async(self, name: str, arguments: dict[str, Any]) -> dict:
        # _extract_payload runs AFTER both `async with` blocks have exited
        # cleanly — raising from inside them would have anyio's task-group
        # __aexit__ wrap it in an opaque ExceptionGroup instead of the
        # clean McpToolError callers actually want to catch.
        try:
            async with streamablehttp_client(self.url, timeout=self.timeout_seconds) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(name, arguments)
        except Exception as e:
            raise McpToolError(f"failed to reach MCP server at {self.url}: {type(e).__name__}: {e}") from e
        return _extract_payload(result)


class InMemoryMcpTransport:
    """Test-only: connects directly to a FastMCP server object in the
    same process, no network. Mirrors HttpMcpTransport's interface
    exactly so provider code never needs to know which one it's using."""

    def __init__(self, server: FastMCP):
        self.server = server

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict:
        return _run_sync(self._call_tool_async, name, arguments)

    async def _call_tool_async(self, name: str, arguments: dict[str, Any]) -> dict:
        async with create_connected_server_and_client_session(self.server) as session:
            result = await session.call_tool(name, arguments)
        return _extract_payload(result)
