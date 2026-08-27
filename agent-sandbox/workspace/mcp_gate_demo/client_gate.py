"""Adapter that makes an MCP tool call look like an ordinary GateStep gate
function. No sandbox_core changes needed for this: GateStep.gate only needs
any importable callable of shape `(outputs: dict[str, str]) -> str`, sync or
async — this module IS that callable, it just happens to fetch its answer
over MCP instead of computing it directly.

Resolvable from a GateStep as "mcp_gate_demo.client_gate:review_via_mcp"
(with agent-sandbox/workspace on PYTHONPATH — see docs/phase3-testing-guide.md).
"""

import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

_SERVER_SCRIPT = Path(__file__).parent / "server.py"

# Lets the MCP server run under a *different* Python/venv than whatever
# process resolves this gate (sandbox-server or the CLI). Defaults to the
# calling process's own interpreter, which is fine for this same-repo demo
# (server.py only needs `mcp`, already a sandbox-core dependency) — but for
# a real agentic-ml-classification server, point this at that project's own
# venv python so its dependencies never need to touch sandbox-core's venv.
_SERVER_PYTHON = os.environ.get("MCP_GATE_SERVER_PYTHON", sys.executable)


async def review_via_mcp(outputs: dict[str, str]) -> str:
    """Opens a fresh stdio MCP session to server.py, calls its
    `review_proposal` tool with every completed step's output, and returns
    the decision string it replies with. One subprocess + session per gate
    execution — fine for a prototype; a long-lived pipeline with many gate
    calls would want a persistent session instead."""
    params = StdioServerParameters(command=_SERVER_PYTHON, args=[str(_SERVER_SCRIPT)])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("review_proposal", arguments={"outputs": outputs})
            if result.isError:
                raise RuntimeError(f"review_proposal MCP tool call failed: {result.content!r}")
            for block in result.content:
                if getattr(block, "type", None) == "text":
                    return block.text
            raise RuntimeError(f"review_proposal returned no text content: {result.content!r}")
