"""The catalog tools as an MCP server over stdio, for an agent spec's stdio binding:

    transport: stdio
    connection: {command: python, args: ["-m", "agentic_pdm.catalog.stdio_server"],
                 env: {PYTHONPATH: <the package's install directory>}}

It runs inside the agent's own container, from the same installed package the
gates use, so the planner and the validate gate always see the same catalog. A
shared HTTP deployment (see tools.py) only knows the catalog it was deployed with.

`mcp` isn't a dependency of this package: the agent's runtime provides it.
"""
from __future__ import annotations

try:
    from mcp.server.mcpserver import MCPServer
except ModuleNotFoundError:  # mcp 1.x called it FastMCP
    from mcp.server.fastmcp import FastMCP as MCPServer

from agentic_pdm.catalog.tools import CATALOG_TOOLS, INSTRUCTIONS


def build_server() -> MCPServer:
    server = MCPServer("agentic-ml-facts", instructions=INSTRUCTIONS)
    for name, fn in CATALOG_TOOLS.items():
        server.tool(name=name)(fn)
    return server


if __name__ == "__main__":
    build_server().run(transport="stdio")
