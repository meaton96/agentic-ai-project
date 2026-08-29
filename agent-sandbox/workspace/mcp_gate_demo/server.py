"""Minimal MCP server standing in for a real gate's decision logic.

Run standalone (`python server.py`) it speaks MCP over stdio and exposes one
tool, `review_proposal`. This is deliberately trivial — a keyword heuristic,
not a real review — the point of this prototype is proving the plumbing
(sandbox gate -> MCP client -> subprocess -> MCP server -> decision string)
works, not the decision logic itself.

For the real agentic-ml-classification integration, this file is the shape
a real server would take: import agentic_ml's own modules (harness.verification,
etc.) and expose their logic as tools, unmodified. That server would run
under agentic-ml-classification's own venv/interpreter (see client_gate.py's
MCP_GATE_SERVER_PYTHON) instead of sandbox-core's — no dependency overlap
required between the two projects.
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("gate-demo")


@mcp.tool()
def review_proposal(outputs: dict[str, str]) -> str:
    """Reviews every completed pipeline step's output so far and decides
    "approved" or "rejected". Demo heuristic only: rejects if any output
    mentions "reject" or "bad", otherwise approves."""
    text = " ".join(outputs.values()).lower()
    if "reject" in text or "bad" in text:
        return "rejected"
    return "approved"


if __name__ == "__main__":
    mcp.run(transport="stdio")
