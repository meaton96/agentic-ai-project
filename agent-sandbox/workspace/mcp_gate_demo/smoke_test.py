"""Standalone check of the MCP round trip, with no pipeline/LLM involved.
Run: sandbox-core/.venv/bin/python workspace/mcp_gate_demo/smoke_test.py
"""

import asyncio

from client_gate import review_via_mcp


async def main() -> None:
    approved = await review_via_mcp({"step_a": "This looks good, ship it."})
    print("expected approved, got:", approved)
    assert approved == "approved"

    rejected = await review_via_mcp({"step_a": "This is bad, please reject."})
    print("expected rejected, got:", rejected)
    assert rejected == "rejected"

    print("MCP round trip OK")


if __name__ == "__main__":
    asyncio.run(main())
