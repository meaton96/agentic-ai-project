#!/usr/bin/env python3
"""
Runs the MCP fact server (agentic_ml.mcp_facts.server) — the
standardized, network-reachable replacement for the in-process Tool
closures in tools/*.py. Serves only JSON facts already written under
runs/<run_id>/facts/ by an McpToolProvider (see mcp_facts/provider.py),
plus the two static, stateless registries (list_templates,
list_feature_ops). Never touches raw data, dataframes, or fitted
pipelines.

`enabled_tools` in the config controls which tools this deployment
serves; a tool left out is never registered.

Usage:
    python scripts/run_mcp_server.py
    python scripts/run_mcp_server.py --config configs/mcp_server.json

Point scripts/run_dynamic_orchestrator.py --use-mcp at this server's
--mcp-url (default http://127.0.0.1:8765/mcp) once it's running. For an
interactive look at the exposed tools:
    npx @modelcontextprotocol/inspector
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agentic_ml.mcp_facts.server import build_server


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mcp_server.json",
                         help="JSON config: name, host, port, enabled_tools")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = json.loads(config_path.read_text()) if config_path.exists() else {}

    server = build_server(config)
    print(f"MCP fact server '{server.name}' listening on "
          f"http://{config.get('host', '127.0.0.1')}:{config.get('port', 8765)}/mcp")
    server.run(transport="streamable-http")


if __name__ == "__main__":
    main()
