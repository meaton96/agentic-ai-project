"""
Read-only view over the pipeline's MCP fact server (agentic_ml.mcp_
facts.server) — its config file, which tools it serves, and whether a
server process happens to be listening. This never runs
scripts/run_mcp_server.py itself; that process (if wanted) is started
independently, the same way it always was, per that script's own
docstring. This module only reads configs/mcp_server.json out of the
pipeline checkout and asks the pipeline's own build_server() what
tools that config would register — the same read-only-inspection
carve-out server/prompts.py and run_manager.py already rely on (see
../CLAUDE.md, "Relationship to the pipeline repo").

resolve_mcp_settings()/mcp_server_url() are also what run_manager.py's
build_argv() calls to resolve --mcp-url for a --use-mcp launch, so a
deployment that edits configs/mcp_server.json's host/port is honored
automatically instead of falling back to run_dynamic_orchestrator.py's
own hardcoded default.
"""
from __future__ import annotations

import asyncio
import json
import socket
from pathlib import Path
from typing import Optional

from server.pipeline_paths import pipeline_repo_root

CONFIG_RELPATH = Path("configs/mcp_server.json")

# Mirrors agentic_ml.mcp_facts.server.build_server's own fallbacks exactly,
# so "no config file on disk" here means the same thing it would mean to
# scripts/run_mcp_server.py itself.
_DEFAULT_NAME = "agentic-ml-facts"
_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8765


def resolve_config_path() -> Path:
    return pipeline_repo_root() / CONFIG_RELPATH


def _load_raw_config() -> dict:
    path = resolve_config_path()
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def resolve_mcp_settings(raw: Optional[dict] = None) -> dict:
    """name/host/port/enabled_tools, defaulted exactly like build_server()
    would default them for this same config dict."""
    from agentic_ml.mcp_facts.server import ALL_TOOL_NAMES

    raw = raw if raw is not None else _load_raw_config()
    return {
        "name": raw.get("name", _DEFAULT_NAME),
        "host": raw.get("host", _DEFAULT_HOST),
        "port": raw.get("port", _DEFAULT_PORT),
        "enabled_tools": list(raw.get("enabled_tools", ALL_TOOL_NAMES)),
    }


def mcp_server_url(settings: dict) -> str:
    return f"http://{settings['host']}:{settings['port']}/mcp"


def check_reachable(host: str, port: int, timeout: float = 0.3) -> bool:
    """Best-effort TCP-level check ("something is listening on host:port"),
    not a full MCP handshake — cheap enough to run synchronously on every
    GET /api/mcp/config without meaningfully blocking the event loop, at
    the cost of not catching a process that accepts connections but
    doesn't actually speak MCP."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def tool_catalog(settings: dict) -> list[dict]:
    """Every tool agentic_ml.mcp_facts.server knows how to serve — name,
    description, and input schema straight off the real FastMCP tool
    registry (not hand-copied), with `enabled` reflecting this config's
    own enabled_tools list. Built with every tool turned on regardless of
    `settings["enabled_tools"]` so the catalog always shows the full
    surface an operator could enable, not just today's subset."""
    from agentic_ml.mcp_facts.server import ALL_TOOL_NAMES, build_server

    server = build_server({
        "name": settings["name"], "host": settings["host"], "port": settings["port"],
        "enabled_tools": list(ALL_TOOL_NAMES),
    })
    tools = asyncio.run(server.list_tools())
    enabled = set(settings["enabled_tools"])
    return [
        {
            "name": t.name,
            "description": t.description or "",
            "input_schema": t.inputSchema,
            "enabled": t.name in enabled,
        }
        for t in tools
    ]


def get_mcp_overview() -> dict:
    config_path = resolve_config_path()
    config_exists = config_path.is_file()
    raw = _load_raw_config()
    settings = resolve_mcp_settings(raw)
    return {
        "config_path": str(config_path),
        "config_exists": config_exists,
        "name": settings["name"],
        "host": settings["host"],
        "port": settings["port"],
        "url": mcp_server_url(settings),
        "enabled_tools": settings["enabled_tools"],
        "reachable": check_reachable(settings["host"], settings["port"]),
        "tools": tool_catalog(settings),
    }
