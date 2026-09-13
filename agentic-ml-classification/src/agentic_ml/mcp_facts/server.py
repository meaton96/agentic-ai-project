"""
FastMCP server exposing harness-computed facts as MCP tools — the
standardized replacement for the in-process Tool closures in
tools/*.py, one seam of this pipeline's "agents propose, harness
decides" architecture (see CLAUDE.md invariant #1).

This process never computes anything a real run's harness code
doesn't already compute, and it never touches raw data, dataframes,
or fitted pipelines. Two tools (list_templates, list_feature_ops) are
static and stateless, so they're recomputed live from the same
registries tools/template_tool.py and tools/feature_tool.py already
call. Every other tool reads a JSON fact under
runs/<run_id>/facts/<name>.json that McpToolProvider (provider.py) or
gate_adapters.py's prepare_*/*_decide gates already wrote, using the
exact same harness calls the corresponding tools/*.py handler makes —
this server is a read-only view over that directory, nothing more.

`enabled_tools` in config is the configurable surface: a tool absent
from it is never registered, so one deployment can serve a restricted
subset (e.g. omit deep-dive evidence) with no code changes.

Auth: build_server() returns a bare FastMCP object (what the in-process
InMemoryMcpTransport connects to — no network, nothing to authenticate).
Anything served over HTTP should go through build_http_app(), which
requires `Authorization: Bearer <token>` matching AUTH_TOKEN_ENV whenever
a token is set, and refuses to build at all for a non-loopback host
without one — so this server can't be exposed beyond the local machine
unauthenticated by accident.
"""
from __future__ import annotations

import hmac
import os
from typing import Optional

from mcp.server.fastmcp import FastMCP
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from agentic_ml.harness.feature_engineering import list_feature_ops
from agentic_ml.mcp_facts.fact_store import read_fact, read_fact_or_default
from agentic_ml.templates.registry import list_template_summaries

AUTH_TOKEN_ENV = "AGENTIC_ML_MCP_AUTH_TOKEN"
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}

# MCP tool name -> fact name written via fact_store.write_fact
_RUN_SCOPED_FACT_TOOLS: dict[str, tuple[str, str]] = {
    "get_raw_schema": (
        "raw_schema",
        "Get the dataset's raw column facts for this run, computed with no "
        "target column assumed yet: names, dtypes, missingness, cardinality, "
        "sample values, and name-based hints for id/group/datetime columns.",
    ),
    "get_dataset_profile": (
        "dataset_profile",
        "Get the deterministic profile of this run's loaded dataset: column "
        "types, missingness, cardinality, likely ID/group/datetime columns, "
        "target class distribution and imbalance, a recommended split "
        "strategy, and any leakage risk flags.",
    ),
    "get_candidate_review_bundle": (
        "review_bundle",
        "Get everything about the candidate under review for this run: the "
        "template used and when it's meant to be used, the agent's config and "
        "explanation, validation metrics with confidence intervals, both "
        "leakage check results, and relevant dataset facts from the profiler.",
    ),
    "get_monitoring_context": (
        "monitoring_context",
        "Get this run's monitoring context for the newly arrived batch: the "
        "drift summary, the current model's test metrics as of its last "
        "(re)training, and how much has accumulated since then.",
    ),
    "get_planning_context": (
        "planning_context",
        "Get this run's planning context: the goal, the full catalog of "
        "available agents, the current run state, and how many planning "
        "iterations remain.",
    ),
    "get_modeling_attempts": (
        "modeling_attempts",
        "Get the modeling candidates already judged in this run: every "
        "template_id tried so far, and each rejection with its reason (a "
        "failed validation or leakage gate, or a verification rejection). "
        "Empty before the first attempt. Use it to avoid proposing a "
        "candidate that was already rejected.",
    ),
}

ALL_TOOL_NAMES: tuple[str, ...] = tuple(_RUN_SCOPED_FACT_TOOLS) + (
    "list_templates", "list_feature_ops", "get_flight_deep_dive_evidence",
)

# fact name -> the MCP tool that serves it (inverse of _RUN_SCOPED_FACT_TOOLS)
FACT_TOOL_NAMES: dict[str, str] = {fact: tool for tool, (fact, _) in _RUN_SCOPED_FACT_TOOLS.items()}


def build_server(config: Optional[dict] = None) -> FastMCP:
    config = config or {}
    enabled = set(config.get("enabled_tools", ALL_TOOL_NAMES))
    unknown = enabled - set(ALL_TOOL_NAMES)
    if unknown:
        raise ValueError(f"Unknown tool(s) in enabled_tools config: {sorted(unknown)}")

    server = FastMCP(
        name=config.get("name", "agentic-ml-facts"),
        host=config.get("host", "127.0.0.1"),
        port=config.get("port", 8765),
    )

    for tool_name, (fact_name, description) in _RUN_SCOPED_FACT_TOOLS.items():
        if tool_name not in enabled:
            continue

        def make_handler(fact_name: str):
            def handler(run_id: str) -> dict:
                # read_fact_or_default only differs from read_fact for facts
                # with a defined empty state (fact_store.FACT_DEFAULTS); every
                # other missing fact is still a structured error.
                return read_fact_or_default(run_id, fact_name)
            return handler

        server.tool(name=tool_name, description=description)(make_handler(fact_name))

    if "list_templates" in enabled:
        @server.tool(name="list_templates", description=(
            "Get the list of available recipe templates: each template's id, "
            "what it does, when to use it, and its config contract (required "
            "and optional keys)."
        ))
        def _list_templates() -> dict:
            return {"templates": list_template_summaries()}

    if "list_feature_ops" in enabled:
        @server.tool(name="list_feature_ops", description=(
            "Get the list of available feature-engineering operations: each "
            "op's id, what it does, required/optional parameters, and what "
            "column dtype it needs."
        ))
        def _list_feature_ops() -> dict:
            return {"feature_ops": list_feature_ops()}

    if "get_flight_deep_dive_evidence" in enabled:
        @server.tool(name="get_flight_deep_dive_evidence", description=(
            "Get the full deep-dive evidence for one flagged flight in this "
            "run: the model's predicted maintenance probability, which sensor "
            "channels drove that prediction (occlusion attribution), "
            "flight-phase segmentation, and independent raw-signal "
            "cross-cylinder imbalance findings."
        ))
        def _get_flight_deep_dive_evidence(run_id: str, flight_id: str) -> dict:
            return read_fact(run_id, f"deep_dive_evidence_{flight_id}")

    return server


class BearerTokenMiddleware:
    """Pure-ASGI shared-secret check. Lifespan events pass through (the
    streamable-HTTP session manager starts there); every other scope must
    present the exact token, compared in constant time."""

    def __init__(self, app: ASGIApp, token: str):
        self.app = app
        self._expected = f"Bearer {token}".encode()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan":
            await self.app(scope, receive, send)
            return
        presented = dict(scope.get("headers") or []).get(b"authorization", b"")
        if not hmac.compare_digest(presented, self._expected):
            response = JSONResponse(
                {"error": "unauthorized"}, status_code=401, headers={"WWW-Authenticate": "Bearer"},
            )
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


def build_http_app(config: Optional[dict] = None, auth_token: Optional[str] = None) -> ASGIApp:
    """The streamable-HTTP ASGI app for build_server(config), behind a
    bearer-token check. `auth_token` defaults to the AUTH_TOKEN_ENV env var —
    deliberately not a config-file key, so the secret never lands in a
    checked-in JSON file. With no token, only a loopback host is allowed."""
    config = config or {}
    token = auth_token or os.environ.get(AUTH_TOKEN_ENV) or None
    host = config.get("host", "127.0.0.1")
    if token is None and host not in _LOOPBACK_HOSTS:
        raise ValueError(
            f"refusing to serve MCP facts on non-loopback host {host!r} without an auth token; "
            f"set {AUTH_TOKEN_ENV}"
        )
    app = build_server(config).streamable_http_app()
    return BearerTokenMiddleware(app, token) if token else app
