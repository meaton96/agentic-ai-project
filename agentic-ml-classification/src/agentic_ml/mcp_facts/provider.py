"""
ToolProvider: the seam between steps/*.py and tool construction. Every
step calls provider.make_*(...) instead of importing a tools/*.py
factory directly, so which channel serves an agent's facts — in-process
closures or the MCP fact server (server.py) — is a choice made once,
outside steps/*.py, never a change to what an agent is told or how it
asks for it.

LocalToolProvider is the default and is a byte-for-byte pass-through
to today's tools/*.py factories — existing behavior, unchanged, for
every script and test that doesn't opt in.

McpToolProvider computes each fact with the exact same harness call
the corresponding tools/*.py handler makes, persists it via
fact_store.write_fact() (servable by server.py, and now a durable,
audit-visible artifact under runs/<run_id>/facts/), and returns a Tool
whose NAME, DESCRIPTION, and SCHEMA are copied verbatim from the local
factory's Tool — only the handler differs, rebound to round-trip
through an McpTransport. This is what makes provider parity a
structural guarantee rather than a hope: see
tests/test_mcp_facts.py::test_provider_parity, which would fail if a
tool's identity (not just its payload) ever diverged between the two
providers.

Nothing here adds an LLM call or widens what an agent can request —
run_id lives in the McpToolProvider instance, never in a schema an
agent could set.
"""
from __future__ import annotations

from typing import Optional, Protocol

import pandas as pd

from agentic_ml.agent_runtime import Tool
from agentic_ml.harness.intake import raw_schema_summary
from agentic_ml.harness.profiler import profile_dataset
from agentic_ml.mcp_facts.fact_store import write_fact
from agentic_ml.mcp_facts.transport import McpTransport
from agentic_ml.tools import (
    deep_dive_tool, feature_tool, intake_tool, planner_tool,
    profiler_tool, retrain_decision_tool, template_tool, verification_tool,
)
from agentic_ml.tools.deep_dive_tool import gather_deep_dive_evidence


class ToolProvider(Protocol):
    def make_raw_schema_tool(self, df: pd.DataFrame) -> Tool: ...

    def make_profiler_tool(self, df: pd.DataFrame, target_column: str) -> Tool: ...

    def make_list_templates_tool(self) -> Tool: ...

    def make_list_feature_ops_tool(self) -> Tool: ...

    def make_review_bundle_tool(self, bundle: dict) -> Tool: ...

    def make_monitoring_context_tool(self, monitoring_context: dict) -> Tool: ...

    def make_planning_context_tool(
        self, goal: str, state_dict: dict, available_agents: list[dict],
        iteration: int, max_iterations: int,
    ) -> Tool: ...

    def make_deep_dive_evidence_tool(
        self, flight_df: pd.DataFrame, feature_row, pipeline,
        feature_columns: list[str], background: pd.Series,
        sample_hz: float = 1.0, loc_thresholds: Optional[dict] = None,
        flight_id: Optional[str] = None,
    ) -> Tool: ...


class LocalToolProvider:
    """Default provider: delegates verbatim to tools/*.py's in-process
    factories. `flight_id` on make_deep_dive_evidence_tool is accepted
    for interface parity with McpToolProvider (which needs it to key a
    fact file) and ignored here."""

    def make_raw_schema_tool(self, df: pd.DataFrame) -> Tool:
        return intake_tool.make_raw_schema_tool(df)

    def make_profiler_tool(self, df: pd.DataFrame, target_column: str) -> Tool:
        return profiler_tool.make_profiler_tool(df, target_column)

    def make_list_templates_tool(self) -> Tool:
        return template_tool.make_list_templates_tool()

    def make_list_feature_ops_tool(self) -> Tool:
        return feature_tool.make_list_feature_ops_tool()

    def make_review_bundle_tool(self, bundle: dict) -> Tool:
        return verification_tool.make_review_bundle_tool(bundle)

    def make_monitoring_context_tool(self, monitoring_context: dict) -> Tool:
        return retrain_decision_tool.make_monitoring_context_tool(monitoring_context)

    def make_planning_context_tool(
        self, goal, state_dict, available_agents, iteration, max_iterations,
    ) -> Tool:
        return planner_tool.make_planning_context_tool(
            goal, state_dict, available_agents, iteration, max_iterations,
        )

    def make_deep_dive_evidence_tool(
        self, flight_df, feature_row, pipeline, feature_columns, background,
        sample_hz=1.0, loc_thresholds=None, flight_id=None,
    ) -> Tool:
        return deep_dive_tool.make_deep_dive_evidence_tool(
            flight_df, feature_row, pipeline, feature_columns, background,
            sample_hz=sample_hz, loc_thresholds=loc_thresholds,
        )


class McpToolProvider:
    """Computes each fact harness-side, persists it for server.py to
    serve, and returns a Tool identical in name/description/schema to
    LocalToolProvider's — built BY constructing the local Tool and
    rebinding only its handler — whose handler calls the matching MCP
    tool through `transport`."""

    def __init__(self, run_id: str, transport: McpTransport):
        self.run_id = run_id
        self.transport = transport

    def _rebind(self, local_tool: Tool, call_args: dict) -> Tool:
        def handler() -> dict:
            return self.transport.call_tool(local_tool.name, call_args)
        return Tool(
            name=local_tool.name, description=local_tool.description,
            parameters=local_tool.parameters, handler=handler,
        )

    def make_raw_schema_tool(self, df: pd.DataFrame) -> Tool:
        write_fact(self.run_id, "raw_schema", raw_schema_summary(df))
        local = intake_tool.make_raw_schema_tool(df)
        return self._rebind(local, {"run_id": self.run_id})

    def make_profiler_tool(self, df: pd.DataFrame, target_column: str) -> Tool:
        fact = profile_dataset(df, target_column=target_column).to_dict()
        write_fact(self.run_id, "dataset_profile", fact)
        local = profiler_tool.make_profiler_tool(df, target_column)
        return self._rebind(local, {"run_id": self.run_id})

    def make_list_templates_tool(self) -> Tool:
        # Static/stateless — server.py computes this live from the same
        # registry, so there's no per-run fact to persist here.
        local = template_tool.make_list_templates_tool()
        return self._rebind(local, {})

    def make_list_feature_ops_tool(self) -> Tool:
        local = feature_tool.make_list_feature_ops_tool()
        return self._rebind(local, {})

    def make_review_bundle_tool(self, bundle: dict) -> Tool:
        write_fact(self.run_id, "review_bundle", bundle)
        local = verification_tool.make_review_bundle_tool(bundle)
        return self._rebind(local, {"run_id": self.run_id})

    def make_monitoring_context_tool(self, monitoring_context: dict) -> Tool:
        write_fact(self.run_id, "monitoring_context", monitoring_context)
        local = retrain_decision_tool.make_monitoring_context_tool(monitoring_context)
        return self._rebind(local, {"run_id": self.run_id})

    def make_planning_context_tool(
        self, goal, state_dict, available_agents, iteration, max_iterations,
    ) -> Tool:
        fact = {
            "goal": goal, "current_state": state_dict,
            "available_agents": available_agents, "iteration": iteration,
            "max_iterations": max_iterations,
        }
        write_fact(self.run_id, "planning_context", fact)
        local = planner_tool.make_planning_context_tool(
            goal, state_dict, available_agents, iteration, max_iterations,
        )
        return self._rebind(local, {"run_id": self.run_id})

    def make_deep_dive_evidence_tool(
        self, flight_df, feature_row, pipeline, feature_columns, background,
        sample_hz=1.0, loc_thresholds=None, flight_id=None,
    ) -> Tool:
        evidence = gather_deep_dive_evidence(
            flight_df, feature_row, pipeline, feature_columns, background,
            sample_hz=sample_hz, loc_thresholds=loc_thresholds,
        )
        fid = flight_id or "current"
        write_fact(self.run_id, f"deep_dive_evidence_{fid}", evidence)
        local = deep_dive_tool.make_deep_dive_evidence_tool(
            flight_df, feature_row, pipeline, feature_columns, background,
            sample_hz=sample_hz, loc_thresholds=loc_thresholds,
        )
        return self._rebind(local, {"run_id": self.run_id, "flight_id": fid})
