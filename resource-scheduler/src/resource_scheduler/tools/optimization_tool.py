"""
Tool exposing aggregated run evidence to the Optimization agent. Same
shape as every other tool in this project: the handler only calls
environment functions and returns their output.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from resource_scheduler.agent_runtime import Tool
from resource_scheduler.environment.policy_evidence import collect_policy_evidence, list_recent_runs


def build_policy_evidence_fact(n_runs: int = 10, runs_dir: Optional[Path] = None) -> dict:
    run_dirs = list_recent_runs(n_runs, runs_dir=runs_dir)
    evidence = collect_policy_evidence(run_dirs)
    evidence["n_run_dirs_considered"] = len(run_dirs)
    return evidence


def make_optimization_tool(evidence_fact: dict) -> Tool:
    def handler() -> dict:
        return evidence_fact

    return Tool(
        name="get_policy_evidence",
        description=(
            "Get aggregated outcomes across recent runs: ranking validity/"
            "consistency rates, allocation acceptance/rejection counts, "
            "reroute acceptance counts. Call this once — it takes no "
            "arguments. Do not invent statistics beyond what this returns."
        ),
        parameters={"type": "object", "properties": {}, "required": []},
        handler=handler,
    )
