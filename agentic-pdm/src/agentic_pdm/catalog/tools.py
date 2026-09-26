"""The catalog's agent-facing tools as plain functions, for an MCP server
to register. This package runs no server and doesn't depend on `mcp`: the
server that exposes these (agentic-mcp's agentic-ml-facts deployment) does

    for name, fn in CATALOG_TOOLS.items():
        server.tool(name=name)(fn)

and the tool's description is the function's docstring.

Read-only and stateless: list entries, describe their parameters, and check
a proposed experiment (schema, bounds, estimated cost). Nothing here trains,
touches data or makes a decision. The pipeline's validate gate re-checks
every proposal, so this is advice that helps the agent get its proposal
right the first time.

A bad argument (unknown kind or entry id) raises ValueError. MCP servers
turn that into an error result the agent can read and fix.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from agentic_pdm.catalog.experiment import DatasetFacts, catalog_summary, describe_entry, validate_experiment

INSTRUCTIONS = (
    "Tools for designing time-series classification experiments from a fixed catalog of models, "
    "augmentations, learning-rate schedules and losses. Start with list_catalog, read the entries you "
    "want with describe_entry, and always check your final proposal with validate_experiment."
)


def list_catalog_tool(kind: Optional[str] = None) -> list[dict]:
    """List catalog entries: id, kind, one-line description and parameter names.
    kind: "architecture", "augmentation", "schedule" or "loss"; omit for all.
    The "input" and "training" sections aren't entries but can be described too."""
    return catalog_summary(kind)


def describe_entry_tool(entry_id: str) -> dict:
    """Full description of one entry: every parameter's type, default, bounds and meaning.
    entry_id: an id from list_catalog, or "input" / "training" for those config sections."""
    return describe_entry(entry_id)


def validate_experiment_tool(experiment: dict[str, Any], dataset_facts: dict[str, Any],
                             calibration: Optional[dict[str, float]] = None) -> dict:
    """Check a proposed experiment without running it. Returns valid (bool), every error at once,
    warnings, the normalized config (defaults filled in), and a cost estimate (parameters, MFLOPs,
    estimated minutes) on this deployment.
    experiment: the proposal JSON object (name, rationale, architecture, augmentations, schedule,
    loss, input, training, folds).
    dataset_facts: the object given in your brief as "dataset_facts" (n_sequences, n_channels,
    seq_len, n_classes, folds).
    calibration: the object given in your brief as "calibration" (measured time-correction factors
    per architecture), so the estimate matches the one the pipeline will enforce."""
    try:
        facts = DatasetFacts(**dataset_facts)
    except TypeError as exc:
        return {"valid": False, "errors": [f"dataset_facts is malformed: {exc}"], "warnings": []}
    return validate_experiment(experiment, facts, calibration=calibration)


# MCP tool name -> function. The names are what agent specs list in allowed_tools.
CATALOG_TOOLS: dict[str, Callable[..., Any]] = {
    "list_catalog": list_catalog_tool,
    "describe_entry": describe_entry_tool,
    "validate_experiment": validate_experiment_tool,
}
