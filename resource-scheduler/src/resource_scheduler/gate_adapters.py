"""agent-sandbox gate adapters for resource-scheduler, for use with
agent-sandbox's NATIVE agent steps (a real AgentSpec: system prompt +
model, created in the UI) rather than resource_scheduler's own
ToolCallingAgent class -- mirrors the prepare_*/*_decide pattern
agentic_ml.gate_adapters already established for the sibling ML pipeline
(see that package; agent-sandbox's README references it directly).

Shape, per resource-scheduler agent:
  prepare_<agent>  (gate)  -- computes the same deterministic facts
                              resource_scheduler's own tools/*.py already
                              exposes to its ToolCallingAgent (reused
                              directly, unmodified), returns them as the
                              step's output text so the next step's
                              task_template can reference them via
                              {{steps.prepare_<agent>.output}}.
  propose_<agent>  (agent) -- a real agent-sandbox AgentSpec (system
                              prompt adapted from resource-scheduler's own
                              prompts/<agent>.md, since there's no tool
                              call in this flow -- the facts arrive as
                              plain text in the task instead).
  <agent>_decide   (gate)  -- reads the agent step's raw text output back
                              and validates/narrates it against the same
                              deterministic facts, the same
                              "agents propose, harness decides" authority
                              resource_scheduler's own step functions
                              already apply -- just split across pipeline
                              steps instead of happening inside one
                              Python function call.

outputs["__task__"] is always the pipeline's seed task (the task-table
CSV path) -- available to every step at every position, not just the
first, so it's read fresh in each prepare_* gate rather than threaded
forward manually through manifests.

v1 scope: Load Monitor only (prepare_load_monitor / load_monitor_decide)
-- proof of concept, matching agent-sandbox's own worked example
(Resource Scheduler - main loop, shared by Mike Eaton) step for step, so
it's directly comparable. The other five agents follow the same shape.
"""

import json
import os
import urllib.request
from pathlib import Path

from resource_scheduler.environment.state import load_task_table
from resource_scheduler.tools.load_monitor_tool import build_resource_snapshot_fact


def _resolve_task_table_path(csv_path_or_url: str) -> str:
    """outputs["__task__"] is a local path for headless/local-dev use, but
    a gate container has no access to a caller's filesystem -- nothing
    outside the installed package and $GATE_SCRATCH_DIR is reachable (see
    agent-sandbox's own docs, "What your gate runs inside"). So on
    agent-sandbox, the seed task is instead the file's raw GitHub URL,
    downloaded once into GATE_SCRATCH_DIR and reused from there on later
    calls -- same "Library data paths" pattern those docs describe.
    Local-dev callers passing a plain path are unaffected."""
    if not csv_path_or_url.startswith(("http://", "https://")):
        return csv_path_or_url

    scratch_dir = os.environ.get("GATE_SCRATCH_DIR", "/tmp")
    dest = Path(scratch_dir) / "resource_scheduler_task_table.csv"
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(csv_path_or_url, dest)
    return str(dest)


def _flags_match(reported: object, authoritative: list[dict]) -> bool:
    """Order-independent comparison on (scope, id, severity) triples --
    same check resource_scheduler.steps.load_monitor_step._flags_match
    already does for the ToolCallingAgent path; duplicated here (not
    imported) to keep this module self-contained rather than reaching
    into another module's private helper."""
    if not isinstance(reported, list):
        return False
    try:
        reported_keys = {(f["scope"], f["id"], f["severity"]) for f in reported}
    except (TypeError, KeyError):
        return False
    authoritative_keys = {(f["scope"], f["id"], f["severity"]) for f in authoritative}
    return reported_keys == authoritative_keys


# -- 1. Load Monitor ------------------------------------------------------


def prepare_load_monitor(outputs: dict[str, str]) -> tuple[str, str]:
    """outputs["__task__"] is the task-table CSV path. Returns the
    deterministic snapshot+thresholds+flags fact as its output TEXT
    (not a file path) -- small enough to inline directly into the next
    step's task_template, same "small object -> plain string, large
    object -> path" judgment call agent-sandbox's own docs describe."""
    csv_path = _resolve_task_table_path(outputs["__task__"])
    df, variance_injected = load_task_table(csv_path)
    fact = build_resource_snapshot_fact(df, variance_injected)
    return "reported", json.dumps(fact, indent=2, default=str)


def load_monitor_decide(outputs: dict[str, str]) -> tuple[str, str]:
    """Reads prepare_load_monitor's fact back (its output IS the JSON,
    per the note above -- no file to read) and propose_load_monitor's
    raw LLM text, and checks the agent copied the authoritative flags
    exactly rather than inventing/altering severities -- same rule
    resource_scheduler.steps.load_monitor_step enforces for the
    ToolCallingAgent path. Load Monitor is read-only/non-rejecting by
    design (nothing to accept or reject, only to narrate), so this
    always decides "reported" -- a mismatch is recorded as data, not a
    hard failure, same "record issues, keep going" philosophy
    run_orchestrator.py uses throughout."""
    fact = json.loads(outputs["prepare_load_monitor"])
    authoritative_flags = fact["flags"]

    raw = outputs.get("propose_load_monitor", "")
    try:
        proposed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        proposed = None

    reported_flags = (proposed or {}).get("flags") if isinstance(proposed, dict) else None
    flag_mismatch = not _flags_match(reported_flags, authoritative_flags)

    report = {
        "n_flags": len(authoritative_flags),
        "flag_mismatch": flag_mismatch,
        "authoritative_flags": authoritative_flags,
        "llm_narrative": proposed,
    }
    return "reported", json.dumps(report, indent=2, default=str)
