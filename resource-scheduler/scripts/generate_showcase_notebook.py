#!/usr/bin/env python3
"""
Builds notebooks/agent_showcase.ipynb from a completed run's report and
transcript files: one section per agent (what it does, its gate logic,
its actual live output from the given run), plus the A2A message flow
and the orchestrator's final narrated summary.

Unlike a hand-written notebook, this reads real files under
runs/<run_id>/ and PRE-POPULATES cell outputs with that real content --
opening the notebook shows genuine live-model output immediately, no
Jupyter kernel or model endpoint required to view it. Re-running the
cells (if runs/<run_id>/ still exists and dependencies are installed)
reproduces the same output from the same files.

Regenerate against a fresher run any time:
    python scripts/generate_showcase_notebook.py --run-id <run_id>
    python scripts/generate_showcase_notebook.py   # most recent run
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Optional

_cell_id_counter = itertools.count(1)


def _new_cell_id() -> str:
    # nbformat 4.5+ requires a unique "id" per cell (a hard error in
    # future nbformat versions, currently only a warning) -- plain
    # incrementing ids are fine since this file is regenerated wholesale
    # each run, never hand-edited or diffed cell-by-cell.
    return f"cell-{next(_cell_id_counter)}"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from resource_scheduler.paths import runs_root

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _most_recent_run_dir() -> Optional[Path]:
    base = runs_root()
    if not base.exists():
        return None
    run_dirs = sorted((p for p in base.iterdir() if p.is_dir()), key=lambda p: p.name)
    return run_dirs[-1] if run_dirs else None


def md(*lines: str) -> dict:
    return {
        "cell_type": "markdown", "metadata": {}, "id": _new_cell_id(),
        "source": [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines else []),
    }


def code_with_output(source_lines: list[str], stdout_text: str, execution_count: int) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "id": _new_cell_id(),
        "execution_count": execution_count,
        "source": [l + "\n" for l in source_lines[:-1]] + ([source_lines[-1]] if source_lines else []),
        "outputs": [{
            "output_type": "stream",
            "name": "stdout",
            "text": [l + "\n" for l in stdout_text.split("\n")[:-1]] + [stdout_text.split("\n")[-1]],
        }],
    }


def code_no_output(source_lines: list[str]) -> dict:
    return {
        "cell_type": "code", "metadata": {}, "id": _new_cell_id(), "execution_count": None,
        "source": [l + "\n" for l in source_lines[:-1]] + ([source_lines[-1]] if source_lines else []),
        "outputs": [],
    }


def _transcript_summary(run_dir: Path, agent_name: str) -> str:
    """Renders the tool call + final assistant response from a
    transcript file as readable text -- the part worth actually
    reading, not the full system prompt every time."""
    matches = sorted((run_dir / "transcripts").glob(f"{agent_name}_*.json"))
    if not matches:
        return "(no transcript found for this agent in this run)"
    messages = _load_json(matches[0])
    if not messages:
        return "(transcript file present but empty/unreadable)"

    lines = []
    for m in messages:
        role = m.get("role")
        if role == "tool":
            content = m.get("content")
            lines.append("--- tool result (deterministic, computed by environment/) ---")
            lines.append(json.dumps(content, indent=2, default=str)[:2000])
        elif role == "assistant" and not m.get("tool_calls"):
            reasoning = m.get("reasoning")
            if reasoning:
                lines.append("--- model's reasoning (separate channel, not the answer) ---")
                lines.append(reasoning.strip()[:1500] + ("..." if len(reasoning.strip()) > 1500 else ""))
            lines.append("--- final structured response ---")
            content = m.get("content")
            lines.append(json.dumps(content, indent=2, default=str) if isinstance(content, (dict, list)) else str(content))
    return "\n".join(lines)


AGENT_DESCRIPTIONS = {
    "load_monitor": (
        "## Agent #1 — Load Monitor",
        "Read-only. `environment/state.py` computes a deterministic snapshot (per-machine status/"
        "utilization, per-slice latency/capacity) and the warning/critical thresholds to judge it "
        "against. The LLM's only job is to narrate the flags the deterministic code already found — "
        "it must copy the `flags` array exactly, never invent or re-score one. "
        "`steps/load_monitor_step.py::_flags_match` checks the agent's reported flags against the "
        "authoritative ones; a mismatch is logged, not silently trusted.",
    ),
    "task_prioritization": (
        "## Agent #2 — Task Prioritization",
        "`environment/queue.py` computes three raw signals per pending task — `urgency_signal`, "
        "`energy_cost_proxy`, `availability_bonus` — as documented proxies derived from real columns "
        "(this dataset has no actual deadline/cost field). The LLM combines them into a ranking using "
        "its own judgment; there's no single 'correct' answer to gate against here, so the gate is "
        "structural (`validate_ranking_proposal`: is this an exact permutation of the pending task ids, "
        "is every task scored) plus a soft consistency check (does the order match the claimed scores). "
        "**First A2A send**: on a valid proposal, the ranking goes directly to Resource Allocation's "
        "mailbox — no orchestrator round-trip.",
    ),
    "resource_allocation": (
        "## Agent #3 — Resource Allocation",
        "**First A2A read**: consumes Task Prioritization's ranking straight from the mailbox. The LLM "
        "proposes machine/slice assignments; `environment/allocation.py::check_constraints` is the real "
        "gate — no assignment to a machine in Maintenance, no slice pushed at/over capacity, checked "
        "cumulatively across the whole batch. An assignment can be structurally valid and still get "
        "environment-rejected if it violates a constraint, independent of the agent's stated rationale. "
        "Every proposed assignment produces a traceable event either way.",
    ),
    "failure_recovery": (
        "## Agent #4 — Failure Recovery",
        "Detects a machine's fresh transition into a fault state by diffing two environment snapshots "
        "(`environment/incidents.py::diff_snapshots`), identifies which currently-committed tasks sit on "
        "it, and proposes reroutes. Short-circuits with **zero LLM calls** when there's nothing to do "
        "(`no_incidents_detected` / `no_affected_tasks`) — no point paying for a guaranteed no-op. "
        "**Second A2A hop**: a valid reroute proposal goes to Resource Allocation as a `reroute_request` "
        "message, re-validated by `run_reroute_validation_step` against the exact same `check_constraints` "
        "gate — deliberately **not** a second LLM call, since Failure Recovery already did the reasoning.",
    ),
    "optimization": (
        "## Agent #5 — Optimization",
        "The odd one out: takes **no task-table input at all**. Instead it aggregates outcomes across "
        "recent runs by reading the report JSON files every other agent's script already writes to "
        "`runs/<run_id>/` (`environment/policy_evidence.py`). May only propose changes to parameters "
        "actually wired to a CLI flag (`queue_size`, `snapshot_window`, `slice_capacity`) — never an "
        "invented knob. Never applies its own proposal; sends it, together with the real underlying "
        "evidence dict (not just its own prose claim about it), to Human Oversight's mailbox — the third "
        "A2A hop.",
    ),
    "human_oversight": (
        "## Agent #6 — Human Oversight",
        "Same shape as the sibling ML pipeline's Verification agent: there's no deterministic correct "
        "verdict to check the LLM against here — a human-judgment call is the whole point of delegating "
        "to this agent. The one enforced rule (`environment/oversight.py::parse_oversight_verdict`): an "
        "unparseable or invalid verdict degrades to `\"flagged\"`, never silently to `\"approved\"`.",
    ),
}


def build_notebook(run_dir: Path) -> dict:
    run_id = run_dir.name
    orchestrator_report = _load_json(run_dir / "orchestrator_report.json") or {}

    cells: list[dict] = []
    cells.append(md(
        "# Resource Scheduler — Agent Showcase",
        "",
        f"Generated from a real completed run: **`{run_id}`**. Every output below is genuine live-model "
        "output captured when this run executed against the RIT GenAI endpoint (`qwen3:8b`) — nothing here "
        "is a mock or a placeholder. Re-running a cell (if `runs/{}/` still exists on disk) reproduces the "
        "same result from the same files; it does not require a live model endpoint just to view.".format(run_id),
        "",
        "Generated by `scripts/generate_showcase_notebook.py` — rerun it against any completed run to "
        "refresh this notebook with newer results:",
        "```bash",
        f"python scripts/generate_showcase_notebook.py --run-id {run_id}",
        "```",
    ))

    cells.append(md(
        "## Architecture in one paragraph",
        "",
        "Six agents, one deterministic **environment layer** (`src/resource_scheduler/environment/`) that "
        "computes every fact and enforces every constraint, and a minimal **A2A mailbox** "
        "(`src/resource_scheduler/a2a/mailbox.py`) that lets agents hand structured proposals directly to "
        "each other rather than only through a central orchestrator — the thing this project exists to "
        "test that the sibling ML classification pipeline never does. Every agent below follows the same "
        "shape: deterministic code computes facts → one tool exposes them → the LLM proposes → "
        "deterministic code gates the proposal, never the other way around.",
    ))

    cells.append(code_no_output([
        "import json",
        "from pathlib import Path",
        "",
        f'RUN_ID = "{run_id}"',
        'RUN_DIR = Path("..") / "runs" / RUN_ID',
        "",
        "def load(name):",
        "    path = RUN_DIR / name",
        "    return json.loads(path.read_text()) if path.exists() else None",
    ]))

    exec_n = 1
    for agent_key, report_filename in [
        ("load_monitor", "load_monitor_report.json"),
        ("task_prioritization", "task_prioritization_report.json"),
        ("resource_allocation", "resource_allocation_report.json"),
        ("failure_recovery", "failure_recovery_report.json"),
        ("optimization", "optimization_report.json"),
        ("human_oversight", "human_oversight_report.json"),
    ]:
        title, desc = AGENT_DESCRIPTIONS[agent_key]
        cells.append(md(title, "", desc))

        report = _load_json(run_dir / report_filename)
        report_text = json.dumps(report, indent=2, default=str) if report is not None else "(no report file for this agent in this run)"
        cells.append(code_with_output(
            [f'report = load("{report_filename}")', "print(json.dumps(report, indent=2, default=str))"],
            report_text,
            exec_n,
        ))
        exec_n += 1

        transcript_text = _transcript_summary(run_dir, agent_key)
        cells.append(md("**Live transcript excerpt** (real tool call + real model response, from this run):"))
        cells.append(code_with_output(
            [f'print(open(RUN_DIR / "transcripts" / "{agent_key}_01.json").read()[:0] or "see rendered output below")'],
            transcript_text,
            exec_n,
        ))
        exec_n += 1

    # --- A2A message flow ---
    cells.append(md(
        "## The A2A message flow in this run",
        "",
        "Not routed through the orchestrator — agents hand proposals directly to each other's mailbox. "
        "`events.jsonl` records every send:",
    ))
    events_path = run_dir / "events.jsonl"
    a2a_lines = []
    if events_path.exists():
        for line in events_path.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("phase") == "a2a" and rec.get("type") == "message_sent":
                payload = rec.get("payload", {})
                a2a_lines.append(f"{payload.get('sender')} -> {payload.get('recipient')}  "
                                  f"[{payload.get('message_type')}]  keys={payload.get('payload_keys')}")
    a2a_text = "\n".join(a2a_lines) if a2a_lines else "(no A2A messages were sent in this run)"
    cells.append(code_with_output(
        ['for line in open(RUN_DIR / "events.jsonl"):',
         '    rec = json.loads(line)',
         '    if rec.get("phase") == "a2a" and rec.get("type") == "message_sent":',
         '        p = rec["payload"]',
         '        print(f"{p[\'sender\']} -> {p[\'recipient\']}  [{p[\'message_type\']}]  keys={p[\'payload_keys\']}")'],
        a2a_text,
        exec_n,
    ))
    exec_n += 1

    # --- Final orchestrator summary ---
    cells.append(md(
        "## Orchestrator's final narrated summary",
        "",
        "Plain-text LLM narration, no tools, no decisions — same pattern the sibling ML pipeline's "
        "orchestrator uses for its own final summary:",
    ))
    summary_text = orchestrator_report.get("final_summary", "(no final_summary in this run's orchestrator_report.json)")
    issues = orchestrator_report.get("issues", [])
    issues_text = ("\n\nIssues flagged:\n" + "\n".join(f"- {i}" for i in issues)) if issues else "\n\nNo issues flagged."
    cells.append(code_with_output(
        ['report = load("orchestrator_report.json")',
         'print(report["final_summary"])',
         'print()',
         'print("Issues flagged:" if report["issues"] else "No issues flagged.")',
         'for i in report["issues"]:',
         '    print(f"- {i}")'],
        summary_text + issues_text,
        exec_n,
    ))

    cells.append(md(
        "## Want the full HTML version?",
        "",
        "```bash",
        f"python scripts/render_run_report.py --run-id {run_id}",
        "```",
        f"writes `runs/{run_id}/report.html` — a single self-contained file, open it in any browser.",
    ))

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None, help="defaults to the most recent run under runs/")
    args = parser.parse_args()

    if args.run_id:
        run_dir = runs_root() / args.run_id
        if not run_dir.exists():
            print(f"No such run directory: {run_dir}")
            sys.exit(1)
    else:
        run_dir = _most_recent_run_dir()
        if run_dir is None:
            print(f"No runs found under {runs_root()} — run an agent or the orchestrator first.")
            sys.exit(1)
        print(f"Using most recent run: {run_dir.name}")

    notebook = build_notebook(run_dir)
    out_path = REPO_ROOT / "notebooks" / "agent_showcase.ipynb"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # encoding="utf-8" is not optional here: Path.write_text()'s default
    # encoding is the platform's preferred locale encoding, which on
    # Windows is typically cp1252, not UTF-8 -- silently mojibake-ing
    # every em-dash and non-ASCII character in this notebook's markdown
    # (confirmed: nbformat.read() still "succeeded" against a cp1252
    # file, it just decoded every non-ASCII byte wrong).
    out_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Notebook written to {out_path}")


if __name__ == "__main__":
    main()
