"""
Assembles the per-run summary served by GET /api/runs/{run_id} (and, in
list form, GET /api/runs). One function used identically for a
server-launched run and a run that only exists on disk from a prior CLI
invocation — the only difference is whether RunManager has a TrackedRun
for it, which just changes where status/started_at/finished_at come
from (RunManager's bookkeeping vs. events.jsonl's own terminal event).

first_event is exposed alongside last_event because it's the only place
the run's launch parameters (target/goal) live — neither orchestrator's
report.json records them, and the run-summary shape has no room to
duplicate that as separate top-level fields without reinventing what's
already in run_started's payload.

`dataset` IS pulled out as its own top-level field, though, because it
can't reliably be read back out of first_event: run_orchestrator.py's
run_started payload includes "data", but run_dynamic_orchestrator.py's
does not (confirmed against real events.jsonl — a real gap in that
script, not something to fix from here; see ../CLAUDE.md's boundary on
../agentic-ml-classification). For a run this RunManager launched, the
dataset path is already known directly from RunConfig regardless of
orchestrator type, so that's preferred; first_event is only a fallback
for a historical/CLI-launched static run.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from agentic_ml.harness.leaderboard import read_leaderboard
from agentic_ml.paths import leaderboard_path as resolve_leaderboard_path
from agentic_ml.paths import run_dir as resolve_run_dir
from agentic_ml.paths import runs_root

from server.events_io import read_events, terminal_status_from_events
from server.run_manager import RunManager

# orchestrator -> report filename each script writes at the end of its run.
_REPORT_FILENAMES = {
    "static": "orchestrator_report.json",
    "dynamic": "dynamic_orchestrator_report.json",
}


def _read_report(run_dir: Path) -> tuple[Optional[str], Optional[dict]]:
    for orchestrator, filename in _REPORT_FILENAMES.items():
        path = run_dir / filename
        if path.exists():
            return orchestrator, json.loads(path.read_text())
    return None, None


def assemble_run_summary(run_id: str, run_manager: RunManager) -> Optional[dict]:
    run_dir = resolve_run_dir(run_id)
    tracked = run_manager.get_run(run_id)

    if tracked is None and not run_dir.exists():
        return None

    events = read_events(run_dir)
    orchestrator, report = _read_report(run_dir)

    if tracked is not None:
        status = tracked.status
        started_at = tracked.started_at
        finished_at = tracked.finished_at
        error = tracked.error
        orchestrator = orchestrator or tracked.config.orchestrator
        dataset = tracked.config.dataset
    else:
        status = terminal_status_from_events(events) or "unknown"
        started_at = None
        finished_at = None
        error = None
        dataset = events[0]["payload"].get("data") if events else None

    leaderboard_entries = [
        entry for entry in read_leaderboard(resolve_leaderboard_path()) if entry.get("run_id") == run_id
    ]

    return {
        "run_id": run_id,
        "orchestrator": orchestrator,
        "status": status,
        "dataset": dataset,
        "started_at": started_at,
        "finished_at": finished_at,
        "error": error,
        "n_events": len(events),
        "first_event": events[0] if events else None,
        "last_event": events[-1] if events else None,
        "report": report,
        "leaderboard_entries": leaderboard_entries,
    }


def list_all_run_ids(run_manager: RunManager) -> set[str]:
    root = runs_root()
    disk_ids = {p.name for p in root.iterdir() if p.is_dir()} if root.exists() else set()
    tracked_ids = {r.run_id for r in run_manager.list_tracked_runs()}
    return disk_ids | tracked_ids
