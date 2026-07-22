"""
Shared event-emission helpers used by every steps/*_step.py function and
both orchestrators. An "event" is a small, JSON-safe dict describing one
meaningful moment in a run (a phase starting, a tool call, a gate
result, a candidate scored, a verdict, a final metric, ...) — never a
dataframe, fitted pipeline, or raw file content, the same restriction
every tool binding in this pipeline already applies to what an agent
gets to see: only facts already exposed via RunStateSummary, the
leaderboard, or transcripts.

on_event is optional and defaults to a no-op everywhere: existing
callers (scripts, notebooks, tests) that don't pass one see no behavior
change at all.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Optional


def emit_event(
    on_event: Optional[Callable[[dict], None]],
    phase: str,
    event_type: str,
    payload: Optional[dict] = None,
) -> None:
    """Calls on_event (if given) with one event dict. ts/run_id are NOT
    set here — make_event_emitter() below stamps those once, at the
    point a run's on_event is actually constructed, so every step
    function only needs to know its own phase/type/payload."""
    if on_event is not None:
        on_event({"phase": phase, "type": event_type, "payload": payload or {}})


def make_event_emitter(
    run_id: str,
    external_on_event: Optional[Callable[[dict], None]] = None,
    persist_fn: Optional[Callable[[dict], None]] = None,
) -> Callable[[dict], None]:
    """Returns the on_event callable actually threaded into a run: stamps
    ts + run_id onto every event a step emits, then fans it out to
    persistence (if given) and an externally supplied callback (if
    given) — composing the two rather than choosing one, so a caller
    streaming events live (e.g. the FastAPI server in the sibling
    agent-frontend repo) still gets them durably persisted to
    events.jsonl at the same time."""
    def emit_full(event: dict) -> None:
        full_event = {"ts": time.time(), "run_id": run_id, **event}
        if persist_fn is not None:
            persist_fn(full_event)
        if external_on_event is not None:
            external_on_event(full_event)
    return emit_full


def make_event_logger(run_dir: Path) -> Callable[[dict], None]:
    """Appends one JSON line per event to runs/<run_id>/events.jsonl."""
    events_path = Path(run_dir) / "events.jsonl"

    def log_event(event: dict) -> None:
        with open(events_path, "a") as f:
            f.write(json.dumps(event, default=str) + "\n")

    return log_event
