"""
Shared event-emission helpers used by every steps/*_step.py function.
Mirrors agentic_ml.events in the sibling agentic-ml-classification
project. An "event" is a small, JSON-safe dict describing one
meaningful moment in a run (a phase starting, a tool call, a flag
raised, an allocation decision, ...) — never raw environment state,
matching the restriction every tool binding in this project applies to
what an agent gets to see.

on_event is optional and defaults to a no-op everywhere.
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
    if on_event is not None:
        on_event({"phase": phase, "type": event_type, "payload": payload or {}})


def make_event_emitter(
    run_id: str,
    external_on_event: Optional[Callable[[dict], None]] = None,
    persist_fn: Optional[Callable[[dict], None]] = None,
) -> Callable[[dict], None]:
    def emit_full(event: dict) -> None:
        full_event = {"ts": time.time(), "run_id": run_id, **event}
        if persist_fn is not None:
            persist_fn(full_event)
        if external_on_event is not None:
            external_on_event(full_event)
    return emit_full


def make_event_logger(run_dir: Path) -> Callable[[dict], None]:
    events_path = Path(run_dir) / "events.jsonl"

    def log_event(event: dict) -> None:
        with open(events_path, "a") as f:
            f.write(json.dumps(event, default=str) + "\n")

    return log_event
