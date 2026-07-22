"""
Single reader over runs/<run_id>/events.jsonl (written by
agentic_ml.events.make_event_logger). Used identically for a run this
server launched and a run that only exists on disk from a prior CLI
invocation: read_events() for GET /api/runs/{id}'s summary,
stream_event_lines() for the SSE endpoint's replay-then-tail. There is
no separate "historical" reader — same file, same parser, either way.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Iterator, Optional


def read_events(run_dir: Path) -> list[dict]:
    events_path = run_dir / "events.jsonl"
    if not events_path.exists():
        return []
    events = []
    with open(events_path) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def terminal_status_from_events(events: list[dict]) -> Optional[str]:
    """Best-effort status for a run this server didn't launch: the last
    run/run_completed or run/run_failed event, if the file has one yet."""
    for event in reversed(events):
        if event.get("type") == "run_completed":
            return "completed"
        if event.get("type") == "run_failed":
            return "failed"
    return None


def stream_event_lines(
    run_dir: Path,
    is_still_running: Callable[[], bool],
    poll_interval: float = 0.25,
    max_wait_for_file: float = 30.0,
) -> Iterator[str]:
    """Yields every line currently in events.jsonl (the replay), then
    tails the file for lines appended while is_still_running() stays
    True. is_still_running() is expected to return False immediately for
    a run this server instance never launched (nothing to tail — replay
    and close), and to reflect RunManager's tracked status for a run it
    did launch (replay-then-follow-live until the process exits)."""
    events_path = run_dir / "events.jsonl"

    waited = 0.0
    while not events_path.exists():
        if not is_still_running() or waited >= max_wait_for_file:
            return
        time.sleep(poll_interval)
        waited += poll_interval

    with open(events_path) as f:
        while True:
            line = f.readline()
            if line:
                stripped = line.strip()
                if stripped:
                    yield stripped
                continue

            if not is_still_running():
                # one last read in case a line landed between this
                # readline() returning "" and the process actually exiting
                line = f.readline()
                stripped = line.strip()
                if stripped:
                    yield stripped
                return

            time.sleep(poll_interval)
