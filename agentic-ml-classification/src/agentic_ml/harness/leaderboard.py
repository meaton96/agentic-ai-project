"""
Append-only leaderboard. Multiple processes/workers may write concurrently
later (Phase 7 parallelization), so writes are file-locked from day one.
"""
from __future__ import annotations

import fcntl
import json
import time
from pathlib import Path


def append_leaderboard_entry(leaderboard_path: Path, entry: dict) -> None:
    leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
    entry = dict(entry)
    entry.setdefault("logged_at_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    # Open in append mode, take an exclusive lock for the duration of the
    # write so concurrent workers never interleave partial JSON lines.
    with open(leaderboard_path, "a") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(json.dumps(entry) + "\n")
            f.flush()
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def read_leaderboard(leaderboard_path: Path) -> list[dict]:
    if not leaderboard_path.exists():
        return []
    entries = []
    with open(leaderboard_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries
