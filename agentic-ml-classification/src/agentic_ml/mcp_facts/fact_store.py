"""
Deterministic JSON persistence for the facts an McpToolProvider hands
to the MCP server (see provider.py). The harness computes each fact
exactly as LocalToolProvider's handlers already do; this module only
writes/reads the resulting JSON to runs/<run_id>/facts/<name>.json —
filesystem-first, same as events.jsonl and transcripts (paths.py).

Writing is atomic (tmp file + rename) so a server read never observes
a partially-written fact. No dataframes, fitted pipelines, or raw file
paths are ever passed to write_fact — only the same JSON-safe payloads
the in-process tool handlers already return.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from agentic_ml.paths import run_dir


class FactNotFoundError(LookupError):
    def __init__(self, run_id: str, name: str):
        self.run_id = run_id
        self.name = name
        super().__init__(f"No fact '{name}' recorded for run '{run_id}'")


def _json_safe(obj):
    """Recursively replace float NaN/Infinity with None. json.dump's
    default allow_nan=True writes literal NaN/Infinity tokens, which are
    valid Python but not valid JSON — an undefined metric (e.g. ROC AUC
    on a single-class fold) previously reached here as float('nan') and
    produced a facts file no standards-compliant JSON reader could parse."""
    if isinstance(obj, float) and (obj != obj or obj in (float("inf"), float("-inf"))):
        return None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def facts_dir(run_id: str) -> Path:
    return run_dir(run_id) / "facts"


def fact_path(run_id: str, name: str) -> Path:
    return facts_dir(run_id) / f"{name}.json"


def write_fact(run_id: str, name: str, payload: dict) -> Path:
    path = fact_path(run_id, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(_json_safe(payload), f, default=str, allow_nan=False)
    os.replace(tmp_path, path)
    return path


def read_fact(run_id: str, name: str) -> dict:
    path = fact_path(run_id, name)
    if not path.exists():
        raise FactNotFoundError(run_id, name)
    with open(path) as f:
        return json.load(f)
