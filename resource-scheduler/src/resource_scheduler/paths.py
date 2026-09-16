"""
Single source of truth for where this project's run state, artifacts,
and datasets live on disk. Every script, step, and orchestrator resolves
these roots through the functions below instead of hardcoding "runs/",
"artifacts/", or "datasets/" relative to the current working directory.
Mirrors agentic_ml.paths in the sibling agentic-ml-classification
project.

artifacts_root() specifically holds state meant to persist ACROSS runs
(e.g. scripts/run_optimization_loop.py's policy_search_history.json,
which the greedy search needs to survive between separate invocations
of the loop) -- runs_root() is scoped per run_id and isn't the right
place for that.

Resolution order per root, checked at CALL time (not import time):
  1. The root-specific env var (RESOURCE_SCHEDULER_RUNS_DIR /
     _ARTIFACTS_DIR / _DATASETS_DIR), if set.
  2. RESOURCE_SCHEDULER_DATA_ROOT/<runs|artifacts|datasets>, if
     RESOURCE_SCHEDULER_DATA_ROOT is set.
  3. "<runs|artifacts|datasets>" relative to the current working
     directory.
"""
from __future__ import annotations

import os
from pathlib import Path


def _resolve_root(specific_env: str, subdir_name: str) -> Path:
    specific = os.environ.get(specific_env)
    if specific:
        return Path(specific)
    data_root = os.environ.get("RESOURCE_SCHEDULER_DATA_ROOT")
    if data_root:
        return Path(data_root) / subdir_name
    return Path(subdir_name)


def runs_root() -> Path:
    return _resolve_root("RESOURCE_SCHEDULER_RUNS_DIR", "runs")


def artifacts_root() -> Path:
    return _resolve_root("RESOURCE_SCHEDULER_ARTIFACTS_DIR", "artifacts")


def datasets_root() -> Path:
    return _resolve_root("RESOURCE_SCHEDULER_DATASETS_DIR", "datasets")


def run_dir(run_id: str) -> Path:
    return runs_root() / run_id
