"""
Single source of truth for where this project's run state and datasets
live on disk. Every script, step, and (eventually) orchestrator resolves
these roots through the functions below instead of hardcoding "runs/" or
"datasets/" relative to the current working directory. Mirrors
agentic_ml.paths in the sibling agentic-ml-classification project.

Resolution order per root, checked at CALL time (not import time):
  1. The root-specific env var (RESOURCE_SCHEDULER_RUNS_DIR /
     _DATASETS_DIR), if set.
  2. RESOURCE_SCHEDULER_DATA_ROOT/<runs|datasets>, if
     RESOURCE_SCHEDULER_DATA_ROOT is set.
  3. "<runs|datasets>" relative to the current working directory.
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


def datasets_root() -> Path:
    return _resolve_root("RESOURCE_SCHEDULER_DATASETS_DIR", "datasets")


def run_dir(run_id: str) -> Path:
    return runs_root() / run_id
