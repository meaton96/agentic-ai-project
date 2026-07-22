"""
Single source of truth for where this pipeline's run state, artifacts,
and datasets live on disk. Every script, step, and orchestrator resolves
these roots through the functions below instead of hardcoding "runs/",
"artifacts/", or "datasets/" relative to the current working directory —
this is what lets an external caller (e.g. the FastAPI server being
built in the sibling agent-frontend repo) point a whole run at a
different filesystem location without touching pipeline code.

Resolution order per root, checked at CALL time (not import time, so
tests that chdir() before invoking a script still get cwd-relative
defaults exactly as before):
  1. The root-specific env var (AGENTIC_ML_RUNS_DIR / _ARTIFACTS_DIR /
     _DATASETS_DIR), if set.
  2. AGENTIC_ML_DATA_ROOT/<runs|artifacts|datasets>, if AGENTIC_ML_DATA_ROOT
     is set.
  3. "<runs|artifacts|datasets>" relative to the current working
     directory — today's exact behavior when nothing is set.
"""
from __future__ import annotations

import os
from pathlib import Path


def _resolve_root(specific_env: str, subdir_name: str) -> Path:
    specific = os.environ.get(specific_env)
    if specific:
        return Path(specific)
    data_root = os.environ.get("AGENTIC_ML_DATA_ROOT")
    if data_root:
        return Path(data_root) / subdir_name
    return Path(subdir_name)


def runs_root() -> Path:
    return _resolve_root("AGENTIC_ML_RUNS_DIR", "runs")


def artifacts_root() -> Path:
    return _resolve_root("AGENTIC_ML_ARTIFACTS_DIR", "artifacts")


def datasets_root() -> Path:
    return _resolve_root("AGENTIC_ML_DATASETS_DIR", "datasets")


def run_dir(run_id: str) -> Path:
    return runs_root() / run_id


def leaderboard_path() -> Path:
    return artifacts_root() / "reports" / "leaderboard.jsonl"


def models_dir() -> Path:
    return artifacts_root() / "models"
