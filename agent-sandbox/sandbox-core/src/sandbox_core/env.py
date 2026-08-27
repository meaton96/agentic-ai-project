"""Loads the repo's .env (if present) and makes GateStep.gate import paths
resolvable without the caller needing to export PYTHONPATH by hand.

Uses python-dotenv's default upward search starting from this file's own
location, so it finds the repo-root .env regardless of the caller's cwd —
same `load_dotenv()`-at-import-time pattern already used by
agentic_ml.cli_common / resource_scheduler.cli_common elsewhere in this
repo.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

GATES_PYTHONPATH_ENV_VAR = "SANDBOX_GATES_PYTHONPATH"

_configured = False


def configure_gate_paths() -> None:
    """Adds every directory in SANDBOX_GATES_PYTHONPATH (os.pathsep-separated,
    same convention as PYTHONPATH itself — ':' on Linux/Mac) to sys.path, so
    a GateStep.gate path like "gates_demo:approve" or
    "agentic_ml.harness.verification:check" resolves the same way regardless
    of whether it's sandbox-server or the CLI running the pipeline. Call once
    at process startup (idempotent — safe to call more than once). A missing
    or unset directory is skipped, not an error: an operator who hasn't set
    up a gates directory yet shouldn't see a startup failure over it."""
    global _configured
    if _configured:
        return
    _configured = True

    load_dotenv()
    raw = os.environ.get(GATES_PYTHONPATH_ENV_VAR, "")
    for entry in raw.split(os.pathsep):
        entry = entry.strip()
        if not entry:
            continue
        path = Path(entry).expanduser().resolve()
        if path.is_dir() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
