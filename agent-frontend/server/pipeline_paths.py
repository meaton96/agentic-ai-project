"""Locates the agentic-ml-classification checkout on disk. Shared by
run_manager.py (to find scripts/*.py) and mcp_config.py (to find
configs/mcp_server.json) — pulled out on its own so neither module has
to import the other just for this."""
from __future__ import annotations

from pathlib import Path


def pipeline_repo_root() -> Path:
    """Locates the checkout via the installed `agentic_ml` package
    itself, rather than assuming a relative path from this repo's cwd
    (../agentic-ml-classification only holds if the server happens to
    run from this repo's directory, which CLAUDE.md says not to
    assume)."""
    import agentic_ml

    # src/agentic_ml/__init__.py -> parents[0]=agentic_ml, [1]=src, [2]=repo root
    return Path(agentic_ml.__file__).resolve().parents[2]
