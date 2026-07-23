"""
Reads/writes prompt overrides for this app's own launched runs. Mirrors
agentic_ml.paths' own resolution pattern (explicit env var, else a
subdirectory of this app's data root, never inside the pipeline repo)
but implemented here rather than in agentic_ml.paths, since that module
belongs to the pipeline repo and this session never edits it — see
../CLAUDE.md.

Deliberately reuses agentic_ml.prompt_loader.PROMPT_OVERRIDE_DIR_ENV_VAR
(AGENTIC_ML_PROMPT_OVERRIDE_DIR) as the env var name: that's the exact
variable agentic_ml.prompt_loader.resolve_prompt_override_dir() falls
back to when a launched run isn't given an explicit --prompt-override-dir.
This module never relies on that fallback, though — server/run_manager.py
always passes the resolved directory explicitly via --prompt-override-dir
when a run opts in, and strips this env var from every launched child's
environment unconditionally, so "opted out" is never at the mercy of
whatever happens to be in this process's own environment. This module's
job is simpler: know where that directory is, for the /api/prompts routes
to read and write.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from agentic_ml.prompt_loader import DEFAULT_PROMPTS_DIR, PROMPT_OVERRIDE_DIR_ENV_VAR


def resolve_override_dir() -> Path:
    """AGENTIC_ML_PROMPT_OVERRIDE_DIR if set, else "prompt_overrides"
    under AGENTIC_ML_DATA_ROOT if set, else "prompt_overrides" relative
    to cwd — the exact same three-tier pattern agentic_ml.paths uses for
    runs/artifacts/datasets, so a deployment that already sets
    AGENTIC_ML_DATA_ROOT gets a sensible override location for free."""
    specific = os.environ.get(PROMPT_OVERRIDE_DIR_ENV_VAR)
    if specific:
        return Path(specific)
    data_root = os.environ.get("AGENTIC_ML_DATA_ROOT")
    if data_root:
        return Path(data_root) / "prompt_overrides"
    return Path("prompt_overrides")


def list_known_agents() -> list[str]:
    """Every agent with a shipped default prompt — derived by scanning
    the pipeline's prompts/ dir rather than hardcoding the list here, so
    this doesn't silently drift if the pipeline adds/renames an agent."""
    return sorted(p.stem for p in DEFAULT_PROMPTS_DIR.glob("*.md"))


def _override_path(agent: str) -> Path:
    return resolve_override_dir() / f"{agent}.md"


def get_prompt_info(agent: str) -> dict:
    default_content = (DEFAULT_PROMPTS_DIR / f"{agent}.md").read_text()
    override_path = _override_path(agent)
    override_content: Optional[str] = override_path.read_text() if override_path.is_file() else None
    return {
        "agent": agent,
        "default_content": default_content,
        "override_content": override_content,
        "has_override": override_content is not None,
    }


def write_override(agent: str, content: str) -> dict:
    override_path = _override_path(agent)
    override_path.parent.mkdir(parents=True, exist_ok=True)
    override_path.write_text(content)
    return get_prompt_info(agent)


def delete_override(agent: str) -> bool:
    """Returns whether an override actually existed to remove."""
    override_path = _override_path(agent)
    if override_path.is_file():
        override_path.unlink()
        return True
    return False
