"""
Loads agent system prompts from prompts/<agent_name>.md at the repo
root, with an optional per-agent override directory. Overrides are
per-agent, not all-or-nothing: if override_dir is given but doesn't
contain a particular agent's file, that agent silently falls back to
the shipped default — this is what lets a UI let someone edit ONE
agent's prompt without having to supply files for every agent.

Resolution of override_dir follows the same "explicit parameter, env-
var-backed default" pattern M0's data-root config (agentic_ml.paths)
and cli_common.resolve_model_endpoint already use: an explicit argument
wins if given, else AGENTIC_ML_PROMPT_OVERRIDE_DIR, else no override
(shipped defaults only).

This module lives under src/agentic_ml/ (not literally inside prompts/)
so it's importable everywhere agentic_ml already is — every script,
notebook, and test in this repo already puts src/ on sys.path before
importing anything from this package, but none of them put the repo
root itself on sys.path. Resolving DEFAULT_PROMPTS_DIR via Path(__file__)
means the actual prompts/*.md text files still live at the repo root
exactly as specified, without requiring every entry point to be updated
just to keep importing steps/*_step.py working.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

DEFAULT_PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"

PROMPT_OVERRIDE_DIR_ENV_VAR = "AGENTIC_ML_PROMPT_OVERRIDE_DIR"


def resolve_prompt_override_dir(explicit: Optional[str] = None) -> Optional[str]:
    """Explicit argument wins if given; else the env var; else None
    (shipped defaults only)."""
    if explicit is not None:
        return explicit
    return os.environ.get(PROMPT_OVERRIDE_DIR_ENV_VAR)


def prompt_source(agent_name: str, override_dir: Optional[str] = None) -> tuple[str, Path]:
    """Returns (source, path): source is "override" if override_dir was
    given AND contains a <agent_name>.md file, else "default". Exposed
    separately from load_prompt() so callers can emit an audit event
    recording which file was actually used without reading it twice."""
    if override_dir:
        override_path = Path(override_dir) / f"{agent_name}.md"
        if override_path.is_file():
            return "override", override_path
    return "default", DEFAULT_PROMPTS_DIR / f"{agent_name}.md"


def load_prompt(agent_name: str, override_dir: Optional[str] = None) -> str:
    """Reads prompts/<agent_name>.md by default, or
    <override_dir>/<agent_name>.md if override_dir is given and that
    file exists there — falling back to the default file if the
    override dir doesn't contain this particular agent's file."""
    _, path = prompt_source(agent_name, override_dir)
    return path.read_text()
