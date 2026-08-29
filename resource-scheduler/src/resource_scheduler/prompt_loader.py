"""
Loads agent system prompts from prompts/<agent_name>.md at the repo
root, with an optional per-agent override directory. Mirrors
agentic_ml.prompt_loader in the sibling agentic-ml-classification
project — see that file's docstring for the reasoning behind the
resolution order.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

DEFAULT_PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"

PROMPT_OVERRIDE_DIR_ENV_VAR = "RESOURCE_SCHEDULER_PROMPT_OVERRIDE_DIR"


def resolve_prompt_override_dir(explicit: Optional[str] = None) -> Optional[str]:
    if explicit is not None:
        return explicit
    return os.environ.get(PROMPT_OVERRIDE_DIR_ENV_VAR)


def prompt_source(agent_name: str, override_dir: Optional[str] = None) -> tuple[str, Path]:
    if override_dir:
        override_path = Path(override_dir) / f"{agent_name}.md"
        if override_path.is_file():
            return "override", override_path
    return "default", DEFAULT_PROMPTS_DIR / f"{agent_name}.md"


def load_prompt(agent_name: str, override_dir: Optional[str] = None) -> str:
    _, path = prompt_source(agent_name, override_dir)
    return path.read_text()
