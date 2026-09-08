"""
Loads agent system prompts from agentic_ml/prompts/<agent_name>.md,
shipped as package data alongside this module, with an optional
per-agent override directory. Overrides are per-agent, not
all-or-nothing: if override_dir is given but doesn't contain a
particular agent's file, that agent silently falls back to the shipped
default — this is what lets a UI let someone edit ONE agent's prompt
without having to supply files for every agent.

Resolution of override_dir follows the same "explicit parameter, env-
var-backed default" pattern M0's data-root config (agentic_ml.paths)
and cli_common.resolve_model_endpoint already use: an explicit argument
wins if given, else AGENTIC_ML_PROMPT_OVERRIDE_DIR, else no override
(shipped defaults only).

DEFAULT_PROMPTS_DIR resolves relative to this module's own file
location (a sibling `prompts/` directory), not the repo root. That's
deliberate, not a convenience: `pip install <this repo> --target ...`
(the mechanism agent-sandbox's Docker-isolated gate execution uses to
import this package directly, see its runner/README.md and
docs/docker.md) installs only this package's own file tree into the
target — it does not carry along anything living outside src/agentic_ml/
in the source checkout. A repo-root prompts/ directory (this module's
original layout) would silently vanish under that install mode with no
error, just a FileNotFoundError deep inside load_prompt() the first
time a gate actually ran — which is exactly what happened. Resolving
via Path(__file__).parent instead means the prompt files travel with
the package under every install mode: editable install, sdist/wheel,
or `--target`. See pyproject.toml's [tool.setuptools.package-data] for
the packaging half of this fix — moving the files alone isn't enough,
setuptools also has to be told to actually include them in a built
wheel.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

DEFAULT_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

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
