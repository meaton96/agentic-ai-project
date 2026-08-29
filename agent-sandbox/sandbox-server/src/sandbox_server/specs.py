"""Reads/writes AgentSpec YAML files under a specs directory — one file per
agent, named `<id>.yaml`. This is the only place on disk agents live in M3;
no database.

Supports the same `${VAR}` env-var placeholder syntax as sandbox_core.cli's
loader (so a spec committed for `sandbox run` and one edited through this
server behave identically) — kept as a small local copy rather than importing
sandbox_core.cli's private helper, since sandbox-core's own CLI internals
aren't part of its public surface.
"""

import os
import re
from pathlib import Path

import yaml
from pydantic import ValidationError

from sandbox_core.schemas.agent_spec import AgentSpec

_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class AgentSpecEnvVarError(RuntimeError):
    pass


class InvalidAgentIdError(ValueError):
    pass


def _expand_env_vars(text: str, source: Path) -> str:
    def replace(match: re.Match) -> str:
        name = match.group(1)
        if name not in os.environ:
            raise AgentSpecEnvVarError(
                f"{source} references ${{{name}}}, but that environment variable is not set"
            )
        return os.environ[name]

    return _ENV_VAR_PATTERN.sub(replace, text)


def agent_spec_path(specs_dir: Path, agent_id: str) -> Path:
    if not agent_id or "/" in agent_id or "\\" in agent_id or agent_id in (".", ".."):
        raise InvalidAgentIdError(f"invalid agent id: {agent_id!r}")
    return specs_dir / f"{agent_id}.yaml"


def _parse(path: Path) -> AgentSpec:
    text = _expand_env_vars(path.read_text(), path)
    return AgentSpec.model_validate(yaml.safe_load(text))


def read_agent_spec(specs_dir: Path, agent_id: str) -> AgentSpec | None:
    path = agent_spec_path(specs_dir, agent_id)
    if not path.exists():
        return None
    return _parse(path)


def list_agent_specs(specs_dir: Path) -> list[AgentSpec]:
    """Skips files that fail to parse or validate rather than failing the
    whole listing — one bad YAML file shouldn't take down the agent list."""
    if not specs_dir.exists():
        return []
    specs = []
    for path in sorted(specs_dir.glob("*.yaml")):
        try:
            specs.append(_parse(path))
        except (yaml.YAMLError, ValidationError, AgentSpecEnvVarError):
            continue
    return specs


def write_agent_spec(specs_dir: Path, spec: AgentSpec) -> None:
    specs_dir.mkdir(parents=True, exist_ok=True)
    path = agent_spec_path(specs_dir, spec.id)
    path.write_text(yaml.safe_dump(spec.model_dump(mode="json"), sort_keys=False))


def delete_agent_spec(specs_dir: Path, agent_id: str) -> bool:
    path = agent_spec_path(specs_dir, agent_id)
    if not path.exists():
        return False
    path.unlink()
    return True
