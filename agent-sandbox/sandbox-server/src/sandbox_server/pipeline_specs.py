"""Reads/writes PipelineSpec YAML files under a specs directory — one file
per pipeline, named `<id>.yaml`. Mirrors specs.py's AgentSpec file store
(same `${VAR}` expansion, same skip-invalid-files-on-list behavior) with its
own local copy of that small helper rather than importing specs.py's private
one — same reasoning specs.py itself gives for not importing cli.py's.
"""

import os
import re
from pathlib import Path

import yaml
from pydantic import ValidationError

from sandbox_core.schemas.pipeline_spec import PipelineSpec

_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class PipelineSpecEnvVarError(RuntimeError):
    pass


class InvalidPipelineIdError(ValueError):
    pass


def _expand_env_vars(text: str, source: Path) -> str:
    def replace(match: re.Match) -> str:
        name = match.group(1)
        if name not in os.environ:
            raise PipelineSpecEnvVarError(
                f"{source} references ${{{name}}}, but that environment variable is not set"
            )
        return os.environ[name]

    return _ENV_VAR_PATTERN.sub(replace, text)


def pipeline_spec_path(specs_dir: Path, pipeline_id: str) -> Path:
    if not pipeline_id or "/" in pipeline_id or "\\" in pipeline_id or pipeline_id in (".", ".."):
        raise InvalidPipelineIdError(f"invalid pipeline id: {pipeline_id!r}")
    return specs_dir / f"{pipeline_id}.yaml"


def _parse(path: Path) -> PipelineSpec:
    text = _expand_env_vars(path.read_text(), path)
    return PipelineSpec.model_validate(yaml.safe_load(text))


def read_pipeline_spec(specs_dir: Path, pipeline_id: str) -> PipelineSpec | None:
    path = pipeline_spec_path(specs_dir, pipeline_id)
    if not path.exists():
        return None
    return _parse(path)


def list_pipeline_specs(specs_dir: Path) -> list[PipelineSpec]:
    if not specs_dir.exists():
        return []
    specs = []
    for path in sorted(specs_dir.glob("*.yaml")):
        try:
            specs.append(_parse(path))
        except (yaml.YAMLError, ValidationError, PipelineSpecEnvVarError):
            continue
    return specs


def write_pipeline_spec(specs_dir: Path, spec: PipelineSpec) -> None:
    specs_dir.mkdir(parents=True, exist_ok=True)
    path = pipeline_spec_path(specs_dir, spec.id)
    path.write_text(yaml.safe_dump(spec.model_dump(mode="json"), sort_keys=False))


def delete_pipeline_spec(specs_dir: Path, pipeline_id: str) -> bool:
    path = pipeline_spec_path(specs_dir, pipeline_id)
    if not path.exists():
        return False
    path.unlink()
    return True
