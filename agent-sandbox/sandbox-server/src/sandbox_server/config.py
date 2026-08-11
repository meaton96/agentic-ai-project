import os
from pathlib import Path

SPECS_DIR_ENV_VAR = "SANDBOX_AGENTS_DIR"
OUTPUT_ROOT_ENV_VAR = "SANDBOX_RUNS_DIR"

DEFAULT_SPECS_DIR = Path("./agents")
DEFAULT_OUTPUT_ROOT = Path("./runs")


def specs_dir() -> Path:
    override = os.environ.get(SPECS_DIR_ENV_VAR)
    return Path(override).expanduser() if override else DEFAULT_SPECS_DIR


def output_root() -> Path:
    override = os.environ.get(OUTPUT_ROOT_ENV_VAR)
    return Path(override).expanduser() if override else DEFAULT_OUTPUT_ROOT
