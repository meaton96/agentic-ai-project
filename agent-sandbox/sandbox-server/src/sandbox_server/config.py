import os
from pathlib import Path

SPECS_DIR_ENV_VAR = "SANDBOX_AGENTS_DIR"
PIPELINES_DIR_ENV_VAR = "SANDBOX_PIPELINES_DIR"
OUTPUT_ROOT_ENV_VAR = "SANDBOX_RUNS_DIR"
PIPELINE_RUNS_ROOT_ENV_VAR = "SANDBOX_PIPELINE_RUNS_DIR"
OPERATIONS_ROOT_ENV_VAR = "SANDBOX_OPERATIONS_DIR"

DEFAULT_SPECS_DIR = Path("./agents")
DEFAULT_PIPELINES_DIR = Path("./pipelines")
DEFAULT_OUTPUT_ROOT = Path("./runs")
# Same "not nested under another root" reasoning as DEFAULT_PIPELINE_RUNS_ROOT
# below: an operation log is keyed by spec id, not run_id, and has nothing to
# do with output_root's per-run directories.
DEFAULT_OPERATIONS_ROOT = Path("./operations")
# Deliberately NOT a subdirectory of DEFAULT_OUTPUT_ROOT: RunManager.list_runs()
# treats every immediate child directory of output_root as a candidate agent
# run (it has no way to tell "not one of mine" apart), so a pipeline-run
# manifest living anywhere under ./runs — even nested — would show up as a
# phantom entry in the plain agent-runs list. Steps' own per-agent-run
# EventLogs still live under output_root, same as any other agent run; only
# the pipeline-level manifest (pipeline.json) needs to live elsewhere.
DEFAULT_PIPELINE_RUNS_ROOT = Path("./pipeline-runs")


def specs_dir() -> Path:
    override = os.environ.get(SPECS_DIR_ENV_VAR)
    return Path(override).expanduser() if override else DEFAULT_SPECS_DIR


def pipelines_dir() -> Path:
    override = os.environ.get(PIPELINES_DIR_ENV_VAR)
    return Path(override).expanduser() if override else DEFAULT_PIPELINES_DIR


def output_root() -> Path:
    override = os.environ.get(OUTPUT_ROOT_ENV_VAR)
    return Path(override).expanduser() if override else DEFAULT_OUTPUT_ROOT


def pipeline_runs_root() -> Path:
    override = os.environ.get(PIPELINE_RUNS_ROOT_ENV_VAR)
    return Path(override).expanduser() if override else DEFAULT_PIPELINE_RUNS_ROOT


def operations_root() -> Path:
    override = os.environ.get(OPERATIONS_ROOT_ENV_VAR)
    return Path(override).expanduser() if override else DEFAULT_OPERATIONS_ROOT
