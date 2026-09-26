"""paths.py's root resolution, in particular the agent-sandbox gate case."""
from pathlib import Path

import pytest

from resource_scheduler.paths import run_dir, runs_root

_VARS = ("RESOURCE_SCHEDULER_RUNS_DIR", "RESOURCE_SCHEDULER_DATA_ROOT", "GATE_SCRATCH_DIR")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in _VARS:
        monkeypatch.delenv(var, raising=False)


def test_cwd_relative_by_default():
    assert runs_root() == Path("runs")


def test_gate_scratch_dir_gets_its_own_subdirectory(monkeypatch):
    # What an agent-sandbox gate container sets. The MCP server reads the
    # same files with RESOURCE_SCHEDULER_DATA_ROOT=/scratch/resource-scheduler.
    monkeypatch.setenv("GATE_SCRATCH_DIR", "/scratch")
    assert run_dir("r1") == Path("/scratch/resource-scheduler/runs/r1")
    monkeypatch.delenv("GATE_SCRATCH_DIR")
    monkeypatch.setenv("RESOURCE_SCHEDULER_DATA_ROOT", "/scratch/resource-scheduler")
    assert run_dir("r1") == Path("/scratch/resource-scheduler/runs/r1")


def test_explicit_settings_win_over_gate_scratch_dir(monkeypatch):
    monkeypatch.setenv("GATE_SCRATCH_DIR", "/scratch")
    monkeypatch.setenv("RESOURCE_SCHEDULER_DATA_ROOT", "/data")
    assert runs_root() == Path("/data/runs")
    monkeypatch.setenv("RESOURCE_SCHEDULER_RUNS_DIR", "/elsewhere")
    assert runs_root() == Path("/elsewhere")
