"""
agentic_ml.paths root resolution. The design claims worth proving:

1. GATE_SCRATCH_DIR alone is enough to place runs/artifacts/datasets on
   agent-sandbox's per-account scratch volume — that's what lets the
   runner stop hardcoding AGENTIC_ML_DATA_ROOT into every gate container.
2. AGENTIC_ML_DATA_ROOT still takes precedence over GATE_SCRATCH_DIR, so
   the in-process/CLI path and explicit local overrides are unaffected.
3. The root-specific vars still beat both, and with nothing set the
   cwd-relative defaults are exactly what they were before.
"""
from pathlib import Path

import pytest

from agentic_ml import paths

ROOT_VARS = (
    "AGENTIC_ML_DATA_ROOT",
    "GATE_SCRATCH_DIR",
    "AGENTIC_ML_RUNS_DIR",
    "AGENTIC_ML_ARTIFACTS_DIR",
    "AGENTIC_ML_DATASETS_DIR",
)

ROOT_FUNCS = [
    (paths.runs_root, "runs"),
    (paths.artifacts_root, "artifacts"),
    (paths.datasets_root, "datasets"),
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ROOT_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.mark.parametrize("root_func,subdir", ROOT_FUNCS)
def test_only_data_root_set_resolves_under_data_root(root_func, subdir, monkeypatch):
    monkeypatch.setenv("AGENTIC_ML_DATA_ROOT", "/data/root")
    assert root_func() == Path("/data/root") / subdir


@pytest.mark.parametrize("root_func,subdir", ROOT_FUNCS)
def test_only_gate_scratch_dir_set_resolves_under_scratch(root_func, subdir, monkeypatch):
    monkeypatch.setenv("GATE_SCRATCH_DIR", "/scratch")
    assert root_func() == Path("/scratch") / subdir


@pytest.mark.parametrize("root_func,subdir", ROOT_FUNCS)
def test_data_root_wins_over_gate_scratch_dir(root_func, subdir, monkeypatch):
    monkeypatch.setenv("AGENTIC_ML_DATA_ROOT", "/data/root")
    monkeypatch.setenv("GATE_SCRATCH_DIR", "/scratch")
    assert root_func() == Path("/data/root") / subdir


@pytest.mark.parametrize("root_func,subdir", ROOT_FUNCS)
def test_neither_set_is_cwd_relative(root_func, subdir):
    assert root_func() == Path(subdir)


def test_empty_data_root_falls_through_to_gate_scratch_dir(monkeypatch):
    monkeypatch.setenv("AGENTIC_ML_DATA_ROOT", "")
    monkeypatch.setenv("GATE_SCRATCH_DIR", "/scratch")
    assert paths.runs_root() == Path("/scratch") / "runs"


@pytest.mark.parametrize("specific_env,root_func", [
    ("AGENTIC_ML_RUNS_DIR", paths.runs_root),
    ("AGENTIC_ML_ARTIFACTS_DIR", paths.artifacts_root),
    ("AGENTIC_ML_DATASETS_DIR", paths.datasets_root),
])
def test_specific_env_wins_over_both_roots(specific_env, root_func, monkeypatch):
    monkeypatch.setenv("AGENTIC_ML_DATA_ROOT", "/data/root")
    monkeypatch.setenv("GATE_SCRATCH_DIR", "/scratch")
    monkeypatch.setenv(specific_env, "/explicit")
    assert root_func() == Path("/explicit")


def test_derived_paths_follow_gate_scratch_dir(monkeypatch):
    monkeypatch.setenv("GATE_SCRATCH_DIR", "/scratch")
    assert paths.run_dir("r1") == Path("/scratch/runs/r1")
    assert paths.leaderboard_path() == Path("/scratch/artifacts/reports/leaderboard.jsonl")
    assert paths.models_dir() == Path("/scratch/artifacts/models")
