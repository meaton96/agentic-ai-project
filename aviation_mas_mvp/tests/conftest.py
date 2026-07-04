"""Shared fixtures for the aviation pipeline test suite.

Adjust REPO/paths if your layout differs. The golden run uses a FIXED workdir so
any path that leaks into output is at least stable across runs.
"""
import os
import sys
from pathlib import Path
import pytest

DATA = Path("data").resolve()
sys.path.insert(0, "../scripts")

UPDATE_GOLDEN = os.getenv("UPDATE_GOLDEN") == "1"


@pytest.fixture(scope="session")
def paths():
    return {
        "flight_dir": DATA / "c28_demo/flights",
        "metadata":   DATA / "c28_demo" / "metadata.csv",
        "model":      DATA / "c28_model.joblib",
        "workdir":    DATA / "tests" / "_golden_work",
        "golden":     DATA / "tests" / "golden" / "aviation_pipeline.json",
        "top_k":      10,
    }


@pytest.fixture(scope="session")
def run_pipeline():
    # EDIT if your package name differs
    from orchestrator_sim import run_pipeline as rp
    return rp
