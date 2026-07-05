"""Shared fixtures for the aviation pipeline test suite.
"""
import os
import sys
from pathlib import Path
import pytest


REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
sys.path.insert(0, str(REPO))           
sys.path.insert(0, str(REPO / "scripts"))



UPDATE_GOLDEN = 0

 
 
@pytest.fixture(scope="session")
def paths():
    return {
        "flight_dir": DATA / "c28_demo/flights",
        "metadata":   DATA / "c28_demo" / "metadata.csv",
        "model":      DATA / "c28_model.joblib",
        "spec":       DATA / "best_spec.json",   # frozen artifact from nb04 (LLM search is upstream)
        "workdir":    DATA / "tests" / "_golden_work",
        "golden":     DATA / "tests" / "golden" / "aviation_pipeline.json",
        "top_k":      10,
    }
 
 
@pytest.fixture(scope="session")
def run_pipeline():
    # EDIT if your package name differs
    from orchestrator_sim import run_pipeline as rp
    return rp

