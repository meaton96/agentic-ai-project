"""
Per-tool contract tests for the aviation pipeline. Runs run_pipeline once
(deterministic), then asserts each tool's invariants on its trace result and the
on-disk artifacts (feats.csv, preds.json, queue). Independent of the frozen golden
values -- these catch contract breaks (esp. featurize<->model columns) that a
value-only comparison could miss.
"""
import sys, csv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pytest
from tools_lib import (load_artifacts, inspect_invariants, featurize_contract,
                       classify_invariants, recommend_invariants)
from golden_lib import clean_workdir


@pytest.fixture(scope="module")
def run(paths, run_pipeline):
    clean_workdir(paths["workdir"])          # recommend appends; start fresh
    out = run_pipeline(flight_dir=paths["flight_dir"], metadata=paths["metadata"],
                       model=paths["model"], workdir=paths["workdir"], top_k=paths["top_k"],
                       spec=paths["spec"])
    art = load_artifacts(paths["workdir"], paths["model"])
    res = {s["step"]: s["result"] for s in out["trace"]}
    with open(paths["metadata"]) as f:
        n_meta = sum(1 for _ in csv.DictReader(f))
    return {"out": out, "res": res, "art": art, "n_meta": n_meta, "top_k": paths["top_k"]}


def test_inspect(run):
    assert not inspect_invariants(run["res"]["inspect"], run["n_meta"])

def test_featurize_model_contract(run):
    n_flights = run["res"]["inspect"]["n_flights"]
    assert not featurize_contract(run["art"]["feats"], run["art"]["model_feats"], n_flights)

def test_classify(run):
    n_flights = run["res"]["inspect"]["n_flights"]
    assert not classify_invariants(run["art"]["preds"], n_flights)

def test_recommend(run):
    assert not recommend_invariants(run["art"]["queue"], run["art"]["preds"], run["top_k"])