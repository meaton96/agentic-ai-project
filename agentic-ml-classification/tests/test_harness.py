"""
Run with: pytest tests/ -v
(from repo root, with src/ on PYTHONPATH — see conftest.py)
"""
import numpy as np
import pandas as pd
import pytest

from agentic_ml.harness.dataset import DatasetSpec, load_dataset
from agentic_ml.harness.splits import make_split
from agentic_ml.harness.leakage import (
    check_duplicate_rows_across_splits,
    check_group_overlap,
    check_time_ordering,
    check_suspicious_feature_correlation,
    label_permutation_test,
)
from agentic_ml.harness.sandbox import static_check, run_candidate_build
from agentic_ml.harness.metrics import compute_metrics
from agentic_ml.harness.profiler import profile_dataset


@pytest.fixture
def synthetic_df():
    rng = np.random.RandomState(0)
    n = 500
    return pd.DataFrame({
        "customer_id": rng.randint(0, 100, size=n),
        "day": rng.randint(0, 365, size=n),
        "x1": rng.normal(size=n),
        "target": rng.randint(0, 2, size=n),
    })


def test_split_determinism(synthetic_df):
    m1 = make_split(synthetic_df, "target", "hash1", strategy="stratified", seed=42)
    m2 = make_split(synthetic_df, "target", "hash1", strategy="stratified", seed=42)
    assert m1.train_idx == m2.train_idx
    assert m1.val_idx == m2.val_idx
    assert m1.test_idx == m2.test_idx


def test_split_different_seed_differs(synthetic_df):
    m1 = make_split(synthetic_df, "target", "hash1", strategy="stratified", seed=42)
    m3 = make_split(synthetic_df, "target", "hash1", strategy="stratified", seed=7)
    assert m1.train_idx != m3.train_idx


def test_split_no_index_overlap(synthetic_df):
    m = make_split(synthetic_df, "target", "hash1", strategy="stratified", seed=42)
    assert set(m.train_idx) & set(m.val_idx) == set()
    assert set(m.train_idx) & set(m.test_idx) == set()
    assert set(m.val_idx) & set(m.test_idx) == set()
    assert set(m.train_idx) | set(m.val_idx) | set(m.test_idx) == set(range(len(synthetic_df)))


def test_group_split_has_no_group_overlap(synthetic_df):
    m = make_split(synthetic_df, "target", "hash1", strategy="group", seed=42, group_column="customer_id")
    check = check_group_overlap(synthetic_df, "customer_id", m.train_idx, m.val_idx, m.test_idx)
    assert check.passed


def test_time_split_is_chronological(synthetic_df):
    m = make_split(synthetic_df, "target", "hash1", strategy="time", seed=42, time_column="day")
    check = check_time_ordering(synthetic_df, "day", m.train_idx, m.val_idx, m.test_idx, strategy="time")
    assert check.passed


def test_duplicate_row_leakage_detected():
    df = pd.DataFrame({"x": [1, 2, 3, 1], "target": [0, 1, 0, 0]})
    # row 0 and row 3 are exact duplicates; put one in train, one in test
    check = check_duplicate_rows_across_splits(df, train_idx=[0, 1], val_idx=[2], test_idx=[3])
    assert not check.passed


def test_suspicious_feature_correlation_catches_direct_leak():
    rng = np.random.RandomState(0)
    y = pd.Series(rng.randint(0, 2, size=200))
    X = pd.DataFrame({"noise": rng.normal(size=200), "leak": y.values})
    check = check_suspicious_feature_correlation(X, y)
    assert not check.passed
    assert "leak" in check.detail


def test_suspicious_feature_correlation_no_false_positive():
    rng = np.random.RandomState(0)
    y = pd.Series(rng.randint(0, 2, size=200))
    X = pd.DataFrame({"noise": rng.normal(size=200)})
    check = check_suspicious_feature_correlation(X, y)
    assert check.passed


def test_sandbox_rejects_forbidden_import():
    code = "import os\ndef build_pipeline(config):\n    return None\n"
    result = static_check(code)
    assert not result.passed


def test_sandbox_rejects_missing_build_pipeline():
    code = "x = 1\n"
    result = static_check(code)
    assert not result.passed


def test_sandbox_accepts_and_builds_legit_candidate():
    code = (
        "from sklearn.linear_model import LogisticRegression\n"
        "def build_pipeline(config):\n"
        "    return LogisticRegression(C=config.get('C', 1.0))\n"
    )
    pipeline, error = run_candidate_build(code, config={"C": 0.3}, timeout_seconds=15)
    assert error is None
    assert pipeline.C == 0.3


def test_sandbox_rejects_at_static_check_before_running():
    code = "import socket\ndef build_pipeline(config):\n    return None\n"
    pipeline, error = run_candidate_build(code, config={}, timeout_seconds=15)
    assert pipeline is None
    assert "static check failed" in error


def test_metrics_ci_bounds_contain_point_estimate():
    rng = np.random.RandomState(0)
    n = 300
    y_true = rng.randint(0, 2, size=n)
    y_proba = np.clip(y_true + rng.normal(0, 0.3, size=n), 0, 1)
    y_pred = (y_proba > 0.5).astype(int)
    results = compute_metrics(y_true, y_pred, y_proba, ["roc_auc"], n_bootstrap=200, seed=1)
    r = results["roc_auc"]
    assert r.ci_low <= r.value <= r.ci_high


def test_profiler_does_not_flag_continuous_float_as_id():
    """Regression test: a continuous float column (e.g. Fare) with near-100%
    unique values must NOT be flagged as an ID column just because of high
    cardinality — that heuristic should only fire for int/string columns."""
    rng = np.random.RandomState(0)
    n = 300
    df = pd.DataFrame({
        "row_id": range(n),                     # true ID (integer, sequential)
        "fare": rng.exponential(30, size=n),      # continuous float, high cardinality, NOT an id
        "target": rng.randint(0, 2, size=n),
    })
    report = profile_dataset(df, target_column="target")
    assert "row_id" in report.likely_id_columns
    assert "fare" not in report.likely_id_columns


def test_profiler_detects_group_and_time_columns():
    rng = np.random.RandomState(0)
    rows = []
    for cust_id in range(50):
        for e in range(rng.randint(1, 5)):
            rows.append({"customer_id": cust_id, "signup_day": e * 3, "x": rng.normal()})
    df = pd.DataFrame(rows)
    df["target"] = rng.randint(0, 2, size=len(df))
    report = profile_dataset(df, target_column="target")
    assert "customer_id" in report.likely_group_columns
    assert "signup_day" in report.likely_datetime_columns
    assert report.recommended_split_strategy == "group_time"


def test_profiler_recommends_stratified_with_no_group_or_time():
    rng = np.random.RandomState(0)
    df = pd.DataFrame({"x": rng.normal(size=100), "target": rng.randint(0, 2, size=100)})
    report = profile_dataset(df, target_column="target")
    assert report.recommended_split_strategy == "stratified"


def test_profiler_flags_high_missingness_column():
    rng = np.random.RandomState(0)
    n = 200
    df = pd.DataFrame({
        "mostly_missing": [None] * int(n * 0.8) + list(rng.normal(size=n - int(n * 0.8))),
        "target": rng.randint(0, 2, size=n),
    })
    report = profile_dataset(df, target_column="target")
    assert any("mostly_missing" in flag for flag in report.leakage_risk_flags)


def test_profiler_flags_direct_feature_leak():
    rng = np.random.RandomState(0)
    y = rng.randint(0, 2, size=200)
    df = pd.DataFrame({"noise": rng.normal(size=200), "leak": y, "target": y})
    report = profile_dataset(df, target_column="target")
    assert len(report.leakage_risk_flags) > 0