"""
Run with: pytest tests/ -v
(from repo root, with src/ on PYTHONPATH — see conftest.py)
"""
import numpy as np
import pandas as pd
import pytest

from agentic_ml.harness.dataset import DatasetSpec, detect_dataset_shape, load_dataset, read_dataframe
from agentic_ml.harness.splits import make_split, resolve_split_columns
from agentic_ml.harness.leakage import (
    check_duplicate_rows_across_splits,
    check_fold_class_presence,
    check_group_overlap,
    check_time_ordering,
    check_suspicious_feature_correlation,
    label_permutation_test,
    run_all_split_leakage_checks,
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


def test_time_ordering_not_enforced_on_strategies_that_never_promised_it():
    """Regression: a 'group' split with a declared time column failed
    check_time_ordering unconditionally — the strict train<val<test
    ordering is a promise only strategy='time' makes, but the check ran
    it for every strategy except group_time. On NGAFID (intake declares
    time_column='date_diff', profiler recommends 'group') this failed
    every split and dead-ended the run."""
    rng = np.random.RandomState(0)
    n = 60
    df = pd.DataFrame({
        "customer_id": rng.randint(0, 12, size=n),
        "day": rng.randint(-2, 3, size=n),
        "target": rng.randint(0, 2, size=n),
    })
    m = make_split(df, "target", "hash1", strategy="group", seed=42, group_column="customer_id")
    # group split makes no ordering promise -> must pass with a note
    check = check_time_ordering(df, "day", m.train_idx, m.val_idx, m.test_idx, strategy="group")
    assert check.passed
    # but the same folds MUST still fail under the strict 'time' contract,
    # proving the check itself wasn't weakened
    check_strict = check_time_ordering(df, "day", m.train_idx, m.val_idx, m.test_idx, strategy="time")
    assert not check_strict.passed


def test_fold_class_presence_passes_when_all_folds_have_both_classes():
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
    check = check_fold_class_presence(y, train_idx=[0, 1, 2, 3], val_idx=[4, 5], test_idx=[6, 7])
    assert check.passed


def test_fold_class_presence_detects_single_class_fold():
    # val fold is entirely class 0 — the exact shape of the NGAFID bug,
    # where roc_auc_score silently returns NaN instead of raising.
    y = pd.Series([0, 1, 0, 1, 0, 0, 0, 1])
    check = check_fold_class_presence(y, train_idx=[0, 1, 2, 3], val_idx=[4, 5], test_idx=[6, 7])
    assert not check.passed
    assert "val" in check.detail


def test_group_time_split_with_target_correlated_group_order_fails_leakage_gate():
    """Reproduces the real NGAFID failure: group_time sorts groups by
    their earliest timestamp, and if that ordering happens to correlate
    with the target (e.g. groups whose only rows are the "after" class
    sort last), the val/test folds end up single-class. The three
    pre-existing leakage checks (duplicates/group-overlap/time-ordering)
    all pass in this scenario — only check_fold_class_presence catches
    it, which is why it has to be part of run_all_split_leakage_checks,
    not just a standalone function nobody calls."""
    rows = []
    for group_id in range(20):
        # groups 0-14: both before(1)/after(0) rows, earliest time early.
        # groups 15-19: ONLY after(0) rows, earliest time late — these
        # sort last under group_time and land in val/test.
        if group_id < 15:
            rows.append({"group": group_id, "t": 0, "target": 1})
            rows.append({"group": group_id, "t": 1, "target": 0})
        else:
            rows.append({"group": group_id, "t": 5, "target": 0})
            rows.append({"group": group_id, "t": 6, "target": 0})
    df = pd.DataFrame(rows)

    m = make_split(df, "target", "hash1", strategy="group_time", seed=42,
                    group_column="group", time_column="t")
    checks = run_all_split_leakage_checks(
        df=df, target_column="target", group_column="group", time_column="t",
        train_idx=m.train_idx, val_idx=m.val_idx, test_idx=m.test_idx, strategy="group_time",
    )
    by_name = {c.check_name: c for c in checks}
    assert not by_name["fold_class_presence"].passed
    assert not all(c.passed for c in checks)


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


def _synthetic_multiclass(n=300, n_classes=3, seed=0):
    rng = np.random.RandomState(seed)
    y_true = rng.randint(0, n_classes, size=n)
    # proba peaked on the true class plus noise, then row-normalized —
    # a stand-in for a reasonably well-calibrated multiclass classifier.
    logits = rng.normal(0, 0.5, size=(n, n_classes))
    logits[np.arange(n), y_true] += 2.0
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    y_proba = exp / exp.sum(axis=1, keepdims=True)
    y_pred = y_proba.argmax(axis=1)
    return y_true, y_pred, y_proba


def test_compute_metrics_handles_multiclass_target():
    y_true, y_pred, y_proba = _synthetic_multiclass()
    results = compute_metrics(
        y_true, y_pred, y_proba,
        ["roc_auc", "pr_auc", "f1", "precision", "recall", "accuracy", "brier"],
        n_bootstrap=50, seed=1,
    )
    for name, r in results.items():
        assert 0.0 <= r.value <= 1.0, f"{name} out of [0, 1]: {r.value}"
        assert r.ci_low <= r.value <= r.ci_high


def test_compute_metrics_multiclass_beats_chance():
    y_true, y_pred, y_proba = _synthetic_multiclass()
    results = compute_metrics(y_true, y_pred, y_proba, ["roc_auc", "accuracy"], n_bootstrap=50, seed=1)
    assert results["roc_auc"].value > 0.7
    assert results["accuracy"].value > 0.5


def test_roc_auc_any_matches_binary_roc_auc_score():
    from sklearn.metrics import roc_auc_score
    from agentic_ml.harness.metrics import roc_auc_any

    rng = np.random.RandomState(2)
    n = 200
    y_true = rng.randint(0, 2, size=n)
    y_proba_pos = np.clip(y_true + rng.normal(0, 0.3, size=n), 0, 1)
    y_proba = np.column_stack([1 - y_proba_pos, y_proba_pos])
    assert roc_auc_any(y_true, y_proba) == pytest.approx(roc_auc_score(y_true, y_proba_pos))


def test_roc_auc_any_handles_multiclass():
    from agentic_ml.harness.metrics import roc_auc_any

    y_true, _, y_proba = _synthetic_multiclass()
    value = roc_auc_any(y_true, y_proba)
    assert 0.7 < value <= 1.0


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


def test_profiler_does_not_treat_relative_offset_column_as_datetime():
    """'date_diff' contains the datetime-name-hint token 'date' but holds
    a small-integer relative OFFSET, not an absolute point in time —
    the exact NGAFID column that got misclassified, causing group_time
    to sort groups by an offset that correlated with the target instead
    of by real calendar time. Only the name-hint fallback is guarded;
    a column that genuinely parses as a date must still be detected."""
    rng = np.random.RandomState(0)
    n = 100
    df = pd.DataFrame({
        "plane_id": rng.randint(0, 20, size=n),
        "date_diff": rng.randint(-2, 3, size=n),
        "x": rng.normal(size=n),
        "target": rng.randint(0, 2, size=n),
    })
    report = profile_dataset(df, target_column="target")
    assert "date_diff" not in report.likely_datetime_columns
    assert report.recommended_split_strategy != "group_time"


def test_profile_fact_marks_declared_columns_the_validator_will_reject():
    """Regression: the modeling agent is told 'never include a declared
    group/time column' but the profile it sees carried no trace of which
    columns were declared — once 'date_diff' stopped being (mis)flagged
    is_likely_datetime, nothing marked it at all, and a real NGAFID run
    burned 8 modeling iterations proposing it (each correctly rejected
    by _validate_candidate_columns). The agent-facing fact's
    excluded_columns must agree with the validator: every excluded
    column, proposed as a feature, must actually be rejected."""
    from agentic_ml.steps.modeling_step import _validate_candidate_columns
    from agentic_ml.tools.profiler_tool import build_profile_fact

    rng = np.random.RandomState(0)
    n = 100
    df = pd.DataFrame({
        "plane_id": rng.randint(0, 20, size=n),
        "date_diff": rng.randint(-2, 3, size=n),
        "x": rng.normal(size=n),
        "target": rng.randint(0, 2, size=n),
    })
    fact = build_profile_fact(df, "target", group_column="plane_id", time_column="date_diff")

    assert fact["declared_group_column"] == "plane_id"
    assert fact["declared_time_column"] == "date_diff"
    # the exact NGAFID blind spot: declared time column, NOT datetime-flagged
    assert "date_diff" in fact["excluded_columns"]
    assert "date_diff" not in fact["likely_datetime_columns"]

    for col in fact["excluded_columns"]:
        errors = _validate_candidate_columns(
            fact, target_column="target", group_column="plane_id", time_column="date_diff",
            config={"numeric_cols": [col, "x"], "categorical_cols": []},
        )
        assert errors, f"validator accepted excluded column {col!r}"
    # sanity: an ordinary feature column still passes
    assert _validate_candidate_columns(
        fact, target_column="target", group_column="plane_id", time_column="date_diff",
        config={"numeric_cols": ["x"], "categorical_cols": []},
    ) == []


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


# --- resolve_split_columns: regression coverage for the bug where the
# profiler's recommended_split_strategy (derived purely from its own
# heuristic detection) outran what intake actually declared as
# group_column/time_column, and make_split() raised a bare ValueError
# deep in the call stack instead of the harness reconciling the two. ---

def _group_time_profile_report():
    rng = np.random.RandomState(0)
    rows = []
    for cust_id in range(50):
        for e in range(rng.randint(1, 5)):
            rows.append({"customer_id": cust_id, "signup_day": e * 3, "x": rng.normal()})
    df = pd.DataFrame(rows)
    df["target"] = rng.randint(0, 2, size=len(df))
    return profile_dataset(df, target_column="target").to_dict()


def test_resolve_split_columns_autoadopts_detected_time_column_when_undeclared():
    report = _group_time_profile_report()
    assert report["recommended_split_strategy"] == "group_time"

    group_column, time_column, notes = resolve_split_columns(
        "group_time", group_column=None, time_column=None, profiler_report=report,
    )
    assert group_column == "customer_id"
    assert time_column == "signup_day"
    assert len(notes) == 2


def test_resolve_split_columns_leaves_already_declared_columns_untouched():
    report = _group_time_profile_report()
    group_column, time_column, notes = resolve_split_columns(
        "group_time", group_column="customer_id", time_column="signup_day", profiler_report=report,
    )
    assert group_column == "customer_id"
    assert time_column == "signup_day"
    assert notes == []


def test_resolve_split_columns_noop_for_stratified():
    report = _group_time_profile_report()
    group_column, time_column, notes = resolve_split_columns(
        "stratified", group_column=None, time_column=None, profiler_report=report,
    )
    assert group_column is None
    assert time_column is None
    assert notes == []


def test_resolve_split_columns_partial_override_only_fills_missing_one():
    """If group_column was explicitly declared but time_column wasn't,
    only the missing one gets auto-resolved."""
    report = _group_time_profile_report()
    group_column, time_column, notes = resolve_split_columns(
        "group_time", group_column="customer_id", time_column=None, profiler_report=report,
    )
    assert group_column == "customer_id"
    assert time_column == "signup_day"
    assert len(notes) == 1
    assert "time_column" in notes[0]


# --- read_dataframe's size guard ---
#
# Regression coverage for a real incident: pointing a raw, long-format
# time-series CSV (28.7M rows, 4GB) directly at this pipeline caused an
# out-of-memory crash, because read_dataframe() — the only code path
# that reads a dataset file — had no guard and attempted a single
# pd.read_csv() over the whole file. These tests use tiny files with an
# explicit max_bytes override rather than an actual multi-GB fixture;
# the file content is deliberately NOT valid CSV, so a passing test
# proves the size check happens before any parse attempt, not just
# that parsing eventually fails.

def test_read_dataframe_rejects_a_file_over_the_size_limit(tmp_path):
    path = tmp_path / "too_big.csv"
    path.write_bytes(b"not,even,valid,csv,content" * 10)  # ~270 bytes
    with pytest.raises(ValueError, match="over the"):
        read_dataframe(path, max_bytes=100)  # smaller than the file


def test_read_dataframe_accepts_a_file_within_the_size_limit(tmp_path):
    path = tmp_path / "fine.csv"
    path.write_text("a,b,target\n1,2,0\n3,4,1\n")
    df = read_dataframe(path, max_bytes=1_000_000)
    assert list(df.columns) == ["a", "b", "target"]
    assert len(df) == 2


def test_read_dataframe_size_limit_overridable_via_env_var(tmp_path, monkeypatch):
    path = tmp_path / "too_big.csv"
    path.write_bytes(b"not,even,valid,csv,content" * 10)
    monkeypatch.setenv("AGENTIC_ML_MAX_DATASET_BYTES", "100")
    with pytest.raises(ValueError, match="over the"):
        read_dataframe(path)  # no explicit max_bytes -> falls back to the env var


# --- detect_dataset_shape: the routing decision upstream of the size guard ---
#
# The size guard above makes an oversized load fail cleanly; it doesn't
# route anything. detect_dataset_shape is the cheap, sample-based peek
# that decides whether a raw long-format time-series file (many
# consecutive rows per example) should be routed through the
# featurize_timeseries agent before ever reaching read_dataframe() at
# all. Uses `min_bytes_to_flag=0` on most of these to keep fixtures
# small/fast; the dedicated size-gate tests below use real file-size
# differences instead, since that gate is the whole point there.

def _write_long_format_csv(path, n_groups=20, rows_per_group=50, seed=0):
    """Synthetic long-format data: contiguous runs of an id column,
    matching the exact structural shape build_flight_feature_table_streaming
    requires (see tests/test_timeseries_features.py's _make_long_csv for
    the same pattern)."""
    rng = np.random.RandomState(seed)
    rows = []
    for gid in range(n_groups):
        for _ in range(rows_per_group):
            rows.append({"id": gid, "group": f"g{gid % 3}", "s1": rng.normal(), "s2": rng.normal()})
    pd.DataFrame(rows).to_csv(path, index=False)


def test_detect_dataset_shape_flags_long_format_repeated_id_column(tmp_path):
    path = tmp_path / "long.csv"
    _write_long_format_csv(path, n_groups=20, rows_per_group=50)
    result = detect_dataset_shape(path, min_bytes_to_flag=0)
    assert result["looks_long_format"] is True
    assert result["repeated_run_column"] == "id"
    assert result["avg_run_length"] == pytest.approx(50.0, rel=0.1)


def test_detect_dataset_shape_does_not_flag_already_tabular_data(synthetic_df, tmp_path):
    path = tmp_path / "tabular.csv"
    synthetic_df.to_csv(path, index=False)  # one row per example, no repeated-run structure
    result = detect_dataset_shape(path, min_bytes_to_flag=0)
    assert result["looks_long_format"] is False


def test_detect_dataset_shape_size_gate_suppresses_a_small_false_positive(tmp_path):
    # A small dataset pre-sorted by a low-cardinality category (e.g. Iris
    # sorted by Species — the actual false positive this test reproduces,
    # found by running detect_dataset_shape against the real datasets/raw/
    # files in this repo) looks exactly like a repeated grouping key
    # structurally. The size gate is what tells them apart: a file this
    # small was never going to cause the memory problem this detector
    # exists to route around.
    path = tmp_path / "small_sorted.csv"
    rows = []
    for species in ("a", "b", "c"):
        for i in range(50):
            rows.append({"sepal_length": float(i), "species": species})
    pd.DataFrame(rows).to_csv(path, index=False)

    default_gate = detect_dataset_shape(path)
    assert default_gate["looks_long_format"] is False
    assert default_gate["avg_run_length"] >= 5.0  # the run-structure signal alone WOULD have fired

    lowered_gate = detect_dataset_shape(path, min_bytes_to_flag=0)
    assert lowered_gate["looks_long_format"] is True  # confirms the size gate, not something else, is what suppressed it


def test_detect_dataset_shape_raises_for_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        detect_dataset_shape(tmp_path / "does_not_exist.csv")