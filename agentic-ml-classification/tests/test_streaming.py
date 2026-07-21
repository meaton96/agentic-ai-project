"""
Harness-level correctness for Phase 9's two new deterministic modules:
harness/streaming.py (batch simulation) and harness/drift.py (drift
measurement). No LLM involved — see tests/test_streaming_monitor.py for
the orchestrator-level (stubbed-client) scenarios.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from agentic_ml.harness.drift import compute_drift_report
from agentic_ml.harness.streaming import simulate_batches


# --- harness/streaming.py ---

def _synthetic_grouped_df(n_groups=30, rows_per_group=4, seed=0):
    rng = np.random.RandomState(seed)
    rows = []
    for g in range(n_groups):
        for r in range(rows_per_group):
            rows.append({
                "id": g * rows_per_group + r,
                "plane_id": g,
                "feature_a": rng.normal(0, 1),
                "label": rng.binomial(1, 0.5),
            })
    return pd.DataFrame(rows)


def test_simulate_batches_never_splits_a_group_across_the_boundary():
    df = _synthetic_grouped_df(n_groups=30, rows_per_group=4)
    initial_df, batches = simulate_batches(
        df, group_column="plane_id", id_column="id",
        n_initial_groups=10, batch_size_groups=5, seed=42,
    )
    outputs = [initial_df] + batches
    group_sets = [set(o["plane_id"]) for o in outputs]

    # every group's rows land together in exactly one output
    all_touched_groups = set()
    for gs in group_sets:
        assert not (gs & all_touched_groups), "a group appeared in more than one output"
        all_touched_groups |= gs
    assert all_touched_groups == set(df["plane_id"])

    # a group never has some rows in one output and some in another
    for g in df["plane_id"].unique():
        n_outputs_containing_g = sum(1 for gs in group_sets if g in gs)
        assert n_outputs_containing_g == 1


def test_simulate_batches_accounts_for_every_row_exactly_once():
    df = _synthetic_grouped_df(n_groups=23, rows_per_group=3)
    initial_df, batches = simulate_batches(
        df, group_column="plane_id", id_column="id",
        n_initial_groups=8, batch_size_groups=4, seed=7,
    )
    all_ids = pd.concat([initial_df] + batches)["id"].sort_values().tolist()
    assert all_ids == sorted(df["id"].tolist())


def test_simulate_batches_is_deterministic_given_the_same_seed():
    df = _synthetic_grouped_df(n_groups=20, rows_per_group=2)
    initial_a, batches_a = simulate_batches(
        df, group_column="plane_id", id_column="id",
        n_initial_groups=5, batch_size_groups=3, seed=123,
    )
    initial_b, batches_b = simulate_batches(
        df, group_column="plane_id", id_column="id",
        n_initial_groups=5, batch_size_groups=3, seed=123,
    )
    pd.testing.assert_frame_equal(initial_a, initial_b)
    assert len(batches_a) == len(batches_b)
    for a, b in zip(batches_a, batches_b):
        pd.testing.assert_frame_equal(a, b)


def test_simulate_batches_different_seeds_produce_different_group_order():
    df = _synthetic_grouped_df(n_groups=20, rows_per_group=2)
    initial_a, _ = simulate_batches(
        df, group_column="plane_id", id_column="id",
        n_initial_groups=5, batch_size_groups=3, seed=1,
    )
    initial_b, _ = simulate_batches(
        df, group_column="plane_id", id_column="id",
        n_initial_groups=5, batch_size_groups=3, seed=2,
    )
    assert set(initial_a["plane_id"]) != set(initial_b["plane_id"])


def test_simulate_batches_last_batch_may_be_smaller():
    df = _synthetic_grouped_df(n_groups=12, rows_per_group=2)  # 12 groups
    _, batches = simulate_batches(
        df, group_column="plane_id", id_column="id",
        n_initial_groups=5, batch_size_groups=4, seed=0,  # 7 remaining groups -> batches of 4, 3
    )
    assert len(batches) == 2
    assert batches[0]["plane_id"].nunique() == 4
    assert batches[1]["plane_id"].nunique() == 3


def test_simulate_batches_rejects_n_initial_groups_leaving_nothing_for_batches():
    df = _synthetic_grouped_df(n_groups=5, rows_per_group=2)
    with pytest.raises(ValueError, match="n_initial_groups"):
        simulate_batches(df, group_column="plane_id", id_column="id",
                          n_initial_groups=5, batch_size_groups=1, seed=0)


def test_simulate_batches_rejects_unknown_group_column():
    df = _synthetic_grouped_df(n_groups=5, rows_per_group=2)
    with pytest.raises(ValueError, match="group_column"):
        simulate_batches(df, group_column="does_not_exist", id_column="id",
                          n_initial_groups=2, batch_size_groups=1, seed=0)


# --- harness/drift.py ---

def _baseline_and_batch(n=200, shift=0.0, seed=0):
    rng = np.random.RandomState(seed)
    baseline = pd.DataFrame({
        "f1": rng.normal(0, 1, n),
        "f2": rng.normal(10, 2, n),
    })
    batch = pd.DataFrame({
        "f1": rng.normal(0 + shift, 1, n // 2),
        "f2": rng.normal(10, 2, n // 2),
    })
    return baseline, batch


def test_compute_drift_report_near_zero_shift_for_identically_distributed_batch():
    baseline, batch = _baseline_and_batch(shift=0.0, seed=1)
    report = compute_drift_report(baseline, batch, feature_columns=["f1", "f2"])
    assert report["n_features_compared"] == 2
    assert report["mean_abs_shift"] < 0.5  # well under the default threshold of 2.0
    assert report["n_features_over_threshold"] == 0
    assert report["n_baseline_examples"] == len(baseline)
    assert report["n_batch_examples"] == len(batch)
    assert report["batch_metrics"] is None


def test_compute_drift_report_detects_a_large_injected_shift():
    baseline, batch = _baseline_and_batch(shift=8.0, seed=1)  # 8 std devs on f1
    report = compute_drift_report(baseline, batch, feature_columns=["f1", "f2"])
    assert report["per_feature_shift"]["f1"] > 4.0  # clearly past the default threshold=2.0
    assert report["n_features_over_threshold"] >= 1
    assert report["fraction_features_over_threshold"] > 0
    assert report["mean_abs_shift"] > report["per_feature_shift"]["f2"]


def test_compute_drift_report_handles_zero_variance_baseline_column_without_crashing():
    baseline = pd.DataFrame({"const": [5.0] * 50})
    batch_same = pd.DataFrame({"const": [5.0] * 20})
    batch_diff = pd.DataFrame({"const": [7.0] * 20})

    report_same = compute_drift_report(baseline, batch_same, feature_columns=["const"])
    assert report_same["per_feature_shift"]["const"] == 0.0

    report_diff = compute_drift_report(baseline, batch_diff, feature_columns=["const"])
    assert report_diff["per_feature_shift"]["const"] == float("inf")
    # the infinite shift is excluded from the finite-shift average, not propagated as NaN/inf
    assert np.isfinite(report_diff["mean_abs_shift"])


def test_compute_drift_report_skips_columns_missing_from_either_side():
    baseline = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "only_in_baseline": [1, 2, 3]})
    batch = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "only_in_batch": [1, 2, 3]})
    report = compute_drift_report(baseline, batch, feature_columns=["f1", "only_in_baseline", "only_in_batch"])
    assert report["n_features_compared"] == 1
    assert "f1" in report["per_feature_shift"]


def test_compute_drift_report_includes_batch_metrics_when_pipeline_and_labels_given():
    rng = np.random.RandomState(0)
    n = 100
    baseline = pd.DataFrame({"f1": rng.normal(0, 1, n), "f2": rng.normal(0, 1, n)})
    baseline_y = (baseline["f1"] + baseline["f2"] > 0).astype(int)
    pipeline = LogisticRegression().fit(baseline[["f1", "f2"]], baseline_y)

    batch = pd.DataFrame({"f1": rng.normal(0, 1, 30), "f2": rng.normal(0, 1, 30)})
    batch_y = (batch["f1"] + batch["f2"] > 0).astype(int)

    report = compute_drift_report(
        baseline, batch, feature_columns=["f1", "f2"],
        pipeline=pipeline, batch_y=batch_y, metric_names=["accuracy", "f1"],
    )
    assert report["batch_metrics"] is not None
    assert set(report["batch_metrics"]) == {"accuracy", "f1"}
    assert 0.0 <= report["batch_metrics"]["accuracy"]["value"] <= 1.0
    assert "replay_simplification_note" in report
