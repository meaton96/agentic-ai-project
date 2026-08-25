"""
Tests for the deterministic environment/state layer -- no LLM involved,
same focus as agentic-ml-classification/tests/test_intake.py etc.:
the deterministic facts an agent's tool call returns are what actually
need to be trustworthy, so that's what's tested directly.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from resource_scheduler.environment.state import (
    JITTERED_COLUMNS,
    compute_flags,
    compute_snapshot,
    compute_thresholds,
    load_task_table,
)

REAL_DATASET = Path(__file__).resolve().parents[1] / "datasets" / "raw" / "industrial_scheduling_dataset.csv"


@pytest.fixture
def synthetic_df():
    rng = np.random.RandomState(0)
    n = 40
    return pd.DataFrame({
        "Task_ID": [f"T{i:04d}" for i in range(n)],
        "Machine_ID": rng.choice(["M01", "M02"], size=n),
        "Network_Slice_ID": rng.choice(["NS_1", "NS_2"], size=n),
        "Task_Type": rng.choice(["Welding", "Cutting"], size=n),
        "Execution_Time": rng.uniform(5, 15, size=n),
        "Machine_Status": rng.choice(["Active", "Overloaded", "Idle"], size=n),
        "Reallocation": rng.choice(["Yes", "No"], size=n),
        "Latency_ms": rng.uniform(2, 6, size=n),
        "Sensor_Temp_C": rng.uniform(60, 80, size=n),
        "URLLC_Score": rng.uniform(0.9, 1.0, size=n),
        "Target": rng.randint(0, 2, size=n),
    })


def test_real_dataset_has_known_constant_columns():
    """Confirms the data-quality finding this module's docstring relies
    on hasn't silently changed -- if this ever fails, the jitter
    injection logic needs revisiting, not just this test."""
    df = pd.read_csv(REAL_DATASET)
    for col in JITTERED_COLUMNS:
        assert df[col].nunique(dropna=True) == 1, f"{col} is no longer constant -- update jitter assumptions"


def test_load_task_table_injects_variance_only_where_constant():
    df, variance_injected = load_task_table(REAL_DATASET, inject_variance=True, seed=1)
    assert all(variance_injected[c] for c in JITTERED_COLUMNS)
    for col in JITTERED_COLUMNS:
        assert df[col].nunique() > 1


def test_load_task_table_no_inject_leaves_columns_constant():
    df, variance_injected = load_task_table(REAL_DATASET, inject_variance=False)
    assert not any(variance_injected.values())
    for col in JITTERED_COLUMNS:
        assert df[col].nunique(dropna=True) == 1


def test_load_task_table_is_deterministic_given_seed():
    df1, _ = load_task_table(REAL_DATASET, inject_variance=True, seed=7)
    df2, _ = load_task_table(REAL_DATASET, inject_variance=True, seed=7)
    pd.testing.assert_frame_equal(df1, df2)


def test_load_task_table_does_not_jitter_already_varying_column(synthetic_df, tmp_path):
    # load_task_table only accepts a path, so exercise the "already
    # varies" branch through a real file rather than an in-memory df.
    path = tmp_path / "synthetic.csv"
    synthetic_df.to_csv(path, index=False)
    df, variance_injected = load_task_table(path, inject_variance=True)
    assert not any(variance_injected.values())


def test_compute_snapshot_shape(synthetic_df):
    snapshot = compute_snapshot(synthetic_df, {c: False for c in JITTERED_COLUMNS}, window=40)
    assert snapshot["window_rows"] == 40
    machine_ids = {m["machine_id"] for m in snapshot["machines"]}
    assert machine_ids == {"M01", "M02"}
    slice_ids = {s["slice_id"] for s in snapshot["slices"]}
    assert slice_ids == {"NS_1", "NS_2"}
    for m in snapshot["machines"]:
        assert 0 <= m["utilization_pct"] <= 100
        assert m["queue_depth"] >= 0


def test_compute_thresholds_are_deterministic(synthetic_df):
    t1 = compute_thresholds(synthetic_df)
    t2 = compute_thresholds(synthetic_df)
    assert t1 == t2
    assert t1["slice_latency_ms"]["critical"] > t1["slice_latency_ms"]["warning"]


def test_compute_thresholds_zero_variance_baseline_does_not_spuriously_flag():
    """Regression test for the --no-inject-variance edge case: a constant
    Latency_ms column has zero std, which would otherwise put
    warning == critical == mean and flag every slice's identical
    reading as critical -- a false alarm with no real signal behind it."""
    df, variance_injected = load_task_table(REAL_DATASET, inject_variance=False)
    thresholds = compute_thresholds(df)
    assert thresholds["slice_latency_ms"]["warning"] is None
    assert thresholds["slice_latency_ms"]["critical"] is None

    snapshot = compute_snapshot(df, variance_injected, window=200)
    flags = compute_flags(snapshot, thresholds)
    latency_flags = [f for f in flags if f["metric"] == "latency_ms"]
    assert latency_flags == []


def test_compute_flags_only_fires_above_threshold():
    snapshot = {
        "machines": [{"machine_id": "M01", "utilization_pct": 50.0}],
        "slices": [{"slice_id": "NS_1", "latency_ms": 1.0, "capacity_used_pct": 10.0}],
    }
    thresholds = {
        "machine_utilization_pct": {"warning": 80.0, "critical": 95.0},
        "slice_latency_ms": {"warning": 5.0, "critical": 8.0},
        "slice_capacity_used_pct": {"warning": 60.0, "critical": 85.0},
    }
    assert compute_flags(snapshot, thresholds) == []

    snapshot["machines"][0]["utilization_pct"] = 96.0
    flags = compute_flags(snapshot, thresholds)
    assert len(flags) == 1
    assert flags[0] == {
        "scope": "machine", "id": "M01", "severity": "critical",
        "metric": "utilization_pct", "value": 96.0, "threshold": 95.0,
    }
