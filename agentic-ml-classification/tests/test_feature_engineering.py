"""
Feature engineering: the deterministic op catalog (harness/
feature_engineering.py) and the agent step that proposes which ops to
apply. Two things worth proving, not just exercising:

1. Every op actually computes what its docstring claims (ratio's
   division-by-zero -> NaN, log1p's negative-input -> NaN,
   datetime_parts' extracted values, missing_indicator's flag).
2. validate_feature_proposal rejects the specific things a
   feature-engineering agent could plausibly get wrong: dropping the
   target/group/time column, using the target as an op input, an op
   applied to the wrong dtype, and a new-column-name collision.
"""
import json

import numpy as np
import pandas as pd
import pytest

from agentic_ml.harness.feature_engineering import (
    apply_datetime_parts,
    apply_feature_op,
    apply_interaction,
    apply_log1p,
    apply_missing_indicator,
    apply_ratio,
    list_feature_ops,
    validate_feature_proposal,
)
from agentic_ml.harness.profiler import profile_dataset
from agentic_ml.model_client import ModelResponse
from agentic_ml.steps.feature_engineering_step import run_feature_engineering_step


# --- deterministic op correctness ---

def test_apply_ratio_computes_division():
    df = pd.DataFrame({"a": [10.0, 20.0], "b": [2.0, 4.0]})
    new_df, names = apply_ratio(df, "a", "b")
    assert names == ["a_over_b"]
    assert new_df["a_over_b"].tolist() == [5.0, 5.0]


def test_apply_ratio_division_by_zero_is_nan_not_inf():
    df = pd.DataFrame({"a": [10.0], "b": [0.0]})
    new_df, _ = apply_ratio(df, "a", "b")
    assert np.isnan(new_df["a_over_b"].iloc[0])


def test_apply_interaction_computes_product():
    df = pd.DataFrame({"a": [2.0, 3.0], "b": [5.0, 4.0]})
    new_df, names = apply_interaction(df, "a", "b")
    assert names == ["a_times_b"]
    assert new_df["a_times_b"].tolist() == [10.0, 12.0]


def test_apply_log1p_negative_is_nan():
    df = pd.DataFrame({"x": [0.0, 1.0, -5.0]})
    new_df, names = apply_log1p(df, "x")
    assert names == ["log1p_x"]
    assert not np.isnan(new_df["log1p_x"].iloc[0])
    assert not np.isnan(new_df["log1p_x"].iloc[1])
    assert np.isnan(new_df["log1p_x"].iloc[2])


def test_apply_datetime_parts_extracts_correct_values():
    df = pd.DataFrame({"d": ["2024-03-15", "2024-03-16"]})  # Fri, Sat
    new_df, names = apply_datetime_parts(df, "d", ["year", "month", "day", "dayofweek"])
    assert set(names) == {"d_year", "d_month", "d_day", "d_dayofweek"}
    assert new_df["d_year"].tolist() == [2024, 2024]
    assert new_df["d_month"].tolist() == [3, 3]
    assert new_df["d_day"].tolist() == [15, 16]
    assert new_df["d_dayofweek"].tolist() == [4, 5]  # Monday=0 -> Friday=4, Saturday=5


def test_apply_missing_indicator_flags_correctly():
    df = pd.DataFrame({"x": [1.0, None, 3.0]})
    new_df, names = apply_missing_indicator(df, "x")
    assert names == ["x_was_missing"]
    assert new_df["x_was_missing"].tolist() == [0, 1, 0]


def test_apply_feature_op_dispatches_by_op_id():
    df = pd.DataFrame({"a": [1.0], "b": [2.0]})
    new_df, names = apply_feature_op(df, "interaction", {"col_a": "a", "col_b": "b"})
    assert names == ["a_times_b"]


def test_apply_feature_op_unknown_op_raises():
    with pytest.raises(KeyError):
        apply_feature_op(pd.DataFrame(), "not_a_real_op", {})


def test_list_feature_ops_matches_catalog_keys():
    ops = list_feature_ops()
    assert {o["op_id"] for o in ops} == {"ratio", "interaction", "log1p", "datetime_parts", "missing_indicator"}


# --- validate_feature_proposal ---

@pytest.fixture
def profile_report():
    rng = np.random.RandomState(0)
    n = 200
    df = pd.DataFrame({
        "customer_id": [f"C{i}" for i in range(n)],
        "age": rng.randint(18, 80, size=n),
        "signup_date": pd.date_range("2020-01-01", periods=n, freq="D"),
        "plan_type": rng.choice(["basic", "premium"], size=n),
        "num_dependents": rng.randint(0, 4, size=n),  # low-cardinality int count, like SibSp/Parch
        "churned": rng.randint(0, 2, size=n),
    })
    return profile_dataset(df, target_column="churned").to_dict()


def test_low_cardinality_integer_count_is_flagged_categorical_not_numeric_by_profiler(profile_report):
    """Sanity-check the premise: the profiler heuristic really does flag a
    genuine integer count as categorical (low cardinality), which is exactly
    why validate_feature_proposal must not gate numeric ops on that flag."""
    entry = next(c for c in profile_report["columns"] if c["name"] == "num_dependents")
    assert entry["is_likely_categorical"] is True
    assert entry["is_likely_numeric"] is False
    assert entry["dtype"].startswith("int")


def test_validate_accepts_reasonable_proposal(profile_report):
    errors = validate_feature_proposal(profile_report, "churned", "customer_id", "signup_date", {
        "drop_columns": [],
        "derived_features": [
            {"op_id": "datetime_parts", "params": {"col": "signup_date", "parts": ["month", "dayofweek"]}},
            {"op_id": "missing_indicator", "params": {"col": "age"}},
        ],
    })
    assert errors == []


def test_validate_rejects_dropping_target(profile_report):
    errors = validate_feature_proposal(profile_report, "churned", None, None, {
        "drop_columns": ["churned"],
    })
    assert any("target/group/time" in e for e in errors)


def test_validate_rejects_dropping_group_column(profile_report):
    errors = validate_feature_proposal(profile_report, "churned", "customer_id", None, {
        "drop_columns": ["customer_id"],
    })
    assert any("target/group/time" in e for e in errors)


def test_validate_rejects_dropping_time_column(profile_report):
    errors = validate_feature_proposal(profile_report, "churned", None, "signup_date", {
        "drop_columns": ["signup_date"],
    })
    assert any("target/group/time" in e for e in errors)


def test_validate_allows_time_column_as_datetime_parts_input(profile_report):
    """The time_column can't be dropped, but IS a legitimate op input."""
    errors = validate_feature_proposal(profile_report, "churned", None, "signup_date", {
        "derived_features": [
            {"op_id": "datetime_parts", "params": {"col": "signup_date", "parts": ["year"]}},
        ],
    })
    assert errors == []


def test_validate_rejects_target_as_op_input(profile_report):
    errors = validate_feature_proposal(profile_report, "churned", None, None, {
        "derived_features": [{"op_id": "missing_indicator", "params": {"col": "churned"}}],
    })
    assert any("target_column" in e for e in errors)


def test_validate_accepts_arithmetic_on_low_cardinality_integer_count(profile_report):
    """Regression test: a real orchestrator run against Titanic proposed
    SibSp * Parch (a standard 'family size' engineered feature) and it was
    incorrectly rejected, because SibSp/Parch are low-cardinality integer
    counts the profiler flags is_likely_categorical=True /
    is_likely_numeric=False (that flag is about one-hot-vs-scaling in a
    baseline model, not about whether arithmetic is valid). Numeric ops must
    gate on actual dtype, not that flag — this is the empirical proof."""
    errors = validate_feature_proposal(profile_report, "churned", None, None, {
        "derived_features": [
            {"op_id": "interaction", "params": {"col_a": "num_dependents", "col_b": "age"}},
            {"op_id": "ratio", "params": {"col_a": "age", "col_b": "num_dependents"}},
        ],
    })
    assert errors == []


def test_validate_rejects_numeric_op_on_categorical_column(profile_report):
    errors = validate_feature_proposal(profile_report, "churned", None, None, {
        "derived_features": [{"op_id": "log1p", "params": {"col": "plan_type"}}],
    })
    assert any("requires a numeric column" in e for e in errors)


def test_validate_rejects_datetime_op_on_non_datetime_column(profile_report):
    errors = validate_feature_proposal(profile_report, "churned", None, None, {
        "derived_features": [{"op_id": "datetime_parts", "params": {"col": "age", "parts": ["year"]}}],
    })
    assert any("requires a datetime-like column" in e for e in errors)


def test_validate_rejects_unknown_op_id(profile_report):
    errors = validate_feature_proposal(profile_report, "churned", None, None, {
        "derived_features": [{"op_id": "not_a_real_op", "params": {}}],
    })
    assert any("unknown op_id" in e for e in errors)


def test_validate_rejects_invalid_datetime_part(profile_report):
    errors = validate_feature_proposal(profile_report, "churned", None, "signup_date", {
        "derived_features": [{"op_id": "datetime_parts", "params": {"col": "signup_date", "parts": ["century"]}}],
    })
    assert any("invalid parts" in e for e in errors)


def test_validate_rejects_new_column_name_collision(profile_report):
    errors = validate_feature_proposal(profile_report, "churned", None, None, {
        "derived_features": [{"op_id": "missing_indicator", "params": {"col": "age", "new_column_name": "age"}}],
    })
    assert any("collides" in e for e in errors)


def test_validate_rejects_missing_required_param(profile_report):
    errors = validate_feature_proposal(profile_report, "churned", None, None, {
        "derived_features": [{"op_id": "ratio", "params": {"col_a": "age"}}],
    })
    assert any("missing required param" in e for e in errors)


def test_validate_rejects_unknown_column_in_drop(profile_report):
    errors = validate_feature_proposal(profile_report, "churned", None, None, {
        "drop_columns": ["does_not_exist"],
    })
    assert any("unknown column" in e for e in errors)


# --- run_feature_engineering_step (stubbed ModelClient) ---

def _resp(text=None, tool_calls=None):
    return ModelResponse(text=text, tool_calls=tool_calls or [], raw=None,
                          latency_seconds=0.01, model="fake-model", input_tokens=1, output_tokens=1)


def _make_fake_client(final_text):
    call_count = {"n": 0}

    class FakeClient:
        def call(self, messages, model=None, tools=None, temperature=0.0, max_tokens=1024):
            call_count["n"] += 1
            n = call_count["n"]
            if n == 1:
                return _resp(tool_calls=[{"id": "1", "name": "get_dataset_profile", "arguments": "{}"}])
            if n == 2:
                return _resp(tool_calls=[{"id": "2", "name": "list_feature_ops", "arguments": "{}"}])
            return _resp(text=final_text)

    return FakeClient()


@pytest.fixture
def sample_df():
    rng = np.random.RandomState(0)
    n = 200
    return pd.DataFrame({
        "age": rng.randint(18, 80, size=n),
        "fare": rng.exponential(30, size=n),
        "cabin": [None] * int(n * 0.8) + list(rng.choice(["A", "B"], size=n - int(n * 0.8))),
        "target": rng.randint(0, 2, size=n),
    })


def test_run_feature_engineering_step_success(sample_df):
    proposal = json.dumps({
        "drop_columns": ["cabin"],
        "derived_features": [{"op_id": "missing_indicator", "params": {"col": "age"}}],
        "explanation": "cabin is mostly missing; flag missing age.",
    })
    result = run_feature_engineering_step(sample_df, "target", _make_fake_client(proposal))
    assert result.ok
    assert result.drop_columns == ["cabin"]
    assert result.new_columns == ["age_was_missing"]
    assert "age_was_missing" in result.df.columns


def test_run_feature_engineering_step_invalid_proposal_fails(sample_df):
    proposal = json.dumps({
        "drop_columns": ["target"],  # invalid: can't drop the target
        "derived_features": [],
    })
    result = run_feature_engineering_step(sample_df, "target", _make_fake_client(proposal))
    assert not result.ok
    assert result.df is None
    assert any("target/group/time" in e for e in result.errors)


def test_run_feature_engineering_step_unparseable_response_fails(sample_df):
    result = run_feature_engineering_step(sample_df, "target", _make_fake_client("not json"))
    assert not result.ok
    assert any("did not parse as JSON" in e for e in result.errors)
