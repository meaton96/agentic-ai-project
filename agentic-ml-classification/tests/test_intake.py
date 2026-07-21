"""
Phase 5 intake: pre-target schema facts + DatasetSpec proposal
validation. The intake agent (steps/intake_step.py) only ever sees
what raw_schema_summary computes, and its proposal is re-validated
here before anything downstream runs — these are the two functions
that make that trust boundary real.
"""
import numpy as np
import pandas as pd
import pytest

from agentic_ml.harness.intake import raw_schema_summary, validate_dataset_spec_proposal


@pytest.fixture
def sample_df():
    rng = np.random.RandomState(0)
    n = 200
    return pd.DataFrame({
        "customer_id": [f"C{i}" for i in range(n)],
        "age": rng.randint(18, 80, size=n),
        "plan_type": rng.choice(["basic", "premium"], size=n),
        "signup_date": pd.date_range("2020-01-01", periods=n, freq="D"),
        "churned": rng.randint(0, 2, size=n),
    })


def test_raw_schema_summary_reports_all_columns(sample_df):
    summary = raw_schema_summary(sample_df)
    assert summary["n_rows"] == 200
    names = {c["name"] for c in summary["columns"]}
    assert names == set(sample_df.columns)


def test_raw_schema_summary_flags_id_by_name(sample_df):
    summary = raw_schema_summary(sample_df)
    by_name = {c["name"]: c for c in summary["columns"]}
    assert by_name["customer_id"]["name_suggests_id"]


def test_raw_schema_summary_flags_datetime(sample_df):
    summary = raw_schema_summary(sample_df)
    by_name = {c["name"]: c for c in summary["columns"]}
    assert by_name["signup_date"]["looks_like_datetime"]


def test_raw_schema_summary_excludes_target_dependent_facts(sample_df):
    """No imbalance/correlation-based leakage facts should appear anywhere
    here — those require a target_column, which intake hasn't decided yet."""
    summary = raw_schema_summary(sample_df)
    assert "target_distribution" not in summary
    for col in summary["columns"]:
        assert "is_likely_id" not in col  # that's profiler.py's target-aware field name


def test_validate_proposal_accepts_valid_binary_target(sample_df):
    errors = validate_dataset_spec_proposal(sample_df, {
        "target_column": "churned", "id_columns": ["customer_id"],
        "group_column": None, "time_column": "signup_date",
    })
    assert errors == []


def test_validate_proposal_rejects_non_dict():
    errors = validate_dataset_spec_proposal(pd.DataFrame(), "not a dict")
    assert errors


def test_validate_proposal_rejects_unknown_target(sample_df):
    errors = validate_dataset_spec_proposal(sample_df, {"target_column": "does_not_exist"})
    assert any("not found" in e for e in errors)


def test_validate_proposal_rejects_high_cardinality_target(sample_df):
    errors = validate_dataset_spec_proposal(sample_df, {"target_column": "age"})
    assert any("distinct class labels" in e for e in errors)


def test_validate_proposal_accepts_multiclass_target(sample_df):
    df = sample_df.assign(species=np.tile(["a", "b", "c"], len(sample_df) // 3 + 1)[:len(sample_df)])
    errors = validate_dataset_spec_proposal(df, {"target_column": "species"})
    assert errors == []


def test_validate_proposal_rejects_group_column_equal_to_target(sample_df):
    errors = validate_dataset_spec_proposal(sample_df, {
        "target_column": "churned", "group_column": "churned",
    })
    assert any("cannot be the same as target_column" in e for e in errors)


def test_validate_proposal_rejects_unknown_group_column(sample_df):
    errors = validate_dataset_spec_proposal(sample_df, {
        "target_column": "churned", "group_column": "nonexistent",
    })
    assert any("not found in dataset columns" in e for e in errors)


def test_validate_proposal_rejects_unknown_id_column(sample_df):
    errors = validate_dataset_spec_proposal(sample_df, {
        "target_column": "churned", "id_columns": ["nonexistent"],
    })
    assert any("id_columns references unknown column" in e for e in errors)


def test_validate_proposal_rejects_target_in_id_columns(sample_df):
    errors = validate_dataset_spec_proposal(sample_df, {
        "target_column": "churned", "id_columns": ["churned"],
    })
    assert any("cannot include the target_column" in e for e in errors)
