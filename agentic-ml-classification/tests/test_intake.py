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


# --- ablation: what actually happens when each check is disabled ---
# See docs/ablation_study_report.md Phase 1c and scripts/run_ablation_study.py
# for the full study this locks in the most severe/interesting findings from.

def test_ablation_group_equals_target_produces_empty_train_fold():
    """The most severe Phase 1c finding: with the collision check
    disabled, group_column=target_column is accepted, and group-based
    splitting then assigns entire classes to entire folds — in this
    fixture, ALL of train ends up empty. check_group_overlap does not
    catch this (there genuinely is no group overlap); only
    check_fold_class_presence does."""
    from agentic_ml.ablation import AblationConfig
    from agentic_ml.harness.leakage import check_fold_class_presence, check_group_overlap
    from agentic_ml.harness.splits import make_split

    rng = np.random.RandomState(0)
    n = 200
    df = pd.DataFrame({"x": rng.normal(size=n), "target": rng.randint(0, 2, size=n)})

    ablation = AblationConfig(skip_group_time_target_collision_check=True)
    errors = validate_dataset_spec_proposal(
        df, {"target_column": "target", "group_column": "target"}, ablation=ablation,
    )
    assert errors == []

    m = make_split(df, target_column="target", data_hash="x", strategy="group",
                    group_column="target", val_frac=0.2, test_frac=0.2, seed=0)
    assert len(m.train_idx) == 0, "train fold should end up completely empty in this fixture"

    overlap = check_group_overlap(df, "target", m.train_idx, m.val_idx, m.test_idx)
    assert overlap.passed, "group_overlap genuinely finds nothing wrong -- it is not the backstop"

    presence = check_fold_class_presence(df["target"], m.train_idx, m.val_idx, m.test_idx)
    assert not presence.passed, "fold_class_presence is the check that actually catches this"


def test_ablation_single_class_target_is_backstopped_by_split_level_check():
    """Disabling the cardinality lower bound accepts a single-class
    target at intake, but the unrelated split-level
    check_fold_class_presence independently catches it — genuine
    defense in depth, unlike the group/target collision case above."""
    from agentic_ml.ablation import AblationConfig
    from agentic_ml.harness.leakage import check_fold_class_presence
    from agentic_ml.harness.splits import make_split

    rng = np.random.RandomState(0)
    n = 200
    df = pd.DataFrame({"x": rng.normal(size=n), "target": 1})

    ablation = AblationConfig(skip_cardinality_check=True)
    errors = validate_dataset_spec_proposal(df, {"target_column": "target"}, ablation=ablation)
    assert errors == []

    m = make_split(df, target_column="target", data_hash="x", strategy="stratified",
                    val_frac=0.2, test_frac=0.2, seed=0)
    presence = check_fold_class_presence(df["target"], m.train_idx, m.val_idx, m.test_idx)
    assert not presence.passed, "an independent split-level check must still catch a single-class target"


def test_ablation_id_columns_as_string_silently_drops_a_real_feature():
    """Disabling the id_columns type check accepts a STRING where a list
    was expected. Python's *string unpacking then iterates it character
    by character, and if any of those single characters happens to match
    a real column name, that column silently vanishes from the feature
    set — no error at any layer."""
    from agentic_ml.ablation import AblationConfig
    from agentic_ml.harness.dataset import DatasetSpec, LoadedDataset

    df = pd.DataFrame({"x": [1, 2], "target": [0, 1], "customer_id": ["a", "b"]})

    ablation = AblationConfig(skip_id_columns_type_check=True)
    errors = validate_dataset_spec_proposal(
        df, {"target_column": "target", "id_columns": "xt"}, ablation=ablation,
    )
    assert errors == []

    spec = DatasetSpec(path="fixture", target_column="target", id_columns="xt")
    loaded = LoadedDataset(df=df, spec=spec, data_hash="fixture")
    assert "x" not in loaded.X.columns, "the genuine feature column 'x' silently disappears"
    assert list(loaded.X.columns) == ["customer_id"]
