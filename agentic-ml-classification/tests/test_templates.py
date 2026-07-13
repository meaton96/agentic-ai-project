"""
Phase 3: recipe templates + registry. Each template must round-trip
through the same sandbox the modeling agent's proposals go through
(static check + subprocess build), then actually fit/predict on a
synthetic mixed dataset. The high-cardinality target-encoding template
also gets a leakage-safety check via label_permutation_test, since it's
the one template whose whole reason for existing is "don't leak the
target through the encoding."
"""
import numpy as np
import pandas as pd
import pytest

from agentic_ml.harness.sandbox import run_candidate_build
from agentic_ml.harness.leakage import label_permutation_test
from agentic_ml.templates.registry import TEMPLATES, get_template, list_template_summaries, validate_config
from sklearn.base import clone
from sklearn.metrics import roc_auc_score


@pytest.fixture
def mixed_df():
    rng = np.random.RandomState(0)
    n = 300
    df = pd.DataFrame({
        "x1": rng.normal(size=n),
        "x2": rng.normal(size=n),
        "cat_low": rng.choice(["a", "b", "c"], size=n),
        "cat_high": rng.choice([f"v{i}" for i in range(50)], size=n),
        "target": rng.randint(0, 2, size=n),
    })
    return df


BASE_CONFIG = {
    "numeric_cols": ["x1", "x2"],
    "categorical_cols": ["cat_low", "cat_high"],
    "seed": 42,
}


def _build_and_fit(template_id, config, df):
    template = get_template(template_id)
    pipeline, error = run_candidate_build(template.read_source(), config, timeout_seconds=30)
    assert error is None, f"{template_id} sandbox build failed: {error}"
    X = df.drop(columns=["target"])
    y = df["target"]
    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    proba = pipeline.predict_proba(X)
    assert len(preds) == len(df)
    assert proba.shape[0] == len(df)
    return pipeline


def test_registry_lists_all_templates():
    summaries = list_template_summaries()
    assert {s["template_id"] for s in summaries} == set(TEMPLATES)


def test_registry_validate_config_missing_required_key():
    errors = validate_config("logistic_numeric", {})
    assert any("numeric_cols" in e for e in errors)


def test_registry_validate_config_wrong_type():
    errors = validate_config("sklearn_mixed_pipeline", {
        "numeric_cols": "x1", "categorical_cols": ["cat_low"],
    })
    assert any("must be a list" in e for e in errors)


def test_registry_unknown_template_raises():
    with pytest.raises(KeyError):
        get_template("does_not_exist")


def test_logistic_numeric_builds_and_fits(mixed_df):
    _build_and_fit("logistic_numeric", {"numeric_cols": ["x1", "x2"], "seed": 42}, mixed_df)


def test_sklearn_mixed_pipeline_logistic(mixed_df):
    config = dict(BASE_CONFIG, classifier="logistic_regression")
    _build_and_fit("sklearn_mixed_pipeline", config, mixed_df)


def test_sklearn_mixed_pipeline_random_forest(mixed_df):
    config = dict(BASE_CONFIG, classifier="random_forest", n_estimators=20)
    _build_and_fit("sklearn_mixed_pipeline", config, mixed_df)


def test_lightgbm_mixed_builds_and_fits(mixed_df):
    pytest.importorskip("lightgbm")
    config = dict(BASE_CONFIG, n_estimators=20)
    _build_and_fit("lightgbm_mixed", config, mixed_df)


def test_xgboost_mixed_builds_and_fits(mixed_df):
    pytest.importorskip("xgboost")
    config = dict(BASE_CONFIG, n_estimators=20)
    _build_and_fit("xgboost_mixed", config, mixed_df)


def test_imbalanced_binary_boosted_builds_and_fits(mixed_df):
    config = dict(BASE_CONFIG, max_iter=20)
    _build_and_fit("imbalanced_binary_boosted", config, mixed_df)


def test_high_cardinality_target_encoding_builds_and_fits(mixed_df):
    config = dict(BASE_CONFIG, classifier="hist_gradient_boosting")
    _build_and_fit("high_cardinality_target_encoding", config, mixed_df)


def test_high_cardinality_target_encoding_not_leaky(mixed_df):
    """The whole point of sklearn's TargetEncoder is internal cross-fitting
    so it never leaks a fold's own label into its own encoding. Confirm
    that fitting the full pipeline on shuffled labels scores at chance,
    the same gate run_modeling_agent.py applies to agent proposals."""
    template = get_template("high_cardinality_target_encoding")
    config = dict(BASE_CONFIG, classifier="hist_gradient_boosting")
    pipeline, error = run_candidate_build(template.read_source(), config, timeout_seconds=30)
    assert error is None

    X = mixed_df.drop(columns=["target"])
    y = mixed_df["target"]
    n = len(X)
    train_idx = np.arange(0, int(n * 0.7))
    val_idx = np.arange(int(n * 0.7), n)

    def fit_and_score(X_tr, y_tr, X_va, y_va):
        p = clone(pipeline)
        p.fit(X_tr, y_tr)
        proba = p.predict_proba(X_va)[:, 1]
        return roc_auc_score(y_va, proba)

    check = label_permutation_test(
        fit_and_score,
        X.iloc[train_idx], y.iloc[train_idx], X.iloc[val_idx], y.iloc[val_idx],
        metric_name="roc_auc", seed=0,
    )
    assert check.passed, check.detail
