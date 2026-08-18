"""
Regression tests for the ablation-study findings in
scripts/run_ablation_study.py (Scenarios 2 and 3). These lock in two
non-obvious empirical claims about label_permutation_test's actual
sensitivity, discovered by the ablation methodology itself rather than
asserted from the docstring:

1. An intermediate component (e.g. a target encoder) that ignores its
   own y_train is NOT reliably caught by the permutation gate, because
   the downstream classifier legitimately relearns from whatever labels
   it's given each permutation round, washing the leak's exploitability
   out.
2. A leak where the FULL pipeline's output is independent of y_train at
   every stage (nothing downstream legitimately depends on the labels
   fit() receives) IS reliably caught — this is the actual shape of
   leak label_permutation_test's implementation is sensitive to, not
   the broader "any preprocessing fit outside its fold" framing its
   docstring suggests.

If either of these ever flips, that's a real change in the gate's
behavior worth knowing about, not a fixture regression.

Also covers train_cv_consistency_check, the third gate added to close
the Scenario 2 gap (see docs/ablation_study_report.md and the module
docstring on train_cv_consistency_check for why v1 — an absolute
train-vs-CV-score gap threshold — had to be redesigned: it produced a
real false positive against tests/test_orchestrator.py's
imbalanced_binary_boosted candidate, a legitimate high-capacity model
whose ordinary overfitting gap (0.54) was larger than any threshold
that would still catch a genuine leak).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from agentic_ml.harness.leakage import (
    check_suspicious_feature_correlation,
    label_permutation_test,
    train_cv_consistency_check,
)
from agentic_ml.harness.metrics import roc_auc_any


class _LeakyMeanEncoder(BaseEstimator, TransformerMixin):
    """fit() ignores the (X, y) it's given and always recomputes its
    mapping from the full (train+val) dataset — see run_ablation_study.py
    Scenario 2."""

    def __init__(self, full_entity_ref, full_y_ref):
        self.full_entity_ref = full_entity_ref
        self.full_y_ref = full_y_ref

    def fit(self, X, y=None):
        merged = pd.DataFrame({"entity_id": self.full_entity_ref, "_y": self.full_y_ref})
        self.mapping_ = merged.groupby("entity_id")["_y"].mean().to_dict()
        self.global_mean_ = float(self.full_y_ref.mean())
        return self

    def transform(self, X):
        col = X["entity_id"] if hasattr(X, "columns") else pd.Series(np.ravel(X))
        return col.map(self.mapping_).fillna(self.global_mean_).to_numpy(dtype=float).reshape(-1, 1)


class _GroundTruthLookupClassifier(BaseEstimator, ClassifierMixin):
    """fit() is a no-op w.r.t. its y argument; predict_proba() always
    answers from a closed-over reference to the true labels — see
    run_ablation_study.py Scenario 3."""

    def __init__(self, true_y_lookup: pd.Series):
        self.true_y_lookup = true_y_lookup

    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        idx = X.index if hasattr(X, "index") else range(len(X))
        p1 = self.true_y_lookup.loc[idx].to_numpy(dtype=float)
        p1 = np.clip(p1 * 0.9 + 0.05, 0.01, 0.99)
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


def test_encoder_that_ignores_its_fold_evades_both_gates():
    """Scenario 2's finding: correlation is structurally blind (non-numeric
    raw column), and — the non-obvious part — permutation is ALSO fooled,
    because the downstream LogisticRegression legitimately refits on
    whatever labels it's given each permutation round."""
    rng = np.random.RandomState(1)
    n, n_categories = 300, 75
    y = pd.Series(rng.randint(0, 2, size=n))
    entity_id = pd.Series(rng.choice([f"ent_{i}" for i in range(n_categories)], size=n), name="entity_id")
    df = pd.DataFrame({"entity_id": entity_id, "target": y})
    X = df[["entity_id"]]
    train_idx, val_idx = list(range(200)), list(range(200, 300))

    encoder = _LeakyMeanEncoder(full_entity_ref=df["entity_id"], full_y_ref=y)
    pipeline = Pipeline([
        ("encode", ColumnTransformer([("cat", encoder, ["entity_id"])])),
        ("clf", LogisticRegression(max_iter=1000)),
    ])

    def fit_and_score(X_tr, y_tr, X_va, y_va):
        candidate = clone(pipeline)
        candidate.fit(X_tr, y_tr)
        return roc_auc_any(y_va, candidate.predict_proba(X_va))

    real_auc = fit_and_score(X.iloc[train_idx], y.iloc[train_idx], X.iloc[val_idx], y.iloc[val_idx])
    assert real_auc > 0.65, "fixture should show real leakage-driven inflation to be meaningful"

    corr_check = check_suspicious_feature_correlation(X.iloc[train_idx], y.iloc[train_idx])
    assert corr_check.passed, "non-numeric column must be structurally invisible to this check"

    perm_check = label_permutation_test(
        fit_and_score, X.iloc[train_idx], y.iloc[train_idx], X.iloc[val_idx], y.iloc[val_idx],
        metric_name="roc_auc", seed=42,
    )
    assert perm_check.passed, (
        "known gap: permutation does not reliably catch an intermediate-component leak "
        "when the downstream classifier legitimately relearns from shuffled labels"
    )


def test_bypass_leak_is_caught_by_permutation_but_not_correlation():
    """Scenario 3's finding: a pipeline whose output never legitimately
    depends on y_train (at any stage, not just one component) IS reliably
    caught by permutation, while correlation still misses it (no raw
    column carries the leak)."""
    rng = np.random.RandomState(4)
    n = 300
    y = pd.Series(rng.randint(0, 2, size=n))
    X = pd.DataFrame({"noise": rng.normal(size=n)})
    train_idx, val_idx = list(range(200)), list(range(200, 300))

    clf = _GroundTruthLookupClassifier(true_y_lookup=y)

    def fit_and_score(X_tr, y_tr, X_va, y_va):
        candidate = clone(clf)
        candidate.fit(X_tr, y_tr)
        return roc_auc_any(y_va, candidate.predict_proba(X_va))

    corr_check = check_suspicious_feature_correlation(X.iloc[train_idx], y.iloc[train_idx])
    assert corr_check.passed, "the leak is not in any raw column, so correlation must miss it"

    perm_check = label_permutation_test(
        fit_and_score, X.iloc[train_idx], y.iloc[train_idx], X.iloc[val_idx], y.iloc[val_idx],
        metric_name="roc_auc", seed=42,
    )
    assert not perm_check.passed, "a prediction pathway independent of fit()'s y must be caught"


def test_train_cv_consistency_does_not_false_positive_on_a_high_capacity_model():
    """A regression test for the exact bug the v1 (absolute-threshold)
    design of this gate introduced: an unregularized DecisionTreeClassifier
    memorizes noise and shows a huge train-vs-holdout gap (~0.5) purely
    from capacity, with NO leak (y is independent random noise) — an
    absolute-threshold gate would reject this legitimate candidate. The
    excess-over-shuffled-baseline design must not."""
    rng = np.random.RandomState(0)
    n = 300
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = pd.Series(rng.randint(0, 2, size=n))
    train_idx = list(range(200))

    clf = DecisionTreeClassifier()

    def fit_and_score(X_tr, y_tr, X_va, y_va):
        candidate = clone(clf)
        candidate.fit(X_tr, y_tr)
        return roc_auc_any(y_va, candidate.predict_proba(X_va))

    result = train_cv_consistency_check(fit_and_score, X.iloc[train_idx], y.iloc[train_idx], metric_name="roc_auc", seed=42)
    assert result.passed, f"high-capacity-but-honest model must not be flagged as leaky: {result.detail}"


def test_train_cv_consistency_does_not_catch_a_bypass_leak():
    """The same bypass-leak fixture from Scenario 3 — a classifier whose
    predict_proba() ignores fit() and answers from a closed-over
    reference to the true labels — has zero train-vs-holdout gap
    difference under permutation for the wrong reason (both the real and
    shuffled passes read the SAME true-label reference), so this is a
    case where train_cv_consistency_check's docstring predicts it should
    NOT catch the leak (only label_permutation_test does, per Scenario
    3). This test locks in that prediction rather than assuming it."""
    rng = np.random.RandomState(4)
    n = 300
    y = pd.Series(rng.randint(0, 2, size=n))
    X = pd.DataFrame({"noise": rng.normal(size=n)})
    train_idx = list(range(200))

    clf = _GroundTruthLookupClassifier(true_y_lookup=y)

    def fit_and_score(X_tr, y_tr, X_va, y_va):
        candidate = clone(clf)
        candidate.fit(X_tr, y_tr)
        return roc_auc_any(y_va, candidate.predict_proba(X_va))

    result = train_cv_consistency_check(fit_and_score, X.iloc[train_idx], y.iloc[train_idx], metric_name="roc_auc", seed=42)
    assert result.passed, (
        "a closed-over-reference leak poisons every fold/permutation identically, so the "
        f"excess gap should stay near zero, not flag it: {result.detail}"
    )
