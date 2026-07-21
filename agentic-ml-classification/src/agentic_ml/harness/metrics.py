"""
Metric calculation, owned entirely by the harness. Agents never compute
these numbers themselves — they only ever see what this module returns.

y_proba is always the FULL predict_proba() matrix (n_samples, n_classes)
— for binary problems that's 2 columns, for multiclass however many
classes are present. Passing the full matrix (never a pre-sliced
positive-class column) lets this module decide binary vs. multiclass
metric formulas itself, from how many classes are actually in y_true,
instead of every caller duplicating that decision.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    brier_score_loss,
)
from sklearn.preprocessing import label_binarize


def roc_auc_any(y_true, y_proba) -> float:
    """AUC that works for both binary and multiclass targets, dispatching
    on how many classes are present in y_true. y_proba must be the full
    predict_proba() matrix, not a sliced column — used both by the
    metric table below and by modeling_step.py's label-permutation
    leakage scorer, so that gate works for multiclass too."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    classes = sorted(np.unique(y_true).tolist())
    if len(classes) > 2:
        return roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro", labels=classes)
    return roc_auc_score(y_true, y_proba[:, 1])


def _multiclass_pr_auc(y_true, y_proba, classes) -> float:
    y_bin = label_binarize(y_true, classes=classes)
    return average_precision_score(y_bin, y_proba, average="macro")


def _multiclass_brier(y_true, y_proba, classes) -> float:
    """Generalization of the binary Brier score: mean squared error
    between one-hot true labels and the predicted probability vector,
    summed over classes. sklearn's brier_score_loss is binary-only."""
    y_bin = label_binarize(y_true, classes=classes)
    return float(np.mean(np.sum((y_proba - y_bin) ** 2, axis=1)))


BINARY_METRIC_FUNCS = {
    "roc_auc": lambda y, p, proba: roc_auc_score(y, proba[:, 1]),
    "pr_auc": lambda y, p, proba: average_precision_score(y, proba[:, 1]),
    "f1": lambda y, p, proba: f1_score(y, p),
    "precision": lambda y, p, proba: precision_score(y, p, zero_division=0),
    "recall": lambda y, p, proba: recall_score(y, p, zero_division=0),
    "accuracy": lambda y, p, proba: accuracy_score(y, p),
    "brier": lambda y, p, proba: brier_score_loss(y, proba[:, 1]),
}


def _make_multiclass_metric_funcs(classes: list) -> dict:
    return {
        "roc_auc": lambda y, p, proba: roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=classes),
        "pr_auc": lambda y, p, proba: _multiclass_pr_auc(y, proba, classes),
        "f1": lambda y, p, proba: f1_score(y, p, average="macro"),
        "precision": lambda y, p, proba: precision_score(y, p, average="macro", zero_division=0),
        "recall": lambda y, p, proba: recall_score(y, p, average="macro", zero_division=0),
        "accuracy": lambda y, p, proba: accuracy_score(y, p),
        "brier": lambda y, p, proba: _multiclass_brier(y, proba, classes),
    }


@dataclass
class MetricResult:
    metric: str
    value: float
    ci_low: float
    ci_high: float
    n_bootstrap: int

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "value": round(float(self.value), 6),
            "ci_low": round(float(self.ci_low), 6),
            "ci_high": round(float(self.ci_high), 6),
            "n_bootstrap": self.n_bootstrap,
        }


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    metric_names: list[str],
    n_bootstrap: int = 1000,
    seed: int = 42,
    ci: float = 0.95,
) -> dict[str, MetricResult]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba)
    if y_proba.ndim == 1:
        # Backward-compat: a caller passed a single positive-class column
        # directly rather than the full matrix. Wrap it so the rest of
        # this function can treat binary/multiclass uniformly.
        y_proba = np.column_stack([1 - y_proba, y_proba])

    classes = sorted(np.unique(y_true).tolist())
    is_multiclass = len(classes) > 2
    metric_funcs = _make_multiclass_metric_funcs(classes) if is_multiclass else BINARY_METRIC_FUNCS

    rng = np.random.RandomState(seed)
    n = len(y_true)
    alpha = 1.0 - ci
    lo_pct = 100 * (alpha / 2)
    hi_pct = 100 * (1 - alpha / 2)

    results = {}
    for name in metric_names:
        if name not in metric_funcs:
            raise ValueError(f"Unknown metric '{name}'. Available: {list(metric_funcs)}")
        func = metric_funcs[name]
        point_estimate = func(y_true, y_pred, y_proba)

        boot_values = []
        for _ in range(n_bootstrap):
            sample_idx = rng.randint(0, n, size=n)
            yt, yp, ypb = y_true[sample_idx], y_pred[sample_idx], y_proba[sample_idx]
            if len(np.unique(yt)) < len(classes):
                continue  # skip bootstrap samples missing a class — degenerate for AUC-like metrics
            try:
                boot_values.append(func(yt, yp, ypb))
            except ValueError:
                continue

        if boot_values:
            ci_low, ci_high = np.percentile(boot_values, [lo_pct, hi_pct])
        else:
            ci_low, ci_high = point_estimate, point_estimate

        results[name] = MetricResult(
            metric=name,
            value=point_estimate,
            ci_low=ci_low,
            ci_high=ci_high,
            n_bootstrap=len(boot_values),
        )

    return results
