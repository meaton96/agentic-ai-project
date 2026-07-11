"""
Metric calculation, owned entirely by the harness. Agents never compute
these numbers themselves — they only ever see what this module returns.
"""
from __future__ import annotations

from dataclasses import dataclass, field

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

BINARY_METRIC_FUNCS = {
    "roc_auc": lambda y, p, proba: roc_auc_score(y, proba),
    "pr_auc": lambda y, p, proba: average_precision_score(y, proba),
    "f1": lambda y, p, proba: f1_score(y, p),
    "precision": lambda y, p, proba: precision_score(y, p, zero_division=0),
    "recall": lambda y, p, proba: recall_score(y, p, zero_division=0),
    "accuracy": lambda y, p, proba: accuracy_score(y, p),
    "brier": lambda y, p, proba: brier_score_loss(y, proba),
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

    rng = np.random.RandomState(seed)
    n = len(y_true)
    alpha = 1.0 - ci
    lo_pct = 100 * (alpha / 2)
    hi_pct = 100 * (1 - alpha / 2)

    results = {}
    for name in metric_names:
        if name not in BINARY_METRIC_FUNCS:
            raise ValueError(f"Unknown metric '{name}'. Available: {list(BINARY_METRIC_FUNCS)}")
        func = BINARY_METRIC_FUNCS[name]
        point_estimate = func(y_true, y_pred, y_proba)

        boot_values = []
        for _ in range(n_bootstrap):
            sample_idx = rng.randint(0, n, size=n)
            yt, yp, ypb = y_true[sample_idx], y_pred[sample_idx], y_proba[sample_idx]
            if len(np.unique(yt)) < 2:
                continue  # skip degenerate bootstrap samples for AUC-like metrics
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
