"""Metrics are computed here and only here. A plugin never sees fold labels
or scores its own predictions."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, log_loss, roc_auc_score
from sklearn.preprocessing import label_binarize


def classification_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    """`proba` is the full (N, K) probability matrix. Binary problems use the
    positive-class column; multiclass uses macro one-vs-rest averages. All
    K model classes are passed to the metrics even if a fold is missing one."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=np.float64)
    k = proba.shape[1]
    labels = list(range(k))
    clipped = np.clip(proba, 1e-7, 1.0)
    out = {
        "accuracy": float(accuracy_score(y_true, proba.argmax(axis=1))),
        "log_loss": float(log_loss(y_true, clipped / clipped.sum(axis=1, keepdims=True), labels=labels)),
    }
    if len(np.unique(y_true)) < 2:
        out["roc_auc"] = out["pr_auc"] = float("nan")
    elif k == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, proba[:, 1]))
        out["pr_auc"] = float(average_precision_score(y_true, proba[:, 1]))
    else:
        out["roc_auc"] = float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=labels))
        out["pr_auc"] = float(average_precision_score(label_binarize(y_true, classes=labels), proba, average="macro"))
    return out
