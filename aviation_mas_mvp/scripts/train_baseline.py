"""
train_baseline.py
=================
Trains a baseline classification model to predict aircraft maintenance 
needs using extracted tabular features.

Implements GroupKFold cross-validation grouped by aircraft tail number 
(`plane_id`) to prevent data leakage and ensure reliable evaluation metrics, 
prior to fitting and saving a final model on the entire dataset.
"""

from __future__ import annotations
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
import joblib

# Columns explicitly excluded from the model training matrix
FEATURE_EXCLUDE = {"label", "group", "filename", "paper_split"}


def feature_columns(table: pd.DataFrame) -> list[str]:
    """Extracts valid numeric feature columns, filtering out metadata and labels."""
    return [c for c in table.columns
            if c not in FEATURE_EXCLUDE and pd.api.types.is_numeric_dtype(table[c])]


def cross_validate(table: pd.DataFrame, n_splits: int = 5, seed: int = 0) -> dict:
    """Performs grouped cross-validation to estimate real-world generalization."""
    cols = feature_columns(table)
    X = table[cols].values
    y = table["label"].values.astype(int)
    groups = table["group"].values

    n_groups = len(np.unique(groups))
    n_splits = min(n_splits, n_groups)
    
    # Enforce strict group-disjoint splits to prevent tail number leakage
    gkf = GroupKFold(n_splits=n_splits)

    roc, pr, acc = [], [], []
    for tr, te in gkf.split(X, y, groups):
        
        # Skip folds that lack both positive and negative samples
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
            
        clf = HistGradientBoostingClassifier(random_state=seed)
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        
        # Calculate standard classification performance metrics
        roc.append(roc_auc_score(y[te], p))
        pr.append(average_precision_score(y[te], p))
        acc.append(accuracy_score(y[te], (p >= 0.5).astype(int)))

    return {
        "n_splits_used": len(roc),
        "roc_auc_mean": float(np.mean(roc)) if roc else None,
        "roc_auc_std": float(np.std(roc)) if roc else None,
        "pr_auc_mean": float(np.mean(pr)) if pr else None,
        "accuracy_mean": float(np.mean(acc)) if acc else None,
        "n_features": len(cols),
        "n_flights": len(table),
        "n_planes": int(n_groups),
        "class_balance": {int(k): int(v) for k, v in
                          zip(*np.unique(y, return_counts=True))},
    }


def fit_final(table: pd.DataFrame, out_path: str | Path, seed: int = 0) -> Path:
    """Trains the production model on all available data and serializes it."""
    cols = feature_columns(table)
    clf = HistGradientBoostingClassifier(random_state=seed)
    clf.fit(table[cols].values, table["label"].values.astype(int))
    
    out_path = Path(out_path)
    # Bundle the model with its required feature schema for later inference
    joblib.dump({"model": clf, "feature_columns": cols}, out_path)
    
    return out_path


def train(table: pd.DataFrame, model_out: str | Path, metrics_out: str | Path | None = None) -> dict:
    """Orchestrates validation, final training, and optional metric logging."""
    metrics = cross_validate(table)
    fit_final(table, model_out)
    
    metrics["model_path"] = str(model_out)
    
    if metrics_out:
        Path(metrics_out).write_text(json.dumps(metrics, indent=2))
        
    return metrics