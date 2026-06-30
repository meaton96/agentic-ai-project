"""
evaluate_features.py
====================
Provides cross-validation and feature evaluation routines for the 
automated feature discovery process.

This module converts a proposed feature specification into a feature table, 
scores its performance (ROC-AUC, PR-AUC) via tail-disjoint holdout splits, 
and logs the execution details and metrics to a JSONL file.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import pandas as pd

try:
    from . import featurize_spec as fs
except ImportError:
    import featurize_spec as fs


# --------------------------------------------------------------------------
# Cross-Validation Adapters
# --------------------------------------------------------------------------

def _fallback_holdout_cv(table: pd.DataFrame, holdout_fold) -> dict:
    """
    Fallback cross-validation routine for offline testing.
    Enforces tail-disjoint splits manually if the primary realdata module is unavailable.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score, average_precision_score
    
    if "paper_split" not in table.columns:
        raise ValueError("table has no 'paper_split' fold column")
        
    cols = fs.feature_columns(table)
    is_hold = table["paper_split"].astype(str) == str(holdout_fold)
    train, test = table[~is_hold], table[is_hold]
    
    # Strictly enforce leakage prevention
    overlap = set(train["group"]) & set(test["group"])
    if overlap:
        raise AssertionError(f"tail leakage: {len(overlap)} tails in both splits")
        
    clf = HistGradientBoostingClassifier(random_state=0).fit(
        train[cols].values, train["label"].values.astype(int))  #type: ignore
        
    p = clf.predict_proba(test[cols].values)[:, 1]
    y = test["label"].values.astype(int)
    
    return {"holdout_roc_auc": float(roc_auc_score(y, p)),  #type: ignore
            "holdout_pr_auc": float(average_precision_score(y, p)),  #type: ignore
            "n_train": int(len(train)), "n_holdout": int(len(test)),
            "train_holdout_plane_overlap": 0}


# Attempt to wire up the primary cross-validation routine from realdata
try:
    try:
        from .realdata import fit_excluding_fold as _real_cv
    except ImportError:
        from scripts.realdata import fit_excluding_fold as _real_cv

    def _default_cv(table, holdout_fold):
        """Wrapper for the primary leakage-free validation function."""
        import tempfile, os
        tmp = os.path.join(tempfile.gettempdir(), "disc_model.joblib")
        m = _real_cv(table, holdout_fold, tmp)
        return m
        
    _CV_SOURCE = "realdata.fit_excluding_fold"
    
except Exception:
    # Revert to standalone offline mode if realdata cannot be loaded
    _default_cv = _fallback_holdout_cv
    _CV_SOURCE = "fallback_holdout_cv (offline)"


# --------------------------------------------------------------------------
# Core Evaluation Tool
# --------------------------------------------------------------------------

def evaluate_features(spec: dict, flight_iter: fs.FlightIter, holdout_fold=0,
                      cv_fn: Callable[[pd.DataFrame, object], dict] | None = None,
                      log_path: str | Path | None = None, note: str = "") -> dict:
    """
    Constructs a feature table from a specification, evaluates it on a held-out 
    tail-disjoint fold, and logs the execution.
    
    Returns normalized AUC keys for consistent downstream consumption.
    """
    cv = cv_fn or _default_cv
    t0 = time.time()
    
    # Standardize specification formatting
    canon = fs.canonical_spec(spec)
    table = fs.build_table(flight_iter, canon)
    n_features = len(fs.feature_columns(table))
    
    # Execute cross-validation
    m = cv(table, holdout_fold)
    roc = m.get("holdout_roc_auc", m.get("roc_auc"))
    
    result = {
        "spec_id": canon["spec_id"],
        "n_features": n_features,
        "roc_auc": round(float(roc), 4) if roc is not None else None,
        "pr_auc": (round(float(m["holdout_pr_auc"]), 4)
                   if m.get("holdout_pr_auc") is not None else None),
        "n_train": m.get("n_train"),
        "n_holdout": m.get("n_holdout"),
        "plane_overlap": m.get("train_holdout_plane_overlap"),
        "holdout_fold": holdout_fold,
        "seconds": round(time.time() - t0, 2),
        "cv_source": _CV_SOURCE,
    }
    
    if log_path:
        _append_log(log_path, canon, result, note)
        
    return result


def _append_log(log_path, canon_spec, result, note):
    """Appends an evaluation result record to the specified JSONL log file."""
    entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": "evaluate_features",
             "spec_id": result["spec_id"], "spec": canon_spec["features"],
             "roc_auc": result["roc_auc"], "n_features": result["n_features"],
             "plane_overlap": result["plane_overlap"], "seconds": result["seconds"],
             "cv_source": result["cv_source"], "note": note}
             
    p = Path(log_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    
    with p.open("a") as f:
        f.write(json.dumps(entry) + "\n")