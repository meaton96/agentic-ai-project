"""
Phase 9 drift measurement: deterministic, no LLM, no new dependency
(plain numpy — not scipy). Compares a newly arrived batch against the
data the currently-served model was trained on, feature by feature.

This is a replay of historical data where labels for the batch are
actually available immediately (see `batch_metrics` below) — that's a
replay-specific simplification of this evaluation harness, explicitly
NOT a claim that a live deployment would have ground truth this fast
(real deployments have label lag). orchestrator/dynamic_loop.py's
retrain_decision branch is the only caller that treats this report as
evidence for a decision; nothing here decides anything itself.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from agentic_ml.harness.metrics import compute_metrics

REPLAY_SIMPLIFICATION_NOTE = (
    "batch_metrics uses labels available immediately in this historical replay; "
    "a live deployment would not have ground truth this fast (label lag is out of "
    "scope for this phase)."
)


def compute_drift_report(
    baseline_df: pd.DataFrame,
    batch_df: pd.DataFrame,
    feature_columns: list[str],
    pipeline: Optional[object] = None,
    batch_y: Optional[pd.Series] = None,
    metric_names: Optional[list[str]] = None,
    shift_threshold: float = 2.0,
) -> dict:
    """Per-feature standardized mean shift: |batch_mean - baseline_mean| /
    baseline_std, plus an aggregate score (mean absolute shift across
    features and the fraction of features past `shift_threshold`
    standard deviations). Non-numeric columns and columns missing from
    either side are silently skipped — this is a numeric-drift measure,
    not a general schema-diff tool. If `pipeline` and `batch_y` are both
    given, also scores the batch with the current model via
    harness/metrics.py::compute_metrics — see REPLAY_SIMPLIFICATION_NOTE.
    """
    per_feature_shift: dict[str, float] = {}
    for col in feature_columns:
        if col not in baseline_df.columns or col not in batch_df.columns:
            continue
        baseline_col = pd.to_numeric(baseline_df[col], errors="coerce")
        batch_col = pd.to_numeric(batch_df[col], errors="coerce")
        baseline_mean = baseline_col.mean()
        baseline_std = baseline_col.std()
        batch_mean = batch_col.mean()
        if pd.isna(baseline_mean) or pd.isna(batch_mean):
            continue  # non-numeric (or all-NaN) column — no mean shift to compute

        if not baseline_std or pd.isna(baseline_std):
            # zero (or undefined) baseline variance: any real difference in the
            # batch mean is an infinite standardized shift, not a divide-by-zero bug
            shift = 0.0 if abs(batch_mean - baseline_mean) < 1e-12 else float("inf")
        else:
            shift = float(abs(batch_mean - baseline_mean) / baseline_std)
        per_feature_shift[col] = round(shift, 6)

    finite_shifts = [s for s in per_feature_shift.values() if np.isfinite(s)]
    n_features = len(per_feature_shift)
    n_over_threshold = sum(1 for s in per_feature_shift.values() if s > shift_threshold)

    report = {
        "n_features_compared": n_features,
        "per_feature_shift": per_feature_shift,
        "mean_abs_shift": round(float(np.mean(finite_shifts)), 6) if finite_shifts else 0.0,
        "shift_threshold": shift_threshold,
        "n_features_over_threshold": n_over_threshold,
        "fraction_features_over_threshold": (
            round(n_over_threshold / n_features, 6) if n_features else 0.0
        ),
        "n_baseline_examples": int(len(baseline_df)),
        "n_batch_examples": int(len(batch_df)),
        "batch_metrics": None,
        "replay_simplification_note": REPLAY_SIMPLIFICATION_NOTE,
    }

    if pipeline is not None and batch_y is not None and metric_names:
        X_batch = batch_df[feature_columns]
        y_pred = pipeline.predict(X_batch)
        y_proba = pipeline.predict_proba(X_batch)
        y_true = batch_y.values if hasattr(batch_y, "values") else np.asarray(batch_y)
        metric_results = compute_metrics(y_true, y_pred, y_proba, metric_names, n_bootstrap=200)
        report["batch_metrics"] = {m: metric_results[m].to_dict() for m in metric_names}

    return report
