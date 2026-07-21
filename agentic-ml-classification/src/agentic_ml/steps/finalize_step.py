"""
Finalize step: deterministic, no LLM. Refits the accepted candidate on
train+val, evaluates it exactly once on the locked test set, and
persists the model bundle (for the deep-dive agent) — extracted out of
run_orchestrator.py's inline logic (unchanged there — this module is
new, additive, used only by the dynamic orchestrator) the same way
steps/split_step.py extracts the split logic next to it.

This is the one irreversible action in either orchestrator: the test
set is touched here and nowhere else. Callers must not call this more
than once per run — orchestrator/dynamic_loop.py enforces that via
required_state={"final_test_metrics_present": False} on the "finalize"
catalog entry, the same precondition-gating every other agent uses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from sklearn.base import clone

from agentic_ml.harness.attribution import compute_background
from agentic_ml.harness.metrics import compute_metrics


@dataclass
class FinalizeStepResult:
    ok: bool
    pipeline: Optional[object]
    test_metrics: Optional[dict]
    model_path: Optional[str]
    feature_columns: Optional[list[str]]
    errors: list[str] = field(default_factory=list)


def run_finalize_step(
    X: pd.DataFrame,
    y: pd.Series,
    manifest,
    candidate_pipeline,
    metric_names: list[str],
    seed: int,
    run_id: str,
) -> FinalizeStepResult:
    train_and_val_idx = sorted(manifest.train_idx + manifest.val_idx)
    final_pipeline = clone(candidate_pipeline)
    final_pipeline.fit(X.iloc[train_and_val_idx], y.iloc[train_and_val_idx])

    y_pred = final_pipeline.predict(X.iloc[manifest.test_idx])
    proba = final_pipeline.predict_proba(X.iloc[manifest.test_idx])
    test_results = compute_metrics(
        y.iloc[manifest.test_idx].values, y_pred, proba, metric_names,
        n_bootstrap=200, seed=seed,
    )
    test_metrics = {m: test_results[m].to_dict() for m in metric_names}

    model_path = None
    # Occlusion attribution (harness/attribution.py) is binary-only for
    # now — same guard run_orchestrator.py's inline version uses.
    if y.iloc[train_and_val_idx].nunique() == 2:
        background = compute_background(
            X.iloc[train_and_val_idx], list(X.columns),
            normal_mask=(y.iloc[train_and_val_idx] == 0),
        )
        path = Path("artifacts/models") / f"{run_id}_model.joblib"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": final_pipeline, "feature_columns": list(X.columns), "background": background},
            path,
        )
        model_path = str(path)

    return FinalizeStepResult(
        ok=True, pipeline=final_pipeline, test_metrics=test_metrics,
        model_path=model_path, feature_columns=list(X.columns),
    )
