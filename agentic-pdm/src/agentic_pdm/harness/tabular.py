"""The harness's path for tabular catalog models, beside trainer.run_cv.

A tabular model reads features computed from each window, not the window
itself. Fold membership and metrics are the same as for any other experiment,
and results.json has run_cv's shape, with one history row per boosting stage
standing in for an epoch. So the pipeline's record gate, leaderboard and
"last"/"best" protocols work unchanged. Validation predictions at every stage
are written to preds.npz for binary tasks.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss

from agentic_pdm.baselines import summary_features
from agentic_pdm.harness.metrics import classification_metrics
from agentic_pdm.harness.trainer import summarize_folds
from agentic_pdm.store import SequenceStore


def _log(msg: str) -> None:
    print(msg, flush=True)


def tabular_features(X: np.ndarray, kind: str) -> np.ndarray:
    """(N, C, T) -> (N, n_features). NaNs stay: gradient boosting handles them."""
    if kind == "last_record":
        return np.asarray(X[:, :, -1], dtype=np.float32)
    if kind == "summary_stats":
        return summary_features(X)
    if kind == "both":
        return np.hstack([np.asarray(X[:, :, -1], dtype=np.float32), summary_features(X)])
    raise ValueError(f"unknown feature set {kind!r}")


def _train_fold(features, y, train_idx, val_idx, n_classes, params, seed, overhead_seconds):
    """Returns (the fold's record, validation probabilities with shape [stages, N_val, K])."""
    t0 = time.perf_counter()
    model = HistGradientBoostingClassifier(
        max_iter=params["max_iter"], learning_rate=params["learning_rate"],
        max_leaf_nodes=params["max_leaf_nodes"], min_samples_leaf=params["min_samples_leaf"],
        l2_regularization=params["l2_regularization"], early_stopping=False, random_state=seed,
    ).fit(features[train_idx], y[train_idx])
    if len(model.classes_) != n_classes:
        raise ValueError(f"training data has {len(model.classes_)} of {n_classes} classes")
    labels = list(range(n_classes))
    proba_val = np.stack(list(model.staged_predict_proba(features[val_idx])))
    train_losses = [log_loss(y[train_idx], p, labels=labels) for p in model.staged_predict_proba(features[train_idx])]
    # The budget charges what history reports, so this includes the feature work.
    per_stage = (time.perf_counter() - t0 + overhead_seconds) / len(proba_val)
    history = [{"epoch": i + 1, "train_loss": float(train_losses[i]), "train_seconds": per_stage,
                **classification_metrics(y[val_idx], proba_val[i])} for i in range(len(proba_val))]
    best = max(history, key=lambda r: -math.inf if math.isnan(r["roc_auc"]) else r["roc_auc"])
    return {"n_train": int(len(train_idx)), "n_val": int(len(val_idx)), "n_params": 0, "ensemble": 1,
            "mean_train_seconds_per_epoch": per_stage, "last": history[-1], "best": best,
            "history": history}, proba_val


def run_tabular_cv(store: SequenceStore, normalized: dict, out_dir: str | Path,
                   log: Callable[[str], None] = _log) -> dict:
    """Cross-validates a normalized gbm experiment, writing results.json after
    every fold and preds.npz at the end."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    params = normalized["architecture"]["params"]
    seed, folds = normalized["training"]["seed"], list(normalized["folds"])

    t0 = time.perf_counter()
    X = store.X[:, :, -normalized["input"]["max_len"]:]
    features = tabular_features(X, params["features"])
    feature_seconds = time.perf_counter() - t0
    log(f"features {params['features']} {features.shape} in {feature_seconds:.1f}s")

    y = store.meta["y"].to_numpy()
    fold_of = store.meta["fold"].to_numpy()
    p = np.full((params["max_iter"], len(y)), np.nan)
    results = {"status": "running", "plugin": normalized["name"], "plugin_path": "catalog",
               "config": {"architecture": normalized["architecture"], "seed": seed,
                          "max_len": normalized["input"]["max_len"]},
               "store": store.manifest.get("source"), "folds": {}, "progress": {}}
    results_path = out_dir / "results.json"

    def write():
        results_path.write_text(json.dumps(results, indent=2, default=str))

    for fold in folds:
        train_idx, val_idx = np.flatnonzero(fold_of != fold), np.flatnonzero(fold_of == fold)
        record, proba = _train_fold(features, y, train_idx, val_idx, store.n_classes, params, seed,
                                    feature_seconds / len(folds))
        results["folds"][fold] = record
        write()
        if store.n_classes == 2:
            p[:, val_idx] = proba[:, :, 1]
        last = record["last"]
        log(f"fold {fold}: train {len(train_idx)}  val {len(val_idx)}  roc {last['roc_auc']:.4f}  "
            f"pr {last['pr_auc']:.4f}  acc {last['accuracy']:.4f}  "
            f"({record['mean_train_seconds_per_epoch'] * len(proba):.1f}s)")

    results["summary"] = summarize_folds(results["folds"])
    results["summary"]["folds"] = [int(f) for f in folds]
    results["status"] = "completed"
    results["wall_seconds"] = time.perf_counter() - t0
    write()
    if store.n_classes == 2:
        np.savez_compressed(out_dir / "preds.npz", id=store.meta["id"].to_numpy(), y=y, fold=fold_of,
                            group=store.meta["group"].to_numpy(), p=p)
    return results
