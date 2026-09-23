"""
Cheap baselines, run under the same folds and metrics as any plugin:

  dummy         — predicts the training-fold class prior. The floor.
  length_only   — gradient boosting on raw sequence length alone. A shortcut
                  probe: if this scores well, the label leaks through how
                  long a sequence is, and every model's number is suspect.
  summary_stats — 12 per-channel summary statistics (the original
                  agentic_ml tabular rollup) into gradient boosting.

Summary statistics are NaN-aware, so left padding and sensor dropout are
ignored rather than treated as zeros.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from agentic_pdm.harness.metrics import classification_metrics
from agentic_pdm.store import SequenceStore

STAT_NAMES = ("mean", "std", "min", "max", "range", "p10", "p50", "p90",
              "slope", "mean_abs_diff", "max_abs_diff", "last")


def summary_features(X: np.ndarray, batch: int = 256) -> np.ndarray:
    """(N, C, T) -> (N, C * 12) summary statistics, computed batch by batch."""
    n, c, t = X.shape
    out = np.empty((n, c, len(STAT_NAMES)), dtype=np.float32)
    time_axis = np.arange(t, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for start in range(0, n, batch):
            x = np.asarray(X[start:start + batch], dtype=np.float64)
            observed = ~np.isnan(x)
            mean = np.nanmean(x, axis=2)
            p10, p50, p90 = np.nanpercentile(x, [10, 50, 90], axis=2)
            lo, hi = np.nanmin(x, axis=2), np.nanmax(x, axis=2)

            count = observed.sum(axis=2)
            t_mean = (observed * time_axis).sum(axis=2) / np.maximum(count, 1)
            dt = np.where(observed, time_axis - t_mean[..., None], 0.0)
            dx = np.where(observed, x - mean[..., None], 0.0)
            var_t = (dt ** 2).sum(axis=2)
            slope = np.where(var_t > 0, (dt * dx).sum(axis=2) / np.where(var_t > 0, var_t, 1), 0.0)

            abs_diff = np.abs(np.diff(x, axis=2))
            last_idx = t - 1 - np.argmax(observed[..., ::-1], axis=2)
            last = np.take_along_axis(x, last_idx[..., None], axis=2)[..., 0]
            last = np.where(count > 0, last, np.nan)

            out[start:start + batch] = np.stack([
                mean, np.nanstd(x, axis=2), lo, hi, hi - lo, p10, p50, p90,
                slope, np.nanmean(abs_diff, axis=2), np.nanmax(abs_diff, axis=2), last,
            ], axis=2)
    return out.reshape(n, c * len(STAT_NAMES))


def feature_names(channels: list[str]) -> list[str]:
    return [f"{ch}__{s}" for ch in channels for s in STAT_NAMES]


def _cv(features: np.ndarray, store: SequenceStore, model_factory, folds) -> dict:
    y = store.meta["y"].to_numpy()
    fold_of = store.meta["fold"].to_numpy()
    per_fold = {}
    for fold in folds:
        train, val = fold_of != fold, fold_of == fold
        if model_factory is None:
            prior = np.bincount(y[train], minlength=store.n_classes) / train.sum()
            proba = np.tile(prior, (val.sum(), 1))
        else:
            model = model_factory().fit(features[train], y[train])
            proba = model.predict_proba(features[val])
        per_fold[int(fold)] = classification_metrics(y[val], proba)
    mean = {k: float(np.nanmean([m[k] for m in per_fold.values()])) for k in next(iter(per_fold.values()))}
    return {"folds": per_fold, "mean": mean}


def run_baselines(store: SequenceStore, folds=None, seed: int = 0, log=print) -> dict:
    folds = folds if folds is not None else store.folds

    def gbm():
        return HistGradientBoostingClassifier(random_state=seed)

    results = {"dummy": _cv(np.zeros((len(store.meta), 1)), store, None, folds)}
    log(f"dummy          {results['dummy']['mean']}")

    lengths = store.meta[["length"]].to_numpy(dtype=np.float32)
    results["length_only"] = _cv(lengths, store, gbm, folds)
    log(f"length_only    {results['length_only']['mean']}")

    log("computing summary statistics ...")
    feats = summary_features(store.X)
    results["summary_stats"] = _cv(feats, store, gbm, folds)
    log(f"summary_stats  {results['summary_stats']['mean']}")
    return results
