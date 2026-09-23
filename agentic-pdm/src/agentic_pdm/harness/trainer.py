"""
Harness-owned cross-validation trainer. The plugin supplies a model, an
augmentation, and optionally an optimizer; this module owns fold
membership, scaling, batching, the loss, evaluation, and metrics.

Two numbers are reported per fold:
  * "last"  — metrics after the final epoch. No validation peeking; this
              is the honest number.
  * "best"  — the epoch with the highest validation ROC-AUC. This matches
              the NGAFID-MC paper's "mean of the best metrics" protocol and
              is optimistic, since it selects on the validation fold itself.
Both are always written so a comparison against the paper is like-for-like
and the stricter number is never hidden.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F

from agentic_pdm.harness.metrics import classification_metrics
from agentic_pdm.harness.normalize import ChannelMinMax
from agentic_pdm.harness.plugin import Plugin, check_augment_output
from agentic_pdm.store import SequenceStore, pool_time


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 64
    pool: int = 4
    seed: int = 0
    threads: Optional[int] = None
    eval_batch_size: int = 256
    plugin_config: dict = field(default_factory=dict)


def _log(msg: str) -> None:
    print(msg, flush=True)


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return -(targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


@torch.no_grad()
def predict_proba(model: torch.nn.Module, X: np.ndarray, batch_size: int) -> np.ndarray:
    model.eval()
    parts = []
    for start in range(0, len(X), batch_size):
        logits = model(torch.from_numpy(X[start:start + batch_size]))
        parts.append(torch.softmax(logits.float(), dim=1).numpy())
    return np.concatenate(parts)


def train_fold(
    X_pooled: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    n_classes: int,
    plugin: Plugin,
    cfg: TrainConfig,
    log: Callable[[str], None] = _log,
    on_epoch: Optional[Callable[[dict], None]] = None,
) -> dict:
    scaler = ChannelMinMax.fit(X_pooled[train_idx])
    X_train = scaler.transform(X_pooled[train_idx])
    X_val = scaler.transform(X_pooled[val_idx])
    y_train, y_val = y[train_idx], y[val_idx]
    targets_train = F.one_hot(torch.from_numpy(y_train).long(), n_classes).float()

    torch.manual_seed(cfg.seed)
    generator = torch.Generator().manual_seed(cfg.seed)
    n_channels, seq_len = X_train.shape[1], X_train.shape[2]
    model = plugin.build_model(n_channels, seq_len, n_classes, dict(cfg.plugin_config))
    optimizer = plugin.configure_optimizer(model, dict(cfg.plugin_config))
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    with torch.no_grad():
        probe = model(torch.from_numpy(X_train[:2]))
    if tuple(probe.shape) != (min(2, len(X_train)), n_classes):
        raise ValueError(f"model output shape {tuple(probe.shape)} != (batch, {n_classes})")

    X_train_t = torch.from_numpy(X_train)
    history = []
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.perf_counter()
        model.train()
        order = torch.randperm(len(X_train_t), generator=generator)
        total, seen = 0.0, 0
        for start in range(0, len(order), cfg.batch_size):
            idx = order[start:start + cfg.batch_size]
            xb, yb = X_train_t[idx], targets_train[idx]
            xa, ya = plugin.augment(xb.clone(), yb.clone(), generator, dict(cfg.plugin_config))
            check_augment_output(xb, yb, xa, ya)
            loss = soft_cross_entropy(model(xa), ya)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at epoch {epoch}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
            seen += len(idx)
        train_seconds = time.perf_counter() - t0

        val_metrics = classification_metrics(y_val, predict_proba(model, X_val, cfg.eval_batch_size))
        record = {"epoch": epoch, "train_loss": total / seen, "train_seconds": train_seconds, **val_metrics}
        history.append(record)
        log(f"  epoch {epoch:3d}/{cfg.epochs}  train_loss {record['train_loss']:.4f}  "
            f"val_loss {val_metrics['log_loss']:.4f}  roc {val_metrics['roc_auc']:.4f}  "
            f"pr {val_metrics['pr_auc']:.4f}  acc {val_metrics['accuracy']:.4f}  ({train_seconds:.1f}s)")
        if on_epoch:
            on_epoch(record)

    def _score(r):
        return -math.inf if math.isnan(r["roc_auc"]) else r["roc_auc"]

    best = max(history, key=_score)
    return {
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_params": int(n_params),
        "mean_train_seconds_per_epoch": float(np.mean([h["train_seconds"] for h in history])),
        "last": history[-1],
        "best": best,
        "history": history,
    }


METRIC_KEYS = ("roc_auc", "pr_auc", "accuracy", "log_loss")


def summarize_folds(fold_results: dict[int, dict]) -> dict:
    summary = {}
    for which in ("last", "best"):
        summary[which] = {
            k: float(np.nanmean([r[which][k] for r in fold_results.values()])) for k in METRIC_KEYS
        }
    summary["mean_train_seconds_per_epoch"] = float(
        np.mean([r["mean_train_seconds_per_epoch"] for r in fold_results.values()]))
    return summary


def run_cv(
    store: SequenceStore,
    plugin: Plugin,
    cfg: TrainConfig,
    out_dir: str | Path,
    folds: Optional[list[int]] = None,
    log: Callable[[str], None] = _log,
) -> dict:
    """Trains one model per requested validation fold (default: all) and
    writes results.json to out_dir, rewritten after every epoch so a
    long-running job always has current progress on disk."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if cfg.threads:
        torch.set_num_threads(cfg.threads)
    folds = folds if folds is not None else store.folds
    unknown = sorted(set(folds) - set(store.folds))
    if unknown:
        raise ValueError(f"unknown folds {unknown}; store has {store.folds}")

    t0 = time.perf_counter()
    X_pooled = pool_time(store.X, cfg.pool)
    log(f"pooled {tuple(store.X.shape)} -> {X_pooled.shape} in {time.perf_counter() - t0:.1f}s")
    y = store.meta["y"].to_numpy()
    fold_of = store.meta["fold"].to_numpy()

    results = {
        "status": "running",
        "plugin": plugin.name,
        "plugin_path": plugin.path,
        "config": asdict(cfg),
        "store": store.manifest.get("source"),
        "folds": {},
        "progress": {},
    }
    results_path = out_dir / "results.json"

    def write():
        results_path.write_text(json.dumps(results, indent=2, default=str))

    for fold in folds:
        train_idx = np.flatnonzero(fold_of != fold)
        val_idx = np.flatnonzero(fold_of == fold)
        log(f"fold {fold}: train {len(train_idx)}  val {len(val_idx)}")

        def on_epoch(record, fold=fold):
            results["progress"] = {"fold": fold, **record}
            write()

        results["folds"][fold] = train_fold(
            X_pooled, y, train_idx, val_idx, store.n_classes, plugin, cfg, log=log, on_epoch=on_epoch)
        write()

    results["summary"] = summarize_folds(results["folds"])
    results["status"] = "completed"
    results["wall_seconds"] = time.perf_counter() - t0
    write()
    return results
