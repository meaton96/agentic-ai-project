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

The loss is the harness's too: soft-target cross-entropy, optionally with
label smoothing or a focal term (TrainConfig.loss/label_smoothing/
focal_gamma). A plugin chooses the model, augmentation, optimizer and LR
schedule, never how its predictions are scored.

`ensemble` > 1 trains that many members per fold (seeds seed, seed+1, ...)
and scores the average of their predicted probabilities at each epoch.
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
    # Keep only the last max_len timesteps (before pooling); None = all.
    max_len: Optional[int] = None
    loss: str = "cross_entropy"  # or "focal"
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    ensemble: int = 1
    plugin_config: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.loss not in LOSSES:
            raise ValueError(f"loss must be one of {LOSSES}, got {self.loss!r}")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError(f"label_smoothing must be in [0, 1), got {self.label_smoothing}")
        if self.ensemble < 1:
            raise ValueError(f"ensemble must be >= 1, got {self.ensemble}")


LOSSES = ("cross_entropy", "focal")


def _log(msg: str) -> None:
    print(msg, flush=True)


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return -(targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def harness_loss(logits: torch.Tensor, targets: torch.Tensor, cfg: TrainConfig) -> torch.Tensor:
    """Soft-target loss per cfg. Label smoothing mixes the targets toward
    uniform; focal loss down-weights examples the model already gets right
    by (1 - p_target)^gamma."""
    k = targets.shape[1]
    if cfg.label_smoothing:
        targets = targets * (1.0 - cfg.label_smoothing) + cfg.label_smoothing / k
    log_p = F.log_softmax(logits, dim=1)
    if cfg.loss == "focal":
        weight = (1.0 - log_p.exp()).clamp(min=0.0) ** cfg.focal_gamma
        return -(targets * weight * log_p).sum(dim=1).mean()
    return -(targets * log_p).sum(dim=1).mean()


@torch.no_grad()
def predict_proba(model: torch.nn.Module, X: np.ndarray, batch_size: int) -> np.ndarray:
    model.eval()
    parts = []
    for start in range(0, len(X), batch_size):
        logits = model(torch.from_numpy(X[start:start + batch_size]))
        parts.append(torch.softmax(logits.float(), dim=1).numpy())
    return np.concatenate(parts)


def _train_member(
    X_train: np.ndarray,
    X_val: np.ndarray,
    targets_train: torch.Tensor,
    n_classes: int,
    plugin: Plugin,
    cfg: TrainConfig,
    seed: int,
):
    """Trains one model, yielding (epoch, train_loss, train_seconds,
    val_proba, n_params) after every epoch."""
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    n_channels, seq_len = X_train.shape[1], X_train.shape[2]
    plugin_config = dict(cfg.plugin_config)
    model = plugin.build_model(n_channels, seq_len, n_classes, plugin_config)
    optimizer = plugin.configure_optimizer(model, plugin_config)
    steps_per_epoch = math.ceil(len(X_train) / cfg.batch_size)
    scheduler = plugin.configure_scheduler(optimizer, plugin_config, steps_per_epoch * cfg.epochs)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    with torch.no_grad():
        model.eval()
        probe = model(torch.from_numpy(X_train[:2]))
    if tuple(probe.shape) != (min(2, len(X_train)), n_classes):
        raise ValueError(f"model output shape {tuple(probe.shape)} != (batch, {n_classes})")

    X_train_t = torch.from_numpy(X_train)
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.perf_counter()
        model.train()
        order = torch.randperm(len(X_train_t), generator=generator)
        total, seen = 0.0, 0
        for start in range(0, len(order), cfg.batch_size):
            idx = order[start:start + cfg.batch_size]
            xb, yb = X_train_t[idx], targets_train[idx]
            xa, ya = plugin.augment(xb.clone(), yb.clone(), generator, plugin_config)
            check_augment_output(xb, yb, xa, ya)
            loss = harness_loss(model(xa), ya, cfg)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at epoch {epoch}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            total += loss.item() * len(idx)
            seen += len(idx)
        train_seconds = time.perf_counter() - t0
        yield epoch, total / seen, train_seconds, predict_proba(model, X_val, cfg.eval_batch_size), n_params


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

    # Per epoch: summed member probabilities, train losses, seconds.
    proba_sum = [None] * cfg.epochs
    losses = [[] for _ in range(cfg.epochs)]
    seconds = [[] for _ in range(cfg.epochs)]
    history = []
    n_params = 0
    for member in range(cfg.ensemble):
        for epoch, train_loss, train_seconds, proba, n_params in _train_member(
            X_train, X_val, targets_train, n_classes, plugin, cfg, seed=cfg.seed + member
        ):
            i = epoch - 1
            proba_sum[i] = proba if proba_sum[i] is None else proba_sum[i] + proba
            losses[i].append(train_loss)
            seconds[i].append(train_seconds)
            member_metrics = classification_metrics(y_val, proba)
            tag = f" member {member + 1}/{cfg.ensemble}" if cfg.ensemble > 1 else ""
            log(f"  epoch {epoch:3d}/{cfg.epochs}{tag}  train_loss {train_loss:.4f}  "
                f"val_loss {member_metrics['log_loss']:.4f}  roc {member_metrics['roc_auc']:.4f}  "
                f"pr {member_metrics['pr_auc']:.4f}  acc {member_metrics['accuracy']:.4f}  ({train_seconds:.1f}s)")
            if on_epoch:
                on_epoch({"member": member, "epoch": epoch, "train_loss": train_loss, **member_metrics})

    for i in range(cfg.epochs):
        ensemble_metrics = classification_metrics(y_val, proba_sum[i] / cfg.ensemble)
        history.append({
            "epoch": i + 1,
            "train_loss": float(np.mean(losses[i])),
            "train_seconds": float(np.sum(seconds[i])),
            **ensemble_metrics,
        })

    def _score(r):
        return -math.inf if math.isnan(r["roc_auc"]) else r["roc_auc"]

    best = max(history, key=_score)
    return {
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_params": int(n_params),
        "ensemble": cfg.ensemble,
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
    X_source = store.X
    if cfg.max_len is not None:
        if not 0 < cfg.max_len <= store.X.shape[2]:
            raise ValueError(f"max_len {cfg.max_len} must be in 1..{store.X.shape[2]}")
        X_source = store.X[:, :, -cfg.max_len:]
    X_pooled = pool_time(X_source, cfg.pool)
    log(f"pooled {tuple(X_source.shape)} -> {X_pooled.shape} in {time.perf_counter() - t0:.1f}s")
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
    results["summary"]["folds"] = [int(f) for f in folds]
    results["status"] = "completed"
    results["wall_seconds"] = time.perf_counter() - t0
    write()
    return results
