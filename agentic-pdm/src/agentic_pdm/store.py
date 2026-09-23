"""Read side of the tensor store that ingest.py writes."""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from agentic_pdm.ingest import MANIFEST_FILENAME, META_FILENAME, X_FILENAME


@dataclass
class SequenceStore:
    X: np.ndarray          # (N, C, T) float32, memory-mapped read-only
    meta: pd.DataFrame     # one row per sequence, aligned with X
    manifest: dict

    @property
    def n_classes(self) -> int:
        return len(self.manifest["classes"])

    @property
    def folds(self) -> list[int]:
        return sorted(self.meta["fold"].unique().tolist())


def load_store(store_dir: str | Path) -> SequenceStore:
    store_dir = Path(store_dir)
    manifest = json.loads((store_dir / MANIFEST_FILENAME).read_text())
    meta = pd.read_csv(store_dir / META_FILENAME)
    X = np.memmap(store_dir / X_FILENAME, dtype=np.float32, mode="r", shape=tuple(manifest["shape"]))
    if len(meta) != X.shape[0]:
        raise ValueError(f"meta has {len(meta)} rows but X has {X.shape[0]} sequences")
    return SequenceStore(X=X, meta=meta, manifest=manifest)


def pool_time(X: np.ndarray, factor: int, batch: int = 256) -> np.ndarray:
    """Non-overlapping NaN-aware mean pooling along time: (N, C, T) ->
    (N, C, T // factor). A window that is entirely NaN (padding, dropout)
    stays NaN. Processes `batch` sequences at a time so a memory-mapped X is
    never fully materialized at full resolution."""
    if factor == 1:
        return np.array(X, dtype=np.float32)
    n, c, t = X.shape
    if t % factor:
        raise ValueError(f"sequence length {t} is not divisible by pool factor {factor}")
    out = np.empty((n, c, t // factor), dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN windows
        for start in range(0, n, batch):
            block = np.asarray(X[start:start + batch], dtype=np.float32)
            out[start:start + batch] = np.nanmean(
                block.reshape(block.shape[0], c, t // factor, factor), axis=3)
    return out
