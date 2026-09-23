"""
Long-format multivariate time-series CSV -> fixed-length sequence tensor store.

Input shape: one row per timestep, many timesteps per sequence (a flight, a
machine cycle), sequences concatenated and identified by an id column, each
sequence carrying one label (and optionally an entity/group id and a
precomputed fold). Output, under `out_dir`:

    X.f32          raw float32, shape (N, C, max_len), C-contiguous. The LAST
                   max_len timesteps of each sequence, left-padded with NaN
                   when shorter (same truncation the NGAFID-MC paper uses).
    meta.csv       one row per sequence: id, y (encoded 0..K-1), label (raw),
                   group, fold, length (raw timesteps), nan_frac
    manifest.json  channels, shape, classes, column roles, source path

Nothing here knows about any particular dataset: every column role comes in
as an argument, and every column not given a role (and not excluded) is a
channel. Memory stays bounded by one CSV chunk plus one sequence, because
each finished sequence is appended straight to X.f32.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

X_FILENAME = "X.f32"
META_FILENAME = "meta.csv"
MANIFEST_FILENAME = "manifest.json"


@dataclass
class IngestConfig:
    id_column: str
    label_column: str
    group_column: Optional[str] = None
    fold_column: Optional[str] = None
    exclude_columns: list[str] = field(default_factory=list)
    # Binary only: the raw label value treated as the positive class (y=1),
    # which is what PR-AUC scores. Defaults to the second sorted value.
    positive_label: Optional[str] = None
    max_len: int = 4096
    chunksize: int = 1_000_000
    max_sequences: Optional[int] = None
    n_folds: int = 5
    seed: int = 0


class _StopIngest(Exception):
    pass


def ingest_long_csv(csv_path: str | Path, out_dir: str | Path, config: IngestConfig) -> dict:
    csv_path, out_dir = Path(csv_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    role_columns = [c for c in (config.id_column, config.label_column,
                                config.group_column, config.fold_column) if c]
    missing = [c for c in role_columns + list(config.exclude_columns) if c not in header]
    if missing:
        raise ValueError(f"columns not found in {csv_path.name}: {missing}")
    channels = [c for c in header if c not in role_columns and c not in config.exclude_columns]
    if not channels:
        raise ValueError("no channel columns left after removing role and excluded columns")

    rows: list[dict] = []
    seen_ids: set = set()
    n_channels, max_len = len(channels), config.max_len

    with open(out_dir / X_FILENAME, "wb") as x_file:

        def emit(seq_id, frames: list[pd.DataFrame]) -> None:
            df = frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)
            per_seq = {}
            for role, col in (("label", config.label_column), ("group", config.group_column),
                              ("fold", config.fold_column)):
                if col is None:
                    continue
                values = df[col].dropna().unique()
                if len(values) != 1:
                    raise ValueError(
                        f"sequence {seq_id!r}: column {col!r} ({role}) must hold exactly one "
                        f"value per sequence, found {values[:5].tolist()}"
                    )
                per_seq[role] = values[0]

            x = df[channels].to_numpy(dtype=np.float32).T  # (C, L)
            length = x.shape[1]
            out = np.full((n_channels, max_len), np.nan, dtype=np.float32)
            tail = x[:, -max_len:]
            out[:, max_len - tail.shape[1]:] = tail
            x_file.write(out.tobytes())

            rows.append({
                "id": seq_id,
                "label": per_seq["label"],
                "group": per_seq.get("group"),
                "fold": per_seq.get("fold"),
                "length": length,
                "nan_frac": float(np.isnan(tail).mean()),
            })
            seen_ids.add(seq_id)
            if config.max_sequences is not None and len(rows) >= config.max_sequences:
                raise _StopIngest

        reader = pd.read_csv(
            csv_path,
            usecols=channels + role_columns,
            dtype={c: np.float32 for c in channels},
            chunksize=config.chunksize,
        )
        pending_id, pending = None, None
        try:
            for chunk in reader:
                ids = chunk[config.id_column].to_numpy()
                change = np.flatnonzero(ids[1:] != ids[:-1]) + 1
                starts = np.r_[0, change]
                ends = np.r_[change, len(ids)]
                for s, e in zip(starts, ends):
                    seq_id = ids[s]
                    segment = chunk.iloc[s:e]
                    if pending is not None and seq_id == pending_id:
                        pending.append(segment)
                        continue
                    if pending is not None:
                        emit(pending_id, pending)
                    if seq_id in seen_ids:
                        raise ValueError(
                            f"sequence id {seq_id!r} appears in two non-contiguous runs; "
                            "rows for one sequence must be contiguous"
                        )
                    pending_id, pending = seq_id, [segment]
            if pending is not None:
                emit(pending_id, pending)
        except _StopIngest:
            pass

    if not rows:
        raise ValueError(f"{csv_path.name} contained no sequences")

    meta = pd.DataFrame(rows)
    classes = sorted(meta["label"].unique().tolist())
    if len(classes) < 2:
        raise ValueError(f"label column {config.label_column!r} has fewer than 2 classes: {classes}")
    if config.positive_label is not None:
        matches = [c for c in classes if str(c) == str(config.positive_label)]
        if len(classes) != 2 or not matches:
            raise ValueError(
                f"positive_label {config.positive_label!r} needs a binary label; classes are {classes}")
        classes = [c for c in classes if c is not matches[0]] + matches
    meta.insert(1, "y", meta["label"].map({c: i for i, c in enumerate(classes)}).astype(int))

    if config.fold_column is None:
        meta["fold"] = assign_group_folds(
            meta["group"] if config.group_column else meta["id"], config.n_folds, config.seed)
        fold_source = f"assigned: {config.n_folds} folds, disjoint by " + (
            "group" if config.group_column else "sequence id")
    else:
        fold_source = f"column {config.fold_column!r}"
    meta["fold"] = meta["fold"].astype(int)
    meta.to_csv(out_dir / META_FILENAME, index=False)

    manifest = {
        "source": str(csv_path),
        "shape": [len(meta), n_channels, max_len],
        "dtype": "float32",
        "channels": channels,
        "classes": [c.item() if hasattr(c, "item") else c for c in classes],
        "positive_label": config.positive_label,
        "columns": {
            "id": config.id_column, "label": config.label_column,
            "group": config.group_column, "fold": config.fold_column,
            "excluded": list(config.exclude_columns),
        },
        "fold_source": fold_source,
        "truncated": bool(config.max_sequences is not None and len(meta) >= config.max_sequences),
    }
    (out_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


def assign_group_folds(groups: pd.Series, n_folds: int, seed: int) -> np.ndarray:
    """Deterministically assigns whole groups to folds, so no group spans two
    folds. Groups are shuffled with `seed`, then dealt round-robin."""
    unique = np.array(sorted(pd.unique(groups), key=str), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    fold_of = {g: i % n_folds for i, g in enumerate(unique)}
    return groups.map(fold_of).to_numpy()
