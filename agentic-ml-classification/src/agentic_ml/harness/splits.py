"""
Split manager. This is the core trust boundary of the whole system:
agents never see this code path, never choose indices, and never see
the test set until the harness explicitly releases final metrics.

Supported strategies:
  - random        : simple random split (only safe for iid tabular data)
  - stratified     : stratified by target column
  - group          : all rows for a given group_column value land in
                      exactly one split (prevents entity leakage)
  - time           : rows sorted by time_column, split by position
                      (train = earliest, test = latest)
  - group_time     : groups assigned to splits based on each group's
                      earliest timestamp (prevents both entity and
                      temporal leakage simultaneously)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold


VALID_STRATEGIES = {"random", "stratified", "group", "time", "group_time"}


@dataclass
class SplitManifest:
    strategy: str
    seed: int
    data_hash: str
    train_idx: list[int]
    val_idx: list[int]
    test_idx: list[int]
    target_distribution: dict = field(default_factory=dict)
    group_overlap_ok: bool = True
    time_range: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "seed": self.seed,
            "data_hash": self.data_hash,
            "n_train": len(self.train_idx),
            "n_val": len(self.val_idx),
            "n_test": len(self.test_idx),
            "train_idx": self.train_idx,
            "val_idx": self.val_idx,
            "test_idx": self.test_idx,
            "target_distribution": self.target_distribution,
            "group_overlap_ok": self.group_overlap_ok,
            "time_range": self.time_range,
        }

    def write(self, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(self.to_dict(), indent=2))


def _target_distribution(y: pd.Series, idx: np.ndarray) -> dict:
    counts = y.iloc[idx].value_counts(normalize=True).to_dict()
    return {str(k): round(float(v), 4) for k, v in counts.items()}


def make_split(
    df: pd.DataFrame,
    target_column: str,
    data_hash: str,
    strategy: str = "stratified",
    seed: int = 42,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    group_column: Optional[str] = None,
    time_column: Optional[str] = None,
) -> SplitManifest:
    if strategy not in VALID_STRATEGIES:
        raise ValueError(f"Unknown split strategy '{strategy}'. Valid: {VALID_STRATEGIES}")

    n = len(df)
    y = df[target_column]
    rng = np.random.RandomState(seed)

    if strategy in ("group", "group_time") and not group_column:
        raise ValueError(f"strategy='{strategy}' requires group_column")
    if strategy in ("time", "group_time") and not time_column:
        raise ValueError(f"strategy='{strategy}' requires time_column")

    if strategy == "random":
        idx = rng.permutation(n)
        n_test = int(n * test_frac)
        n_val = int(n * val_frac)
        test_idx = idx[:n_test]
        val_idx = idx[n_test : n_test + n_val]
        train_idx = idx[n_test + n_val :]

    elif strategy == "stratified":
        idx = np.arange(n)
        # two-stage stratified split: carve off test, then val from remainder
        test_idx, remainder_idx = _stratified_two_way(idx, y.values, test_frac, seed)
        val_relative_frac = val_frac / (1.0 - test_frac)
        val_idx, train_idx = _stratified_two_way(
            remainder_idx, y.values[remainder_idx], val_relative_frac, seed + 1
        )

    elif strategy == "group":
        groups = df[group_column].values
        unique_groups = pd.unique(groups)
        rng.shuffle(unique_groups)
        n_test_groups = max(1, int(len(unique_groups) * test_frac))
        n_val_groups = max(1, int(len(unique_groups) * val_frac))
        test_groups = set(unique_groups[:n_test_groups])
        val_groups = set(unique_groups[n_test_groups : n_test_groups + n_val_groups])
        train_groups = set(unique_groups[n_test_groups + n_val_groups :])

        test_idx = np.where(np.isin(groups, list(test_groups)))[0]
        val_idx = np.where(np.isin(groups, list(val_groups)))[0]
        train_idx = np.where(np.isin(groups, list(train_groups)))[0]

    elif strategy == "time":
        order = df[time_column].values.argsort(kind="stable")
        n_test = int(n * test_frac)
        n_val = int(n * val_frac)
        train_idx = order[: n - n_test - n_val]
        val_idx = order[n - n_test - n_val : n - n_test]
        test_idx = order[n - n_test :]

    elif strategy == "group_time":
        # earliest timestamp per group decides which split the whole
        # group falls into, so no group straddles a time boundary
        group_first_time = df.groupby(group_column)[time_column].min().sort_values()
        ordered_groups = group_first_time.index.values
        n_groups = len(ordered_groups)
        n_test_groups = max(1, int(n_groups * test_frac))
        n_val_groups = max(1, int(n_groups * val_frac))
        train_groups = ordered_groups[: n_groups - n_test_groups - n_val_groups]
        val_groups = ordered_groups[
            n_groups - n_test_groups - n_val_groups : n_groups - n_test_groups
        ]
        test_groups = ordered_groups[n_groups - n_test_groups :]

        groups_col = df[group_column].values
        train_idx = np.where(np.isin(groups_col, train_groups))[0]
        val_idx = np.where(np.isin(groups_col, val_groups))[0]
        test_idx = np.where(np.isin(groups_col, test_groups))[0]

    else:  # pragma: no cover - guarded above
        raise AssertionError

    group_overlap_ok = True
    if group_column:
        train_g = set(df[group_column].iloc[train_idx])
        val_g = set(df[group_column].iloc[val_idx])
        test_g = set(df[group_column].iloc[test_idx])
        group_overlap_ok = not (
            (train_g & val_g) or (train_g & test_g) or (val_g & test_g)
        )

    time_range = {}
    if time_column:
        time_range = {
            "train": [str(df[time_column].iloc[train_idx].min()), str(df[time_column].iloc[train_idx].max())],
            "val": [str(df[time_column].iloc[val_idx].min()), str(df[time_column].iloc[val_idx].max())],
            "test": [str(df[time_column].iloc[test_idx].min()), str(df[time_column].iloc[test_idx].max())],
        }

    return SplitManifest(
        strategy=strategy,
        seed=seed,
        data_hash=data_hash,
        train_idx=sorted(int(i) for i in train_idx),
        val_idx=sorted(int(i) for i in val_idx),
        test_idx=sorted(int(i) for i in test_idx),
        target_distribution={
            "train": _target_distribution(y, np.array(train_idx)),
            "val": _target_distribution(y, np.array(val_idx)),
            "test": _target_distribution(y, np.array(test_idx)),
        },
        group_overlap_ok=group_overlap_ok,
        time_range=time_range,
    )


def _stratified_two_way(idx: np.ndarray, y: np.ndarray, frac: float, seed: int):
    """Split idx into (small_frac_part, remainder_part), stratified by y."""
    rng = np.random.RandomState(seed)
    small_parts = []
    remainder_parts = []
    for cls in np.unique(y):
        cls_idx = idx[y == cls]
        rng.shuffle(cls_idx)
        n_small = max(1, int(len(cls_idx) * frac)) if len(cls_idx) > 1 else 0
        small_parts.append(cls_idx[:n_small])
        remainder_parts.append(cls_idx[n_small:])
    small = np.concatenate(small_parts) if small_parts else np.array([], dtype=int)
    remainder = np.concatenate(remainder_parts) if remainder_parts else np.array([], dtype=int)
    return small, remainder


def make_cv_folds(
    df: pd.DataFrame,
    target_column: str,
    n_splits: int = 5,
    seed: int = 42,
    group_column: Optional[str] = None,
):
    """
    Returns a list of (train_idx, val_idx) tuples for cross-validation,
    to be used *within* the training portion only (after test set is
    already carved out by make_split). Never call this on the full
    dataset including the locked test set.
    """
    y = df[target_column].values
    if group_column:
        groups = df[group_column].values
        splitter = GroupKFold(n_splits=n_splits)
        return list(splitter.split(df, y, groups=groups))
    try:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(splitter.split(df, y))
    except ValueError:
        # falls back if a class has fewer members than n_splits
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(splitter.split(df))
