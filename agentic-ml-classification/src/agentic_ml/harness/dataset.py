"""
Dataset loading, schema validation, and content hashing.

This module owns the *only* code path that reads raw dataset files.
Agents never receive raw file paths — they receive profiler summaries
and, later, in-memory X/y arrays handed to them by the harness.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class DatasetSpec:
    """Declares what a dataset is, before any modeling happens."""

    path: str
    target_column: str
    task: str = "binary_classification"
    id_columns: list[str] = field(default_factory=list)
    group_column: Optional[str] = None
    time_column: Optional[str] = None
    positive_label: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "target_column": self.target_column,
            "task": self.task,
            "id_columns": self.id_columns,
            "group_column": self.group_column,
            "time_column": self.time_column,
            "positive_label": self.positive_label,
        }


@dataclass
class LoadedDataset:
    df: pd.DataFrame
    spec: DatasetSpec
    data_hash: str

    @property
    def X(self) -> pd.DataFrame:
        drop_cols = [self.spec.target_column, *self.spec.id_columns]
        return self.df.drop(columns=[c for c in drop_cols if c in self.df.columns])

    @property
    def y(self) -> pd.Series:
        return self.df[self.spec.target_column]


def _hash_dataframe(df: pd.DataFrame) -> str:
    """
    Deterministic content hash. Sorted-column order + row-order-preserving
    byte hash, so the same file always produces the same hash regardless
    of which machine loaded it, but differs if row order changes (which
    matters for time-series datasets where row order is meaningful).
    """
    hasher = hashlib.sha256()
    hasher.update(",".join(sorted(df.columns.astype(str))).encode("utf-8"))
    # pandas.util.hash_pandas_object gives a fast, stable per-row hash
    row_hashes = pd.util.hash_pandas_object(df, index=False).values
    hasher.update(row_hashes.tobytes())
    return hasher.hexdigest()


def read_dataframe(path: str | Path) -> pd.DataFrame:
    """CSV/parquet reading, split out of load_dataset() so callers that
    need to look at a dataframe before a target_column/DatasetSpec exists
    yet (e.g. the intake step, which has to propose the target column in
    the first place) don't duplicate the format-branching logic."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    elif path.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported dataset format: {path.suffix}")


def load_dataset(spec: DatasetSpec) -> LoadedDataset:
    df = read_dataframe(spec.path)

    if spec.target_column not in df.columns:
        raise ValueError(
            f"target_column '{spec.target_column}' not found in dataset columns: "
            f"{list(df.columns)}"
        )

    for required_col in (spec.group_column, spec.time_column):
        if required_col and required_col not in df.columns:
            raise ValueError(f"Declared column '{required_col}' not found in dataset")

    data_hash = _hash_dataframe(df)
    return LoadedDataset(df=df, spec=spec, data_hash=data_hash)


def write_dataset_spec(spec: DatasetSpec, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(spec.to_dict(), indent=2))
