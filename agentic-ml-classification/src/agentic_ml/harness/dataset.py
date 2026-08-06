"""
Dataset loading, schema validation, and content hashing.

This module owns the *only* code path that reads raw dataset files.
Agents never receive raw file paths — they receive profiler summaries
and, later, in-memory X/y arrays handed to them by the harness.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

# read_dataframe() is the single entry point every script/notebook/step
# uses to load a dataset — a raw long-format time-series file (one row
# per timestep, not per example) pointed at this pipeline directly
# will silently try to load the whole thing into one DataFrame, which
# is exactly what happened running datasets/raw/C28.csv (4GB, 28.7M
# rows) through it: a real OOM that took the machine down, not just a
# slow load. This is a cheap stat()-based guard, checked before any
# read is attempted, so a too-large file fails fast with a clear
# message instead of silently exhausting memory. Overridable per-call
# or via AGENTIC_ML_MAX_DATASET_BYTES for a dataset that's legitimately
# this large (e.g. a real production-scale engineered table).
DEFAULT_MAX_DATASET_BYTES = 500_000_000  # 500 MB


def _max_dataset_bytes() -> int:
    override = os.environ.get("AGENTIC_ML_MAX_DATASET_BYTES")
    return int(override) if override else DEFAULT_MAX_DATASET_BYTES


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


def hash_dataframe(df: pd.DataFrame) -> str:
    """
    Deterministic content hash. Sorted-column order + row-order-preserving
    byte hash, so the same file always produces the same hash regardless
    of which machine loaded it, but differs if row order changes (which
    matters for time-series datasets where row order is meaningful).

    Public (not just load_dataset()'s internal helper) so callers that
    already have an in-memory dataframe from somewhere other than a
    plain file read — e.g. dynamic_loop.py's featurize_timeseries branch,
    which builds its table via harness/timeseries_features.py's streaming
    rollup rather than read_dataframe() — can hash it without a redundant
    disk re-read through load_dataset().
    """
    hasher = hashlib.sha256()
    hasher.update(",".join(sorted(df.columns.astype(str))).encode("utf-8"))
    # pandas.util.hash_pandas_object gives a fast, stable per-row hash
    row_hashes = pd.util.hash_pandas_object(df, index=False).values
    hasher.update(row_hashes.tobytes())
    return hasher.hexdigest()


# detect_dataset_shape()'s sample size: large enough to see many
# consecutive-run repeats even when a group (e.g. one flight) spans
# hundreds of rows, small enough that pd.read_csv(path, nrows=...) stays
# cheap and bounded regardless of total file size — this is the only
# thing safe to do before deciding whether the file is even safe to load
# in full via read_dataframe().
DEFAULT_SHAPE_SAMPLE_ROWS = 5000
DEFAULT_MIN_AVG_RUN_LENGTH = 5.0
# Below this, don't bother flagging long-format at all, regardless of run
# structure — a file this small was never going to cause the memory
# problem this detector exists to route around, and flagging it anyway
# is a pure false-positive risk with no offsetting safety benefit. This
# is what tells apart a genuinely huge sensor log from a small dataset
# that just happens to be pre-sorted by a low-cardinality category (e.g.
# Iris sorted by Species, 50-row contiguous blocks) — confirmed by
# testing against the real files this project uses: Titanic (61KB) and
# Iris (5KB) both fall far under this; the real NGAFID raw file (4.2GB)
# and even a deliberately small long-format fixture well past this
# threshold both fall well over it.
DEFAULT_MIN_BYTES_TO_FLAG = 5_000_000  # 5 MB


def detect_dataset_shape(
    path: str | Path,
    sample_rows: int = DEFAULT_SHAPE_SAMPLE_ROWS,
    min_avg_run_length: float = DEFAULT_MIN_AVG_RUN_LENGTH,
    min_bytes_to_flag: int = DEFAULT_MIN_BYTES_TO_FLAG,
) -> dict:
    """
    Cheap, streaming-safe peek at whether a CSV looks like already-
    tabular data (one row per example — Titanic, Iris) or raw long-
    format time-series data (many consecutive rows per example — e.g.
    NGAFID sensor logs, one row per timestep). Never loads more than
    `sample_rows` rows, so this is safe to call on a file of any size,
    unlike read_dataframe() (which this function exists specifically to
    be called BEFORE, so a routing decision can be made without ever
    attempting a full load of a file that turns out not to need one).

    Detection signal: for each sampled column, the average length of
    consecutive equal-value runs. A column that repeats the same value
    for many rows in a row (few "change points" relative to the sample
    size) is exactly the structural shape
    harness/timeseries_features.py's rollup engine already requires —
    it groups by contiguous runs of an id column and raises if a run is
    ever non-contiguous. An already-tabular dataset (one row per
    example) has no column that behaves this way; a long-format sensor
    log's id/group columns do. Gated by `min_bytes_to_flag` (see above)
    so a small dataset that happens to be sorted by a repeating category
    doesn't get misflagged — this isn't just a heuristic patch, it's the
    actual scope of the problem: file size is what determines whether
    skipping straight to read_dataframe() is even risky in the first
    place.

    This is a structural heuristic, not dataset-specific knowledge — it
    doesn't know what NGAFID's columns are named. A remaining false
    positive (a large, legitimately-tabular, pre-sorted dataset) just
    means an unnecessary featurization attempt gets proposed, which
    harness/timeseries_features.py's own non-contiguity check would then
    reject — not a silent wrong answer. This is not a substitute for
    read_dataframe()'s size guard, which still protects the actual load
    regardless of what this function decides.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    size = path.stat().st_size

    sample = pd.read_csv(path, nrows=sample_rows)
    n = len(sample)
    best_col: Optional[str] = None
    best_avg_run = 1.0
    for col in sample.columns:
        s = sample[col]
        n_unique = s.nunique(dropna=True)
        if n == 0 or n_unique <= 1 or n_unique >= n:
            continue  # constant, empty, or all-unique columns can't show run structure
        n_change_points = int((s != s.shift()).sum())
        avg_run_length = n / n_change_points if n_change_points else float(n)
        if avg_run_length > best_avg_run:
            best_col, best_avg_run = col, avg_run_length

    looks_long_format = size >= min_bytes_to_flag and best_avg_run >= min_avg_run_length
    return {
        "file_size_bytes": size,
        "sampled_rows": n,
        "repeated_run_column": best_col,
        "avg_run_length": best_avg_run,
        "looks_long_format": looks_long_format,
    }


def read_dataframe(path: str | Path, max_bytes: Optional[int] = None) -> pd.DataFrame:
    """CSV/parquet reading, split out of load_dataset() so callers that
    need to look at a dataframe before a target_column/DatasetSpec exists
    yet (e.g. the intake step, which has to propose the target column in
    the first place) don't duplicate the format-branching logic.

    max_bytes=None uses AGENTIC_ML_MAX_DATASET_BYTES / the 500MB default
    (see _max_dataset_bytes) — pass an explicit value to override just
    this call."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    limit = max_bytes if max_bytes is not None else _max_dataset_bytes()
    size = path.stat().st_size
    if size > limit:
        raise ValueError(
            f"Dataset {path} is {size / 1e6:.0f}MB, over the {limit / 1e6:.0f}MB limit "
            "for a single in-memory load — refusing to read it (this pipeline has no "
            "chunked/streaming path for general tabular data, so attempting this can "
            "exhaust memory rather than just being slow). If this is raw long-format "
            "time-series/sensor data (many rows per example, not one), roll it up into "
            "one row per example first — see scripts/featurize_ngafid_flights.py for a "
            "worked example — before handing it to this pipeline. If this file is "
            "legitimately meant to be loaded whole, raise the limit via the "
            "AGENTIC_ML_MAX_DATASET_BYTES env var or read_dataframe's max_bytes arg."
        )

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

    data_hash = hash_dataframe(df)
    return LoadedDataset(df=df, spec=spec, data_hash=data_hash)


def write_dataset_spec(spec: DatasetSpec, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(spec.to_dict(), indent=2))
