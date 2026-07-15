"""
Long-format, grouped time-series -> flight/episode-level tabular rollup.

Some datasets don't arrive as one row per labeled example — they arrive as
many timesteps per example (a flight's sensor trace, a session's event
log), concatenated into one long file and identified by an id column, with
examples further grouped by a higher-level entity (a plane, a customer)
that a group-aware split must respect. A standard row-per-example
train/val/test split can't run directly against data shaped like that.

This module is the deterministic (no LLM, no agent) fix: it rolls up N
timesteps sharing an id into exactly one row of summary-statistic features
per channel, so the output is an ordinary tabular table the rest of this
pipeline (intake/profiler/modeling/harness) already knows how to handle
unchanged, via `--target/--group-column/--strategy group`. Same trust-
boundary reasoning as harness/dataset.py: this is data preparation, not a
modeling decision, so no agent is involved and nothing here is a
"proposal" that needs re-validation.

Deliberately generic: no dataset-specific column names or label
vocabulary live here (see scripts/featurize_ngafid_flights.py for the
NGAFID-specific adapter that supplies those) — this module only knows
"roll up rows sharing an id column into per-channel stats," which is
reusable for any long-format grouped time-series dataset.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd

CHANNEL_FEATURE_KEYS = [
    "mean", "std", "min", "max", "range", "p10", "p50", "p90",
    "slope", "mean_abs_diff", "max_abs_diff", "last",
]


def channel_features(x: np.ndarray) -> dict:
    """Computes 12 summary statistics for one sensor/value channel over
    one example's timesteps. NaN/Inf values are dropped before computing
    (sensor dropout is common in real time-series data); an all-missing
    or empty channel degrades to zeros rather than raising, since a
    dropped-out channel is still valid input to a downstream classifier."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if x.size == 0:
        return {k: 0.0 for k in CHANNEL_FEATURE_KEYS}
    if x.size == 1:
        v = float(x[0])
        return {"mean": v, "std": 0.0, "min": v, "max": v, "range": 0.0,
                "p10": v, "p50": v, "p90": v, "slope": 0.0,
                "mean_abs_diff": 0.0, "max_abs_diff": 0.0, "last": v}

    t = np.arange(x.size)
    slope = float(np.polyfit(t, x, 1)[0])
    d = np.abs(np.diff(x))

    return {
        "mean": float(np.mean(x)), "std": float(np.std(x)),
        "min": float(np.min(x)), "max": float(np.max(x)),
        "range": float(np.max(x) - np.min(x)),
        "p10": float(np.percentile(x, 10)), "p50": float(np.percentile(x, 50)),
        "p90": float(np.percentile(x, 90)), "slope": slope,
        "mean_abs_diff": float(d.mean()), "max_abs_diff": float(d.max()),
        "last": float(x[-1]),
    }


def resolve_label(value, label_map: dict) -> int:
    """Maps a raw label value to an integer via `label_map` (case-
    insensitive string match). Raises on anything unmapped rather than
    silently coercing, since a mislabeled row here corrupts the target
    column, not a feature."""
    key = str(value).strip().lower()
    if key not in label_map:
        raise ValueError(
            f"Unrecognised label value {value!r}; known values: "
            f"{sorted(label_map)}. Extend label_map to cover it."
        )
    return int(label_map[key])


def _iter_flights(
    csv_path: str | Path,
    value_columns: list[str],
    id_column: str,
    group_column: Optional[str],
    label_column: Optional[str],
    extra_columns: list[str],
    chunksize: int,
) -> Iterator[tuple]:
    """Streams a large CSV chunk-by-chunk, yielding one
    (id, group_value, label_value, extra_values, value_df) tuple per
    contiguous run of `id_column`. Rows for a single id MUST be
    contiguous in the file (the common case for a file that was written
    one example at a time); a non-contiguous id raises rather than
    silently under-aggregating that example. A chunk boundary can split
    one id's rows across two reads — the trailing partial run of the
    current chunk is carried over and prepended to the next chunk before
    grouping, so no id is ever double-counted or dropped at a boundary."""
    usecols = list(value_columns) + [id_column]
    if group_column:
        usecols.append(group_column)
    if label_column:
        usecols.append(label_column)
    usecols += [c for c in extra_columns if c not in usecols]

    carry = None
    seen: set = set()
    reader = pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize)

    def _emit(fid, g):
        if fid in seen:
            raise ValueError(
                f"id {fid!r} is non-contiguous in {csv_path}; sort the "
                f"file by '{id_column}' first."
            )
        seen.add(fid)
        group_val = g[group_column].iloc[0] if group_column else None
        label_val = g[label_column].iloc[0] if label_column else None
        extra_vals = {c: g[c].iloc[0] for c in extra_columns}
        return fid, group_val, label_val, extra_vals, g[value_columns]

    for chunk in reader:
        if carry is not None:
            chunk = pd.concat([carry, chunk], ignore_index=True)
            carry = None
        if len(chunk) == 0:
            continue

        # Group by contiguous-run position, not id value: two separate,
        # non-contiguous runs that happen to share an id value (e.g. id
        # 0's rows appear, then id 1's, then id 0 reappears) must stay
        # distinguishable so the non-contiguity check below can catch
        # them. A same-value-match on just the trailing rows (the
        # simpler approach this was first written with) would silently
        # merge such cases into one row instead of raising.
        ids = chunk[id_column].to_numpy()
        is_new_run = np.empty(len(ids), dtype=bool)
        is_new_run[0] = True
        is_new_run[1:] = ids[1:] != ids[:-1]
        run_ids = np.cumsum(is_new_run)
        last_run = run_ids[-1]

        for run_id, g in chunk.groupby(run_ids, sort=True):
            if run_id == last_run:
                # last run may continue into the next chunk; carry it
                # forward rather than emitting it as complete.
                carry = g
            else:
                yield _emit(g[id_column].iloc[0], g)

    if carry is not None and len(carry):
        yield _emit(carry[id_column].iloc[0], carry)


def build_flight_feature_table_streaming(
    csv_path: str | Path,
    sensor_columns: list[str],
    id_column: str = "id",
    group_column: Optional[str] = None,
    label_column: Optional[str] = None,
    label_map: Optional[dict] = None,
    extra_columns: Optional[list[str]] = None,
    chunksize: int = 500_000,
    max_flights: Optional[int] = None,
    progress_every: int = 500,
) -> pd.DataFrame:
    """Streams a long-format, id-grouped time-series CSV and rolls it up
    into one row per id: `{sensor}__{stat}` for every sensor x
    channel_features stat, `__n_steps`, the group/label/extra columns
    passed through (label resolved via `label_map` if given, else passed
    through as-is), and `id_column` itself. Memory use is bounded by
    `chunksize`, not file size — this is what makes multi-GB source
    files tractable."""
    extra_columns = extra_columns or []
    rows, n = [], 0

    for fid, group_val, label_val, extra_vals, value_df in _iter_flights(
        csv_path, sensor_columns, id_column, group_column, label_column,
        extra_columns, chunksize,
    ):
        feats = {}
        for c in sensor_columns:
            for k, v in channel_features(value_df[c].values).items():
                feats[f"{c}__{k}"] = v
        feats["__n_steps"] = int(len(value_df))
        feats[id_column] = fid
        if group_column:
            feats[group_column] = group_val
        if label_column:
            feats[label_column] = (
                resolve_label(label_val, label_map) if label_map is not None else label_val
            )
        feats.update(extra_vals)
        rows.append(feats)
        n += 1

        if progress_every and n % progress_every == 0:
            print(f"  ...featurized {n} rows")
        if max_flights and n >= max_flights:
            break

    table = pd.DataFrame(rows)
    feature_cols = [c for c in table.columns if "__" in c and c != "__n_steps"]
    if feature_cols:
        table[feature_cols] = table[feature_cols].fillna(0.0)
    return table


def extract_single_flight_raw(
    csv_path: str | Path,
    flight_id,
    sensor_columns: list[str],
    id_column: str = "id",
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Streams a raw long-format CSV and returns just one id's raw
    timestep rows, without loading the rest of a multi-GB file. Used by
    the deep-dive agent, which needs one flagged example's raw trace
    (not the rolled-up feature row build_flight_feature_table_streaming
    produces) to segment phases and localize anomalies. Reuses
    _iter_flights, so a non-contiguous id for the requested flight
    raises the same ValueError it would during full featurization,
    rather than silently returning a corrupted partial trace."""
    target = str(flight_id)
    for fid, _, _, _, value_df in _iter_flights(
        csv_path, sensor_columns, id_column, None, None, [], chunksize,
    ):
        if str(fid) == target:
            return value_df.reset_index(drop=True)
    raise ValueError(f"id {flight_id!r} not found in {csv_path}")
