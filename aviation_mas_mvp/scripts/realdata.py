"""
realdata.py
===========
An adapter module designed to handle the large-scale NGAFID-MC dataset format.

Instead of individual per-flight files, this dataset aggregates flight data 
sequentially into unified CSV files demarcated by an `id` column. This module 
provides streaming utilities to build tabular features and slice subsets of flights 
without loading the multi-gigabyte files entirely into memory.
"""

from __future__ import annotations
from pathlib import Path
from typing import Iterator
import numpy as np
import pandas as pd

from .featurize import channel_features
from .synthdata import NGAFID_SENSORS  # canonical 22-sensor list

NGAFID_NON_SENSOR = {"id", "plane_id", "split", "date_diff", "before_after"}

# Normalize variations of positive and negative class representations
_LABEL_MAP = {
    "before": 1, "pre": 1, "pre-maintenance": 1, "1": 1, "true": 1,
    "after": 0, "post": 0, "post-maintenance": 0, "0": 0, "false": 0,
}


def resolve_label(value) -> int:
    """Maps varied string flags to a strict binary integer label (0 or 1)."""
    s = str(value).strip().lower()
    if s not in _LABEL_MAP:
        raise ValueError(
            f"Unrecognised before_after value {value!r}. Known: {sorted(set(_LABEL_MAP))}. "
            f"Add a mapping or pass label_col-appropriate values."
        )
    return _LABEL_MAP[s]


def peek_ngafid(csv_path: str | Path, nrows: int = 200_000) -> dict:
    """
    Performs a cheap inspection on the initial chunk of a large file 
    to validate column schemas, sensor counts, and data contiguity.
    """
    head = pd.read_csv(csv_path, nrows=nrows)
    cols = list(head.columns)
    sensors = [c for c in cols if c not in NGAFID_NON_SENSOR]
    ids = head["id"].values
    
    # Check for contiguity: individual flight records must be consecutive runs
    runs = np.sum(ids[1:] != ids[:-1]) + 1
    contiguous = runs == head["id"].nunique()
    
    return {
        "columns": cols,
        "n_sensor_cols": len(sensors),
        "sensors": sensors,
        "unexpected_sensors": [c for c in sensors if c not in NGAFID_SENSORS],
        "label_values_in_sample": sorted(map(str, head["before_after"].unique())),
        "label_map_preview": {str(v): resolve_label(v) for v in head["before_after"].unique()},
        "n_flights_in_sample": int(head["id"].nunique()),
        "n_planes_in_sample": int(head["plane_id"].nunique()),
        "id_contiguous_in_sample": bool(contiguous),
        "rows_per_flight_sample": int(np.median(head.groupby("id").size())),
        "note": "If id_contiguous_in_sample is False, sort the file by id before streaming.",
    }


def _iter_flights(csv_path: str | Path, sensors: list[str], id_col: str,
                  plane_col: str, label_col: str, split_col: str | None,
                  chunksize: int) -> Iterator[tuple]:
    """
    Streams a large CSV chunk-by-chunk and yields structural partitions per flight.
    Applies a carry-over mechanism to combine flights split across chunk boundaries.
    """
    usecols = sensors + [id_col, plane_col, label_col] + ([split_col] if split_col else [])
    dtypes = {c: "float32" for c in sensors}
    dtypes[plane_col] = "string"
    dtypes[label_col] = "string"
    if split_col:
        dtypes[split_col] = "string"

    carry = None
    seen: set = set()
    reader = pd.read_csv(csv_path, usecols=usecols, dtype=dtypes, chunksize=chunksize)
    
    for chunk in reader:
        # Prepend residual rows left over from the trailing edge of the previous chunk
        if carry is not None:
            chunk = pd.concat([carry, chunk], ignore_index=True)
            carry = None
            
        last_id = chunk[id_col].iloc[-1]
        is_last = chunk[id_col] == last_id
        
        # Isolate the incomplete final flight to evaluate with the next read pass
        carry = chunk[is_last]                       
        complete = chunk[~is_last]
        
        for fid, g in complete.groupby(id_col, sort=False):
            if fid in seen:
                raise ValueError(f"flight id {fid} is non-contiguous; sort the CSV by '{id_col}' first.")
            seen.add(fid)
            yield fid, g[plane_col].iloc[0], g[label_col].iloc[0], \
                (g[split_col].iloc[0] if split_col else None), g[sensors]
                
    # Flush any remaining rows following EOF
    if carry is not None and len(carry):
        fid = carry[id_col].iloc[0]
        if fid in seen:
            raise ValueError(f"flight id {fid} is non-contiguous; sort the CSV by '{id_col}' first.")
        yield fid, carry[plane_col].iloc[0], carry[label_col].iloc[0], \
            (carry[split_col].iloc[0] if split_col else None), carry[sensors]


def build_feature_table_streaming(
    csv_path: str | Path,
    sensors: list[str] | None = None,
    id_col: str = "id",
    plane_col: str = "plane_id",
    label_col: str = "before_after",
    split_col: str | None = "split",
    chunksize: int = 500_000,
    max_flights: int | None = None,
    progress_every: int = 500,
) -> pd.DataFrame:
    """
    Streams and aggregates continuous flight signals directly into a single unified 
    tabular feature table row-by-row, optimizing memory overhead.
    """
    csv_path = Path(csv_path)
    if sensors is None:
        header = pd.read_csv(csv_path, nrows=0)
        sensors = [c for c in header.columns if c not in NGAFID_NON_SENSOR]

    rows, n = [], 0
    for fid, plane, raw_label, split, sens_df in _iter_flights(
            csv_path, sensors, id_col, plane_col, label_col, split_col, chunksize):
        
        # Calculate localized 12-stat summary metrics for each extracted channel
        feats = {}
        for c in sensors:
            for k, v in channel_features(sens_df[c].values).items():
                feats[f"{c}__{k}"] = v
        feats["__n_steps"] = int(len(sens_df))
        feats["label"] = resolve_label(raw_label)
        feats["group"] = str(plane)
        feats["filename"] = str(fid)
        
        if split_col:
            feats["paper_split"] = str(split)
            
        rows.append(feats)
        n += 1
        
        if progress_every and n % progress_every == 0:
            print(f"  ...featurized {n} flights")
        if max_flights and n >= max_flights:
            break

    table = pd.DataFrame(rows).fillna(0.0)
    bal = table["label"].value_counts().to_dict()
    print(f"[realdata] {len(table)} flights | {table['group'].nunique()} tails | "
          f"balance {bal} | {len([c for c in table.columns if '__' in c])} features")
    return table


def export_flights_to_dir(
    csv_path: str | Path,
    out_dir: str | Path,
    max_flights: int | None = 50,
    id_col: str = "id",
    plane_col: str = "plane_id",
    label_col: str = "before_after",
    split_col: str | None = "split",
    only_fold: int | str | None = None,
    chunksize: int = 500_000,
) -> tuple[Path, Path]:
    """
    Splits out a subset of contiguous flights from the master CSV file and materializes 
    them as discrete, individual flight CSVs alongside an associated metadata index.
    """
    out_dir = Path(out_dir)
    flight_dir = out_dir / "flights"
    flight_dir.mkdir(parents=True, exist_ok=True)
    sensors = [c for c in pd.read_csv(csv_path, nrows=0).columns if c not in NGAFID_NON_SENSOR]
    meta, n = [], 0
    use_split_col = split_col if only_fold is not None else None
    
    for fid, plane, raw_label, split_val, sens_df in _iter_flights(
            csv_path, sensors, id_col, plane_col, label_col, use_split_col, chunksize):
        
        # Filter explicitly by evaluation partitions if instructed
        if only_fold is not None and str(split_val) != str(only_fold):
            continue
            
        fn = f"flight_{fid}.csv"
        sens_df.to_csv(flight_dir / fn, index=False)
        meta.append({"filename": fn, "label": resolve_label(raw_label),
                     "plane_id": str(plane)})
        n += 1
        if max_flights and n >= max_flights:
            break
            
    meta_path = out_dir / "metadata.csv"
    pd.DataFrame(meta).to_csv(meta_path, index=False)
    fold_note = f" (fold={only_fold} only)" if only_fold is not None else ""
    print(f"[realdata] exported {n} flights{fold_note} -> {flight_dir} (+ metadata.csv)")
    return flight_dir, meta_path


def fit_excluding_fold(
    table: pd.DataFrame,
    holdout_fold: int | str,
    model_out: str | Path,
    fold_col: str = "paper_split",
    seed: int = 0,
) -> dict:
    """
    Fits an isolated estimator while enforcing a complete structural holdout 
    on an isolated fold slice to allow validation on unexposed tail categories.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
    import joblib as _joblib

    if fold_col not in table.columns:
        raise KeyError(f"{fold_col!r} not in table; pass split_col to "
                       f"build_feature_table_streaming to carry it through.")
                       
    # Filter columns to only include feature variables
    cols = [c for c in table.columns if c not in {"label", "group", "filename", fold_col}
            and pd.api.types.is_numeric_dtype(table[c])]
            
    is_holdout = table[fold_col].astype(str) == str(holdout_fold)
    train_tbl, hold_tbl = table[~is_holdout], table[is_holdout]
    if hold_tbl.empty:
        raise ValueError(f"no rows with {fold_col}=={holdout_fold!r}; values present: "
                         f"{sorted(table[fold_col].unique())}")

    clf = HistGradientBoostingClassifier(random_state=seed)
    clf.fit(train_tbl[cols].values, train_tbl["label"].values.astype(int))
    
    # Generate probability maps against testing slices
    p = clf.predict_proba(hold_tbl[cols].values)[:, 1]
    y = hold_tbl["label"].values.astype(int)
    
    metrics = {
        "holdout_fold": str(holdout_fold),
        "n_train": int(len(train_tbl)), "n_holdout": int(len(hold_tbl)),
        "n_train_planes": int(train_tbl["group"].nunique()),
        "n_holdout_planes": int(hold_tbl["group"].nunique()),
        "holdout_roc_auc": float(roc_auc_score(y, p)) if len(set(y)) == 2 else None,
        "holdout_pr_auc": float(average_precision_score(y, p)) if len(set(y)) == 2 else None,
        "holdout_accuracy": float(accuracy_score(y, (p >= 0.5).astype(int))),
        "n_features": len(cols),
    }
    
    # Track cross-contamination between modeling partitions
    overlap = set(train_tbl["group"]) & set(hold_tbl["group"])
    metrics["train_holdout_plane_overlap"] = len(overlap)  
    
    _joblib.dump({"model": clf, "feature_columns": cols}, model_out)
    print(f"[realdata] fit excluding fold {holdout_fold}: train={metrics['n_train']} rows / "
          f"{metrics['n_train_planes']} planes, holdout={metrics['n_holdout']} rows / "
          f"{metrics['n_holdout_planes']} planes, plane overlap={metrics['train_holdout_plane_overlap']}, "
          f"holdout AUC={metrics['holdout_roc_auc']}")
    return metrics


def _main():
    import argparse, json
    ap = argparse.ArgumentParser(description="Offline real-data prep for NGAFID-MC big CSVs.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("peek");      p.add_argument("--csv", required=True); p.add_argument("--nrows", type=int, default=200_000)
    p = sub.add_parser("featurize"); p.add_argument("--csv", required=True); p.add_argument("--out", required=True)
    p.add_argument("--chunksize", type=int, default=500_000); p.add_argument("--max-flights", type=int, default=None)
    p = sub.add_parser("export");    p.add_argument("--csv", required=True); p.add_argument("--out-dir", required=True)
    p.add_argument("--max-flights", type=int, default=50)
    p.add_argument("--only-fold", type=str, default=None)

    a = ap.parse_args()
    if a.cmd == "peek":
        print(json.dumps(peek_ngafid(a.csv, a.nrows), indent=2))
    elif a.cmd == "featurize":
        t = build_feature_table_streaming(a.csv, chunksize=a.chunksize, max_flights=a.max_flights)
        t.to_csv(a.out, index=False)
        print(json.dumps({"rows": int(len(t)), "out": a.out}, indent=2))
    elif a.cmd == "export":
        fd, mp = export_flights_to_dir(
            a.csv, 
            a.out_dir, 
            max_flights=a.max_flights, 
            only_fold=a.only_fold
        )
        print(json.dumps({"flight_dir": str(fd), "metadata": str(mp)}, indent=2))


if __name__ == "__main__":
    _main()