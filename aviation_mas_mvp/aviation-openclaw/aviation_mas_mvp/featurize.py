"""
featurize.py
============
Converts multi-channel flight time series data into a tabular feature set. 
It computes summary statistics across each sensor channel per flight to create 
a fixed-width feature vector suitable for standard tabular machine learning models.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

# Columns to exclude from sensor channel detection
DEFAULT_NON_SENSOR = {"time", "timestamp", "seconds", "unixtime", "lcl_date",
                      "lcl_time", "utc_date", "utc_time", "id", "index"}


def channel_features(x: np.ndarray) -> dict:
    """Computes summary statistics for a single sensor channel."""
    x = np.asarray(x, dtype=float)
    
    # Filter out NaN/Inf values
    x = x[np.isfinite(x)]
    
    keys = ["mean", "std", "min", "max", "range", "p10", "p50", "p90",
            "slope", "mean_abs_diff", "max_abs_diff", "last"]
            
    # Handle edge cases with insufficient data
    if x.size == 0:
        return {k: 0.0 for k in keys}
    if x.size == 1:
        v = float(x[0])
        return {"mean": v, "std": 0.0, "min": v, "max": v, "range": 0.0,
                "p10": v, "p50": v, "p90": v, "slope": 0.0,
                "mean_abs_diff": 0.0, "max_abs_diff": 0.0, "last": v}
                
    # Calculate time-series specific metrics (trends and deltas)
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


def detect_channels(df: pd.DataFrame, non_sensor=DEFAULT_NON_SENSOR) -> list[str]:
    """Identifies numeric columns that are likely sensor data."""
    out = []
    for c in df.columns:
        if c.strip().lower() in non_sensor:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            out.append(c)
    return out


def featurize_flight(df: pd.DataFrame, channels: list[str] | None = None) -> dict:
    """Converts a flight DataFrame into a flat dictionary of features."""
    if channels is None:
        channels = detect_channels(df)
        
    feats = {}
    for c in channels:
        for k, v in channel_features(df[c].values).items():
            # Flatten the nested structure (e.g., 'altitude__mean')
            feats[f"{c}__{k}"] = v
            
    # Track total sequence length for structural insight
    feats["__n_steps"] = int(len(df))
    return feats


def build_feature_table(
    flight_dir: str | Path,
    metadata: pd.DataFrame,
    file_col: str = "filename",
    label_col: str = "label",
    group_col: str = "plane_id",
    channels: list[str] | None = None,
    max_flights: int | None = None,
) -> pd.DataFrame:
    """
    Processes a directory of flight CSVs and merges them with metadata
    to create a comprehensive tabular dataset for training/evaluation.
    """
    flight_dir = Path(flight_dir)
    rows = []
    md = metadata
    
    if max_flights:
        md = md.head(max_flights)

    # Lock channel set from the first readable flight so all rows align perfectly
    locked = channels
    
    for _, m in md.iterrows():
        fp = flight_dir / str(m[file_col])
        if not fp.exists():
            continue
            
        try:
            df = pd.read_csv(fp)
        except Exception:
            continue
            
        if locked is None:
            locked = detect_channels(df)
            
        # Extract features and append metadata labels
        feats = featurize_flight(df, channels=locked)
        feats["label"] = int(m[label_col])
        feats["group"] = m[group_col]
        feats["filename"] = str(m[file_col])
        rows.append(feats)

    # Convert to DataFrame and sanitize any lingering missing data
    table = pd.DataFrame(rows).fillna(0.0)
    return table